"""
1. extract embeddings using AutoGluon
2. construct multi-view graphs
3. use multiplex graph convolution 
    - tab MLP can refer to https://github.com/awslabs/autogluon/blob/master/multimodal/src/autogluon/multimodal/models/categorical_mlp.py
"""
import os
import json
import argparse
import time
import random
from datetime import datetime 
from typing import Tuple, Dict, Any
from collections import OrderedDict
import itertools

import pandas as pd
import numpy as np
import torch as th
import torch.nn.functional as F
from autogluon.tabular import TabularDataset, FeatureMetadata
from autogluon.core.utils import infer_problem_type
from autogluon.multimodal.data.infer_types import infer_column_types
from autogluon.core.constants import BINARY, MULTICLASS
from autogluon.multimodal import MultiModalPredictor
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from autogluon.multimodal.data import MultiModalFeaturePreprocessor
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import dgl
from scipy.sparse import coo_matrix

from ..utils import get_exp_constraint, prepare_ag_dataset, IMG_COL
from ..autogluon.exec import get_metric_names
from .models import TabEncoder, MLP, FUSION_HIDDEN_SIZE, MultiplexGNN

__version__ = '0.1'


def merge_train_dev_test_dfs(
        train_df: TabularDataset, 
        dev_df: TabularDataset, 
        test_df: TabularDataset,
        col_label: str
        ) -> Tuple[pd.DataFrame, pd.Series, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    all_data = pd.concat([train_df, dev_df, test_df])
    train_mask = np.zeros((all_data.shape[0]), dtype=bool)
    train_mask[:train_df.shape[0]] = 1
    assert train_mask.sum() == train_df.shape[0]
    dev_mask = np.zeros((all_data.shape[0]), dtype=bool)
    dev_mask[train_df.shape[0]:train_df.shape[0]+dev_df.shape[0]] = 1
    assert dev_mask.sum() == dev_df.shape[0]
    test_mask = np.zeros((all_data.shape[0]), dtype=bool)
    test_mask[train_df.shape[0]+dev_df.shape[0]:] = 1
    assert test_mask.sum() == test_df.shape[0]
    all_data, all_label = all_data.drop(columns=col_label), all_data[col_label]
    return all_data, all_label, (train_mask, dev_mask, test_mask)


def generate_tab_feature_by_tabular_pipeline(
                         train_data: TabularDataset, 
                         dev_data: TabularDataset, 
                         test_data: TabularDataset,
                         col_label: str,
                         ) -> Tuple[np.ndarray, pd.Series, Tuple, Tuple]:
    from autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch import TabularNeuralNetTorchModel
    auto_ml_pipeline_feature_generator = AutoMLPipelineFeatureGenerator(
            enable_text_special_features=False, 
            enable_text_ngram_features=False, 
            enable_vision_features=False,
            )
    auto_ml_pipeline_feature_generator.fit(train_data.drop(columns=col_label))
    auto_ml_pipeline_feature_generator.print_feature_metadata_info(log_level=40)
    train_feat = auto_ml_pipeline_feature_generator.transform(train_data.drop(columns=col_label))
    le = preprocessing.LabelEncoder()
    le.fit(train_data[col_label])
    train_label = le.transform(train_data[col_label])

    nn = TabularNeuralNetTorchModel()
    # set missing properties
    problem_type = infer_problem_type(y=train_data[col_label])
    assert problem_type in [BINARY, MULTICLASS]
    nn.problem_type = problem_type
    nn.quantile_levels = None
    # End set missing properties
    nn._set_default_params()
    params = nn._get_model_params()
    processor_kwargs, optimizer_kwargs, fit_kwargs, loss_kwargs, params = nn._prepare_params(params)
    nn._preprocess_set_features(train_feat)
    train_dataset, _ = nn._generate_datasets(train_feat, train_label, processor_kwargs)   # to create processor
    # train_dataset is a pytorch dataset
    print(f"Transformed features for Tabular-NN: "
          f"{train_dataset.num_examples} examples, {train_dataset.num_features} features "
          f"({len(train_dataset.feature_groups['vector'])} vector, {len(train_dataset.feature_groups['embed'])} embedding)")
    num_categs_per_feature = train_dataset.getNumCategoriesEmbeddings()
    # _types_of_features, dev_feat_post = nn._get_types_of_features(dev_feat)
    # print(_types_of_features)
    # print(dev_feat_post)

    all_data, all_label, masks_tuple = merge_train_dev_test_dfs(train_data, dev_data, test_data, col_label)
    all_feat = auto_ml_pipeline_feature_generator.transform(all_data)
    all_feat = nn.processor.transform(all_feat)
    all_label = le.transform(all_label)
    return all_feat, all_label, masks_tuple,\
            (num_categs_per_feature, nn.feature_arraycol_map, nn.feature_type_map, le.classes_.shape[0])


def generate_tab_feature_by_mm_pipeline(
        train_df: TabularDataset, 
        dev_df: TabularDataset, 
        test_df: TabularDataset,
        col_label: str,
        ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], Tuple, MultiModalFeaturePreprocessor]:
    """
    @note: deprecated due to poor performance
    """
    from autogluon.multimodal.utils import init_df_preprocessor, get_config
    column_types = infer_column_types(
            data=train_df,
            valid_data=dev_df,
            label_columns=col_label,
        )
    config = get_config()
    #config.data.categorical.minimum_cat_count = 2
    #config.data.categorical.maximum_num_cat = 256
    config.data.categorical.convert_to_text = False
    df_preprocessor = init_df_preprocessor(
            config=config.data,
            column_types=column_types,
            label_column=col_label,
            train_df_x=train_df.drop(columns=col_label),
            train_df_y=train_df[col_label],
        )
    # print(df_preprocessor.text_feature_names)
    print(f'[INFO] categorical features: {df_preprocessor.categorical_feature_names}')
    print(f'[INFO] numerical features: {df_preprocessor.numerical_feature_names}')
    all_data, all_label, masks_tuple = merge_train_dev_test_dfs(train_df, dev_df, test_df, col_label)
    all_cat_feat = df_preprocessor.transform_categorical(all_data)
    all_num_feat = df_preprocessor.transform_numerical(all_data)
    all_label = df_preprocessor.transform_label(all_data)
    return all_cat_feat, all_num_feat, all_label, (train_mask, dev_mask, test_mask), df_preprocessor


def pack_cate_text_cols_to_one_sent(elem: Dict[str, Any]) -> str:
    sentence = ''
    for k, v in elem.items():
        sentence = f'{sentence}{k}: {v}. '
    return sentence


def generate_text_image_feature_by_pretrained_CLIP(
         train_df: TabularDataset, 
         dev_df: TabularDataset, 
         test_df: TabularDataset,
         col_label: str,
         ) -> Tuple[np.ndarray, np.ndarray, pd.Series, Tuple]:
    mm_predictor = MultiModalPredictor(hyperparameters={"model.names": ["clip"]}, problem_type="zero_shot")
    column_types = infer_column_types(
            data=train_df.drop(columns=col_label),
            valid_data=dev_df.drop(columns=col_label),
        )
    print(f'[INFO] column_types={column_types} by infer_column_types()')
    text_cols = [_[0] for _ in column_types.items() if _[1] in ['text', 'categorical']]
    print(f'[INFO] Text raw input collected from columns: {text_cols}')
    all_data, all_label, masks_tuple = merge_train_dev_test_dfs(train_df, dev_df, test_df, col_label)
    text_cols_raw = all_data[text_cols].to_dict('records')
    text_cols_raw = [pack_cate_text_cols_to_one_sent(_) for _ in text_cols_raw]
    img_paths = all_data[IMG_COL].tolist()
    clip_embeddings = mm_predictor.extract_embedding({"image": img_paths, 'text': text_cols_raw})
    text_feats = clip_embeddings['text']
    image_feats = clip_embeddings['image']
    le = preprocessing.LabelEncoder()
    le.fit(train_df[col_label])
    all_label = le.transform(all_label)
    return text_feats, image_feats, all_label, masks_tuple


cal_sparsity = lambda A: 1.0 - ( np.count_nonzero(A) / A.size)

def construct_sim_based_graph(feats: np.ndarray, expect_sparsity: float = 0.5):
    """
    Returns a simlarity matrix removing diagonal elements (self-loop edges)
    """
    cosine_sim_matrix = cosine_similarity(feats)
    np.fill_diagonal(cosine_sim_matrix, 0.0)
    threshold = np.percentile(cosine_sim_matrix, expect_sparsity*100)
    cosine_sim_matrix[cosine_sim_matrix < threshold] = 0.0
    return cosine_sim_matrix


def main(args: argparse.Namespace):
    if not os.path.exists(args.exp_save_dir):
        os.makedirs(args.exp_save_dir)
    ts_duration = time.time()
    random.seed(args.seed)
    # load task configure
    with open(os.path.join(args.dataset_dir, 'info.json')) as fopen:
        info_dict = json.load(fopen)
    col_label = info_dict['label']
    eval_metric = info_dict['eval_metric']
    # load train, dev, test
    train_df, dev_df, test_df, feature_metadata = prepare_ag_dataset(args.dataset_dir)

    # ===========
    # Text and Image Emb from CLIP
    text_feats, image_feats, ti_labels, ti_masks_tuple = generate_text_image_feature_by_pretrained_CLIP(train_df, dev_df, test_df, col_label)
    print(f'[info] text feats shape={text_feats.shape}, image feats shape={image_feats.shape}')
    # TODO: HPO for graph construction...., sparsity level..
    expect_sparsity = 0.75
    text_graph_Adj = construct_sim_based_graph(text_feats, expect_sparsity)
    print(f'[INFO] sparsity of simlarity based graph from text and categorical features = {cal_sparsity(text_graph_Adj)}')
    image_graph_Adj = construct_sim_based_graph(image_feats, expect_sparsity)
    print(f'[INFO] sparsity of simlarity based graph from image features = {cal_sparsity(image_graph_Adj)}')

    # ===========
    # Tab feature from ag Tabular AutoMLPipelineFeatureGenerator
    tab_feats, all_labels, (train_mask, dev_mask, test_mask), \
       (num_categs_per_feature, feature_arraycol_map, feature_type_map, num_classes) \
            = generate_tab_feature_by_tabular_pipeline(train_df, dev_df, test_df, col_label)
    print(f'[info] tabular feats shape={tab_feats.shape}, label shape={all_labels.shape}')
    assert ti_labels.tolist() == all_labels.tolist()
    # print(np.asarray(np.unique(all_labels, return_counts=True)).T)
    tab_graph_Adj = construct_sim_based_graph(tab_feats, expect_sparsity)
    print(f'[INFO] sparsity of simlarity based graph from tabular features = {cal_sparsity(tab_graph_Adj)}')

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    # ===========
    # create model
    # create tab model
    vector_idcies = [indices for ((_, indices), (_, f_type)) in zip(feature_arraycol_map.items(), feature_type_map.items()) if f_type == 'vector']
    vector_idcies = list(itertools.chain(*vector_idcies))
    vector_dims = len(vector_idcies)
    multiplex_gnn = MultiplexGNN(num_categs_per_feature, vector_dims, text_feats.shape[1], image_feats.shape[1], num_classes)

    opt = th.optim.Adam(multiplex_gnn.parameters(), lr=3e-4)
    multiplex_gnn.to(device)

    # ===========
    # prepare data
    data_batch = {}
    if multiplex_gnn.tab_encoder.has_vector_features:
        data_batch['vector'] = th.tensor(tab_feats[:, vector_idcies]).to(device)
    if multiplex_gnn.tab_encoder.has_embed_features:
        all_embed_feats = []
        for ((_, indices), (_, f_type)) in zip(feature_arraycol_map.items(), feature_type_map.items()):
            if f_type != 'embed':
                continue
            assert len(indices) == 1
            all_embed_feats.append(th.tensor(tab_feats[:, indices[0]], dtype=th.int32).to(device))
        data_batch['embed'] = all_embed_feats
    data_batch['text'] = th.tensor(text_feats).to(device)
    data_batch['image'] = th.tensor(image_feats).to(device)
    tab_g = dgl.add_self_loop(dgl.from_scipy(coo_matrix(tab_graph_Adj)))
    tab_g = tab_g.to(device)
    txt_g = dgl.add_self_loop(dgl.from_scipy(coo_matrix(text_graph_Adj)))
    txt_g = txt_g.to(device)
    img_g = dgl.add_self_loop(dgl.from_scipy(coo_matrix(image_graph_Adj)))
    img_g = img_g.to(device)
    all_labels = th.tensor(all_labels).to(device)
    train_mask = th.tensor(train_mask, dtype=th.bool).to(device)
    dev_mask = th.tensor(dev_mask, dtype=th.bool).to(device)
    test_mask = th.tensor(test_mask, dtype=th.bool).to(device)

    for epoch in range(500):
        multiplex_gnn.train()
        logits = multiplex_gnn(data_batch, tab_g, txt_g, img_g)
        loss = F.cross_entropy(logits[train_mask], all_labels[train_mask])
        _, preds = th.max(logits, dim=1)
        acc = accuracy_score(all_labels[train_mask].cpu(), preds[train_mask].cpu())
        if epoch % 20 == 0:
            print(f'[DEBUG] {epoch=} {loss.item()=}, {acc=}')
        # backward propagation
        opt.zero_grad()
        loss.backward()
        opt.step()
        # do eval
        if epoch % 20 == 0:
            multiplex_gnn.eval()
            logits = multiplex_gnn(data_batch, tab_g, txt_g, img_g)
            _, preds = th.max(logits, dim=1)
            acc = accuracy_score(all_labels[test_mask].cpu(), preds[test_mask].cpu())
            print(f'[DEBUG] {epoch=} test_acc={acc}')
            #print(np.asarray(np.unique(preds[dev_mask].detach().cpu().numpy(), return_counts=True)).T)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AutoGluon Multimodal predictor arguments to set")
    # required arguments
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--exp_save_dir', type=str, required=True)
    # optional arguments
    parser.add_argument('--do_load_ckpt', action='store_true')
    # please refer to https://auto.gluon.ai/dev/tutorials/multimodal/beginner_multimodal.html
    parser.add_argument('--fit_time_limit', type=int, default=3600,
            help="TabularPredictor.fit(). how long fit() should run for (wallclock time in seconds).")
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    print(f'[INFO] Exp arguments: {args}')
    main(args)
