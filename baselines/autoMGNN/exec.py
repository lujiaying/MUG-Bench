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
from typing import Tuple, Dict
from collections import OrderedDict
import itertools

import pandas as pd
import numpy as np
import torch as th
import torch.nn.functional as F
from autogluon.tabular import TabularDataset, FeatureMetadata
from autogluon.core.utils import infer_problem_type
from autogluon.core.constants import BINARY, MULTICLASS
from autogluon.multimodal.data import MultiModalFeaturePreprocessor
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import dgl
from scipy.sparse import coo_matrix

from ..utils import get_exp_constraint, prepare_ag_dataset
from ..autogluon.exec import get_metric_names
from .models import tabGNN, TabEncoder, GCN

__version__ = '0.1'


def generate_tab_feature_by_tabular_pipeline(
                         train_data: TabularDataset, 
                         dev_data: TabularDataset, 
                         test_data: TabularDataset,
                         col_label: str,
                         ) -> Tuple[np.ndarray, np.ndarray, Tuple, Tuple]:
    from autogluon.features.generators import AutoMLPipelineFeatureGenerator
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

    all_data = pd.concat([train_data, dev_data, test_data])
    train_mask = np.zeros((all_data.shape[0]), dtype=bool)
    train_mask[:train_data.shape[0]] = 1
    assert train_mask.sum() == train_data.shape[0]
    dev_mask = np.zeros((all_data.shape[0]), dtype=bool)
    dev_mask[train_data.shape[0]:train_data.shape[0]+dev_data.shape[0]] = 1
    assert dev_mask.sum() == dev_data.shape[0]
    test_mask = np.zeros((all_data.shape[0]), dtype=bool)
    test_mask[train_data.shape[0]+dev_data.shape[0]:] = 1
    assert test_mask.sum() == test_data.shape[0]
    all_data, all_label = all_data.drop(columns=col_label), all_data[col_label]
    all_feat = auto_ml_pipeline_feature_generator.transform(all_data)
    all_feat = nn.processor.transform(all_feat)
    all_label = le.transform(all_label)
    return all_feat, all_label, (train_mask, dev_mask, test_mask),\
            (num_categs_per_feature, nn.feature_arraycol_map, nn.feature_type_map, le.classes_.shape[0])


def generate_tab_feature_by_mm_pipeline(
        train_df: TabularDataset, 
        dev_df: TabularDataset, 
        test_df: TabularDataset,
        col_label: str,
        ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], Tuple, MultiModalFeaturePreprocessor]:
    from autogluon.multimodal.utils import init_df_preprocessor, get_config
    from autogluon.multimodal.data.infer_types import infer_column_types
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
    all_data = pd.concat([train_df, dev_df, test_df])
    train_mask = np.zeros((all_data.shape[0]), dtype=np.int8)
    train_mask[:train_df.shape[0]] = 1
    assert train_mask.sum() == train_df.shape[0]
    dev_mask = np.zeros((all_data.shape[0]), dtype=np.int8)
    dev_mask[train_df.shape[0]:train_df.shape[0]+dev_df.shape[0]] = 1
    assert dev_mask.sum() == dev_df.shape[0]
    test_mask = np.zeros((all_data.shape[0]), dtype=np.int8)
    test_mask[train_df.shape[0]+dev_df.shape[0]:] = 1
    assert test_mask.sum() == test_df.shape[0]
    all_cat_feat = df_preprocessor.transform_categorical(all_data)
    all_num_feat = df_preprocessor.transform_numerical(all_data)
    all_label = df_preprocessor.transform_label(all_data)
    return all_cat_feat, all_num_feat, all_label, (train_mask, dev_mask, test_mask), df_preprocessor


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
    # Tab feature from ag Tabular AutoMLPipelineFeatureGenerator
    tab_feats, all_labels, (train_mask, dev_mask, test_mask), \
       (num_categs_per_feature, feature_arraycol_map, feature_type_map, num_classes) \
            = generate_tab_feature_by_tabular_pipeline(train_df, dev_df, test_df, col_label)
    print(f'[info] tabular feats shape={tab_feats.shape}, label shape={all_labels.shape}')
    # print(np.asarray(np.unique(all_labels, return_counts=True)).T)
    tab_graph_Adj = construct_sim_based_graph(tab_feats, expect_sparsity=0.5)
    print(f'[INFO] sparsity of simlarity based graph from tabular features = {cal_sparsity(tab_graph_Adj)}')

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    # ===========
    # create model
    vector_idcies = [indices for ((_, indices), (_, f_type)) in zip(feature_arraycol_map.items(), feature_type_map.items()) if f_type == 'vector']
    vector_idcies = list(itertools.chain(*vector_idcies))
    vector_dims = len(vector_idcies)
    tab_encoder = TabEncoder(num_categs_per_feature, vector_dims)
    gcn = GCN(tab_encoder.out_feats, num_classes)
    opt = th.optim.Adam(list(tab_encoder.parameters()) + list(gcn.parameters()), lr=3e-4)
    tab_encoder.to(device)
    gcn.to(device)

    # ===========
    # prepare data
    data_batch = {'vector': None, 'embed': None}
    if tab_encoder.has_vector_features:
        data_batch['vector'] = th.tensor(tab_feats[:, vector_idcies]).to(device)
    if tab_encoder.has_embed_features:
        all_embed_feats = []
        for ((_, indices), (_, f_type)) in zip(feature_arraycol_map.items(), feature_type_map.items()):
            if f_type != 'embed':
                continue
            assert len(indices) == 1
            all_embed_feats.append(th.tensor(tab_feats[:, indices[0]], dtype=th.int32).to(device))
        data_batch['embed'] = all_embed_feats
    tab_graph_Adj = coo_matrix(tab_graph_Adj)
    g = dgl.from_scipy(tab_graph_Adj, eweight_name='w')
    g.edata['w'] = g.edata['w'].float()
    g = dgl.add_self_loop(g)
    g = g.to(device)
    all_labels = th.tensor(all_labels).to(device)
    train_mask = th.tensor(train_mask, dtype=th.bool).to(device)
    dev_mask = th.tensor(dev_mask, dtype=th.bool).to(device)
    test_mask = th.tensor(test_mask, dtype=th.bool).to(device)

    print(all_labels[train_mask])
    for epoch in range(1000):
        tab_encoder.train()
        gcn.train()
        node_feats = tab_encoder(data_batch)
        logits = gcn(node_feats, g)
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
            tab_encoder.eval()
            gcn.eval()
            node_feats = tab_encoder(data_batch)
            logits = gcn(node_feats, g)
            _, preds = th.max(logits, dim=1)
            acc = accuracy_score(all_labels[test_mask].cpu(), preds[test_mask].cpu())
            print(f'[DEBUG] {epoch=} test_acc={acc}')
            #print(np.asarray(np.unique(preds[dev_mask].detach().cpu().numpy(), return_counts=True)).T)


    """
    # Tab feature from ag MultiModalFeaturePreprocessor
    # cat_feats, num_feats, all_labels, mask_tuple, df_preprocessor \
    #         = generate_tab_feature_by_mm_pipeline(train_df, dev_df, test_df, col_label)
    train_mask, dev_mask, test_mask = mask_tuple
    train_mask = train_mask.astype(bool)
    dev_mask = dev_mask.astype(bool)
    test_mask = test_mask.astype(bool)
    # print(np.asarray(np.unique(all_labels[col_label], return_counts=True)).T)
    all_labels = th.tensor(all_labels[col_label])
    tab_feats = np.stack(list(cat_feats.values()) + list(num_feats.values()), axis=1)   # shape=(N, num_cat_f+num_num_f)
    tab_graph_Adj = construct_sim_based_graph(tab_feats)
    print(f'[INFO] sparsity of simlarity based graph from tabular features = {cal_sparsity(tab_graph_Adj)}')

    # ===========
    # Create model
    num_categories = df_preprocessor.categorical_num_categories
    num_numerical_feats = len(df_preprocessor.numerical_feature_names)
    num_classes = len(df_preprocessor.label_generator.classes_)
    tab_gnn = tabGNN(num_categories, num_numerical_feats, num_classes)
    opt = th.optim.Adam(tab_gnn.parameters(), lr=3e-4)

    # ==========
    # Do Train
    batch = {}
    batch[tab_gnn.cat_mlp.categorical_key] = [th.tensor(_) for _ in cat_feats.values()]
    batch[tab_gnn.num_mlp.numerical_key] = th.tensor(np.stack(list(num_feats.values()), axis=1))
    tab_graph_Adj = coo_matrix(tab_graph_Adj)
    g = dgl.from_scipy(tab_graph_Adj, eweight_name='w')
    g.edata['w'] = g.edata['w'].float()
    g = dgl.add_self_loop(g)

    for epoch in range(50):
        tab_gnn.train()
        logits = tab_gnn(batch, g)
        loss = F.cross_entropy(logits[train_mask], all_labels[train_mask])
        _, preds = th.max(logits, dim=1)
        acc = accuracy_score(all_labels[train_mask], preds[train_mask])
        print(f'[DEBUG] {epoch=} {loss.item()=}, {acc=}')
        # backward propagation
        opt.zero_grad()
        loss.backward()
        opt.step()
        # do eval
        tab_gnn.eval()
        logits = tab_gnn(batch, g)
        _, preds = th.max(logits, dim=1)
        acc = accuracy_score(all_labels[dev_mask], preds[dev_mask])
        print(f'[DEBUG] {epoch=} val_acc={acc}')
        # print(np.asarray(np.unique(all_labels[dev_mask], return_counts=True)).T)
        print(np.asarray(np.unique(preds[dev_mask].detach().numpy(), return_counts=True)).T)
    """


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
