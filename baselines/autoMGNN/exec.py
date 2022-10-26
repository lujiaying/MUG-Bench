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
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing
import dgl
from scipy.sparse import coo_matrix
from ray import tune
from hyperopt import hp

from ..utils import get_exp_constraint, prepare_ag_dataset, IMG_COL, get_multiclass_metrics, EarlyStopping, get_exp_resource
from ..autogluon.exec import get_metric_names
from .models import MultiplexGNN


__version__ = '0.1'
CKPT_FNAME = 'model.pt'


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
                         ) -> Tuple[np.ndarray, np.ndarray, Tuple, Tuple]:
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


def train_mgnn(config: dict,
               multimodal_feats: Tuple[np.ndarray, np.ndarray, np.ndarray],
               data_batch: Dict[str, th.Tensor],
               all_labels: pd.Series, 
               num_classes: int, 
               masks_tuple: Tuple[np.ndarray, np.ndarray, np.ndarray],
               tab_feat_params: dict,
               eval_metric: str,
               num_epochs: int = 1000, 
               ):
    """
    Args:
        config: current trial configuration
            - "sim_graph_sparsity": float, expected sparsity of the generated similarity-based graph
            - "lr": float, learning rate for optimizer
    """
    # ===========
    # generate graphs
    tab_feats, text_feats, image_feats = multimodal_feats
    expect_sparsity = config['sim_graph_sparsity']
    tab_graph_Adj = construct_sim_based_graph(tab_feats, expect_sparsity)
    print(f'[INFO] sparsity of simlarity based graph from tabular features = {cal_sparsity(tab_graph_Adj)}')
    text_graph_Adj = construct_sim_based_graph(text_feats, expect_sparsity)
    print(f'[INFO] sparsity of simlarity based graph from text and categorical features = {cal_sparsity(text_graph_Adj)}')
    image_graph_Adj = construct_sim_based_graph(image_feats, expect_sparsity)
    print(f'[INFO] sparsity of simlarity based graph from image features = {cal_sparsity(image_graph_Adj)}')
    # ===========
    # create model
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    num_categs_per_feature = tab_feat_params['num_categs_per_feature']
    vector_dims = tab_feat_params['vector_dims']
    multiplex_gnn = MultiplexGNN(num_categs_per_feature, vector_dims, text_feats.shape[1], image_feats.shape[1], num_classes)
    multiplex_gnn.to(device)
    if config['optimizer'] == 'adamw':
        opt = th.optim.AdamW(multiplex_gnn.parameters())  # default lr=1e-3, weight_decay=1e-2
        early_stop_patience = 25
        scheduler = th.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=50, eta_min=1e-6)
    elif config['optimizer'] == 'sgd':
        # warm up with AdamW
        opt = th.optim.SGD(multiplex_gnn.parameters(), lr=0.1, momentum=0.9, nesterov=True)
        early_stop_patience = 40
        scheduler = th.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-2, total_steps=num_epochs)
    else:
        raise ValueError(f'[ERROR] Not support optimizer={config["optimizer"]}')
    loss_fn = th.nn.CrossEntropyLoss()

    # ===========
    # prepare data
    tab_g = dgl.add_self_loop(dgl.from_scipy(coo_matrix(tab_graph_Adj)))
    tab_g = tab_g.to(device)
    txt_g = dgl.add_self_loop(dgl.from_scipy(coo_matrix(text_graph_Adj)))
    txt_g = txt_g.to(device)
    img_g = dgl.add_self_loop(dgl.from_scipy(coo_matrix(image_graph_Adj)))
    img_g = img_g.to(device)
    train_mask, dev_mask, _ = masks_tuple
    train_labels = th.tensor(all_labels[train_mask]).to(device)
    dev_labels = th.tensor(all_labels[dev_mask]).to(device)
    train_mask = th.tensor(train_mask, dtype=th.bool).to(device)
    dev_mask = th.tensor(dev_mask, dtype=th.bool).to(device)
    label_space = np.unique(all_labels)

    # ===========
    # do training
    global CKPT_FNAME
    early_stop = EarlyStopping(patience=early_stop_patience, path=f'./{CKPT_FNAME}')
    for epoch in range(num_epochs):
        multiplex_gnn.train()
        logits = multiplex_gnn(data_batch, tab_g, txt_g, img_g, train_mask)
        loss = loss_fn(logits, train_labels)
        pred_probas = F.softmax(logits, dim=1)
        train_metric_scores = get_multiclass_metrics(train_labels.cpu().numpy(), pred_probas.detach().cpu().numpy(),
                                                     label_space)
        # if epoch % log_per_epoch == 0:
        #     print(f'[DEBUG] TRAIN {epoch=} loss={loss.item()}, metrics={train_metric_scores}')
        # backward propagation
        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()
        # do eval
        with th.no_grad():
            multiplex_gnn.eval()
            logits = multiplex_gnn(data_batch, tab_g, txt_g, img_g, dev_mask)
            loss = loss_fn(logits, dev_labels)
            pred_probas = F.softmax(logits, dim=1)
            val_metric_scores = get_multiclass_metrics(dev_labels.cpu().numpy(), pred_probas.detach().cpu().numpy(), 
                                                       label_space)
            early_stop(loss.item(), multiplex_gnn)
            val_score = val_metric_scores[eval_metric]
            train_score = train_metric_scores[eval_metric]
            if eval_metric == 'log_loss':
                val_score = -val_score   # log_loss the less the better
                train_score = -train_score
            tune.report(val_score=val_score, val_acc=val_metric_scores['accuracy'],
                        train_score=train_score, train_acc=train_metric_scores['accuracy'])
            # if epoch % log_per_epoch == 0:
            #     print(f'[DEBUG] VAL {epoch=} loss={loss.item()}, metrics={val_metric_scores}')
            #print(np.asarray(np.unique(preds[dev_mask].detach().cpu().numpy(), return_counts=True)).T)
        if early_stop.early_stop is True:
            print(f'[INFO] early stop at Epoch={epoch}')
            break


def do_test(config: dict,
            ckpt_path: str,
            multimodal_feats: Tuple[np.ndarray, np.ndarray, np.ndarray],
            data_batch: Dict[str, th.Tensor],
            all_labels: pd.Series, 
            num_classes: int, 
            masks_tuple: Tuple[np.ndarray, np.ndarray, np.ndarray],
            tab_feat_params: dict,
            ) -> dict:
    # ===========
    # generate graphs
    tab_feats, text_feats, image_feats = multimodal_feats
    expect_sparsity = config['sim_graph_sparsity']
    tab_graph_Adj = construct_sim_based_graph(tab_feats, expect_sparsity)
    print(f'[INFO] sparsity of simlarity based graph from tabular features = {cal_sparsity(tab_graph_Adj)}')
    text_graph_Adj = construct_sim_based_graph(text_feats, expect_sparsity)
    print(f'[INFO] sparsity of simlarity based graph from text and categorical features = {cal_sparsity(text_graph_Adj)}')
    image_graph_Adj = construct_sim_based_graph(image_feats, expect_sparsity)
    print(f'[INFO] sparsity of simlarity based graph from image features = {cal_sparsity(image_graph_Adj)}')
    # ===========
    # create model
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    num_categs_per_feature = tab_feat_params['num_categs_per_feature']
    vector_dims = tab_feat_params['vector_dims']
    multiplex_gnn = MultiplexGNN(num_categs_per_feature, vector_dims, text_feats.shape[1], image_feats.shape[1], num_classes)
    state_dict = th.load(ckpt_path)
    print(f'[INFO] load_state_dict from {ckpt_path}')
    multiplex_gnn.load_state_dict(state_dict)
    multiplex_gnn.to(device)
    # ===========
    # prepare data
    tab_g = dgl.add_self_loop(dgl.from_scipy(coo_matrix(tab_graph_Adj)))
    tab_g = tab_g.to(device)
    txt_g = dgl.add_self_loop(dgl.from_scipy(coo_matrix(text_graph_Adj)))
    txt_g = txt_g.to(device)
    img_g = dgl.add_self_loop(dgl.from_scipy(coo_matrix(image_graph_Adj)))
    img_g = img_g.to(device)
    _, _, test_mask = masks_tuple
    test_labels = th.tensor(all_labels[test_mask]).to(device)
    test_mask = th.tensor(test_mask, dtype=th.bool).to(device)
    label_space = np.unique(all_labels)
    # ===========
    # do test
    with th.no_grad():
        multiplex_gnn.eval()
        logits = multiplex_gnn(data_batch, tab_g, txt_g, img_g, test_mask)
        pred_probas = F.softmax(logits, dim=1)
        test_metric_scores = get_multiclass_metrics(test_labels.cpu().numpy(), pred_probas.detach().cpu().numpy(),
                                                    label_space)
    return test_metric_scores
    

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
    # ===========
    # Tab feature from ag Tabular AutoMLPipelineFeatureGenerator
    tab_feats, all_labels, masks_tuple, \
       (num_categs_per_feature, feature_arraycol_map, feature_type_map, num_classes) \
            = generate_tab_feature_by_tabular_pipeline(train_df, dev_df, test_df, col_label)
    vector_indices = [indices for ((_, indices), (_, f_type)) in zip(feature_arraycol_map.items(), feature_type_map.items()) if f_type == 'vector']
    vector_indices = list(itertools.chain(*vector_indices))
    vector_dims = len(vector_indices)
    tab_feat_params = dict(
            num_categs_per_feature=num_categs_per_feature,
            vector_dims=vector_dims,
            )
    print(f'[info] tabular feats shape={tab_feats.shape}, label shape={all_labels.shape}')
    assert ti_labels.tolist() == all_labels.tolist()
    # ===========
    # prepare data
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    data_batch = {}
    if vector_dims > 0:
        data_batch['vector'] = th.tensor(tab_feats[:, vector_indices]).to(device)
    if len(num_categs_per_feature) > 0:
        all_embed_feats = []
        for ((_, indices), (_, f_type)) in zip(feature_arraycol_map.items(), feature_type_map.items()):
            if f_type != 'embed':
                continue
            assert len(indices) == 1
            all_embed_feats.append(th.tensor(tab_feats[:, indices[0]], dtype=th.int32).to(device))
        data_batch['embed'] = all_embed_feats
    data_batch['text'] = th.tensor(text_feats).to(device)
    data_batch['image'] = th.tensor(image_feats).to(device)

    # ===========
    # Train model using HPO
    search_space = {
            "sim_graph_sparsity": tune.grid_search([0.5, 0.75, 0.9]),
            "optimizer": tune.grid_search(['adamw']),
            }
    ray_dir = os.path.join(args.exp_save_dir, 'ray_results')
    ts = time.time()
    # ray version=1.13.0 due to autogluon
    analysis = tune.run(
            tune.with_parameters(train_mgnn, 
                                 multimodal_feats=(tab_feats, text_feats, image_feats),
                                 data_batch=data_batch,
                                 all_labels=all_labels,
                                 num_classes=num_classes,
                                 masks_tuple=masks_tuple,
                                 tab_feat_params=tab_feat_params,
                                 eval_metric=eval_metric,
                                 ),
            config=search_space,
            time_budget_s=args.fit_time_limit,
            max_concurrent_trials=1,
            local_dir=ray_dir,
            resources_per_trial=get_exp_resource(),
            metric='val_score',
            mode='max',
            )
    te = time.time()
    training_duration = round(te-ts, 1)
    # Get a dataframe for the max accuracy seen for each trial
    dfs = analysis.dataframe(metric='val_score', mode='max')
    print(f'[INFO] Max val_score seen for each trial:')
    print(dfs)
    best_trial_config = analysis.get_best_config(metric='val_score', mode='max', scope='all')
    print(f'[INFO] best_trial_config={best_trial_config}')
    logdir = analysis.get_best_logdir(metric='val_score', mode="max", scope='all')
    print(f'[INFO] best logdir={logdir}')
    # do test
    ts = time.time()
    global CKPT_FNAME
    test_metric_scores = do_test(best_trial_config, 
                                 os.path.join(logdir, CKPT_FNAME),
                                 multimodal_feats=(tab_feats, text_feats, image_feats),
                                 data_batch=data_batch,
                                 all_labels=all_labels,
                                 num_classes=num_classes,
                                 masks_tuple=masks_tuple,
                                 tab_feat_params=tab_feat_params,
                                 )
    te = time.time()
    predict_duration = round(te-ts, 1)
    print(f'[DEBUG] {test_metric_scores=}')
    te_duration = time.time()
    duration = round(te_duration-ts_duration, 1)
    params_to_save = args.__dict__
    params_to_save['hpo_best_config'] = best_trial_config
    result = dict(
                task=info_dict['task'],
                framework=f'AutoMultiplexGNN',
                constraint=get_exp_constraint(args.fit_time_limit),
                type='multiclass',
                params=params_to_save,
                framework_version=__version__,
                utc=datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S'),
                duration=duration,
                training_duration=training_duration,
                predict_duration=predict_duration,
                seed=args.seed,
            )
    result.update(test_metric_scores)
    exp_result_save_path = os.path.join(args.exp_save_dir, 'results.csv')
    result_df = pd.DataFrame.from_records([result])
    result_df.to_csv(exp_result_save_path, index=False)
    print(f'[INFO] test result saved into {exp_result_save_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AutoGluon Multimodal predictor arguments to set")
    # required arguments
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--exp_save_dir', type=str, required=True)
    # optional arguments
    parser.add_argument('--fit_time_limit', type=int, default=3600,
            help="TabularPredictor.fit(). how long fit() should run for (wallclock time in seconds).")
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    print(f'[INFO] Exp arguments: {args}')
    main(args)
