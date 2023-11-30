import argparse
import random
import os
import json
from typing import Tuple
import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.multimodal.data.infer_types import infer_column_types
from autogluon.multimodal import MultiModalPredictor
from PIL import Image
from ray import tune
import torch as th
import dgl

from .utils import prepare_ag_dataset
from .MuGNet.exec import generate_tab_feature_by_tabular_pipeline, pack_cate_text_cols_to_one_sent, prepare_graph_ingredients, CKPT_FNAME, construct_graph_from_features
from .MuGNet.models import MuGNet


def get_tabular_embs(
        train_data: TabularDataset, 
        dev_data: TabularDataset, 
        test_data: TabularDataset,
        col_label: str,
        ) -> Tuple[np.ndarray, list]:
    tab_feats, all_labels, masks_tuple, \
       (num_categs_per_feature, feature_arraycol_map, feature_type_map, num_classes) \
            = generate_tab_feature_by_tabular_pipeline(train_data, dev_data, test_data, col_label)
    print(feature_arraycol_map)
    print(feature_type_map)
    test_mask = masks_tuple[-1]
    test_feats = tab_feats[test_mask]
    # get original test labels
    test_labels = test_data[col_label].tolist()
    assert(len(test_feats) == len(test_labels))
    return test_feats, test_labels


def get_tabular_embs_from_model(
        model_ckpt_dir: str,
        test_data: TabularDataset,
        col_label: str,
        ) -> Tuple[np.ndarray, list]:
    predictor = TabularPredictor.load(model_ckpt_dir)
    # prepare data in predictor level
    labels_p = predictor.transform_labels(test_data[col_label])
    test_data_p = predictor.transform_features(test_data)
    # prepare data in model level
    model_obj = predictor._trainer.load_model('NeuralNetTorch')
    test_dataset_p = model_obj._process_test_data(df=model_obj.preprocess(test_data_p), labels=labels_p)
    test_dataloader = test_dataset_p.build_loader(512, 1, is_test=True)  # batch_size, worker
    model = model_obj.model
    model.eval()
    all_tab_feats = []
    with th.no_grad():
        for data_batch in test_dataloader:
            # get emb before Sequential
            input_data = []
            if model.has_vector_features:
                input_data.append(data_batch['vector'].to(model.device))
            if model.has_embed_features:
                embed_data = data_batch['embed']
                for i in range(len(model.embed_blocks)):
                    input_data.append(model.embed_blocks[i](embed_data[i].to(model.device)))
            if len(input_data) > 1:
                input_data = th.cat(input_data, dim=1)
            else:
                input_data = input_data[0]
            # get embs
            batch_tab_embs = model.main_block[:-1](input_data)
            all_tab_feats.append(batch_tab_embs.detach().cpu())
    all_tab_feats = th.cat(all_tab_feats, axis=0).numpy()
    # get original test labels
    test_labels = test_data[col_label].tolist()
    assert(len(all_tab_feats) == len(test_labels))
    return all_tab_feats, test_labels


def get_text_embs(
        train_data: TabularDataset, 
        dev_data: TabularDataset, 
        test_data: TabularDataset,
        col_label: str,
        ) -> Tuple[np.ndarray, list]:
    """
    Bag of words representation
    Then PCA into 50 dims
    """
    column_types = infer_column_types(
            data=train_data.drop(columns=col_label),
            valid_data=dev_data.drop(columns=col_label),
        )
    print(f'[INFO] column_types={column_types} by infer_column_types()')
    text_cols = [_[0] for _ in column_types.items() if _[1] in ['text', 'categorical']]
    text_cols_raw = test_data[text_cols].to_dict('records')
    text_cols_raw = [pack_cate_text_cols_to_one_sent(_) for _ in text_cols_raw]
    print(f'{len(text_cols_raw)=}')
    count_vect = CountVectorizer()
    test_bag_of_words = count_vect.fit_transform(text_cols_raw)
    print(f'{test_bag_of_words.shape=}')
    pca_n_components = 50
    if test_bag_of_words.shape[1] > pca_n_components:
        svd = TruncatedSVD(n_components=pca_n_components)
        test_embs = svd.fit_transform(test_bag_of_words)
    else:
        test_embs = test_bag_of_words
    print(f'text {test_embs.shape=}')
    test_labels = test_data[col_label].tolist()
    return test_embs, test_labels


def get_text_embs_from_model(
        model_ckpt_dir: str,
        test_data: TabularDataset,
        col_label: str,
        ) -> Tuple[np.ndarray, list]:
    predictor = MultiModalPredictor.load(model_ckpt_dir)
    test_embs = predictor.extract_embedding(test_data)
    print(f'text model {test_embs.shape=}')
    test_labels = test_data[col_label].tolist()
    return test_embs, test_labels


def get_image_embs(
        train_data: TabularDataset, 
        dev_data: TabularDataset, 
        test_data: TabularDataset,
        col_label: str,
        ) -> Tuple[np.ndarray, list]:
    """
    Get raw image features
    Then PCA into K * 3 dims
    """
    test_images = []
    for image_path in test_data['Image Path'].tolist():
        with Image.open(image_path) as im:
            test_image = np.array(im.resize((224, 224)).convert('RGB'))
            test_images.append(test_image)
    test_images = np.stack(test_images, axis=0)  # (N, width, height, 3)
    N = test_images.shape[0]
    pca_n_components = 30
    test_embs_all_channels = []
    for c in range(3):
        test_images_c = test_images[:, :, :, c].reshape(N, -1)
        test_images_c = preprocessing.normalize(test_images_c)
        pca = PCA(n_components=pca_n_components)
        test_embs_c = pca.fit_transform(test_images_c)  # (N, n_components)
        test_embs_all_channels.append(test_embs_c)
    test_embs = np.stack(test_embs_all_channels, axis=2).reshape(N, -1)
    print(f'final image {test_embs.shape=}')
    test_labels = test_data[col_label].tolist()
    return test_embs, test_labels


def get_image_embs_from_model(
        image_model_ckpt_dir: str,
        test_data: TabularDataset,
        col_label: str,
        ) -> Tuple[np.ndarray, list]:
    predictor = MultiModalPredictor.load(image_model_ckpt_dir)
    #image_paths = test_data['Image Path'].tolist()
    test_embs = predictor.extract_embedding(test_data)
    print(f'{test_embs.shape=}')
    test_labels = test_data[col_label].tolist()
    return test_embs, test_labels


def get_fused_raw_embs(out_save_dir: str) -> Tuple[np.ndarray, list]:
    tab_emb_path = os.path.join(out_save_dir, 'tab_feats.tsv')
    tab_df = pd.read_csv(tab_emb_path, sep='\t', header=None)
    txt_emb_path = os.path.join(out_save_dir, 'txt_feats.tsv')
    txt_df = pd.read_csv(txt_emb_path, sep='\t', header=None)
    img_emb_path = os.path.join(out_save_dir, 'img_feats.tsv')
    img_df = pd.read_csv(img_emb_path, sep='\t', header=None)
    # assertion
    assert len(tab_df) == len(txt_df) == len(img_df)
    tab_feats = tab_df.iloc[:, 1:].to_numpy()
    txt_feats = txt_df.iloc[:, 1:].to_numpy()
    img_feats = img_df.iloc[:, 1:].to_numpy()
    print(f'{tab_feats.shape=} {txt_feats.shape=} {img_feats.shape=}')
    fused_feats = np.concatenate((tab_feats, txt_feats, img_feats), axis=1)
    print(f'{fused_feats.shape=}')
    # pca = PCA(n_components=80)
    # fused_embs = pca.fit_transform(fused_feats)
    fused_embs = fused_feats
    print(f'{fused_embs.shape=}')
    test_labels = tab_df.iloc[:, 0].tolist()
    return fused_embs, test_labels


def get_fusion_embs_from_gnn_model(
        model_ckpt_dir: str,
        train_df: TabularDataset, 
        dev_df: TabularDataset, 
        test_df: TabularDataset,
        col_label: str,
        ) -> Tuple[np.ndarray, list]:
    analysis = tune.ExperimentAnalysis(model_ckpt_dir)
    best_trial_config = analysis.get_best_config(metric='val_score', mode='max', scope='all')
    print(f'[INFO] best_trial_config={best_trial_config}')
    logdir = analysis.get_best_logdir(metric='val_score', mode="max", scope='all')
    print(f'[INFO] best logdir={logdir}')
    # prepare data meets required format
    tab_feats, text_feats, image_feats,\
            all_labels, masks_tuple, tab_feat_params,\
            num_classes, data_batch\
            = prepare_graph_ingredients(train_df, dev_df, test_df, col_label)
    # prepare gnn model
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    num_categs_per_feature = tab_feat_params['num_categs_per_feature']
    vector_dims = tab_feat_params['vector_dims']
    multiplex_gnn = MultiplexGNN(num_categs_per_feature, vector_dims, text_feats.shape[1], image_feats.shape[1], num_classes)
    ckpt_path = os.path.join(logdir, CKPT_FNAME)
    state_dict = th.load(ckpt_path)
    print(f'[INFO] load_state_dict from {ckpt_path}')
    multiplex_gnn.load_state_dict(state_dict)
    print(multiplex_gnn)
    multiplex_gnn.to(device)
    # generate graphs
    print('Construct Test Graphs...')
    tab_graph_Adj = construct_graph_from_features(tab_feats, **best_trial_config['g_const'])
    text_graph_Adj = construct_graph_from_features(text_feats, **best_trial_config['g_const'])
    image_graph_Adj = construct_graph_from_features(image_feats, **best_trial_config['g_const'])
    tab_g = dgl.add_self_loop(dgl.from_scipy(tab_graph_Adj)).to(device)
    txt_g = dgl.add_self_loop(dgl.from_scipy(text_graph_Adj)).to(device)
    img_g = dgl.add_self_loop(dgl.from_scipy(image_graph_Adj)).to(device)
    _, _, test_mask = masks_tuple
    test_labels = th.tensor(all_labels[test_mask]).to(device)
    test_mask = th.tensor(test_mask, dtype=th.bool).to(device)
    # ===========
    # get embeddings
    multiplex_gnn.eval()
    with th.no_grad():
        embeddings = multiplex_gnn.extract_fused_embedding(data_batch, tab_g, txt_g, img_g)
        embeddings = embeddings[test_mask]
    test_embs = embeddings.detach().cpu().numpy()
    print(f'fused embedding {test_embs.shape=}')
    test_labels = test_df[col_label].tolist()
    return test_embs, test_labels


def save_to_tsv(
        test_labels: list,
        test_feats: np.ndarray,
        output_path: str
        ):
    with open(output_path, 'w') as fwrite:
        for label, feats in zip(test_labels, test_feats):
            out_line = str(label) + '\t'
            out_line += '\t'.join([str(f) for f in feats]) + '\n'
            fwrite.write(out_line)


def main(args: argparse.Namespace):
    random.seed(args.seed)
    if not os.path.exists(args.out_save_dir):
        os.makedirs(args.out_save_dir)
    # load task configure
    with open(os.path.join(args.dataset_dir, 'info.json')) as fopen:
        info_dict = json.load(fopen)
    col_label = info_dict['label']
    eval_metric = info_dict['eval_metric']
    # load train, dev, test
    train_df, dev_df, test_df, feature_metadata = prepare_ag_dataset(args.dataset_dir)

    """
    # tabular raw embs
    test_feats, test_labels = get_tabular_embs(train_df, dev_df, test_df, col_label)
    tab_feat_out_path = os.path.join(args.out_save_dir, 'tab_feats.tsv')
    save_to_tsv(test_labels, test_feats, tab_feat_out_path)
    """

    if args.tab_model_ckpt_dir:
        tab_feats, test_labels = get_tabular_embs_from_model(args.tab_model_ckpt_dir, test_df, col_label)
        tab_feat_out_path = os.path.join(args.out_save_dir, 'tabMLP_tab_feats.tsv')
        save_to_tsv(test_labels, tab_feats, tab_feat_out_path)

    """
    # text raw embs
    text_feats, test_labels = get_text_embs(train_df, dev_df, test_df, col_label)
    text_feat_out_path = os.path.join(args.out_save_dir, 'txt_feats.tsv')
    save_to_tsv(test_labels, text_feats, text_feat_out_path)
    """
    # image embs from trained model
    if args.text_model_ckpt_dir:
        text_feats, test_labels = get_text_embs_from_model(args.text_model_ckpt_dir, test_df, col_label)
        text_feat_out_path = os.path.join(args.out_save_dir, 'roberta_txt_feats.tsv')
        save_to_tsv(test_labels, text_feats, text_feat_out_path)

    """
    # image raw embs
    image_feats, test_labels = get_image_embs(train_df, dev_df, test_df, col_label)
    image_feat_out_path = os.path.join(args.out_save_dir, 'img_feats.tsv')
    save_to_tsv(test_labels, image_feats, image_feat_out_path)
    """

    # image embs from trained model
    if args.image_model_ckpt_dir:
        image_feats, test_labels = get_image_embs_from_model(args.image_model_ckpt_dir, test_df, col_label)
        image_feat_out_path = os.path.join(args.out_save_dir, 'vit_img_feats.tsv')
        save_to_tsv(test_labels, image_feats, image_feat_out_path)

    # fused raw embs
    fused_feats, test_labels = get_fused_raw_embs(args.out_save_dir)
    feat_out_path = os.path.join(args.out_save_dir, 'fused_raw_feats.tsv')
    save_to_tsv(test_labels, fused_feats, feat_out_path)

    # fusion embs from trained model
    if args.fusion_model_ckpt_dir:
        fused_feats, test_labels = get_fusion_embs_from_gnn_model(args.fusion_model_ckpt_dir, train_df, dev_df, test_df, col_label)
        fused_feat_out_path = os.path.join(args.out_save_dir, 'gnn_fusion_feats.tsv')
        save_to_tsv(test_labels, fused_feats, fused_feat_out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Scripts for get embeddings from different modalities")
    # required arguments
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--out_save_dir', type=str, required=True)
    # optional arguments
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--tab_model_ckpt_dir', type=str, default='')
    parser.add_argument('--image_model_ckpt_dir', type=str, default='')
    parser.add_argument('--text_model_ckpt_dir', type=str, default='')
    parser.add_argument('--fusion_model_ckpt_dir', type=str, default='')

    args = parser.parse_args()
    print(f'[INFO] Exp arguments: {args}')
    main(args)
