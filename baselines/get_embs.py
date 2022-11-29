import argparse
import random
import os
import json
from typing import Tuple
import numpy as np

from sklearn import preprocessing
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from autogluon.tabular import TabularDataset
from autogluon.multimodal.data.infer_types import infer_column_types
from PIL import Image

from .utils import prepare_ag_dataset
from .autoMGNN.exec import generate_tab_feature_by_tabular_pipeline, pack_cate_text_cols_to_one_sent


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
    print(f'{test_embs.shape=}')
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
    Then PCA into 25 * 3 dims
    """
    test_images = []
    for image_path in test_data['Image Path'].tolist():
        with Image.open(image_path) as im:
            test_image = np.array(im.resize((224, 224)).convert('RGB'))
            test_images.append(test_image)
    test_images = np.stack(test_images, axis=0)  # (N, width, height, 3)
    N = test_images.shape[0]
    pca_n_components = 25
    test_embs_all_channels = []
    for c in range(3):
        test_images_c = test_images[:, :, :, c].reshape(N, -1)
        test_images_c = preprocessing.normalize(test_images_c)
        pca = PCA(n_components=pca_n_components)
        test_embs_c = pca.fit_transform(test_images_c)  # (N, n_components)
        test_embs_all_channels.append(test_embs_c)
    test_embs = np.stack(test_embs_all_channels, axis=2).reshape(N, -1)
    print(f'final {test_embs.shape=}')
    test_labels = test_data[col_label].tolist()
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
    # tabular embs
    test_feats, test_labels = get_tabular_embs(train_df, dev_df, test_df, col_label)
    tab_feat_out_path = os.path.join(args.out_save_dir, 'tab_feats.tsv')
    save_to_tsv(test_labels, test_feats, tab_feat_out_path)
    """

    """
    # text embs
    text_feats, test_labels = get_text_embs(train_df, dev_df, test_df, col_label)
    text_feat_out_path = os.path.join(args.out_save_dir, 'txt_feats.tsv')
    save_to_tsv(test_labels, text_feats, text_feat_out_path)
    """

    image_feats, test_labels = get_image_embs(train_df, dev_df, test_df, col_label)
    image_feat_out_path = os.path.join(args.out_save_dir, 'img_feats.tsv')
    save_to_tsv(test_labels, image_feats, image_feat_out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Scripts for get embeddings from different modalities")
    # required arguments
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--out_save_dir', type=str, required=True)
    # optional arguments
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    print(f'[INFO] Exp arguments: {args}')
    main(args)
