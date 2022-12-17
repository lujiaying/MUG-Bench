import os
import json
import argparse
from typing import List, Tuple
from collections import Counter

import numpy as np
import pandas as pd

from baselines.utils import prepare_ag_dataset


def get_Shannon_equitability(all_labels: pd.Series) -> float:
    k = len(all_labels.unique())
    max_entropy = np.log(k)
    freqs = all_labels.value_counts(normalize=True).to_numpy()
    entropy = (-freqs * np.log(freqs)).sum()
    E = entropy / max_entropy
    print(f'{k=}')
    print(f'Shannon equitability = {E:.4f}')
    return E


def analysis_numerical_features(all_data: pd.DataFrame):
    with pd.option_context('display.max_columns', 40):
        # for Pokemon
        # print(f'{all_data["generation"].unique().shape=}')
        # # for CSGO
        # all_data['Min Price'] = pd.to_numeric(all_data['Min Price'].str.replace('$', '').str.replace(',', ''))
        # all_data['Max Price'] = pd.to_numeric(all_data['Max Price'].str.replace('$', '').str.replace(',', ''))
        # print(f'{all_data["Availability"].unique().shape=}')
        print(all_data.describe(include='all'))
        # print(pd.to_numeric(all_data['Price'], errors='coerce').describe())   # for LOL


def analysis_text_features(all_data: pd.DataFrame, txt_cols: List[str]):
    txt_cat = all_data[txt_cols].apply(lambda row: ' '.join(row.values.astype(str)), axis=1).tolist()
    word_counter = Counter()
    for words in txt_cat:
        length = len(words.strip().split(' '))
        word_counter[length] += 1
    print(word_counter)
    word_cnt_percent = {k: v / len(txt_cat) for k, v in word_counter.items()}
    # print(word_cnt_percent)
    for k in sorted(word_cnt_percent.keys()):
        print(f'({k},{round(word_cnt_percent[k]*100, 2)})', end=' ')
    print('')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # required arguments
    parser.add_argument('--dataset_dir', type=str, required=True)

    args = parser.parse_args()
    print(f'[INFO] Exp arguments: {args}')

    # load task configure
    with open(os.path.join(args.dataset_dir, 'info.json')) as fopen:
        info_dict = json.load(fopen)
    col_label = info_dict['label']

    # load train, dev, test
    train_df, dev_df, test_df, feature_metadata = prepare_ag_dataset(args.dataset_dir)
    all_data = pd.concat([train_df, dev_df, test_df])
    all_labels = all_data[col_label]
    # E = get_Shannon_equitability(all_labels)

    analysis_numerical_features(all_data)
    # CSGO
    # analysis_text_features(all_data, ['Skin Name'])
    # LOL
    # analysis_text_features(all_data, ['SkinName', 'Concept', 'Model', 'Particles' ,'Animations', 'Sounds', 'Release date'])
    # HS
    # analysis_text_features(all_data, ['id', 'name', 'artist', 'text', 'mechanics'])
    # PKM
    analysis_text_features(all_data, ['name', 'species', 'ability_1', 'ability_2', 'ability_hidden'])
