import os
import json
import argparse
from typing import List, Tuple

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
    all_data.describe()


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
