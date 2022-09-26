import os
import argparse

from scipy.stats import ttest_rel
import pandas as pd


def main(args: argparse.Namespace):
    dataset = pd.read_csv(args.rawdata_path)
    assert all(col in dataset.columns.tolist() for col in args.label_cols)
    dataset_shffule = dataset.sample(frac=1, random_state=args.seed)
    train_size = int(len(dataset) * 0.8)
    dev_size = int(len(dataset) * 0.05)
    train_data = dataset_shffule.iloc[:train_size]
    dev_data = dataset_shffule.iloc[train_size:train_size+dev_size]
    test_data = dataset_shffule.iloc[train_size+dev_size:]
    print(f'dataset.shape={dataset.shape}, train.shape={train_data.shape}, dev.shape={dev_data.shape}, test.shape={test_data.shape}')
    for col in args.label_cols:
        print(f'Now comparing col=`{col}`')
        all_dist = dataset[col].value_counts(normalize=True)
        train_dist = train_data[col].value_counts(normalize=True)
        dev_dist = dev_data[col].value_counts(normalize=True)
        test_dist = test_data[col].value_counts(normalize=True)
        # print(f'train vs all: {ttest_rel(train_dist, all_dist)}')
        cate_order = all_dist.index.tolist()
        all_dist = [all_dist.loc[cate] for cate in cate_order]
        train_dist = [train_dist.loc[cate] if cate in train_dist else 0.0 for cate in cate_order]
        dev_dist = [dev_dist.loc[cate] if cate in dev_dist else 0.0 for cate in cate_order]
        test_dist = [test_dist.loc[cate] if cate in test_dist else 0.0 for cate in cate_order]
        # print(all_dist, train_dist, dev_dist, test_dist)
        print(f'train vs all: {ttest_rel(train_dist, all_dist)}')
        print(f'dev vs all: {ttest_rel(dev_dist, all_dist)}')
        print(f'test vs all: {ttest_rel(test_dist, all_dist)}')
        print(f'train vs dev: {ttest_rel(train_dist, dev_dist)}')
        print(f'train vs test: {ttest_rel(train_dist, test_dist)}')
    # write to disk
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    train_data.to_csv(os.path.join(args.output_dir, 'train.csv'), index=False)
    dev_data.to_csv(os.path.join(args.output_dir, 'dev.csv'), index=False)
    test_data.to_csv(os.path.join(args.output_dir, 'test.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Split raw data into train, dev, test sets")
    # required arguments
    parser.add_argument('--rawdata_path', type=str, required=True)
    parser.add_argument('--label_cols', nargs='+', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    # optional arguments
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    print(f'[INFO] Input arguments: {args}')
    main(args)
