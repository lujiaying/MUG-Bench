import os
import json
import argparse

import pandas as pd


def main(args: argparse.Namespace):
    train_df = pd.read_csv(os.path.join(args.dataset_dir, 'train.csv'))
    dev_df = pd.read_csv(os.path.join(args.dataset_dir, 'dev.csv'))
    test_df = pd.read_csv(os.path.join(args.dataset_dir, 'test.csv'))
    # criterion 0: `info.json` exists and is valid
    info_json_path = os.path.join(args.dataset_dir, 'info.json')
    if not os.path.exists(info_json_path):
        raise AssertionError(f'path={info_json_path} NOT exists!')
    with open(info_json_path) as fopen:
        info_dict = json.load(fopen)
    for key in ['task', 'label', 'id_col', 'eval_metric']:
        if key not in info_dict:
            raise AssertionError(f'{key} should EXIST in info_dict')
    id_col = info_dict['id_col']
    # criterion 1: no overlaps
    dev_overlap = dev_df[dev_df[id_col].isin(train_df[id_col])]
    if len(dev_overlap) > 0:
        print('*** Please remove the overlaps ***')
        print(dev_overlap)
        raise AssertionError('Above rows from **dev** EXIST in train!!')
    test_overlap = test_df[test_df[id_col].isin(train_df[id_col])]
    if len(test_overlap) > 0:
        print('*** Please remove the overlaps ***')
        print(test_overlap)
        raise AssertionError('Above rows from **test** EXIST in train!!')
    print('[PASS C1] No overlaps among train, dev, test sets')
    # criterion 2: same feature space; size split 80%, 5%, 15%
    train_len, train_feat_cnt = train_df.shape
    dev_len, dev_feat_cnt = dev_df.shape
    test_len, test_feat_cnt = test_df.shape
    assert train_feat_cnt == dev_feat_cnt == test_feat_cnt
    print('[PASS C2.a] train, dev, test sets have same feature space')
    total_len = train_len + dev_len + test_len
    print(f'{train_len=}, {dev_len=}, {test_len=}, {total_len=}')
    assert abs(train_len - int(total_len * 0.8)) <= 2
    assert abs(dev_len - int(total_len * 0.05)) <= 2
    assert abs(test_len - int(total_len * 0.15)) <= 2
    print('[PASS C2.b] train, dev, test sets follows 80%, 5%, 15% size split')
    # criterion 3: train and test set contains all labels (i.e. no zero shot)
    label = info_dict['label']
    train_labels = set(train_df[label].unique())
    dev_labels = set(dev_df[label].unique())
    test_labels = set(test_df[label].unique())
    total_labels = train_labels.union(dev_labels).union(test_labels)
    if train_labels != total_labels:
        print('[DEBUG] train_label')
        print(train_df[label].value_counts())
        print(pd.concat([train_df, dev_df, test_df])[label].value_counts())
        print('[DEBUG] all_label')
        print(pd.concat([train_df, dev_df, test_df])[label].value_counts())
        raise AssertionError(f'For label={label}, train set missing label={total_labels-train_labels} from total_labels')
    if test_labels != total_labels:
        print('[DEBUG] test_label')
        print(test_df[label].value_counts())
        print('[DEBUG] all_label')
        print(pd.concat([train_df, dev_df, test_df])[label].value_counts())
        raise AssertionError(f'For label={label}, test set missing label={total_labels-test_labels} from total_labels')
    print(f'[PASS C3] For label={label}, train and test set ALL contains every label value')
    # criterion 4: image path valid
    image_col = 'Image Path'
    for set_name, df in [('train', train_df), ('dev', dev_df), ('test', test_df)]:
        if image_col not in df:
            raise AssertionError(f'{image_col} NOT exist in {set_name}')
        for image_rel_path in df[image_col].tolist():
            if not os.path.exists(os.path.join(args.dataset_dir, image_rel_path)):
                raise AssertionError(f'{set_name} Image Path={image_rel_path} could not be located. ')
    print(f"[PASS C4] Every row's image path exists")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Validate whether a train, dev, test split is as expected")
    # required arguments
    parser.add_argument('--dataset_dir', type=str, required=True)

    args = parser.parse_args()
    print(f'[INFO] Input arguments: {args}')
    main(args)
