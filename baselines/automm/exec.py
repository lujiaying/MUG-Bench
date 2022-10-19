import os
import json
import argparse
import time
import random
from datetime import datetime 
from typing import List

import pandas as pd
from autogluon.tabular import TabularDataset, FeatureMetadata
from autogluon.multimodal import MultiModalPredictor, __version__

from ..utils import get_exp_constraint
from ..autogluon.exec import get_metric_names


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
    train_data = TabularDataset(os.path.join(args.dataset_dir, 'train.csv'))
    dev_data = TabularDataset(os.path.join(args.dataset_dir, 'dev.csv'))
    test_data = TabularDataset(os.path.join(args.dataset_dir, 'test.csv'))
    feature_metadata = FeatureMetadata.from_df(train_data)
    image_col = 'Image Path'
    image_id_to_path_func = lambda image_id: os.path.join(args.dataset_dir, image_id)
    train_data[image_col] = train_data[image_col].apply(image_id_to_path_func)
    dev_data[image_col] = dev_data[image_col].apply(image_id_to_path_func)
    test_data[image_col] = test_data[image_col].apply(image_id_to_path_func)
    feature_metadata = feature_metadata.add_special_types({image_col: ['image_path']})
    # prepare predictor
    model_save_dir = os.path.join(args.exp_save_dir, 'ag_ckpt')
    if args.do_load_ckpt:
        predictor = MultiModalPredictor.load(model_save_dir)
    else:
        predictor = MultiModalPredictor(label=col_label, path=model_save_dir, eval_metric=eval_metric)
        # do train
        ts = time.time()
        predictor.fit(train_data=train_data, tuning_data=dev_data, 
                time_limit=args.fit_time_limit,
                seed=args.seed,
                )
        te = time.time()
        training_duration = te - ts
    # do test
    metric_names = get_metric_names(predictor.problem_type)
    ts = time.time()
    test_metric_res = predictor.evaluate(test_data, metrics=metric_names)
    te = time.time()
    predict_duration = te - ts
    print(f'Test metrics={test_metric_res}')
    te_duration = time.time()
    if not args.do_load_ckpt:
        result = dict(
                task=info_dict['task'],
                framework=f'AutoMM',
                constraint=get_exp_constraint(args.fit_time_limit),
                type=predictor.problem_type,
                params=args.__dict__,
                framework_version=__version__,
                utc=datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S'),
                duration=round(te_duration-ts_duration, 1),
                training_duration=round(training_duration, 1),
                predict_duration=round(predict_duration, 1),
                seed=args.seed,
                )
        result.update(test_metric_res)
        exp_result_save_path = os.path.join(args.exp_save_dir, 'results.csv')
        result_df = pd.DataFrame.from_records([result])
        result_df.to_csv(exp_result_save_path, index=False)


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
