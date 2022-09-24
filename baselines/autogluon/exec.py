import os
import json
import argparse
import time
import random
from datetime import datetime 

import pandas as pd
from autogluon.tabular import TabularPredictor, TabularDataset, __version__, FeatureMetadata
from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config

from ..utils import get_exp_constraint


def main(args: argparse.Namespace):
    if not os.path.exists(args.exp_save_dir):
        os.makedirs(args.exp_save_dir)
    """
    # store exp arguments for reproducibility
    exp_arg_save_path = os.path.join(args.exp_save_dir, 'exp_arguments.json')
    if args.do_load_ckpt:
        with open(exp_arg_save_path) as fopen:
            exp_args_disk = json.load(fopen)
        print(f'[INFO] experiment arguments= {exp_args_disk}, load from {exp_arg_save_path}')
    else:
        with open(exp_arg_save_path, 'w') as fwrite:
            json.dump(args.__dict__, fwrite, indent=2)
        print(f'[INFO] experiment arguments store into {exp_arg_save_path}')
    """
    random.seed(args.seed)
    ts_duration = time.time()
    # load train, dev, test
    train_data = TabularDataset(os.path.join(args.dataset_dir, 'train.csv'))
    dev_data = TabularDataset(os.path.join(args.dataset_dir, 'dev.csv'))
    test_data = TabularDataset(os.path.join(args.dataset_dir, 'test.csv'))
    feature_metadata = FeatureMetadata.from_df(train_data)
    if args.fit_hyperparameters == 'multimodal':
        image_col = 'Image Path'
        image_id_to_path_func = lambda image_id: os.path.join(args.dataset_dir, image_id)
        train_data[image_col] = train_data[image_col].apply(image_id_to_path_func)
        dev_data[image_col] = dev_data[image_col].apply(image_id_to_path_func)
        test_data[image_col] = test_data[image_col].apply(image_id_to_path_func)
        feature_metadata = feature_metadata.add_special_types({image_col: ['image_path']})
    # prepare predictor
    model_save_dir = os.path.join(args.exp_save_dir, 'ag_ckpt')
    if args.do_load_ckpt:
        predictor = TabularPredictor.load(model_save_dir)
    else:
        predictor = TabularPredictor(label=args.col_label, path=model_save_dir)
        # do train
        ts = time.time()
        hyperparameters = get_hyperparameter_config(args.fit_hyperparameters)
        predictor.fit(train_data=train_data, tuning_data=dev_data, 
                hyperparameters=hyperparameters, presets=args.fit_presets,
                time_limit=args.fit_time_limit,
                feature_metadata=feature_metadata,
                )
        te = time.time()
        training_duration = te - ts
    predictor.leaderboard()

    # do test
    predictor.persist_models('best')   # load model from disk into memory
    ts = time.time()
    test_metric_res = predictor.evaluate(test_data)
    te = time.time()
    predict_duration = te - ts
    ## show feature importance
    print(predictor.feature_importance(test_data))
    te_duration = time.time()
    if not args.do_load_ckpt:
        result = dict(
                task=args.task_name,
                framework='AutoGluon',
                constraint=get_exp_constraint(args.fit_time_limit),
                type=predictor.problem_type,
                params=json.dumps(args.__dict__),
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
    parser = argparse.ArgumentParser(description="AutoGluon arguments to set")
    # required arguments
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--exp_save_dir', type=str, required=True)
    parser.add_argument('--col_label', type=str, required=True)
    parser.add_argument('--task_name', type=str, required=True)
    # optional arguments
    parser.add_argument('--do_load_ckpt', action='store_true')
    # Please refer to https://auto.gluon.ai/stable/api/autogluon.predictor.html#autogluon.tabular.TabularPredictor.fit
    parser.add_argument('--fit_hyperparameters', default='default', 
            help="TabularPredictor.fit(). Choices include 'default', 'multimodal'")
    parser.add_argument('--fit_time_limit', type=int, default=3600,
            help="TabularPredictor.fit(). how long fit() should run for (wallclock time in seconds).")
    parser.add_argument('--fit_presets', type=str, default='best_quality',
            help="TabularPredictor.fit(). Available Presets: [‘best_quality’, ‘high_quality’, ‘good_quality’, ‘medium_quality’, ‘optimize_for_deployment’, ‘interpretable’, ‘ignore_text’]")
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    print(f'[INFO] Exp arguments: {args}')
    main(args)
