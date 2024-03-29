import os
import json
import argparse
import time
import random
from datetime import datetime 
from typing import List, Tuple

import pandas as pd
from autogluon.tabular import TabularPredictor, TabularDataset, __version__, FeatureMetadata
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config

from ..utils import get_exp_constraint, prepare_ag_dataset


def get_metric_names(problem_type: str) -> List[str]:
    """
    All metrics that need to calculate for particular problem_type
    """
    if problem_type == 'binary':
        return ['accuracy', 'roc_auc', 'f1', 'log_loss']
    elif problem_type == 'multiclass':
        return ['accuracy', 'balanced_accuracy', 'mcc', 'log_loss']
    else:
        raise ValueError(f'problem_type={problem_type} get_metric_names() NOT implemented yet')


def my_get_hyperparameter_config(fit_hyperparameters: str) -> dict:
    if fit_hyperparameters == 'default':
        hyperparameters = get_hyperparameter_config(fit_hyperparameters)
        return hyperparameters
    elif fit_hyperparameters == 'multimodal':
        hyperparameters = get_hyperparameter_config(fit_hyperparameters)
        return hyperparameters
    elif fit_hyperparameters == 'GBMLarge':
        hyperparameters = {'GBM': ['GBMLarge']}
        return hyperparameters
    elif fit_hyperparameters == 'tabMLP':
        hyperparameters = {'NN_TORCH': {}}
        return hyperparameters
    else:
        raise ValueError(f'fit_hyperparameters-name={fit_hyperparameters} my_get_hyperparameter_config() NOT implemented yet')


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
    include_image_col_flag = args.fit_hyperparameters == 'multimodal'
    train_data, dev_data, test_data, feature_metadata = prepare_ag_dataset(args.dataset_dir, include_image_col_flag)
    # prepare predictor
    model_save_dir = os.path.join(args.exp_save_dir, 'ag_ckpt')
    # use TabularPredictor
    if args.do_load_ckpt:
        predictor = TabularPredictor.load(model_save_dir)
    else:
        predictor = TabularPredictor(label=col_label, path=model_save_dir, eval_metric=eval_metric)
        # do train
        ts = time.time()
        hyperparameters = my_get_hyperparameter_config(args.fit_hyperparameters)
        if args.fit_presets in ['high_quality', 'best_quality']:
            # high or best NOT support tuning_data
            train_data = pd.concat([train_data, dev_data])
            dev_data = None
        predictor.fit(train_data=train_data, tuning_data=dev_data, 
                hyperparameters=hyperparameters, 
                presets=args.fit_presets,
                time_limit=args.fit_time_limit,
                feature_metadata=feature_metadata,
                ag_args_ensemble=dict(fold_fitting_strategy='sequential_local'),
                )
        te = time.time()
        training_duration = te - ts
    predictor.leaderboard()
    # do test
    predictor.persist_models('all')   # load model from disk into memory
    metric_names = get_metric_names(predictor.problem_type)
    ts = time.time()
    # WARNING: leaderboard() actuall do predict with every trained models (which can be slower)
    # we have to use leaderboard() because we want to have extra_metrics
    leaderboard = predictor.leaderboard(test_data, extra_metrics=metric_names)
    te = time.time()
    predict_duration = te - ts
    best_model_row = leaderboard.set_index('model').loc[predictor.get_model_best()]
    test_metric_res = {m: best_model_row.loc[m] for m in metric_names}
    test_metric_res['log_loss'] = - test_metric_res['log_loss']   # ag use flipped log_loss, we should align with sklearn
    print(f'Test metrics={test_metric_res}')
    ## show feature importance
    # print(predictor.feature_importance(test_data))
    te_duration = time.time()
    if not args.do_load_ckpt:
        result = dict(
                task=info_dict['task'],
                framework=f'AutoGluon-{args.fit_hyperparameters}-{args.fit_presets}',
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
    parser = argparse.ArgumentParser(description="AutoGluon arguments to set")
    # required arguments
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Which dataset to use. Expect a directory contains csvs and images.')
    parser.add_argument('--exp_save_dir', type=str, required=True,
                        help='the directory to save model checkpoints and exp result csv')
    # optional arguments
    # Please refer to https://auto.gluon.ai/stable/api/autogluon.predictor.html#autogluon.tabular.TabularPredictor.fit
    parser.add_argument('--fit_hyperparameters', default='default', 
            choices=['default', 'multimodal', 'GBMLarge', 'tabMLP'],
            help="TabularPredictor.fit(). Choices include 'default', 'multimodal', 'GBMLarge', 'tabMLP'")
    parser.add_argument('--fit_time_limit', type=int, default=3600,
            help="TabularPredictor.fit(). how long fit() should run for (wallclock time in seconds). default=3600 (1 hour)")
    parser.add_argument('--fit_presets', type=str, default='best_quality',
            help="TabularPredictor.fit(). Available Presets: [‘best_quality’, ‘high_quality’, ‘good_quality’, ‘medium_quality’, ‘optimize_for_deployment’, ‘interpretable’, ‘ignore_text’]")
    parser.add_argument('--seed', type=int, default=0,
                        help="global random seed. default=0")
    parser.add_argument('--do_load_ckpt', action='store_true',
                        help='a flag. If set, model will be loaded from `exp_save_dir`, and training process will be skipped. default=False.')

    args = parser.parse_args()
    print(f'[INFO] Exp arguments: {args}')
    main(args)
