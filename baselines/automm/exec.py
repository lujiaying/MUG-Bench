import os
import json
import argparse
import time
import random
from datetime import datetime 

import pandas as pd
from autogluon.multimodal import MultiModalPredictor, __version__
from sklearn.metrics import log_loss

from ..utils import get_exp_constraint, prepare_ag_dataset
from ..autogluon.exec import get_metric_names


def get_fit_hyperparameters(model_names: str) -> dict:
    if model_names == 'fusion':
        hyperparameters = {
                "model.names": ["hf_text", "timm_image", "clip", "categorical_mlp", "numerical_mlp", "fusion_mlp"],
                "optimization.max_epochs": 1000,
                }
    elif model_names == 'clip':
        hyperparameters = {
                "model.names": ["clip"],
                "data.categorical.convert_to_text": True,
                "data.numerical.convert_to_text": True,
                "optimization.max_epochs": 1000,
                }
    else:
        raise ValueError(f'Not support model_names={model_names}')
    return hyperparameters


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
    train_data, dev_data, test_data, feature_metadata = prepare_ag_dataset(args.dataset_dir)
    # prepare predictor
    model_save_dir = os.path.join(args.exp_save_dir, 'ag_ckpt')
    if args.do_load_ckpt:
        predictor = MultiModalPredictor.load(model_save_dir)
    else:
        hyperparameters = get_fit_hyperparameters(args.fit_setting)
        predictor = MultiModalPredictor(label=col_label, 
                                        path=model_save_dir, 
                                        eval_metric=eval_metric,
                                        )
        # do train
        ts = time.time()
        predictor.fit(train_data=train_data, 
                      tuning_data=dev_data, 
                      time_limit=args.fit_time_limit,
                      hyperparameters=hyperparameters,
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
    if 'log_loss' in test_metric_res:
        # mm_predictor log_loss has some issue
        y_pred_proba = predictor.predict_proba(test_data)
        test_metric_res['log_loss'] = log_loss(test_data[col_label], y_pred_proba)
    print(f'Test metrics={test_metric_res}')
    te_duration = time.time()
    if not args.do_load_ckpt:
        result = dict(
                task=info_dict['task'],
                framework=f'AutoMM-{args.fit_setting}',
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
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Which dataset to use. Expect a directory contains csvs and images.')
    parser.add_argument('--exp_save_dir', type=str, required=True,
                        help='the directory to save model checkpoints and exp result csv')
    # optional arguments
    # please refer to https://auto.gluon.ai/dev/tutorials/multimodal/beginner_multimodal.html
    parser.add_argument('--fit_setting', type=str, default='fusion',
                        choices=['fusion', 'clip'],
                        help="Use which models. `fusion` represents multimodal fusion method AutoMM; `clip` represent txt-img model CLIP. default=fusion.", 
                        )
    parser.add_argument('--fit_time_limit', type=int, default=3600,
            help="How long training should run for (wallclock time in seconds). default=3600 (1 hour)")
    parser.add_argument('--seed', type=int, default=0,
                        help="global random seed. default=0")
    parser.add_argument('--do_load_ckpt', action='store_true',
                        help='a flag. If set, model will be loaded from `exp_save_dir`, and training process will be skipped. default=False.')

    args = parser.parse_args()
    print(f'[INFO] Exp arguments: {args}')
    main(args)
