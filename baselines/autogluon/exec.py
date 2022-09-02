import os
import argparse

from autogluon.tabular import TabularPredictor, TabularDataset


def main(args: argparse.Namespace):
    # load train, dev, test
    train_data = TabularDataset(os.path.join(args.dataset_dir, 'train.csv'))
    dev_data = TabularDataset(os.path.join(args.dataset_dir, 'dev.csv'))
    test_data = TabularDataset(os.path.join(args.dataset_dir, 'test.csv'))
    # TODO: add support for GPU model (potentially one argument from parser)
    # prepare predictor
    if not os.path.exists(args.exp_save_dir):
        os.makedirs(args.exp_save_dir)
    model_save_dir = os.path.join(args.exp_save_dir, 'ag_ckpt')
    predictor = TabularPredictor(label=args.col_label, path=model_save_dir)
    
    # do train
    predictor.fit(train_data=train_data, tuning_data=dev_data, 
            hyperparameters=args.fit_hyperparameters, presets=args.fit_presets,
            time_limit=args.fit_time_limit,
            )
    # do test
    predictor.persist_models('all')   # load model from disk into memory
    predictor.leaderboard(test_data)

    # TODO: cal multiple eval metrics on test_data
    # TODO: save eval metrics into csv under args.exp_save_dir (recommend using pandas)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AutoGluon arguments to set")
    # required arguments
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--exp_save_dir', type=str, required=True)
    parser.add_argument('--col_label', type=str, required=True)
    # optional arguments
    # Please refer to https://auto.gluon.ai/stable/api/autogluon.predictor.html#autogluon.tabular.TabularPredictor.fit
    parser.add_argument('--fit_hyperparameters', default='default', 
            help="TabularPredictor.fit(). Choices include 'default', 'multimodal'")
    parser.add_argument('--fit_time_limit', type=int, default=3600,
            help="TabularPredictor.fit(). how long fit() should run for (wallclock time in seconds).")
    parser.add_argument('--fit_presets', type=str, default='best_quality',
            help="TabularPredictor.fit(). Available Presets: [‘best_quality’, ‘high_quality’, ‘good_quality’, ‘medium_quality’, ‘optimize_for_deployment’, ‘interpretable’, ‘ignore_text’]")

    args = parser.parse_args()
    print(f'Exp arguments: {args}')

    main(args)
