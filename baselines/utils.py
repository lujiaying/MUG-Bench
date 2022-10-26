import os
from typing import Final

import numpy as np
import torch


IMG_COL: Final[str] = 'Image Path'


def get_exp_constraint(time_limit_in_second: int) -> str:
    hour = int(round(time_limit_in_second/3600, 0))
    resource = get_exp_resource()
    cpu_count = resource['cpu']
    gpu_count = resource['gpu']
    constraint = f'{hour}h{cpu_count}c'
    if gpu_count > 0:
        constraint = f'{constraint}-{gpu_count}g'
    return constraint


def get_exp_resource() -> dict:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if visible_devices is not None:
        gpu_count = len(visible_devices.split(','))
    else:
        gpu_count = 0
    resource = dict(
            cpu=len(os.sched_getaffinity(0)),
            gpu=gpu_count,
            )
    return resource


def prepare_ag_dataset(dataset_dir: str, include_image_col: bool = True) -> tuple:
    from autogluon.tabular import TabularDataset, FeatureMetadata
    global IMG_COL
    train_data = TabularDataset(os.path.join(dataset_dir, 'train.csv'))
    dev_data = TabularDataset(os.path.join(dataset_dir, 'dev.csv'))
    test_data = TabularDataset(os.path.join(dataset_dir, 'test.csv'))
    feature_metadata = FeatureMetadata.from_df(train_data)
    if include_image_col:
        image_id_to_path_func = lambda image_id: os.path.join(dataset_dir, image_id)
        train_data[IMG_COL] = train_data[IMG_COL].apply(image_id_to_path_func)
        dev_data[IMG_COL] = dev_data[IMG_COL].apply(image_id_to_path_func)
        test_data[IMG_COL] = test_data[IMG_COL].apply(image_id_to_path_func)
        feature_metadata = feature_metadata.add_special_types({IMG_COL: ['image_path']})
    return train_data, dev_data, test_data, feature_metadata


def get_multiclass_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray, labels: list) -> dict:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, matthews_corrcoef, log_loss
    y_pred = np.argmax(y_pred_proba, axis=1)
    results = dict(
            accuracy=accuracy_score(y_true, y_pred),
            balanced_accuracy=balanced_accuracy_score(y_true, y_pred),
            mcc=matthews_corrcoef(y_true, y_pred),
            log_loss=log_loss(y_true, y_pred_proba, labels=labels)
            )
    return results


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


if __name__ == '__main__':
    time_limit = 3600
    constraint = get_exp_constraint(time_limit)
    print(f'[DEBUG] constraint={constraint} for input time_limit={time_limit}')
    print(f'[DEBUG] {get_exp_resource()=}')
