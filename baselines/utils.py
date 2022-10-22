import os
from py3nvml import py3nvml


def get_exp_constraint(time_limit_in_second: int) -> str:
    #cpu_count = psutil.cpu_count()
    cpu_count = len(os.sched_getaffinity(0))
    hour = int(round(time_limit_in_second/3600, 0))
    py3nvml.nvmlInit()
    device_count = py3nvml.nvmlDeviceGetCount()
    # TODO: cpu_count need to be specified by SBATCH..
    constraint = f'{hour}h{cpu_count}c'
    if device_count > 0:
        handle = py3nvml.nvmlDeviceGetHandleByIndex(0)
        constraint = f'{constraint}-{py3nvml.nvmlDeviceGetName(handle)}'
    return constraint


def prepare_ag_dataset(dataset_dir: str, include_image_col: bool = True) -> tuple:
    from autogluon.tabular import TabularDataset, FeatureMetadata
    train_data = TabularDataset(os.path.join(dataset_dir, 'train.csv'))
    dev_data = TabularDataset(os.path.join(dataset_dir, 'dev.csv'))
    test_data = TabularDataset(os.path.join(dataset_dir, 'test.csv'))
    feature_metadata = FeatureMetadata.from_df(train_data)
    if include_image_col:
        image_col = 'Image Path'
        image_id_to_path_func = lambda image_id: os.path.join(dataset_dir, image_id)
        train_data[image_col] = train_data[image_col].apply(image_id_to_path_func)
        dev_data[image_col] = dev_data[image_col].apply(image_id_to_path_func)
        test_data[image_col] = test_data[image_col].apply(image_id_to_path_func)
        feature_metadata = feature_metadata.add_special_types({image_col: ['image_path']})
    return train_data, dev_data, test_data, feature_metadata


if __name__ == '__main__':
    time_limit = 3600
    constraint = get_exp_constraint(time_limit)
    print(f'[DEBUG] constraint={constraint} for input time_limit={time_limit}')
