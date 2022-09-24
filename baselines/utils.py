import multiprocessing
from py3nvml import py3nvml


def get_exp_constraint(time_limit_in_second: int) -> str:
    cpu_count = multiprocessing.cpu_count()
    hour = int(round(time_limit_in_second/3600, 0))
    py3nvml.nvmlInit()
    device_count = py3nvml.nvmlDeviceGetCount()
    constraint = f'{hour}h{cpu_count}c'
    if device_count > 0:
        handle = py3nvml.nvmlDeviceGetHandleByIndex(0)
        constraint = f'{constraint}-{py3nvml.nvmlDeviceGetName(handle)}'
    return constraint


if __name__ == '__main__':
    time_limit = 3600
    constraint = get_exp_constraint(time_limit)
    print(f'[DEBUG] constraint={constraint} for input time_limit={time_limit}')