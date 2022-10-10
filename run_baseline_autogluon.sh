#!/bin/bash
#SBATCH --job-name=MUG-exp
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/MUG-trial

# Oct 10
# Note: example below is just for DEBUG purpose
# for benchmark, please set input argument properly
python -m baselines.autogluon.exec \
    --dataset_dir datasets/cardClass \
    --exp_save_dir exps/Hearthstone-All-set_medium \
    --fit_time_limit 240 \
    --fit_presets medium_quality
