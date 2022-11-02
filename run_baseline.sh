#!/bin/bash
#SBATCH --job-name=MUG-exp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/MUG-trial

# **Note**
# some critical arguments to set
# --fit_time_limit 28800: max training time 8 hours
# --exp_save_dir: the directory to save model checkpoints and experiment results csv

# Run tab model (AG-tab)
# Note: Plz use best_quality for max performance
python -m baselines.autogluon.exec \
    --dataset_dir datasets/Pokemon_PrimaryType \
    --exp_save_dir exps/Pokemon_PrimaryType_AG-tab-best \
    --fit_time_limit 28800 \
    --fit_hyperparameters default \
    --fit_presets best_quality

# Run txt-image model (CLIP)
python -m baselines.automm.exec \
    --dataset_dir datasets/Pokemon_PrimaryType \
    --exp_save_dir exps/Pokemon_PrimaryType_CLIP \
    --fit_time_limit 28800 \
    --fit_setting clip

# Run MM-Ensemble model (AG-MM)
python -m baselines.autogluon.exec \
    --dataset_dir datasets/Pokemon_PrimaryType \
    --exp_save_dir exps/Pokemon_PrimaryType_AG-tab-best \
    --fit_time_limit 28800 \
    --fit_hyperparameters multimodal \
    --fit_presets best_quality

# Run MM-Fusion model (AutoMM)
python -m baselines.automm.exec \
    --dataset_dir datasets/Pokemon_PrimaryType \
    --exp_save_dir exps/Pokemon_PrimaryType_AutoMM-Fusion \
    --fit_time_limit 28800 \
    --fit_setting fusion
