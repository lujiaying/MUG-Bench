#!/bin/bash
#SBATCH --job-name=MUG-exp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/MUG-trial

# **Note**
# some critical arguments to set
# --fit_time_limit 28800: max training time 8 hours
# --exp_save_dir: the directory to save model checkpoints and experiment results csv

# Run Multimodal
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

## Run MuGNET modal
python -m baselines.automgnn.exec \
        --dataset_dir datasets/Pokemon_PrimaryType \
        --exp_save_dir exps/Pokemon_PrimaryType_mugnet \
        --fit_time_limit 28800 \

# Run unimodal
# Run tab-model (GBM)
python -m baselines.autogluon.exec \
        --dataset_dir datasets/Pokemon_PrimaryType \
        --exp_save_dir exps/Pokemon_PrimaryType_GBMLarge \
        --fit_time_limit 28800 \
        --fit_setting GBMLarge

# Run tab-model (tabMLP)
python -m baselines.autogluon.exec \
        --dataset_dir datasets/Pokemon_PrimaryType \
        --exp_save_dir exps/Pokemon_PrimaryType_tabMLP \
        --fit_time_limit 28800 \
        --fit_setting tabMLP
        
# Run txt model (Roberta)
python -m baselines.automm.exec \
        --dataset_dir datasets/Pokemon_PrimaryType \
        --exp_save_dir exps/Pokemon_PrimaryType_roberta \
        --fit_time_limit 28800 \
        --fit_setting roberta
        
# Run txt model (Electra)
python -m baselines.automm.exec \
        --dataset_dir datasets/Pokemon_PrimaryType \
        --exp_save_dir exps/Pokemon_PrimaryType_electra \
        --fit_time_limit 28800 \
        --fit_setting electra

# Run img model (ViT)
python -m baselines.automm.exec \
        --dataset_dir datasets/Pokemon_PrimaryType \
        --exp_save_dir exps/Pokemon_PrimaryType_vit  \
        --fit_time_limit 28800 \
        --fit_setting vit
# Run img model (Swin)
python -m baselines.automm.exec \
        --dataset_dir datasets/Pokemon_PrimaryType \
        --exp_save_dir exps/Pokemon_PrimaryType_swin \
        --fit_time_limit 28800 \
        --fit_setting swin
