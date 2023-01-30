# **Note**
# some critical arguments to set
# --fit_time_limit 28800: max training time 8 hours
# --exp_save_dir: the directory to save model checkpoints and experiment results csv


# ======== Unimodal Classifiers =======
# Run tab-model1 (GBM)
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


# ======== Multimodal Classifiers =======
# Run AutoGluon
python -m baselines.autogluon.exec \
    --dataset_dir datasets/Pokemon_PrimaryType \
    --exp_save_dir exps/Pokemon_PrimaryType_AutoGluon \
    --fit_time_limit 28800 \
    --fit_hyperparameters multimodal \
    --fit_presets best_quality

# Run AutoMM
python -m baselines.automm.exec \
    --dataset_dir datasets/Pokemon_PrimaryType \
    --exp_save_dir exps/Pokemon_PrimaryType_AutoMM \
    --fit_time_limit 28800 \
    --fit_setting fusion

# Run MuGNet modal
python -m baselines.MuGNet.exec \
        --dataset_dir datasets/Pokemon_PrimaryType \
        --exp_save_dir exps/Pokemon_PrimaryType_mugnet \
        --fit_time_limit 28800 
