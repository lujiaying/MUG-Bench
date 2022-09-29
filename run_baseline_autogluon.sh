#!/bin/bash
#SBATCH --job-name=MUG-trial
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/MUG-trial

# Sep 24
# dataset_name="AnimalCrossing_Species"
# python -m baselines.autogluon.exec \
#     --dataset_dir datasets/${dataset_name} \
#     --exp_save_dir exps/${dataset_name}_multimodal_medium \
#     --task_name ${dataset_name} \
#     --col_labels Species \
#     --fit_hyperparameters multimodal \
#     --fit_presets medium_quality

# Sep 26
# dataset_name="Pokemon_Type1_Type2"
# python -m baselines.autogluon.exec \
#     --dataset_dir datasets/pokemon_0421_removed \
#     --exp_save_dir exps/${dataset_name}_medium \
#     --task_name ${dataset_name} \
#     --col_labels type_1 type_2 \
#     --fit_presets medium_quality

# Sep 28
# python -m baselines.autogluon.exec \
#      --dataset_dir datasets/Hearthstone-All/cardClass \
#      --eval_metric log_loss \
#      --exp_save_dir exps/Hearthstone-All-cardClass_medium \
#      --task_name Hearthstone-All-cardClass \
#      --col_labels cardClass \
#      --fit_presets medium_quality
# python -m baselines.autogluon.exec \
#     --dataset_dir datasets/Hearthstone-All/cardClass \
#     --eval_metric log_loss \
#     --exp_save_dir exps/Hearthstone-All-cardClass_mm_medium \
#     --task_name Hearthstone-All-cardClass \
#     --col_labels cardClass \
#     --fit_hyperparameters multimodal \
#     --fit_presets medium_quality

python -m baselines.autogluon.exec \
    --dataset_dir datasets/Hearthstone-All/rarity \
    --eval_metric log_loss \
    --exp_save_dir exps/Hearthstone-All-rarity_medium \
    --task_name Hearthstone-All-rarity \
    --col_labels rarity \
    --fit_presets medium_quality
