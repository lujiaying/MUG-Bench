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
dataset_name="Pokemon_Type1_Type2"
python -m baselines.autogluon.exec \
    --dataset_dir datasets/pokemon_0421_removed \
    --exp_save_dir exps/${dataset_name}_medium \
    --task_name ${dataset_name} \
    --col_labels type_1 type_2 \
    --fit_presets medium_quality
