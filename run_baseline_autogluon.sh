# Sep 24
dataset_name="AnimalCrossing_Species"
python -m baselines.autogluon.exec \
    --dataset_dir datasets/${dataset_name} \
    --exp_save_dir exps/${dataset_name}_multimodal_medium \
    --task_name ${dataset_name} \
    --col_label Species \
    --fit_hyperparameters multimodal \
    --fit_presets medium_quality
