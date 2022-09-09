# Sep 2
# DEBUG
# dataset_name="AnimalCrossing_Gender"
# python baselines/autogluon/exec.py \
#     --dataset_dir datasets/${dataset_name} \
#     --exp_save_dir exps/${dataset_name}_DEBUG \
#     --col_label Gender \
#     --fit_presets medium_quality

# dataset_name="AnimalCrossing_Species"
# python baselines/autogluon/exec.py \
#     --dataset_dir datasets/${dataset_name} \
#     --exp_save_dir exps/${dataset_name}_DEBUG \
#     --col_label Species \
#     --fit_presets medium_quality

# Sep 9
dataset_name="AnimalCrossing_Species"
python baselines/autogluon/exec.py \
    --dataset_dir datasets/${dataset_name} \
    --exp_save_dir exps/${dataset_name}_multimodal_medium \
    --col_label Species \
    --fit_presets medium_quality
