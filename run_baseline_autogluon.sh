# Sep 2
# DEBUG
python baselines/autogluon/exec.py \
    --dataset_dir datasets/AnimalCrossing_Gender \
    --exp_save_dir exps/AnimalCrossing_Gender_DEBUG \
    --col_label Gender \
    --fit_presets medium_quality
