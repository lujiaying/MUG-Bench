# Sep 26
# Notes: seed 0, 27 all failed to pass validation
# python dutils/split_dataset.py \
#     --seed 1105 \
#     --rawdata_path datasets/pokemon_0421.csv \
#     --label_cols type_1 type_2 \
#     --output_dir datasets/pokemon_0421
python dutils/validate_dataset.py \
    --dataset_dir datasets/pokemon_0421 \
    --id_col name \
    --label_cols type_1 type_2 
