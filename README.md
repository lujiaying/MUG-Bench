# MuG

MuG: a Multimodal Classification Benchmark on Game Data with Tabular, Textual, and Visual Fields

**Table of Contents**

- [Datasets](#datasets)
- [Prerequisites](#prerequisites)
- [Reproduce Results](#reproduce-results)

## Datasets

The eight datasets used in paper can be downloaded from https://doi.org/10.6084/m9.figshare.21454413.
After downloading and decompressing them under `./datasets/` directory, the directory looks like:

```
📁 ./dataset
|-- 📁 Pokemon-primary_type
|-- 📁 Pokemon-secondary_type
|-- 📁 Hearthstone-All-cardClass
|-- 📁 Hearthstone-All-set
|-- 📁 Hearthstone-Minion-race
|-- 📁 Hearthstone-Spell-spellSchool
|-- 📁 LeagueOfLegends-Skin-category
|-- 📁 CSGO-Skin-quality
|-- CHANGELOG
```

And each subdirectory represents one dataset, for instance:

```
📁 ./dataset/Pokemon-primary_type
|-- info.json
|-- train.csv
|-- dev.csv
|-- test.csv
|-- train_images.zip
|-- dev_images.zip
|-- test_images.zip
```

where `info.json` stores meta information of the dataset;
`train/dev/test.csv` store raw tabular and text features of each sample;
`train_images/dev_images/test_images.zip` represent a compressed directory of raw images.

## Prerequisites

```
python>=3.9.12
autogluon==0.5.2
```

For installation instructions of autogluon 0.5.2 (GPU version recommended), please refer to https://auto.gluon.ai/dev/versions.html.
We mainly use it for feature preprocessing and baselines. And it has specific dependecies on pytorch(==1.12) and ray(==1.13.0).

## Reproduce Results

Example scripts to run unimodal classifiers and multimodal classifiers are listed in [run_baseline.sh](run_baseline.sh).

For instance, we can use the following script to reproduce the proposed MuGNet model:

```Shell
# Run MuGNet modal
python -m baselines.MuGNet.exec \
        --dataset_dir datasets/Pokemon_PrimaryType \
        --exp_save_dir exps/Pokemon_PrimaryType_mugnet \
        --fit_time_limit 28800 
```
where `--dataset_dir` specifies the dataset directory, `--exp_save_dir` specifies the destination to store the final output, `--fit_time_limit` specifies the time limit for the model to run in seconds.
