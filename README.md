# MuG-Bench

Data and code of the manuscript "[MuG: A Multimodal Classification Benchmark on Game Data with Tabular, Textual, and Visual Fields](https://arxiv.org/abs/2302.02978)".
For any suggestion/question, please feel free to create an issue or drop an email @ ([jiaying.lu@emory.edu](mailto:jiaying.lu@emory.edu)).


**Table of Contents**

- [Datasets](#datasets)
- [Prerequisites](#prerequisites)
- [Reproduce Results](#reproduce-results)
- [Citing Our Work](#citing-our-work)

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

All dependecies are listed in [conda_env.yml](conda_env.yml). We recommend using `conda` to manage the environment.

```
conda env create -n MuG_env --file conda_env.yml
```


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

Another example to run GBM model:

```Shell
python -m baselines.autogluon.exec \
        --dataset_dir datasets/Pokemon_PrimaryType \
        --exp_save_dir exps/Pokemon_PrimaryType_GBMLarge \
        --fit_time_limit 28800 \
        --fit_setting GBMLarge
```

## Citing Our Work

```bibtex
@article {lu2023MuG,
  title = {MuG: A Multimodal Classification Benchmark on Game Data with Tabular, Textual, and Visual Fields},
  author = {Jiaying Lu, Yongchen Qian, Shifan Zhao, Yuanzhe Xi, Carl Yang},
  year = {2023},
  URL = {https://arxiv.org/abs/2302.02978},
  journal = {arXiv preprint arXiv:2302.02978}
}
```
