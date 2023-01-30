import argparse

import numpy as np
import pandas as pd
from matplotlib.cbook import boxplot_stats


def analyze_every_model(df: pd.DataFrame):
    task_order = [
            'Pokemon-primary_type',
            'Pokemon-secondary_type',
            'Hearthstone-All-cardClass',
            'Hearthstone-All-set',
            'Hearthstone-Minions-race',
            'Hearthstone-Spell-spellSchool',
            'LOL-Skins-Category',
            'CSGO-Skin-quality',
            ]
    df = df.set_index('task').loc[task_order]
    accuracy = df['accuracy'].tolist()
    print('accuracy')
    print('& '.join([f'{(round(_, 3)):.3f}' for _ in accuracy]))
    print(f'mean accuracy {round(df.accuracy.mean(), 4)}')
    logloss = df['log_loss'].tolist()
    print('logloss')
    print('& '.join([f'{(round(_, 3)):.3f}' for _ in logloss]))
    print(f'mean log_loss {round(df.log_loss.mean(), 4)}')

    print('training duration for boxplot')
    stats = boxplot_stats(df.training_duration)[0]
    print(f'lower whisker={stats["whislo"]}, lower quartile={stats["q1"]},\nmedian={stats["med"]}, upper quartile={stats["q3"]},\nupper whisker={stats["whishi"]}')

    print('testing duration and accuracy tradeoffs')
    # task-wise accuracy need min-max normalization
    mean_acc = df['accuracy'].mean()
    mean_test_duration = df['predict_duration'].mean()
    print(round(mean_test_duration, 2), '\t', round(mean_acc, 3), )


def analyze_merged_result(df: pd.DataFrame):
    agg_df = df.groupby('dataset_name').agg(
            max_acc=pd.NamedAgg(column='accuracy', aggfunc=max),
            min_acc=pd.NamedAgg(column='accuracy', aggfunc=min),
            )
    df['normed_acc'] = df.apply(lambda row: (row['accuracy']-agg_df.loc[row['dataset_name']].loc['min_acc']) / (agg_df.loc[row['dataset_name']].loc['max_acc']-agg_df.loc[row['dataset_name']].loc['min_acc']), axis=1)
    res_df = df.groupby('classifier_name').agg(
            mean_normed_acc=pd.NamedAgg(column='normed_acc', aggfunc=np.mean),
            mean_acc=pd.NamedAgg(column='accuracy', aggfunc=np.mean),
            )
    print(res_df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("convert exp results files into latex used format")
    parser.add_argument('--in_file', type=str, required=True)

    args = parser.parse_args()
    df = pd.read_csv(args.in_file)
    
    # analyze_every_model(df)
    analyze_merged_result(df)
