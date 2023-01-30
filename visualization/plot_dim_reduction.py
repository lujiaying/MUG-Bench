import os
import argparse
from typing import Tuple

import pandas as pd
import umap
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns


def get_feats_labels(in_file_path: str,
                     algorithm: str
                     ) -> Tuple[pd.DataFrame]:
    valid_labels = ["BEAST", "DRAGON", "DEMON", "ELEMENTAL", "MECHANICAL", "MURLOC"]
    df = pd.read_csv(in_file_path, sep='\t', header=None)
    df = df.loc[df[0].isin(valid_labels)]
    labels = df.iloc[:, 0]
    feats = df.iloc[:, 1:]
    print(f'{in_file_path=}, {feats.shape=} {labels.shape=}')
    if algorithm == 'umap':
        # === UMAP ===
        reducer = umap.UMAP(n_neighbors=20)
        embedding = reducer.fit_transform(feats)
    elif algorithm == 'tsne':
        # === tSNE ===
        embedding = TSNE(learning_rate='auto', init='pca').fit_transform(feats)
    else:
        raise ValueError(f'Not support {algorithm=}')
    df_draw = pd.DataFrame(data={'x': embedding[:,0], 'y': embedding[:,1], 'label': labels})
    return df_draw


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str, required=True)
    parser.add_argument('--out_file', type=str, required=True)
    parser.add_argument('--algorithm', type=str, default='umap')
    args = parser.parse_args()
    print(f'exp {args=}')

    sns.set_context("paper", rc={"font.size":12, "axes.titlesize":20, "axes.labelsize":13})
    sns.set_theme(style="ticks")
    fig, axs = plt.subplots(ncols=4, nrows=2)
    fig.figure.set_size_inches(16, 8.3)

    in_file = os.path.join(args.in_dir, 'tab_feats.tsv')
    df_draw = get_feats_labels(in_file, args.algorithm)
    sns_plot = sns.scatterplot(data=df_draw, x='x', y='y', hue='label', s=48, linewidth=0.25, alpha=0.85, ax=axs[0, 0])
    sns_plot.legend([],[], frameon=False)  # remove legend
    sns_plot.set(xlabel=None, ylabel=None)  # remove axis labels
    sns_plot.set_title("Tab raw feats", fontsize=18)

    in_file = os.path.join(args.in_dir, 'tabMLP_tab_feats.tsv')
    df_draw = get_feats_labels(in_file, args.algorithm)
    sns_plot = sns.scatterplot(data=df_draw, x='x', y='y', hue='label', s=48, linewidth=0.25, alpha=0.85, ax=axs[1, 0])
    sns_plot.legend([],[], frameon=False)  # remove legend
    sns_plot.set(xlabel=None, ylabel=None)  # remove axis labels
    sns_plot.set_title("Tab trained embs", fontsize=18)

    in_file = os.path.join(args.in_dir, 'txt_feats.tsv')
    df_draw = get_feats_labels(in_file, args.algorithm)
    sns_plot = sns.scatterplot(data=df_draw, x='x', y='y', hue='label', s=48, linewidth=0.25, alpha=0.85, ax=axs[0, 1])
    sns_plot.legend([],[], frameon=False)  # remove legend
    sns_plot.set(xlabel=None, ylabel=None)  # remove axis labels
    sns_plot.set_title("Txt raw feats", fontsize=18)

    in_file = os.path.join(args.in_dir, 'roberta_txt_feats.tsv')
    df_draw = get_feats_labels(in_file, args.algorithm)
    sns_plot = sns.scatterplot(data=df_draw, x='x', y='y', hue='label', s=48, linewidth=0.25, alpha=0.85, ax=axs[1, 1])
    sns_plot.legend([],[], frameon=False)  # remove legend
    sns_plot.set(xlabel=None, ylabel=None)  # remove axis labels
    sns_plot.set_title("Txt trained embs", fontsize=18)

    in_file = os.path.join(args.in_dir, 'img_feats.tsv')
    df_draw = get_feats_labels(in_file, args.algorithm)
    sns_plot = sns.scatterplot(data=df_draw, x='x', y='y', hue='label', s=48, linewidth=0.25, alpha=0.85, ax=axs[0, 2])
    sns_plot.legend([],[], frameon=False)  # remove legend
    sns_plot.set(xlabel=None, ylabel=None)  # remove axis labels
    sns_plot.set_title("Img raw feats", fontsize=18)

    in_file = os.path.join(args.in_dir, 'vit_img_feats.tsv')
    df_draw = get_feats_labels(in_file, args.algorithm)
    sns_plot = sns.scatterplot(data=df_draw, x='x', y='y', hue='label', s=48, linewidth=0.25, alpha=0.85, ax=axs[1, 2])
    sns_plot.legend([],[], frameon=False)  # remove legend
    sns_plot.set(xlabel=None, ylabel=None)  # remove axis labels
    sns_plot.set_title("Img trained embs", fontsize=18)
    
    in_file = os.path.join(args.in_dir, 'fused_raw_feats.tsv')
    df_draw = get_feats_labels(in_file, args.algorithm)
    sns_plot = sns.scatterplot(data=df_draw, x='x', y='y', hue='label', s=48, linewidth=0.25, alpha=0.85, ax=axs[0, 3])
    sns_plot.legend([],[], frameon=False)  # remove legend
    sns_plot.set(xlabel=None, ylabel=None)  # remove axis labels
    sns_plot.set_title("Fused raw feats", fontsize=18)

    in_file = os.path.join(args.in_dir, 'gnn_fusion_feats.tsv')
    df_draw = get_feats_labels(in_file, args.algorithm)
    sns_plot = sns.scatterplot(data=df_draw, x='x', y='y', hue='label', s=48, linewidth=0.25, alpha=0.85, ax=axs[1, 3])
    # sns_plot.legend([],[], frameon=False)  # remove legend
    sns.move_legend(sns_plot, "lower center", bbox_to_anchor=(-1.25, -0.3), ncol=6, title=None, fontsize=14)
    sns_plot.set(xlabel=None, ylabel=None)  # remove axis labels
    sns_plot.set_title("Fused trained embs", fontsize=18)

    sns.despine()
    plt.subplots_adjust(hspace=0.3)
    fig.figure.savefig(args.out_file, bbox_inches='tight')
