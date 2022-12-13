import argparse

import pandas as pd
import umap
import umap.plot
from sklearn.manifold import TSNE
import seaborn as sns

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', type=str, required=True)
    parser.add_argument('--out_file', type=str, required=True)
    parser.add_argument('--title', type=str, required=True)
    parser.add_argument('--algorithm', type=str, default='tsne')
    args = parser.parse_args()
    print(f'exp {args=}')

    # sns.set_context("paper", rc={"font.size":8, "axes.titlesize":12, "axes.labelsize":10})
    sns.set_context("paper", rc={"font.size":12, "axes.titlesize":16, "axes.labelsize":13})
    df = pd.read_csv(args.in_file, sep='\t', header=None)
    # do filter to select only 5 classes
    # df = df.loc[df[0].isin(["MAGE", "HUNTER", "PRIEST", "WARRIOR", "ROGUE"])]
    df = df.loc[df[0].isin(["BEAST", "DRAGON", "DEMON", "ELEMENTAL", "MECHANICAL", "MURLOC"])]
    labels = df.iloc[:, 0]
    feats = df.iloc[:, 1:]
    print(f'{args.in_file=}, {feats.shape=} {labels.shape=}')

    # === UMAP ===
    if args.algorithm == 'umap':
        reducer = umap.UMAP(n_neighbors=20)
        embedding = reducer.fit_transform(feats)
        df_umap = pd.DataFrame(data={'UMAP1': embedding[:,0], 'UMAP2': embedding[:,1], 'label': labels})
        # sns_plot = sns.scatterplot(data=df_umap, x='UMAP1', y='UMAP2', hue='label', s=48)
        sns_plot = sns.scatterplot(data=df_umap, x='UMAP1', y='UMAP2', hue='label', s=36)

    # === tSNE ===
    if args.algorithm == 'tsne':
        embedding = TSNE(learning_rate='auto', init='pca').fit_transform(feats)
        df_tsne = pd.DataFrame(data={'tSNE1': embedding[:,0], 'tSNE2': embedding[:,1], 'label': labels})
        sns_plot = sns.scatterplot(data=df_tsne, x='tSNE1', y='tSNE2', hue='label')

    sns_plot.set(title=args.title)
    # sns_plot.legend([],[], frameon=False)  # remove legend
    sns.move_legend(sns_plot, "lower center", bbox_to_anchor=(0.5, -0.5), ncol=2, title=None, fontsize=12)
    sns_plot.figure.set_size_inches(4, 4)
    sns_plot.figure.savefig(args.out_file, bbox_inches='tight')
