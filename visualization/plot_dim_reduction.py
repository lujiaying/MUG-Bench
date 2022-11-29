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

    df = pd.read_csv(args.in_file, sep='\t', header=None)
    labels = df.iloc[:, 0]
    feats = df.iloc[:, 1:]
    print(f'{args.in_file=}, {feats.shape=} {labels.shape=}')

    # === UMAP ===
    if args.algorithm == 'umap':
        reducer = umap.UMAP()
        # reducer = umap.UMAP(n_neighbors=5).fit(feats)  # small dataset
        reducer = umap.UMAP().fit(feats)    # default n_neighbors=15
        ax = umap.plot.points(reducer, labels=labels)
        ax.figure.savefig(args.out_file)

    # === tSNE ===
    if args.algorithm == 'tsne':
        embedding = TSNE(learning_rate='auto', init='pca').fit_transform(feats)
        df_tsne = pd.DataFrame(data={'tSNE1': embedding[:,0], 'tSNE2': embedding[:,1], 'label': labels})
        sns_plot = sns.scatterplot(data=df_tsne, x='tSNE1', y='tSNE2', hue='label')
        sns_plot.set(title=args.title)
        sns.move_legend(sns_plot, "upper left", bbox_to_anchor=(1, 1))
        sns_plot.figure.set_size_inches(11.7, 8.27)
        sns_plot.figure.savefig(args.out_file, bbox_inches='tight')
