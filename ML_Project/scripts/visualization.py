import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

def tsne_visualization(df):
    embeddings = np.array([np.array(emb) for emb in df['embedding']])
    print("Embeddings Shape:", embeddings.shape)  
 
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=200)
    tsne_result = tsne.fit_transform(embeddings)
    df['tsne_1'] = tsne_result[:, 0]
    df['tsne_2'] = tsne_result[:, 1]

    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x='tsne_1', y='tsne_2',
        hue='syndrome_id',
        palette='tab10',
        data=df,
        legend='full',
        alpha=0.7
    )
    plt.title('t-SNE Visualization of Embeddings')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    plt.savefig('../visuals/tsne_visualization.png')
    plt.show()

if __name__ == '__main__':
    df = pd.read_pickle('../data/flattened_data.pkl')
    tsne_visualization(df)
