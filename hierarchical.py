""" Agglomerative (Bottom-Up) Clustering Algorithm"""
import numpy as np
np.random.seed(0)

import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist
from scipy.cluster import hierarchy

from kmeans import compute_wcc, compute_sc, compute_nmi, plot_clusters

colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink', 'brown', \
    'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise', 'darkgreen', 'tan', 'salmon', \
    'gold', 'lightpurple', 'darkred', 'darkblue']

def hierarchical(data, linkage):
    ytdist = pdist(data)
    Z = hierarchy.linkage(ytdist, linkage)
    return Z

def plot_dendrograms(Z1, Z2, Z3):
    _, axes = plt.subplots(3, 1, figsize=(15, 8))
    axes[0].set_title('Dendrogram using Single Linkage')
    axes[1].set_title('Dendrogram using Complete Linkage')
    axes[2].set_title('Dendrogram using Average Linkage')
    axes[0].set_ylabel('Distances')
    axes[1].set_ylabel('Distances')
    axes[2].set_ylabel('Distances')
    dn1 = hierarchy.dendrogram(Z1, ax=axes[0])
    dn2 = hierarchy.dendrogram(Z2, ax=axes[1])
    dn3 = hierarchy.dendrogram(Z3, ax=axes[2])
    plt.tight_layout()
    plt.show()

def sample_dataset(df):
    new_df = pd.DataFrame()
    for idx in range(10):
        new_df = new_df.append(df[df[1] == idx].sample(n=10), ignore_index=True)
    return new_df

def compute_and_plot_metrics(data, cutree_dataset, n_clusters, linkage, plot=True):
    wcc_array = []
    sc_array = []
    if isinstance(n_clusters, int):
        clusters_ = cutree_dataset[:, 0]
        centroids_ = []
        for k in range(n_clusters):
            centroids_.append(np.mean(data[clusters_ == k], axis = 0))
    else:
        for idx, K in enumerate(n_clusters):
            clusters_ = cutree_dataset[:, idx]
            centroids_ = []
            for k in range(K):
                centroids_.append(np.mean(data[clusters_ == k], axis = 0))
            wcc = compute_wcc(data, np.array(centroids_), clusters_, K)
            sc = compute_sc(data, clusters_, K)
            wcc_array.append(wcc)
            sc_array.append(sc)
    # plot the metrics
    if plot:
        _, ax = plt.subplots(1,2, figsize=[10,5])
        ax[0].plot(n_clusters, wcc_array)
        ax[0].set_title('WC SSD vs Number of Clusters (K)')
        ax[0].set_xlabel('Number of Clusters (K)')
        ax[0].set_ylabel('Within-Cluster Sum Squared Distance')
        ax[0].set_xticks(n_clusters)
        ax[0].grid(True, linestyle='--', linewidth=0.5)
        ax[1].plot(n_clusters, sc_array)
        ax[1].set_title('SC vs Number of Clusters (K)')
        ax[1].set_xlabel('Number of Clusters (K)')
        ax[1].set_ylabel('Silhoutte Coefficient')
        ax[1].set_xticks(n_clusters)
        ax[1].grid(True, linestyle='--', linewidth=0.5)
        plt.suptitle(f'Hierarchical Clustering using {linkage} linkage.')
        plt.show()

    return clusters_, centroids_

def plot_clusters(linkage, data, centroids, clusters, K):
    plt.figure(figsize=(10,8))
    plt.title(f'MNIST Dataset Hierarchical Clusters, Linkage {linkage}')
    area = (20)**2
    for idx in range(K):
        plt.scatter(data[clusters == idx][:,0], data[clusters == idx][:,1], c=colors[idx])          
        plt.scatter(centroids[idx][0], centroids[idx][1], s=area, marker='^', edgecolors='white', c=colors[idx])
    plt.show()

def analysis_hierarchical_nmi(data, distances, class_labels, K, linkage):
    cutree_dataset = hierarchy.cut_tree(distances, n_clusters=K)
    clusters, centroids = compute_and_plot_metrics(data, cutree_dataset, K, 'single', plot=False)
    nmi = compute_nmi(class_labels, clusters, K)
    print(f'Hierarchical Clustering NMI: {round(nmi,4)} Linkage: {linkage}')
    plot_clusters(linkage, data, centroids, clusters, K)

if __name__ == '__main__':
    filename = 'digits-embedding.csv'
    digits_df = pd.read_csv(filename, header = None)
    sampled_data = sample_dataset(digits_df)
    data = sampled_data.iloc[:,2:].to_numpy()
    class_labels = sampled_data.iloc[:, 1].to_numpy()
    Z1 = hierarchical(data, 'single')
    Z2 = hierarchical(data, 'complete')
    Z3 = hierarchical(data, 'average')
    plot_dendrograms(Z1, Z2, Z3)

    # 3.4
    n_clusters = [2,4,8,16,32]
    cutree_dataset1 = hierarchy.cut_tree(Z1, n_clusters=n_clusters)
    cutree_dataset2 = hierarchy.cut_tree(Z2, n_clusters=n_clusters)
    cutree_dataset3 = hierarchy.cut_tree(Z3, n_clusters=n_clusters)
    compute_and_plot_metrics(data, cutree_dataset1, n_clusters, 'single')
    compute_and_plot_metrics(data, cutree_dataset2, n_clusters, 'complete')
    compute_and_plot_metrics(data, cutree_dataset3, n_clusters, 'average')

    # 3.5
    analysis_hierarchical_nmi(data, Z1, class_labels, 8, 'single')
    analysis_hierarchical_nmi(data, Z2, class_labels, 8, 'complete')
    analysis_hierarchical_nmi(data, Z3, class_labels, 8, 'average')

