""" K-Means Clustering Algorithm"""
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from kmeans import kmeans, compute_wcc, compute_sc, compute_nmi, plot_clusters

def analysis_kmeans_helper(data, K, seed_ = 0):
    np.random.seed(seed_)
    clusters, centroids = kmeans(data, K)
    return clusters, centroids

def analysis_kmeans(dataset_num, data, K_array, seeds = None):
    wcc_array = []
    sc_array = []
    wcc_std_array = []
    sc_std_array = []
    for K in K_array:
        if seeds is None:
            clusters, centroids = analysis_kmeans_helper(data, K)
            avg_wcc = compute_wcc(data, centroids, clusters, K)
            avg_sc = compute_sc(data, clusters, K)
        else:
            wcc_ = []
            sc_ = []
            for seed_ in seeds:
                clusters, centroids = analysis_kmeans_helper(data, K, seed_)
                wcc = compute_wcc(data, centroids, clusters, K)
                sc = compute_sc(data, clusters, K)
                wcc_.append(wcc)
                sc_.append(sc)
            avg_wcc = sum(wcc_)/len(wcc_)
            avg_sc = sum(sc_)/len(sc_)
            std_wc = (sum([((x - avg_wcc) ** 2) for x in wcc_]) / len(wcc_)) ** 0.5
            std_sc = (sum([((x - avg_sc) ** 2) for x in sc_]) / len(sc_)) ** 0.5
            wcc_std_array.append(std_wc)
            sc_std_array.append(std_sc)
        wcc_array.append(avg_wcc)
        sc_array.append(avg_sc)
    if seeds is None:
        plot_curve(dataset_num, K_array, wcc_array, sc_array)
    else:
        plot_error_curve(dataset_num, K_array, wcc_array, sc_array, wcc_std_array, sc_std_array)

def plot_curve(dataset_num, K_array, wcc_array, sc_array):
    _, ax = plt.subplots(1,2, figsize=[10,5])
    ax[0].plot(K_array, wcc_array)
    ax[0].set_title('WC SSD vs Number of Clusters (K)')
    ax[0].set_xlabel('Number of Clusters (K)')
    ax[0].set_ylabel('Within-Cluster Sum Squared Distance')
    ax[0].set_xticks(K_array)
    ax[0].grid(True, linestyle='--', linewidth=0.5)
    ax[1].plot(K_array, sc_array)
    ax[1].set_title('SC vs Number of Clusters (K)')
    ax[1].set_xlabel('Number of Clusters (K)')
    ax[1].set_ylabel('Silhoutte Coefficient')
    ax[1].set_xticks(K_array)
    ax[1].grid(True, linestyle='--', linewidth=0.5)
    plt.suptitle(f'Dataset {dataset_num}')

def plot_error_curve(dataset_num, K_array, wcc_array, sc_array, wcc_std_array, sc_std_array):
    _, ax = plt.subplots(1,2, figsize=[10,5])
    ax[0].errorbar(K_array, wcc_array, yerr = wcc_std_array, marker='o')
    ax[0].set_title('Error Plot WC-SSD Number of Clusters (K) for different initial samples')
    ax[0].set_xlabel('Number of Clusters (K)')
    ax[0].set_ylabel('Average Within-Cluster Sum Squared Distance and Std. Deviation')
    ax[0].set_xticks(K_array)
    ax[0].grid(True, linestyle='--', linewidth=0.5)
    ax[1].errorbar(K_array, sc_array, yerr = sc_std_array, marker='o')
    ax[1].set_title('Error Plot SC Number of Clusters (K) for different initial samples')
    ax[1].set_xlabel('Number of Clusters (K)')
    ax[1].set_ylabel('Average Silhoutte Coefficient and Std. Deviation')
    ax[1].set_xticks(K_array)
    ax[1].grid(True, linestyle='--', linewidth=0.5)
    plt.suptitle(f'Dataset {dataset_num}')

def analysis_kmeans_datasets(df_1, df_2, df_3, K_array):
    analysis_kmeans(1, df_1, K_array) # Dataset 1
    analysis_kmeans(2, df_2, K_array) # Dataset 2
    analysis_kmeans(3, df_3, K_array) # Dataset 3
    plt.show()

def analysis_kmeans_seed_sensitivity(df_1, df_2, df_3, K_array):
    seeds = np.arange(1,11)
    analysis_kmeans(1, df_1, K_array, seeds) # Dataset 1
    analysis_kmeans(2, df_2, K_array, seeds) # Dataset 2
    analysis_kmeans(3, df_3, K_array, seeds) # Dataset 3
    plt.show()

def analysis_kmeans_nmi(dataset_num, data, class_labels, K):
    clusters, centroids = analysis_kmeans_helper(data, K, 0)
    nmi = compute_nmi(class_labels, clusters, K)
    print(f'K-Means Clustering NMI: {round(nmi,4)} Dataset: {dataset_num}')
    # plot 1000 randomly chosen examples
    sampled_idxs = np.random.choice(len(data), 1000, replace = False)
    clusters = clusters[sampled_idxs]
    data = data[sampled_idxs, :]
    plot_clusters(dataset_num, data, centroids, clusters, K)

if __name__ == '__main__':
    s = time.time()
    filename = 'digits-embedding.csv'
    digits_df = pd.read_csv(filename, header = None)
    # Constructing Datasets
    df_1 = digits_df.iloc[:, 2:].to_numpy()
    class_labels1 = digits_df.iloc[:, 1].astype('int')
    df_2 = digits_df.loc[digits_df[1].isin([2,4,6,7])]
    class_labels2 = df_2.iloc[:, 1].astype('int')
    df_2 = df_2.iloc[:, 2:].to_numpy()
    df_3 = digits_df.loc[digits_df[1].isin([6,7])]
    class_labels3 = df_3.iloc[:, 1].astype('int')
    df_3 = df_3.iloc[:, 2:].to_numpy()
    # K-Means Analysis
    K_array = [2, 4, 8, 16, 32]
    # 2.2 (1)
    analysis_kmeans_datasets(df_1, df_2, df_3, K_array)
    # 2.2 (4)
    analysis_kmeans_nmi(1, df_1, class_labels1, 8)
    analysis_kmeans_nmi(2, df_2, class_labels2, 4)
    analysis_kmeans_nmi(3, df_3, class_labels3, 2)
    # 2.2 (3)
    print(f'Analysis of K-Means sensitivity to initial seeds. This may take a while...')
    analysis_kmeans_seed_sensitivity(df_1, df_2, df_3, K_array)
