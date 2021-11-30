""" K-Means Clustering Algorithm"""
import sys

import numpy as np
np.random.seed(0)

import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

max_iter = 50

def recompute_clusters(data, centroids):
    clusters = []
    distances = cdist(data, centroids, metric='euclidean')
    for idx in range(len(data)):
        clusters.append(np.argmin(distances[idx, :]))
    return np.array(clusters)

def recompute_centroids(data, centroids, clusters, K):
    for idx in range(K):
        centroids[idx] = np.mean(data[clusters == idx], axis = 0)
    return centroids

def plot_clusters(dataset_num, data, centroids, clusters, K):
    plt.figure(figsize=(10,8))
    plt.title(f'MNIST Dataset {dataset_num} K-Means Clusters')
    area = (20)**2
    for idx in range(K):
        plt.scatter(data[clusters == idx][:,0], data[clusters == idx][:,1], c=colors[idx])          
        plt.scatter(centroids[idx][0], centroids[idx][1], s=area, marker='^', edgecolors='white', c=colors[idx])
    plt.show()

def kmeans(data, K = 10):
    # start with k randomly chosen centroids
    sampled_idxs = np.random.choice(len(data), K, replace = False)
    centroids = data[sampled_idxs, :]
    for _ in range(max_iter):
        # assign instances to closest centroid
        clusters =  recompute_clusters(data, centroids)
        # recompute cluster centroids
        centroids = recompute_centroids(data, centroids, clusters, K)
    # plot_clusters(1, data, centroids, clusters, K)
    return np.array(clusters), np.array(centroids)

def compute_wcc(data, centroids, clusters, K):
    # computing within-cluster sum of squared distances
    wc_ck = []
    for idx in range(K):
        wc_ck.append(np.sum(np.square(cdist(data[clusters == idx], centroids[idx,:].reshape(1,-1), metric='euclidean'))))
    wcc = np.sum(wc_ck)
    return wcc

def compute_sc(data, clusters, K):
    # computing silhoutte coefficient
    sc_array = []
    distances = cdist(data, data, metric='euclidean')
    for idx in range(len(data)):
        cluster_membership = clusters[idx]
        A = np.mean(distances[idx][np.where(clusters == cluster_membership)])
        di_Cj = []
        for cl_k in range(K):
            if cl_k == cluster_membership:
                continue
            di_Cj.append(np.mean(distances[idx][np.where(clusters == cl_k)]))
        B = np.min(di_Cj)
        sc_array.append((B-A)/(max(A,B)))
    sc = np.mean(sc_array)
    return sc

def compute_entropy(data):
    probabilities = np.bincount(data)/len(data)
    result =  - np.sum(probabilities * np.log(probabilities, where = probabilities > 0))
    return result

def compute_nmi(class_labels, clusters, K):
    # entropy of class labels H(Y): This value is constant and doesn't depend on clustering
    h_y = compute_entropy(class_labels)
    # entropy of cluster labels H(C): This value changes as clusters change
    h_c = compute_entropy(clusters)
    # mutual information between Y and C: I(Y,C) = H(Y) - H(Y|C)
    h_yc = sum(len(class_labels[clusters == idx])/len(class_labels) * \
                compute_entropy(class_labels[clusters == idx]) for idx in range(K))
    i_yc = h_y - h_yc
    # computing normalized mutual information gain NMI = I(Y,C)/[H(Y) + H(C)]
    nmi = i_yc / (h_y + h_c)
    return nmi

def output_kmeans(wcc, sc, nmi):
    print(f'WC-SSD: {round(wcc,3)}')
    print(f'SC: {round(sc,3)}')
    print(f'NMI: {round(nmi,3)}')

if __name__ == '__main__':
    filename, K = sys.argv[1], int(sys.argv[2])
    digits_df = pd.read_csv(filename, header = None)
    # Constructing Datasets
    df_1 = digits_df.iloc[:, 2:].to_numpy()
    df_2 = digits_df.loc[digits_df[1].isin([2,4,6,7])]
    df_2 = df_2.iloc[:, 2:].to_numpy()
    df_3 = digits_df.loc[digits_df[1].isin([6,7])]
    df_3 = df_3.iloc[:, 2:].to_numpy()
    class_labels = digits_df.iloc[:, 1].astype('int')
    # K-Means Algorithm and Results/Analysis
    clusters, centroids = kmeans(df_1, K)
    wcc = compute_wcc(df_1, centroids, clusters, K)
    sc = compute_sc(df_1, clusters, K)
    nmi = compute_nmi(class_labels, clusters, K)
    output_kmeans(wcc, sc, nmi)
