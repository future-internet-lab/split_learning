from sklearn.cluster import KMeans, AffinityPropagation
from sklearn.metrics import silhouette_score
import copy
import numpy as np
from src.Utils import num_client_in_cluster


def clustering_algorithm(list_performance, num, client_cluster_config):
    return clustering_AffinityPropagation(list_performance, num, client_cluster_config["AffinityPropagation"])


def clustering_AffinityPropagation(list_performance, num, config):
    label_counts = []
    for i in list_performance:
        if i != -1:
            label_counts.append(i)
    damping = config['damping']
    max_iter = config['max_iter']
    affinity_propagation = AffinityPropagation(damping=damping, max_iter=max_iter)
    affinity_propagation.fit(np.array(label_counts).reshape(-1, 1))

    cluster_centers_indices = affinity_propagation.cluster_centers_indices_
    labels = affinity_propagation.labels_
    labels = labels.tolist()
    labels = [0, 0, 1, 1]
    if num == 2:
        cluster_layer_2 = [0, 1]
    else:
        cluster_layer_2 = [0]

    infor_cluster = num_client_in_cluster(labels)
    print(infor_cluster)
    infor_layer_2 = [[1], [1]]
    for i in range(len(infor_cluster)):
        infor_cluster[i].append(infor_layer_2[i][0])
    for idx, i in enumerate(list_performance):
        if i != -1:
            list_performance[idx] = labels.pop(0)
        else:
            list_performance[idx] = cluster_layer_2.pop()
    list_cut_layer = [[14], [24]]
    return list_performance, infor_cluster, 2, list_cut_layer
