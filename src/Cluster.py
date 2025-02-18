from sklearn.cluster import KMeans, AffinityPropagation
from sklearn.metrics import silhouette_score


# def clustering_algorithm(label_counts, client_cluster_config):
#     cluster_name = client_cluster_config['cluster']
#     if cluster_name == 'KMeans':
#         cluster_config = client_cluster_config['KMeans']
#         return clustering_KMeans(label_counts, cluster_config)
#     elif cluster_name == 'AffinityPropagation':
#         cluster_config = client_cluster_config['AffinityPropagation']
#         return clustering_AffinityPropagation(label_counts, cluster_config)
#     else:
#         raise ValueError(f"Cluster '{cluster_name}' algorithm not contain in Cluster processing.")

def clustering_algorithm(label_counts, client_cluster_config):
    return clustering_AffinityPropagation(label_counts, client_cluster_config["AffinityPropagation"])

#
# def clustering_KMeans(label_counts, config):
#     mode = config['mode']
#     if mode == 'auto':
#         range_n_clusters = list(range(2, len(label_counts)))
#         silhouette_avg_max = 0.0
#         num_clusters = None
#         labels = None
#
#         for k in range_n_clusters:
#             kmeans = KMeans(n_clusters=k)
#             cluster_labels = kmeans.fit_predict(label_counts)
#
#             silhouette_avg = silhouette_score(label_counts, cluster_labels)
#             if silhouette_avg > silhouette_avg_max:
#                 silhouette_avg_max = silhouette_avg
#                 num_clusters = k
#                 labels = kmeans.labels_
#
#         return num_clusters, labels, silhouette_avg_max
#     elif mode.isnumeric():
#         k = int(mode)
#         if k > len(label_counts) or k < 1:
#             raise ValueError(f"K = '{k}' is not valid.")
#         else:
#             kmeans = KMeans(n_clusters=k)
#             labels = kmeans.labels_
#
#             return k, labels, None
#     else:
#         raise ValueError(f"KMeans mode '{mode}' is not valid.")


def clustering_AffinityPropagation(label_counts, config):
    damping = config['damping']
    max_iter = config['max_iter']

    affinity_propagation = AffinityPropagation(damping=damping, max_iter=max_iter)
    affinity_propagation.fit(label_counts)

    cluster_centers_indices = affinity_propagation.cluster_centers_indices_
    labels = affinity_propagation.labels_

    return len(cluster_centers_indices), labels, None
