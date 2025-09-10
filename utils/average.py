from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize, StandardScaler
from sklearn import cluster
import faiss
from typing import List

import numpy as np
from tqdm import tqdm
from collections import defaultdict

def apply_pca(
    word_features: List[np.ndarray], num_components: int = 350
) -> List[np.ndarray]:
    """
    Applies PCA for dimensionality reduction on word-level feature sequences.

    Args:
        word_features: List of word-level feature sequences, 
            where each item is a (T, D) array (T = time steps, D = feature dim).
        num_components: Number of PCA components to retain.

    Returns:
        split_features: List of word-level features after PCA,
            each reduced to (T, num_components).
    """
    pca = PCA(n_components=num_components)
    scaler = StandardScaler()
    
    lengths = [len(feat) for feat in word_features]
    word_features = np.vstack(word_features)
    word_features = scaler.fit_transform(word_features)

    pca.fit(word_features)
    word_features = pca.transform(word_features)

    split_features = []
    index = 0

    for length in tqdm(lengths, desc="Reducing dimensionality with PCA"):
        split_features.append(word_features[index:index+length, :])
        index += length

    return split_features


def pooling(word_features: List[np.ndarray]) -> np.ndarray:
    """
    Pools word-level feature sequences into fixed-size embeddings using mean pooling,
    followed by L2 normalization.

    Args:
        word_features: List of word-level feature sequences,
            where each item is (T, D) after PCA.

    Returns:
        pooled_features: Array of pooled embeddings, shape (N, D),
            where N = number of words.
    """

    pooled_features = []
    for feat in tqdm(word_features, desc="Pooling features"):
        if len(feat.shape) == 1:
            feat = feat.reshape(1, -1)
        pooled_features.append(np.mean(feat, axis=0).astype(np.float32))

    
    pooled_features = normalize(
                np.stack(pooled_features, axis=0), axis=1, norm="l2"
            ).astype(np.float64)
    
    print(f"Features shape after PCA and normalisation: {pooled_features.shape}")

    return pooled_features


def cluster_features_kmeans(word_features, n_clusters):
    """
    Cluster feature vectors using k-means (FAISS implementation).

    Args:
        word_features: 2D array of feature vectors (shape = [n_samples, n_features]).
        n_clusters: Number of clusters to form.

    Returns:
        labels: 1D array of cluster assignments for each feature vector.
    """
    sk_kmeans, _ = cluster.kmeans_plusplus(word_features, n_clusters, random_state=0)
    initial_centroids = sk_kmeans.astype(np.float32)

    acoustic_model = faiss.Kmeans(
        word_features.shape[1], n_clusters, niter=15, nredo=3, verbose=True
    )
    acoustic_model.train(word_features, init_centroids=initial_centroids)

    _, Index = acoustic_model.index.search(word_features, 1)
    labels = Index.flatten()
    return labels


def cluster_features_birch(word_features, n_clusters):
    """
    Cluster feature vectors using the BIRCH algorithm.

    Args:
        word_features: 2D array of feature vectors (shape = [n_samples, n_features]).
        n_clusters: Number of clusters to form.

    Returns:
        labels: 1D array of cluster assignments for each feature vector.
    """
    word_features = word_features.astype(np.float32)
    clustering = cluster.Birch(
        n_clusters=n_clusters, threshold=0.25, compute_labels=True
    )
    clustering.fit(word_features)
    labels = clustering.labels_
    return labels


def cluster_features_agglomerative(word_features, n_clusters):
    """
    Cluster feature vectors using agglomerative hierarchical clustering.

    Args:
        word_features: 2D array of feature vectors (shape = [n_samples, n_features]).
        n_clusters: Number of clusters to form.

    Returns:
        labels: 1D array of cluster assignments for each feature vector.
    """
    clustering = cluster.AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    labels = clustering.fit_predict(word_features)
    return labels


def convert_cluster_ids_to_partition(cluster_labels):
    """
    Convert flat cluster labels into a partition (list of clusters).

    Args:
        cluster_labels: 1D iterable of cluster IDs, one per node.

    Returns:
        partition: List of clusters, where each cluster is a list of node indices.
    """
    partition_dict = defaultdict(list)

    for node_id, cluster_id in tqdm(
        enumerate(cluster_labels),
        total=len(cluster_labels),
        desc="Converting cluster IDs to partition",
    ):
        partition_dict[cluster_id].append(node_id)

    partition = list(partition_dict.values())
    return partition
