from utils.partition_graph import CPM_partition, write_partition, find_partition
from utils.evaluate_partition import evaluate_partition_file
from utils.build_graph import chunk_indices
from utils.average import cluster_features_kmeans, convert_cluster_ids_to_partition, apply_pca, pooling

import pandas as pd
import numpy as np
from tqdm import tqdm   
import igraph as ig
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import IncrementalPCA
import rapidfuzz.distance.Levenshtein as editdistance
from joblib import Parallel, delayed
from typing import List, Tuple
from pathlib import Path


def load_hacked_units(
    language: str = "english",
    dataset: str = "test",
    model_name: str = "wavlm-large",
    layer: int = 21,
    k: int = 500,
    lmbda: float = 100
) -> Tuple[List[np.ndarray], List[str]]:
    """
    Load "hacked" unit embeddings for a dataset by sampling one random example per unique word
    and duplicating it for all occurrences of that word.

    Args:
        language: Dataset language (e.g. "english", "mandarin").
        dataset: Dataset split (e.g. "train", "dev", "test").
        model_name: Model used for feature extraction.
        layer: Layer index used for embeddings.
        k: Number of clusters used during unit extraction.
        lmbda: Regularization weight for DPWFST segmentation.

    Returns:
        Tuple containing:
            - List of feature embeddings (np.ndarray) for each token in the dataset.
            - List of identifiers (filename_wordid) corresponding to each embedding.
    """
    
    boundary_df = pd.read_csv(f"Data/alignments/{language}/{dataset}_boundaries.csv")
    grouped_by_text = boundary_df.groupby('text')

    heacked_features, paths = [], []
    for group in tqdm(grouped_by_text, desc = "Loading hacked units"):
        random_row = group[1].sample(n=1, random_state=42).iloc[0]
        file = random_row['filename']
        word_id = random_row['word_id']

        embedding = np.load(f"units/{language}/{dataset}/{model_name}/{layer}/k{k}/{lmbda}/{file}_{word_id}.npy")
        for _, row in group[1].iterrows():
            heacked_features.append(embedding)
            paths.append(f"{row['filename']}_{row['word_id']}")
    
    return heacked_features, paths
        

def load_hacked_features(
    language: str = "english",
    dataset: str = "test",
    model_name: str = "wavlm-large",
    layer: int = 21,
) -> Tuple[List[np.ndarray], List[str]]:
    """
    Load features and create a 'hacked' version where each token embedding is replaced
    by its word-type mean plus some random noise. Also compute the true mean embeddings
    for each word type (to be used as centroids).

    Args:
        language: Dataset language (e.g., "english", "mandarin").
        dataset: Dataset split (e.g., "train", "dev", "test").
        model_name: Feature extraction model name.
        layer: Layer index from which to load embeddings.

    Returns:
        Tuple containing:
            - List of np.ndarray: 'Hacked' embeddings per token (mean + noise).
            - List of str: Identifiers per token in "filename_wordid" format.
            - np.ndarray: True mean embeddings per word type (centroids).
    """

    boundary_df = pd.read_csv(f"Data/alignments/{language}/{dataset}_boundaries.csv")
    grouped_by_text = boundary_df.groupby('text')

    
    feature_dir = Path(f"cut_features/{language}/{dataset}/{model_name}/{layer}/")
    file_paths = sorted(feature_dir.glob("*.npy"))
    file_paths.sort()

    all_features = []
    for path in tqdm(file_paths, desc="Loading Features"):
        features = np.load(path, mmap_mode="r")  
        all_features.append(features)

    all_features = apply_pca(all_features)
    all_features = pooling(all_features)

    print(f"Loaded and PCA-reduced: {len(all_features)} features with {all_features[0].shape} shape each")
    assert len(all_features) == len(file_paths), "Mismatch in number of feature files and paths"

    path_stems = [p.stem for p in file_paths]   
    path_to_idx = {p: i for i, p in enumerate(path_stems)}
    hacked_features = []
    hacked_paths = []

    for group in tqdm(grouped_by_text, desc="Creating hacked features"):
        group_df = group[1]

        token_features = []

        for _, row in group_df.iterrows():
            filename = row['filename']
            word_id = row['word_id']
            token_id = f"{filename}_{word_id}"
            if token_id in path_to_idx:
                hacked_paths.append(token_id)
                idx = path_to_idx[token_id]
                token_features.append(all_features[idx])

        if len(token_features) == 0:
            continue

        token_features = np.vstack(token_features)
        word_mean = np.mean(token_features, axis=0)

        for _ in range(len(group_df)):
            noise_std = np.random.uniform(0.005, 0.015)
            hacked_feat = np.random.normal(word_mean, noise_std, size=word_mean.shape[0])
            hacked_features.append(hacked_feat)

    assert len(hacked_features) == len(file_paths), f"Mismatch in total hacked features and paths: {len(hacked_features)} vs {len(file_paths)} vs  {len(hacked_paths)}"
    print(f"Total hacked features created: {len(hacked_features)} with shape: {hacked_features[0].shape}")

    hacked_features = np.stack(hacked_features, axis=0).astype(np.float32)
    return hacked_features, hacked_paths

    
def build_hacked_ed_graph(
    language: str = "english",
    dataset: str = "test",
    model_name: str = "wavlm-large",
    layer: int = 21,
    threshold: float = 0.65,
    k: int = 500,
    lmbda: float = 100,
    batch_size: int = 1_000
) -> ig.Graph:
    """
    Build a graph where nodes are "hacked" units and edges are added based on
    normalized edit distance between sequences.

    Args:
        language: Dataset language.
        dataset: Dataset split.
        model_name: Model used for feature extraction.
        layer: Layer index for embeddings.
        threshold: Maximum normalized edit distance to add an edge.
        k: Number of clusters used during unit extraction.
        lmbda: Regularization weight for DPWFST segmentation.
        batch_size: Number of nodes to process per batch when computing edges.

    Returns:
        ig.Graph: Graph with nodes as embeddings and weighted edges based on similarity.
    """

    graph_dir = Path(f"hacked_representations_graphs/{language}/{dataset}/{model_name}/{layer}/k{k}/{lmbda}/")
    graph_dir.mkdir(parents=True, exist_ok=True)

    graph_pattern = graph_dir / f"graph_ed_{threshold:.2f}_*.pkl"
    existing_graphs = list(graph_pattern.parent.glob(graph_pattern.name))
    if existing_graphs:
        print(f"Graph already exists at {existing_graphs[0]}, loading it.")
        return ig.Graph.Read_Pickle(str(existing_graphs[0]))

    else:

        features, paths = load_hacked_units(language, dataset, model_name, layer, k, lmbda)
        g = ig.Graph()
        g.add_vertices(len(features))
        g.vs["name"] = paths

        def compute_edges_batch(i_range):
            """
            Compute similarity edges for a batch of utterance indices.

            Args:
                i_range: Range of node indices (subset of utterances) to process.

            Returns:
                batch_edges: List of (i, j, weight) tuples, where
                    - i, j: Node indices for connected utterances.
                    - weight: Similarity score = 1 - normalized edit distance.
            """
            batch_edges = []
            for i in i_range:
                seq_i = features[i]
                for j in range(i + 1, len(features)):
                    seq_j = features[j]

                    dist = editdistance.normalized_distance(seq_i, seq_j)

                    if dist < threshold:
                        batch_edges.append((i, j, 1 - dist))
            return batch_edges
        
        edge_batches = Parallel(n_jobs=-1)(
            delayed(compute_edges_batch)(batch)
            for batch in tqdm(list(chunk_indices(len(features), batch_size)), desc="Computing edges")
        )
        edges = [edge for batch in edge_batches for edge in batch]
        g.add_edges([(i, j) for i, j, _ in edges])
        g.es["weight"] = [w for _, _, w in edges]

        graph_path = graph_dir / f"graph_ed_{threshold:.2f}_0.0.pkl"
        g.write_pickle(str(graph_path))

    return g


def hacked_ed_graph(res: float = None) -> None:
    """
    Main pipeline for building a hacked edit-distance graph, partitioning it using CPM,
    saving the partition, and evaluating it.

    """
    
    boundary_df = pd.read_csv("Data/alignments/english/test_boundaries.csv")
    num_clusters = len(set(boundary_df['text'].tolist()))   
    partition_file = find_partition(
        partition_type="hacked_graph",
        language="english",
        dataset="test",
        model_name="wavlm-large",
        layer=21,
        distance_type="ed",
        threshold=0.65,
        k= 500,
        lmbda=100
    )
    if res is not None:
        print(f"Repartitioning with resolution {res}")
        partition_file = None
    if partition_file is None:
        g = build_hacked_ed_graph(
            language="english",
            dataset="test",
            model_name="wavlm-large",
            layer=21,
            threshold=0.65,
            k=500,
            lmbda=100.0,
            batch_size=1_000
        )
        if res is None:
            res = 0.3

        membership, _ = CPM_partition(g, num_clusters, res)
        partition_file = write_partition(
            partition_type="hacked_graph",
            partition_membership=membership,   
            language="english",
            dataset="test",
            model_name="wavlm-large",
            layer=21,
            distance_type="ed",
            threshold=0.65,
            word_info = g.vs["name"],
            total_time = 0.0,
            k=500,
            lmbda=0
        )

    evaluate_partition_file(
        partition_file, "english", "test", "wavlm-large", 0.0, threshold=0.65, k=500, lmbda=100
    )


def hacked_kmeans() -> None:
    """
    Main pipeline for loading hacked continuous feature embeddings, clustering them using KMeans,
    saving the partition, and evaluating it.

    """
    
    boundary_df = pd.read_csv("Data/alignments/english/test_boundaries.csv")
    num_clusters = len(set(boundary_df['text'].tolist()))   

    partition_file = find_partition(
        partition_type="hacked_kmeans",
        language="english",
        dataset="test",
        model_name="wavlm-large",
        layer=21,
        distance_type=None,
        threshold=None,
        k=None,
        lmbda=None
    )
    if partition_file is None:
        features, paths = load_hacked_features()
        
        cluster_ids = cluster_features_kmeans(features, num_clusters)
        membership = convert_cluster_ids_to_partition(cluster_ids)

        partition_file = write_partition(
            partition_type="hacked_kmeans",
            partition_membership=membership,   
            language="english",
            dataset="test",
            model_name="wavlm-large",
            layer=21,
            distance_type=None,
            threshold=None,
            word_info = paths,
            total_time = 0.0,
            k=None,
            lmbda=None
        )

    evaluate_partition_file(
        partition_file, "english", "test", "wavlm-large", 0.0
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Builds a similarity graph from hacked unit sequences.")
    parser.add_argument("partition_type", type=str, choices=["graph", "kmeans"], help="Type of partition to perform.")
    parser.add_argument("--res", type=float, default=None, help="If set, only evaluate existing partition without rebuilding graph.")
    args = parser.parse_args()
    
    if args.partition_type == "graph":
        if args.res is not None:
            hacked_ed_graph(res=args.res)
        else:
            hacked_ed_graph()
    elif args.partition_type == "kmeans":
        hacked_kmeans()
    else:
        raise ValueError(f"Unsupported partition type: {args.partition_type}")