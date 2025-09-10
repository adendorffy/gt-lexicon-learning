import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time
from typing import Generator, List, Tuple, Optional
from joblib import Parallel, delayed
import igraph as ig
# import editdistance
import rapidfuzz.distance.Levenshtein as editdistance
import h5py
import json

from utils.cut_features import load_cut_features
from utils.discretise import load_units
from utils.average import apply_pca, pooling
from utils.dtw_distances import compute_dtw_distances


def chunk_indices(n: int, batch_size: int) -> Generator[List[int], None, None]:
    """
    Yield index batches of size `batch_size` until `n` is exhausted.

    Args:
        n: Total number of items.
        batch_size: Number of items per batch.

    Yields:
        List of indices for the current batch.
    """
    for i in range(0, n, batch_size):
        yield range(i, min(i + batch_size, n))


def cos_graph(
    language: str,
    dataset: str,
    model_name: str,
    layer: int,
    threshold: float,
    batch_size: int = 1_000,
) -> Tuple[ig.Graph, float]:
    """
    Build or load a cosine similarity graph for word-level features.

    Args:
        language: Dataset language ("english" or "mandarin").
        dataset: Dataset split (e.g. "train", "dev", "test").
        model_name: Model used for features.
        layer: Layer index for features.
        threshold: Cosine distance threshold (similarity > 1 - threshold).
        batch_size: Number of nodes per parallel batch.

    Returns:
        g: iGraph graph with vertices = words and weighted edges for similarities.
        total_time: Time taken to compute (seconds).
    """
    graph_dir = Path(f"graphs/{language}/{dataset}/{model_name}/{layer}")
    graph_dir.mkdir(parents=True, exist_ok=True)

    graph_pattern = graph_dir / f"cos_{threshold}_*.pkl"
    existing_graphs = list(graph_pattern.parent.glob(graph_pattern.name))
    if existing_graphs:
        graph_time = float(existing_graphs[0].stem.split("_")[-1])
        print(f"Graph with threshold {threshold} already exists, time taken {graph_time}: . Loading from disk.")
        g = ig.Graph.Read_Pickle(existing_graphs[0])
        return g, graph_time
    
    word_features, paths = load_cut_features(language, dataset, model_name, layer)
    print(f"Loaded {len(word_features)} word features.")

    word_features = apply_pca(word_features)
    word_features = pooling(word_features)

    start_time = time.time()

    g = ig.Graph()
    g.add_vertices(len(word_features))
    g.vs["name"] = [path for path in paths]

    def compute_edges_batch(
        batch_indices: List[int], features: List[np.ndarray], threshold: float
    ) -> List[Tuple[int, int, float]]:
        """
        Compute similarity edges for a batch of nodes.

        Args:
            batch_indices: List of node indices in the batch.
            features: List of feature vectors (already L2-normalized).
            threshold: Distance threshold (similarity > 1 - threshold).

        Returns:
            List of (i, j, weight) edges.
        """
        batch_edges = []
        for i in batch_indices:
            vec_i = features[i]
            sims = (
                features[i + 1 :] @ vec_i.T
            )  

            above_thresh = np.where(sims > 1 - threshold)[0]
            for idx in above_thresh:
                j = i + 1 + idx
                batch_edges.append((i, j, sims[idx]))
        return batch_edges

    edge_batches = Parallel(n_jobs=-1)(
        delayed(compute_edges_batch)(batch, word_features, threshold)
        for batch in tqdm(list(chunk_indices(len(word_features), batch_size)), desc="Computing edges")
    )

    edges = [edge for batch in edge_batches for edge in batch]
    g.add_edges([(i, j) for i, j, _ in edges])
    g.es["weight"] = [w for _, _, w in edges]

    total_time = round(time.time() - start_time)
    graph_path = graph_dir / f"cos_{threshold}_{total_time}.pkl"
    g.write_pickle(graph_path)

    return g, total_time


def ed_graph(language: str,
    dataset: str,
    model_name: str,
    layer: int,
    threshold: float,
    k: int,
    lmbda: float,
    batch_size: int = 1_000,
) -> Tuple[ig.Graph, float]:
    """
    Build or load an edit-distance similarity graph for unit sequences.

    Args:
        language: Dataset language (e.g. "english", "mandarin").
        dataset: Dataset split (e.g. "train", "dev", "test").
        model_name: Model used for feature extraction.
        layer: Layer index used for features.
        threshold: Normalized edit distance threshold 
            (edge added if distance < threshold).
        k: Number of clusters used in unit extraction.
        lmbda: Regularization weight for DPWFST segmentation.
        batch_size: Number of nodes per parallel batch during edge computation.

    Returns:
        g: iGraph graph with vertices = utterances and weighted edges 
            (weight = 1 - normalized edit distance).
        total_time: Time taken to compute the graph (in seconds).
    """
    graph_dir = Path(f"graphs/{language}/{dataset}/{model_name}/{layer}")
    graph_dir.mkdir(parents=True, exist_ok=True)

    graph_pattern = graph_dir / f"ed_{k}_{lmbda}_{threshold}_*.pkl"
    existing_graphs = list(graph_pattern.parent.glob(graph_pattern.name))
    if existing_graphs:
        print(f"Graph with threshold {threshold} already exists. Loading from disk.")
        g = ig.Graph.Read_Pickle(existing_graphs[0])
        return g, float(existing_graphs[0].stem.split("_")[-1])
    
    
    features, paths = load_units(language, dataset, model_name, layer, k, lmbda)
    
    start_time = time.time()
    g = ig.Graph()
    g.add_vertices(len(features))
    g.vs["name"] = [path for path in paths]

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

                # dist = editdistance.eval(seq_i, seq_j) / max(len(seq_i), len(seq_j))
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

    total_time = time.time() - start_time
    total_time = round(total_time)

    graph_path = graph_dir / f"ed_{threshold}_{k}_{lmbda}_{total_time}.pkl"
    g.write_pickle(graph_path)

    return g, total_time


def dtw_graph(language: str,
    dataset: str,
    model_name: str,
    layer: int,
    threshold: float,
    batch_size: int = 1_000,
) -> Tuple[ig.Graph, float]:
    """
    Build a similarity graph from DTW distances between word-level features.

    The graph is constructed by connecting nodes (words) whose pairwise DTW 
    distance is below a given threshold. Edges are weighted as `1 - dist`, so 
    closer pairs have higher weights. Graphs are cached on disk to avoid 
    recomputation.

    Args:
        language (str): Language identifier (e.g., "english", "mandarin").
        dataset (str): Dataset name (used to locate features).
        model_name (str): Model used to extract features.
        layer (int): Model layer from which features are taken.
        threshold (float): Distance threshold for connecting nodes.
        batch_size (int, optional): Number of distance entries to process per 
            batch when building the graph. Defaults to 1,000.

    Returns:
        Tuple[ig.Graph, float]:
            - The constructed igraph graph object.
            - The total runtime in seconds (rounded).
    """
    graph_dir = Path(f"graphs/{language}/{dataset}/{model_name}/{layer}")
    graph_dir.mkdir(parents=True, exist_ok=True)

    graph_pattern = graph_dir / f"dtw_{threshold}_*.pkl"
    existing_graphs = list(graph_pattern.parent.glob(graph_pattern.name))
    if existing_graphs:
        print(f"Graph with threshold {threshold} already exists. Loading from disk.")
        g = ig.Graph.Read_Pickle(existing_graphs[0])
        return g, float(existing_graphs[0].stem.split("_")[-1])
    
    distance_file, times_file, paths = compute_dtw_distances(
        language, dataset, model_name, layer, batch_size=batch_size
    )

    g = ig.Graph()
    g.add_vertices(len(paths))
    g.vs["name"] = [path for path in paths]

    with open(times_file) as f:
        chunks = json.load(f)

    distances_time = sum(chunks)
    print(f"DTW distances computed in {distances_time:.2f} seconds.")
    start_time = time.time()

    with h5py.File(distance_file, "r") as f:
        dists = f["dist"]
        total = dists.shape[0]
        print(f"Reading {total} distance entries from HDF5 file...")

        for idx in tqdm(range(0, total, batch_size), desc="Building graph"):
            count = min(batch_size, total - idx)
            chunk = dists[idx : idx + count]

            mask = chunk["dist"] < threshold
            if not np.any(mask):
                continue

            i = chunk["i"][mask]
            j = chunk["j"][mask]
            edges = list(zip(i, j))
            
            dist = chunk["dist"][mask]
            weights = 1 - dist

            g.add_edges(edges)
            g.es[-len(edges) :]["weight"] = weights.tolist()
    
    total_time = time.time() - start_time + distances_time
    total_time = round(total_time)

    graph_path = graph_dir / f"dtw_{threshold}_{total_time}.pkl"
    g.write_pickle(graph_path)
    return g, total_time


def build_graph(
    language: str,
    dataset: str,
    model_name: str,
    layer: int,
    threshold: float,
    distance_type: str,
    k: Optional[int] = None,
    lmbda: Optional[float] = None,
) -> Tuple[ig.Graph, float]:
    """
    Build a similarity graph based on the specified distance type.

    Args:
        language: Dataset language.
        dataset: Dataset split.
        model_name: Model used for features.
        layer: Layer index for features.
        threshold: Threshold for similarity graph.
        distance_type: "cos" (cosine) or other (future extensions).
        k: Optional (for k-NN graphs).
        lmbda: Optional (for exponential scaling graphs).

    Returns:
        g: Constructed graph.
        total_time: Time taken (seconds).
    """
    if distance_type == "cos":
        return cos_graph(language, dataset, model_name, layer, threshold)
    
    elif distance_type == "ed":
        if k is None or lmbda is None:
            raise ValueError("k and lmbda must be provided for edit distance graphs.")
        return ed_graph(language, dataset, model_name, layer, threshold, k, lmbda)
    
    elif distance_type == "dtw":
        return dtw_graph(language, dataset, model_name, layer, threshold)
    
    else:
        raise ValueError(f"Unsupported distance type: {distance_type}")
    
