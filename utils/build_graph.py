import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time
from typing import Generator, List, Tuple, Optional
from joblib import Parallel, delayed
import igraph as ig

from utils.cut_features import load_cut_features
from utils.average import apply_pca, pooling


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
    word_features, paths = load_cut_features(language, dataset, model_name, layer)
    print(f"Loaded {len(word_features)} word features.")

    word_features = apply_pca(word_features)
    word_features = pooling(word_features)

    graph_dir = Path(f"graphs/{language}/{dataset}/{model_name}/{layer}")
    graph_dir.mkdir(parents=True, exist_ok=True)
    graph_pattern = graph_dir / f"cos_{threshold}_*.pkl"
    existing_graphs = list(graph_pattern.parent.glob(graph_pattern.name))
    if existing_graphs:
        print(f"Graph with threshold {threshold} already exists. Loading from disk.")
        g = ig.Graph.Read_Pickle(existing_graphs[0])
        return g, float(existing_graphs[0].stem.split("_")[-1])

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
    
    units, paths = load_units(language, dataset, model_name, layer, k, lmbda)


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
    
    else:
        raise ValueError(f"Unsupported distance type: {distance_type}")