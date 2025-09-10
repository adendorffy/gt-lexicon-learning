from utils.partition_graph import CPM_partition, write_partition
from utils.evaluate_partition import evaluate_partition_file
from utils.build_graph import chunk_indices
from utils.discretise import load_units

from argparse import Namespace
import pandas as pd
import numpy as np
from tqdm import tqdm   
import igraph as ig
import rapidfuzz.distance.Levenshtein as editdistance
from joblib import Parallel, delayed
from typing import List


def build_ed_graph_no_saving(
    language: str = "english",
    dataset: str = "test",
    model_name: str = "wavlm-large",
    layer: int = 21,
    threshold: float = 0.65,
    k: int = 500,
    lmbda: float = 0,
    batch_size: int = 1_000
) -> ig.Graph:
    """
    Build a graph where edges are added based on
    normalized edit distance between sequences without saving the resulting graph.

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
    features, paths = load_units(language, dataset, model_name, layer, k, lmbda)
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

    return g, paths


def get_true_types_from_paths(paths: List[str], boundary_df: pd.DataFrame) -> List[int]:
    """
    Retrieve the true word types (text IDs) for a list of node paths using a boundary DataFrame.

    Args:
        paths: List of identifiers in the format "filename_wordid" corresponding to nodes.
        boundary_df: DataFrame containing columns ['filename', 'word_id', 'text'], representing
                     ground-truth word boundaries and type IDs.

    Returns:
        List[int]: List of true word types for each path. If a path does not match any row in
                   the boundary_df, -1 is appended and a warning is printed.
    """
    true_types = []
    for path in paths:
        filename = path.split("_")[0]
        word_id = int(path.split("_")[1])
        row = boundary_df[(boundary_df['filename'] == filename) & (boundary_df['word_id'] == word_id)]
        if not row.empty:
            true_types.append(row['text'].values[0])
        else:
            print(f"Warning: No matching row found for {path}")
            true_types.append(-1)  
    
    return true_types
        

def main(args: Namespace) -> None:
    """
    Main pipeline for building a hacked edit-distance graph, partitioning it using CPM,
    saving the partition, and evaluating it.

    Args:
        args: Namespace containing configuration parameters:
            - language, dataset, model_name, layer
            - threshold, k, lmbda
            - res: CPM resolution parameter
    """
    
    boundary_df = pd.read_csv(f"Data/alignments/{args.language}/{args.dataset}_boundaries.csv")
    num_clusters = len(set(boundary_df['text'].tolist()))   

    g, paths = build_ed_graph_no_saving(
        language=args.language,
        dataset=args.dataset,
        model_name=args.model_name,
        layer=args.layer,
        threshold=args.threshold,
        k=args.k,
        lmbda=args.lmbda,
    )
    true_types = get_true_types_from_paths(paths, boundary_df)

    membership, _ = CPM_partition(g, num_clusters, args.res, initialise_with=true_types)
    partition_file = write_partition(
        partition_type="hacked_graph",
        partition_membership=membership,   
        language=args.language,
        dataset=args.dataset,
        model_name=args.model_name,
        layer=args.layer,
        distance_type="ed",
        threshold=args.threshold,
        word_info = g.vs["name"],
        total_time = 0.0,
        k=args.k,
        lmbda=args.lmbda
    )

    evaluate_partition_file(
        partition_file, args.language, args.dataset, args.model_name, 0.0,
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Builds a similarity graph from hacked unit sequences.")
    parser.add_argument(
        "--language",
        type=str,
        choices=["english", "mandarin"],
        default="english",
        help="Dataset language.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["train", "dev", "test"],
        default="test",
        help="Dataset split.",
    )
    parser.add_argument(
        "model_name",
        type=str,
        default="wavlm-large",
        help="Model used for feature extraction.",
    )
    parser.add_argument(
        "--layer", type=int, default=21, help="Layer index used for features."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.65,
        help="Distance threshold used during graph construction.",
    )
    parser.add_argument(
        "--k", type=int, default=500, help="Number of clusters used during unit extraction."
    )
    parser.add_argument(
        "--lmbda", type=float, default=0, help="Regularization weight for DPWFST segmentation."
    )
    parser.add_argument(
        "--res",
        type=float,
        default=0.3,
        help="Resolution parameter for the CPM graph partitioning algorithm.",
    )

    args = parser.parse_args()
    main(args)