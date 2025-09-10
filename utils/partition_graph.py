import leidenalg as la
import igraph as ig
from time import time
from pathlib import Path
import pandas as pd
from typing import List, Tuple

def CPM_partition(
    g: ig.Graph,
    n_clusters: int = 8_216,
    resolution: float = 0.3,
    max_iterations: int = 25,
    diff_tolerance: int = 10,
    initialise_with: List[int] = None
) -> Tuple[List[List[int]], float]:
    """
    Apply the Constant Potts Model (CPM) for community detection on a graph,
    adjusting the resolution parameter until the target number of clusters
    is reached (within tolerance).

    Uses Leiden with warm-starting (`initial_membership`) so that each iteration
    refines the previous clustering instead of restarting.

    Args:
        g: iGraph graph object.
        n_clusters: Desired number of clusters (default = 8_216).
        resolution: Initial resolution parameter for CPM.
        max_iterations: Maximum number of iterations for the optimiser.
        diff_tolerance: Acceptable difference between target and found clusters.
        initialise_with: Optional initial membership list.

    Returns:
        Tuple of:
            - Best partition (leidenalg.Partition).
            - Runtime in seconds.
    """
    start_time = time()
    partition_type = la.CPMVertexPartition
    lr = 0.1
    best_diff = float("inf")
    best_partition = None
    patience_counter = 0

    partition = la.find_partition(
        g,
        partition_type,
        weights="weight",
        resolution_parameter=resolution,
        initial_membership=initialise_with,
        seed=42,
    )

    for i in range(max_iterations):
        curr_clusters = len(set(partition.membership))
        diff = curr_clusters - n_clusters

        print(f"[Iter {i+1:02}] res={partition.resolution_parameter:.6f}, "
                  f"clusters={curr_clusters}, diff={diff:+d}")

        if abs(diff) <= diff_tolerance:
            print(f"✅ Acceptable resolution found. Res: {resolution:.8f}")
            best_partition = partition
            break
        
        if abs(diff) < abs(best_diff):
            best_diff = diff
            best_partition = partition
            patience_counter = 0
        elif abs(diff) == abs(best_diff):
            print("Same difference as best, keeping previous best partition.")
        else:
            patience_counter += 1


        if patience_counter >= 1:
            lr *= 0.5
            print(f"⚠️ Reducing learning rate to {lr:.6f}")
            patience_counter = 0

        diff_clipped = max(min(diff, 2 * n_clusters), -2 * n_clusters)
        step = lr * (diff_clipped / n_clusters)
        resolution = min(max(resolution - step, -10), 10)
        lr *= 0.9

        partition = la.find_partition(
            g,
            partition_type,
            weights="weight",
            resolution_parameter=resolution,
            initial_membership=partition.membership,
            seed=42,
        )

    end_time = time()
    return best_partition, end_time - start_time


def write_partition(
    partition_type, 
    partition_membership,
    language,
    dataset,
    model_name,
    layer,
    distance_type,
    threshold,
    word_info,
    total_time,
    k=None,
    lmbda=None
):
    """
    Write clustering or graph partition results to a text file.

    Args:
        partition_type: Type of partition (e.g. "graph", "kmeans").
        partition_membership: List of clusters, where each cluster is a list of node indices.
        language: Dataset language (e.g. "english", "mandarin").
        dataset: Dataset split (e.g. "train", "dev", "test").
        model_name: Model used for feature extraction.
        layer: Layer index used for features.
        distance_type: Distance metric used (e.g. "ed", "cos", "dtw").
        threshold: Distance threshold used during graph construction or clustering.
        word_info: List of identifiers (filename_wordid) for each node.
        total_time: Runtime (in seconds) for the partitioning step.
        k: (Optional) Number of clusters used during unit extraction.
        lmbda: (Optional) Regularization weight for DPWFST segmentation.

    Returns:
        partitions_path: Path to the saved partition file.
    """
      
    boundary_df = pd.read_csv(f"Data/alignments/{language}/{dataset}_boundaries.csv")
    boundary_lookup = {
        (row["filename"], row["word_id"]): row for _, row in boundary_df.iterrows()
    }

    partitions_dir = (
        Path("partitions") / partition_type / language / dataset / model_name / str(layer)
    )
    partitions_dir.mkdir(parents=True, exist_ok=True)

    if partition_type == "graph":
        if k is not None and lmbda is not None:
            partitions_path = partitions_dir / f"{distance_type}_{k}_{lmbda}_{threshold}_{round(total_time)}.txt"
        else:
            partitions_path = partitions_dir / f"{distance_type}_{threshold}_{round(total_time)}.txt"

    elif partition_type in ["kmeans", "birch", "agg"]: 
        partitions_path = partitions_dir / f"{partition_type}_{round(total_time)}.txt"
    

    with open(partitions_path, "w") as f:
        for id, cluster in enumerate(partition_membership):
            f.write(f"Class {id}\n")
            lines = []
            for node in cluster:
                fname, word_id = (
                    word_info[node].split("_")[0],
                    int(word_info[node].split("_")[1]),
                )
                if (fname, word_id) in boundary_lookup:
                    row = boundary_lookup[(fname, word_id)]
                    lines.append(
                        f"{row['filename']} {row['word_start']} {row['word_end']} {row['text']} {row['phones']}\n"
                    )
                else:
                    lines.append(f"{fname}_{word_id}\tError\n")
            f.writelines(lines)
            f.write("\n")

    print(f"Partition saved to {partitions_path}")

    return partitions_path


def find_partition(partition_type, language, dataset, model_name, layer, distance_type, threshold, k=None, lmbda=None):
    """
    Locate the most recent saved partition file for a given configuration.

    Args:
        partition_type: Type of partition ("graph", "kmeans", "birch", "agg").
        language: Dataset language (e.g. "english", "mandarin").
        dataset: Dataset split (e.g. "train", "dev", "test").
        model_name: Model used for feature extraction.
        layer: Layer index used for features.
        distance_type: Distance metric used (e.g. "ed", "cosine").
        threshold: Distance threshold used in graph construction or clustering.
        k: (Optional) Number of clusters used in unit extraction.
        lmbda: (Optional) Regularization weight for DPWFST segmentation.

    Returns:
        latest_file: Path to the most recent partition file if found,
            otherwise None.
    """
    partitions_dir = (
        Path("partitions") / partition_type / language / dataset / model_name / str(layer)
    )

    if partition_type == "graph":
        if k is not None and lmbda is not None:
            partition_path = partitions_dir / f"{distance_type}_{k}_{lmbda}_{threshold}_*.txt"
        else:
            partition_path = partitions_dir / f"{distance_type}_{threshold}_*.txt"

    elif partition_type in ["kmeans", "birch", "agg"]: 
        partition_path = partitions_dir / f"{partition_type}_*.txt"
    
    else:
        raise ValueError(f"Unsupported partition type: {partition_type}")
    
    matching_files = list(partitions_dir.glob(partition_path.name))
    if matching_files:
        latest_file = max(matching_files, key=lambda f: f.stat().st_mtime)
        print(f"Found existing partition file: {latest_file}")
        return latest_file
    else:
        print("No existing partition file found.")
        return None
    
