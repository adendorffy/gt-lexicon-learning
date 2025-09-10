import time
from pathlib import Path
import numpy as np
import h5py
import json
from typing import List, Tuple, Union

from utils.average import apply_pca
from utils.cython_dtw import _dtw
from utils.cut_features import load_cut_features


def pair_index_to_ij(k: int, n: int) -> Tuple[int, int]:
    """
    Convert a linearized pair index into its corresponding (i, j) indices
    for the upper triangular matrix of pairwise combinations.

    This is useful when iterating over all unique pairs of `n` items without
    explicitly constructing the full matrix of pairs.

    Args:
        k (int): The linearized pair index (0 ≤ k < n*(n-1)/2).
        n (int): The total number of items.

    Returns:
        Tuple[int, int]: The corresponding pair of indices (i, j) such that
        0 ≤ i < j < n.
    """
    i = int(n - 2 - np.floor(np.sqrt(-8 * k + 4 * n * (n - 1) - 7) / 2.0 - 0.5))
    j = int(k + i + 1 - n * (n - 1) // 2 + (n - i) * ((n - i) - 1) // 2)
    return i, j


def compute_dtw_distances(
    language: str,
    dataset: str,
    model_name: str,
    layer: Union[int, str],
    batch_size: int = 1_000_000,
) -> Tuple[Path, Path, List[str]]:
    """
    Compute pairwise DTW (Dynamic Time Warping) distances between word-level
    features and save the results incrementally to an HDF5 file.

    Features are first loaded and optionally transformed (via PCA),
    and then all unique pairs are iterated over. The computation supports
    checkpointing so that previously computed distances are not recalculated.

    Args:
        language (str): The language identifier (e.g., "english", "mandarin").
        dataset (str): Dataset name to load features from.
        model_name (str): The model used to extract features.
        layer (Union[int, str]): The layer of the model from which to extract features.
        batch_size (int, optional): Number of pairwise distances to compute
            and store in memory before flushing to disk. Defaults to 1,000,000.

    Returns:
        Tuple[Path, Path, List[str]]:
            - Path to the HDF5 file containing distances.
            - Path to the JSON file with per-chunk timing information.
            - List of feature paths corresponding to the word features.
    """
    word_features, paths = load_cut_features(language, dataset, model_name, layer)
    if not word_features:
        raise ValueError("No word features found.")
    
    n = len(word_features)
    total_pairs = (n * (n - 1)) // 2

    distances_dir = Path(f"distances/{language}/{dataset}/{model_name}/{layer}")
    distances_dir.mkdir(parents=True, exist_ok=True)

    distance_file = distances_dir / f"dtw_distances.h5"
    buffer = np.empty((batch_size,), dtype=[("i", "i4"), ("j", "i4"), ("dist", "f4")])

    if distance_file.exists():
        with h5py.File(distance_file, "r+") as f:
            dset = f["dist"]
            current_size = dset.shape[0]

            if current_size >= total_pairs:
                print("All pairs already computed, exiting.")
                return distance_file, times_file, paths
            
            print(f"Resuming from {current_size} pairs, total pairs: {total_pairs}")

    else:
        with h5py.File(distance_file, "w") as f:
            dset = f.create_dataset(
                "dist",
                shape=(0,),
                maxshape=(None,),
                dtype=buffer.dtype,
                compression="gzip",
                chunks=(batch_size,),
            )
            current_size = 0
        print(f"Created new dataset at {distance_file}")

    chunck_times = []
    times_file  = distances_dir / f"dtw_times.json"
    if times_file.exists():
        with open(times_file, "r") as f:
            chunk_times = json.load(f)
    

    word_features = apply_pca(word_features)
    word_features = np.array([feat.astype(np.float64) for feat in word_features])

    batch_idx = current_size // batch_size
    buff_idx = 0

    
    with h5py.File(distance_file, "r+") as f:
        dset = f["dist"]
        chunk_start_time = time.time()

        for pair_idx in range(current_size, total_pairs):
            i, j = pair_index_to_ij(pair_idx, n)
            dist = _dtw.multivariate_dtw_cost_cosine(word_features[i], word_features[j], True)

            buffer[buff_idx] = (i, j, dist)
            buff_idx += 1

            if buff_idx == batch_size or pair_idx == total_pairs - 1:
                dset.resize(dset.shape[0] + buff_idx, axis=0)
                dset[-buff_idx:] = buffer[:buff_idx]
                f.flush()

                buff_idx = 0
                batch_end_time = time.time()

                chunk_duration = batch_end_time - chunk_start_time
                chunck_times.append(chunk_duration)
                with open(times_file, "w") as tf:
                    json.dump(chunck_times, tf)

                print(f"Processed {pair_idx + 1}/{total_pairs} pairs. Chunk time: {chunk_duration:.2f}s")
                chunk_start_time = time.time()
                batch_idx += 1
    
    return distance_file, times_file, paths

