from utils.cut_features import load_cut_features
from utils.kmeans_model import build_model
from utils.dpwfst import dpwfst

from typing import List, Tuple
import numpy as np
from pathlib import Path
import pandas as pd
import torch

def load_units(
    language: str,
    dataset: str,
    model_name: str,
    layer: int,
    k: int,
    lmbda: float
) -> Tuple[np.ndarray, List[str]]:
    """
    Load unit sequences (discretized speech features) for a given configuration.

    Parameters
    ----------
    language : str
        Language namee (e.g., 'english', 'mandarin').
    dataset : str
        Dataset name (e.g., 'dev').
    model_name : str
        SSL model identifier (e.g., 'wavlm-large').
    layer : int
        Model layer index to extract features from.
    k : int
        Number of clusters for quantization.
    lmbda : float
        Regularization weight for the DPWFST algorithm.

    Returns
    -------
    features : list of np.ndarray
        List of unit sequences (one per utterance).
    paths : list of str
        Corresponding utterance identifiers (file stems).
    """
    units_dir = Path(f"units/{language}/{dataset}/{model_name}/{layer}/k{k}/{lmbda}")
    paths = [file.stem for file in units_dir.glob("*.npy")]
    paths.sort()

    boundary_df = pd.read_csv(f"Data/alignments/{language}/{dataset}_boundaries.csv")
    if len(boundary_df) != len(paths):
        print(f"Warning: Number of unit files ({len(paths)}) does not match number of boundaries ({len(boundary_df)}).")
        extract_units(language, dataset, model_name, layer, k, lmbda)
        paths = [file.stem for file in units_dir.glob("*.npy")]
    
    features = []
    for file in paths:
        embedding = np.load(file)
        features.append(embedding)
    
    return features, paths

def extract_units(
    language: str,
    dataset: str,
    model_name: str,
    layer: int,
    k: int,
    lmbda: float
) -> None:
    """
    Extract unit sequences from cut features and save them to disk.

    Parameters
    ----------
    language : str
        Language namee (e.g., 'english', 'mandarin').
    dataset : str
        Dataset name.
    model_name : str
        SSL model identifier.
    layer : int
        Model layer index.
    k : int
        Number of clusters for quantization.
    lmbda : float
        Regularization weight for DPWFST.

    Notes
    -----
    - Requires pre-extracted cut features (`load_cut_features`).
    - Saves unit sequences to `units/{language}/{dataset}/{model_name}/{layer}/k{k}/{lmbda}/`.
    """
    features, paths = load_cut_features(language, dataset, model_name, layer)
    if not features:
        print("No cut features found. Please run cut_features first.")
        return

    model = build_model(language, model_name, layer, k)
    codebook = torch.from_numpy(model.cluster_centers_).type(torch.float32)

    units_dir = Path(f"units/{language}/{dataset}/{model_name}/{layer}/k{k}/{lmbda}")
    units_dir.mkdir(parents=True, exist_ok=True)

    for feature, path in zip(features, paths):
        if not isinstance(feature, torch.Tensor):
            feature = torch.tensor(feature, dtype=torch.float32)
            
        units = dpwfst(
            feature,
            codebook,
            lmbda,
            num_neighbors=10,
            deduplicate=False,
        )[0].numpy()

        save_path = units_dir / f"{path}.npy"
        np.save(save_path, units)

