from utils.cut_features import load_cut_features
from utils.kmeans_model import build_model
from utils.dpwfst import dpwfst

from typing import List, Tuple, Optional
import numpy as np
from pathlib import Path

def load_units(language: str, dataset: str, model_name: str, layer: int, k: int, lmbda: float) -> Tuple[np.ndarray, List[str]]:
    units_dir = Path(f"units/{language}/{dataset}/{model_name}/{layer}/k{k}/{lmbda}")
    unit_files = list(units_dir.glob("*.npy"))
    unit_files.sort()

    