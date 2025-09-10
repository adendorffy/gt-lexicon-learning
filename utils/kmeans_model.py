
from sklearn.cluster import MiniBatchKMeans
from pathlib import Path
import joblib
from tqdm import tqdm
import numpy as np
from typing import Generator, List

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

def build_model(
    language: str,
    model_name: str,
    layer: int,
    k: int,
    batch_size: int = 10_000
) -> MiniBatchKMeans:
    """
    Build or load a KMeans model for clustering word-level features.

    Args:
        language: "english" or "mandarin".
        model_name: Name of the feature extraction model.
        layer: Layer index of the features.
        k: Number of clusters.
        batch_size: Batch size for MiniBatchKMeans.

    Returns:
        Trained MiniBatchKMeans model.
    """

    model_dir = Path(f"models/kmeans/{language}/")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"{model_name}_{layer}_k{k}.joblib"
    if model_path.exists():
        print(f"KMeans model with k={k} already exists. Loading from disk.")
        kmeans = MiniBatchKMeans(n_clusters=k)
        kmeans = joblib.load(model_path)
        return kmeans
    
    if language == "english":
        dataset = "train"
    elif language == "mandarin":
        dataset = "dev"
    else:
        raise ValueError("Language must be 'english' or 'mandarin'.")
    
    kmeans_model = MiniBatchKMeans(
        n_clusters=k, random_state=42, batch_size=batch_size, max_iter=100
    )

    for batch_features in tqdm(
        load_features_batched(language, dataset, model_name, layer, batch_size)[0],
        total=load_features_batched(language, dataset, model_name, layer, batch_size)[1],
        desc="Training KMeans model",
    ):
        kmeans_model.partial_fit(batch_features)

    joblib.dump(kmeans_model, model_path)
    print(f"KMeans model with k={k} saved to {model_path}.")

    return kmeans_model

def load_features_batched(
    language: str,
    dataset: str,
    model_name: str,
    layer: int,
    batch_size: int = 10_000
) -> Generator[np.ndarray, None, None]:
    """
    Generator that yields batches of features from disk for a given language, dataset,
    model, and layer, suitable for MiniBatchKMeans training.

    Args:
        language: "english" or "mandarin".
        dataset: Name of the dataset, e.g., "train" or "dev".
        model_name: Name of the feature extraction model.
        layer: Layer index of features to load.
        batch_size: Number of feature files to load per batch.

    Yields:
        np.ndarray: A 2D array of shape (batch_size, feature_dim) containing the features
                    for the current batch. Last batch may be smaller than batch_size.
    """
    if language == "english":
        feature_dir = Path(f"utterance_features/{language}/{dataset}/{model_name}/{layer}")
    elif language == "mandarin":
        feature_dir = Path(f"segment_features/{language}/{dataset}/{model_name}/{layer}")

    feature_files = list(feature_dir.glob("*.npy"))
    total_batches = (len(feature_files) + batch_size - 1) // batch_size
    if not feature_files:
        raise ValueError(f"No feature files found in {feature_dir}")
    
    for i in range(0, len(feature_files), batch_size):
        batch_files = feature_files[i:i+batch_size]
        batch_features = [np.load(f) for f in batch_files]
        yield np.vstack(batch_features).astype(np.float64), total_batches

