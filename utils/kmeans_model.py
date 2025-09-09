
from utils.extract_features import load_features
from utils.build_graph import chunk_indices
from sklearn.cluster import MiniBatchKMeans
from pathlib import Path
import joblib
from tqdm import tqdm
import numpy as np

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
        features, paths = load_features(language, "train", model_name, layer, for_kmeans=True)
    elif language == "mandarin":
        features, paths = load_features(language, "dev", model_name, layer)
    else:
        raise ValueError("Unsupported language. Choose 'english' or 'mandarin'.")
    
    kmeans_model = MiniBatchKMeans(
        n_clusters=k, random_state=42, batch_size=batch_size, max_iter=100
    )

    for batch in tqdm(
        chunk_indices(len(features), batch_size),
        total=(len(features) + batch_size - 1) // batch_size,
        desc="Training KMeans model",
    ):
        batch_features = np.vstack(batch).astype(np.float64)
        kmeans_model.partial_fit(batch_features)

    joblib.dump(kmeans_model, model_path)
    print(f"KMeans model with k={k} saved to {model_path}.")

    return kmeans_model

