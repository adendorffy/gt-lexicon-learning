from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize, StandardScaler
from typing import List

import numpy as np
from tqdm import tqdm


def apply_pca(
    word_features: List[np.ndarray], num_components: int = 350
) -> List[np.ndarray]:
    """
    Applies PCA for dimensionality reduction on word-level feature sequences.

    Args:
        word_features: List of word-level feature sequences, 
            where each item is a (T, D) array (T = time steps, D = feature dim).
        num_components: Number of PCA components to retain.

    Returns:
        split_features: List of word-level features after PCA,
            each reduced to (T, num_components).
    """
    pca = PCA(n_components=num_components)
    scaler = StandardScaler()
    
    lengths = [len(feat) for feat in word_features]
    word_features = np.vstack(word_features)
    word_features = scaler.fit_transform(word_features)

    pca.fit(word_features)
    word_features = pca.transform(word_features)

    split_features = []
    index = 0

    for length in tqdm(lengths, desc="Reducing dimensionality with PCA"):
        split_features.append(word_features[index:index+length, :])
        index += length

    return split_features


def pooling(word_features: List[np.ndarray]) -> np.ndarray:
    """
    Pools word-level feature sequences into fixed-size embeddings using mean pooling,
    followed by L2 normalization.

    Args:
        word_features: List of word-level feature sequences,
            where each item is (T, D) after PCA.

    Returns:
        pooled_features: Array of pooled embeddings, shape (N, D),
            where N = number of words.
    """

    pooled_features = []
    for feat in tqdm(word_features, desc="Pooling features"):
        if len(feat.shape) == 1:
            feat = feat.reshape(1, -1)
        pooled_features.append(np.mean(feat, axis=0).astype(np.float32))

    
    pooled_features = normalize(
                np.stack(pooled_features, axis=0), axis=1, norm="l2"
            ).astype(np.float64)
    
    print(f"Features shape after PCA and normalisation: {pooled_features.shape}")

    return pooled_features