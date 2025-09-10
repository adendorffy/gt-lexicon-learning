from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm
import pandas as pd
import numpy as np
from utils.extract_features import load_features

def cut_features(language: str, dataset: str, model_name: str, layer: int) -> None:
    """
    Entry point to cut word-level embeddings from pre-extracted features.

    Args:
        language: Language of the dataset ("english" or "mandarin").
        dataset: Dataset split (e.g. "train", "dev", "test").
        model_name: Model used for feature extraction.
        layer: Transformer layer index to use for embeddings.
    """
    if language == "english":
        cut_utterance_features(dataset, model_name, layer)
    elif language == "mandarin":
        cut_segment_features(dataset, model_name, layer)
   

def cut_utterance_features(dataset: str, model_name: str, layer: int) -> None:
    """
    Cuts embeddings at word boundaries for English utterances.

    Args:
        dataset: Dataset split.
        model_name: Model used for feature extraction.
        layer: Transformer layer index to use for embeddings.
    """
    
    features, paths = load_features("english", dataset, model_name, layer)
    boundaries_df = pd.read_csv(f"Data/alignments/english/{dataset}_boundaries.csv")
    boundary_groups = boundaries_df.groupby("filename")

    save_dir = Path(f"cut_features/english/{dataset}/{model_name}/{layer}")
    save_dir.mkdir(parents=True, exist_ok=True)

    existing_files = [path.stem for path in save_dir.glob("*.npy")]

    for fname, embedding in tqdm(
        zip(paths, features),
        total=len(features),
        desc="Cut out word-level embeddings",
    ):
        if fname not in boundary_groups.groups:
            print(f"No alignments found for {fname}. Skipping.")
            continue

        boundaries = boundary_groups.get_group(fname)

        for _, row in boundaries.iterrows():

            word_start = row["word_start"]
            word_end = row["word_end"]
            save_path = save_dir / f"{row['filename']}_{row['word_id']}.npy"
            
            if save_path.stem in existing_files:
                continue
            
            start_frame = int(word_start * 50)
            end_frame = int(word_end * 50)
            word_embedding = embedding[start_frame:end_frame]
            np.save(save_path, word_embedding)


def cut_segment_features(dataset: str, model_name: str, layer: int) -> None:
    """
    Cuts embeddings at word boundaries for Mandarin segments.

    Args:
        dataset: Dataset split.
        model_name: Model used for feature extraction.
        layer: Transformer layer index to use for embeddings.
    """
    features, paths = load_features("mandarin", dataset, model_name, layer)

    vad_df = pd.read_csv(f"Data/boundaries/mandarin/{dataset}_vad.csv")
    vad_groups = vad_df.groupby(["filename", "seg_id"])

    boundaries_df = pd.read_csv(f"Data/boundaries/mandarin/{dataset}_boundaries.csv")
    boundary_groups = boundaries_df.groupby("filename")

    save_dir = Path(f"cut_features/mandarin/{dataset}/{model_name}/{layer}")
    save_dir.mkdir(parents=True, exist_ok=True)

    existing_files = [path.stem for path in save_dir.glob("*.npy")]

    for path, embedding in tqdm(
        zip(paths, features),
        total=len(features),
        desc="Cut out word-level embeddings from segments",
    ):
        (fname, seg_id) = path.stem.split("_")[1]

        if fname not in boundary_groups.groups:
            print(f"No alignments found for {fname}. Skipping.")
            continue

        if (fname, seg_id) not in vad_groups.groups:
            print(f"No VAD found for {fname} segment {seg_id}. Skipping.")
            continue
        
        vad = vad_groups.get_group((fname, seg_id)).iloc[0]
        boundaries = boundary_groups.get_group(fname)

        seg_start = vad["seg_start"]
        seg_end = vad["seg_end"]

        for _, row in boundaries.iterrows():
            
            word_start = row["word_start"]
            word_end = row["word_end"]
            save_path = save_dir / f"{row['filename']}_{row['word_id']}.npy"
            
            if save_path.stem in existing_files:
                continue
            
            if word_start < seg_start or word_end > seg_end:
                continue
            
            relative_start = word_start - seg_start
            relative_end = word_end - seg_end
            start = int(relative_start * 50)
            end = int(relative_end * 50)

            word_embedding = embedding[start:end]
            np.save(save_path, word_embedding)


def load_cut_features(
    language: str, dataset: str, model_name: str, layer: int
) -> Tuple[List[np.ndarray], List[str]]:
    """
    Loads previously cut word-level embeddings from disk.

    Args:
        language: Language of the dataset ("english" or "mandarin").
        dataset: Dataset split.
        model_name: Model used for feature extraction.
        layer: Transformer layer index.

    Returns:
        features: List of embeddings (as NumPy arrays).
        paths: List of corresponding filenames (stems).
    """
    if language == "english" or language == "mandarin":
        cut_dir = Path(f"cut_features/{language}/{dataset}/{model_name}/{layer}")
    else:
        print(f"Language {language} not supported.")
        return [], []
    
    cut_files = list(cut_dir.glob("*.npy"))
    cut_files.sort()

    boundary_df = pd.read_csv(f"Data/alignments/{language}/{dataset}_boundaries.csv")
    if len(boundary_df) != len(cut_files):
        print(f"Warning: Number of cut features ({len(cut_files)}) does not match number of boundaries ({len(boundary_df)}).")
        cut_features(language, dataset, model_name, layer)
        cut_files = list(cut_dir.glob("*.npy"))
        cut_files.sort()

    features = []
    paths = []

    for file in tqdm(cut_files, desc="Loading cut features"):
        embedding = np.load(file)
        features.append(embedding)
        paths.append(file.stem)

    return features, paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("language", type=str, choices=["english", "mandarin"], help="Language of the dataset.")
    parser.add_argument("dataset", type=str, help="Name of the dataset.")
    parser.add_argument("model_name", type=str, help="Name of the model used to extract features (e.g., hubert).")
    parser.add_argument("layer", type=int, help="Layer of the model from which features were extracted (e.g., 9).")
            
    args = parser.parse_args()

    cut_features(args.language, args.dataset, args.model_name, args.layer)
