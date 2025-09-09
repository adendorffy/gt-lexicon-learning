import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, AutoModel
from tqdm import tqdm
import numpy as np
from pathlib import Path
import pandas as pd
from typing import List


def get_audio_files(language: str, dataset: str) -> List[Path]:
    """
    Retrieve audio file paths for a given language and dataset.

    Args:
        language (str): The language ("english" or "mandarin").
        dataset (str): The dataset split (e.g., "train", "dev", "test").

    Returns:
        List[Path]: List of audio file paths.
    """

    if language == "english":
        audio_dir = Path(f"Data/audio/english/{dataset}")
        audio_files = list(audio_dir.rglob("**/*.flac"))
    elif language == "mandarin":
        audio_dir = Path(f"Data/audio/mandarin/{dataset}")
        audio_files = list(audio_dir.rglob("**/*.wav"))
    else:
        print(f"Language {language} not supported.")
        return []
    
    return audio_files


def load_features(language: str, dataset: str, model_name: str, layer: int, for_kmeans: bool = False) -> List[np.ndarray]:
    """
    Load all feature files from a directory.

    Args:
        language (str): Language ("english" or "mandarin").
        dataset (str): Dataset split (e.g., "train", "dev").
        model_name (str): Name of the model used for feature extraction.
        layer (int): Model layer from which features were extracted.
        for_kmeans (bool): If True, extract 50h of features for k-means training.

    Returns:
        List[np.ndarray]: List of loaded feature arrays.
    """
    if language == "english":
        feature_dir = Path(f"utterance_features/{language}/{dataset}/{model_name}/{layer}")
    elif language == "mandarin":
        feature_dir = Path(f"segment_features/{language}/{dataset}/{model_name}/{layer}")
    else:
        print(f"Language {language} not supported.")
        return []
    
    audio_files = get_audio_files(language, dataset)
    features = [np.load(file).astype(np.float32) for file in feature_dir.glob("*.npy")]
    paths = [path.stem for path in feature_dir.glob("*.npy")]

    if len(features) < len(audio_files):
        print(f"Warning: Expected at least {len(audio_files)} features, but found {len(features)} in {feature_dir}. Re-extracting features.")
        extract_features(language, dataset, model_name, layer, for_kmeans)
        features = [np.load(file).astype(np.float32) for file in feature_dir.glob("*.npy")]
        paths = [path.stem for path in feature_dir.glob("*.npy")]

    return features, paths


def extract_features(language: str, dataset: str, model_name: str, layer: int, for_kmeans: bool = False) -> None:
    """
    Extract features for a dataset using the specified model.

    Args:
        language (str): Language ("english" or "mandarin").
        dataset (str): Dataset split (e.g., "train", "dev").
        model_name (str): Name of the model to use.
        layer (int): Model layer to extract features from.
        for_kmeans (bool): If True, extract 50h of features for k-means training.
    """

    audio_files = get_audio_files(language, dataset)

    if language == "english":
        feature_dir = Path(f"utterance_features/{language}/{dataset}/{model_name}/{layer}")
        feature_dir.mkdir(parents=True, exist_ok=True)
        extract_utterance_features(audio_files, feature_dir, model_name, layer, for_kmeans)

    elif language == "mandarin":
        feature_dir = Path(f"segment_features/{language}/{dataset}/{model_name}/{layer}")
        feature_dir.mkdir(parents=True, exist_ok=True)
        extract_segment_features(audio_files, feature_dir, model_name, layer)

    else:
        print(f"Language {language} not supported for feature extraction.")
        return 


def extract_utterance_features(audio_files: List[Path], feature_dir: Path, model_name: str, layer: int, for_kmeans: bool = False) -> None:
    """
    Extract utterance-level features for English audio.

    Args:
        audio_files (List[Path]): List of audio files.
        feature_dir (Path): Directory to save features.
        model_name (str): Model to use.
        layer (int): Layer index for feature extraction.
        for_kmeans (bool): If True, extract 50h of features for k-means training.
    """
    paths = [path.stem for path in feature_dir.glob("*.npy")]
    features = [np.load(file).astype(np.float32) for file in feature_dir.glob("*.npy")]

    if feature_dir.exists() and len(features) >= len(audio_files):
        print(f"Features already extracted for {model_name} in {feature_dir}.")
        return 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name == "wavlm-large":
        model_path = "microsoft/wavlm-large"
        model = AutoModel.from_pretrained(model_path, output_hidden_states=True)
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)

    elif model_name == "hubert-large":
        model_path = "facebook/hubert-large-ll60k"
        model = AutoModel.from_pretrained(model_path, output_hidden_states=True)
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)

    elif model_name == "hubert-soft":
        model = torch.hub.load("bshall/hubert:main", "hubert_soft", trust_repo=True)

    elif model_name == "mhubert":
        model_path = "utter-project/mHuBERT-147"
        model = AutoModel.from_pretrained(model_path, output_hidden_states=True)
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)

    elif model_name == "chinese-hubert":
        model_path = "TencentGameMate/chinese-hubert-large"
        model = AutoModel.from_pretrained(model_path, output_hidden_states=True)
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)

    else:
        raise ValueError(f"No model found for {model_name}")


    model = model.to(device)
    model.eval()

    total_hours = 0.0
    for audio_file in tqdm(audio_files, desc="Extracting features"):
        if audio_file.stem in paths:
            continue

        waveform, sr = torchaudio.load(str(audio_file))
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform)
            
        waveform = waveform.to(device)
        total_hours += waveform.size()[1] / 16000 / 3600
        if for_kmeans and total_hours > 50.0:
            break
        
        layer = int(layer)

        if model_name == "hubert-soft":
            if waveform.ndim == 1:
                    waveform = waveform.unsqueeze(0).unsqueeze(0)  
            elif waveform.ndim == 2:
                waveform = waveform.unsqueeze(0)  
            
            with torch.inference_mode():
                outputs = model.units(waveform)
                encoding = outputs.squeeze(0).cpu()

        else:
            input_values = feature_extractor(
                waveform.cpu().numpy(), return_tensors="pt", sampling_rate=16000
            ).input_values
            input_values = input_values.to(device)

            with torch.inference_mode():
                outputs = model(input_values).hidden_states
                encoding = outputs[layer - 1].squeeze(0).cpu()

        features.append(encoding)
        paths.append(audio_file.stem)

        feature_path = feature_dir / f"{audio_file.stem}.npy"
        np.save(feature_path, encoding)


def extract_segment_features(audio_files: List[Path], feature_dir: Path, model_name: str, layer: int) -> None:
    """
    Extract segment-level features for Mandarin audio using VAD segmentation.

    Args:
        audio_files (List[Path]): List of audio files.
        feature_dir (Path): Directory to save features.
        model_name (str): Model to use.
        layer (int): Layer index for feature extraction.
    """

    vad_df = pd.read_csv("Data/alignments/mandarin/dev_vad.csv")
    paths = [path.stem for path in feature_dir.glob("*.npy")]
    features = [np.load(file).astype(np.float32) for file in feature_dir.glob("*.npy")]

    if feature_dir.exists() and len(features) >= len(vad_df):
        print(f"Features already extracted for {model_name} in {feature_dir}.")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name == "wavlm-large":
        model_path = "microsoft/wavlm-large"
        model = AutoModel.from_pretrained(model_path, output_hidden_states=True)
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)

    elif model_name == "hubert-large":
        model_path = "facebook/hubert-large-ll60k"
        model = AutoModel.from_pretrained(model_path, output_hidden_states=True)
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)

    elif model_name == "hubert-soft":
        model = torch.hub.load("bshall/hubert:main", "hubert_soft", trust_repo=True)

    elif model_name == "mhubert":
        model_path = "utter-project/mHuBERT-147"
        model = AutoModel.from_pretrained(model_path, output_hidden_states=True)
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)

    elif model_name == "chinese-hubert":
        model_path = "TencentGameMate/chinese-hubert-large"
        model = AutoModel.from_pretrained(model_path, output_hidden_states=True)
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)

    else:
        raise ValueError(f"No model found for {model_name}")


    model = model.to(device)
    model.eval()

    filename_groups = vad_df.groupby("filename")

    for audio_file in tqdm(audio_files, desc="Extracting features"):
        group = filename_groups.get_group(audio_file.stem)
        if len(group) == 0:
            continue

        if feature_dir / f"{audio_file.stem}_{group['seg_id'].max()}.npy" in paths:
            continue

        waveform, sr = torchaudio.load(str(audio_file))
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform)

        waveform = waveform.to(device)  

        layer = int(layer)
        for _, row in group.iterrows():
            feature_path = feature_dir / f"{audio_file.stem}_{row['seg_id']}.npy"
            if feature_path.stem in paths:
                continue

            start = int(row["start"] * 16000)
            end = int(row["end"] * 16000)
            seg_waveform = waveform[:, start:end]

            if seg_waveform.size()[1] == 0:
                print(f"{row['filename']}, {row['seg_id']} at {row['start']} - {row['end']} is empty. Skipping.\tWaveform length: {waveform.size()[1]}, Segment length: {start}-{end}")
                continue

            if model_name == "hubert-soft":
                if seg_waveform.ndim == 1:
                    seg_waveform = seg_waveform.unsqueeze(0).unsqueeze(0)  
                elif seg_waveform.ndim == 2:
                    seg_waveform = seg_waveform.unsqueeze(0)  
                
                with torch.inference_mode():
                    outputs = model.units(seg_waveform)
                    encoding = outputs.squeeze(0).cpu()

            else:
                input_values = feature_extractor(
                    seg_waveform.cpu().numpy(), return_tensors="pt", sampling_rate=16000
                ).input_values
                input_values = input_values.to(device)

                with torch.inference_mode():
                    outputs = model(input_values).hidden_states
                    encoding = outputs[layer - 1].squeeze(0).cpu().numpy().astype(np.float64)

            np.save(feature_path, encoding)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract features from audio segments."
    )
    parser.add_argument("language", type=str, help="Language code")
    parser.add_argument("dataset", type=str, help="Dataset name")
    parser.add_argument("model_name", type=str, help="Model name")
    parser.add_argument("layer", type=str, help="Layer number to extract features from")

    args = parser.parse_args()

    extract_features(args.language, args.dataset, args.model_name, args.layer)
    