from tqdm import tqdm
import itertools
import numpy as np
import pandas as pd
from collections import defaultdict

import statistics
from intervaltree import Interval, IntervalTree
from sklearn import metrics
import editdistance
from collections import Counter
from typing import Dict, List, Tuple, Any


def distance(p: List[str], q: List[str]) -> float:
    """
    Compute normalized edit distance between two phone sequences.

    Args:
        p (List[str]): First phone sequence.
        q (List[str]): Second phone sequence.

    Returns:
        float: Normalized edit distance in [0, 1].
    """
    if p and q:
        return editdistance.eval(p, q) / max(len(p), len(q))  
    print(f"⚠️ One of the phone sequences is empty: p={p}, q={q}")
    return 1.0


def check_boundary(gold: Interval, disc: Interval) -> bool:
    """
    Check whether a discovered interval sufficiently overlaps a gold interval.

    Args:
        gold (Interval): Ground-truth interval.
        disc (Interval): Discovered interval.

    Returns:
        bool: True if overlap is sufficient to count as a match.
    """
    if gold.contains_interval(disc):
        return True

    gold_duration = round(gold.end - gold.begin, 2)
    overlap_duration = round(gold.overlap_size(disc), 2)
    overlap_percentage = overlap_duration / gold_duration if gold_duration > 0 else 0

    return (gold_duration >= 0.06 and overlap_duration >= 0.03) or (
        gold_duration < 0.06 and overlap_percentage > 0.5
    )


def build_speaker_trees(
    gold_fragments: List[Tuple[str, Interval, List[str], str]]
) -> Dict[str, IntervalTree]:
    """
    Build an interval tree for each speaker from gold fragments.

    Args:
        gold_fragments (List[Tuple[str, Interval, List[str], str]]):
            Tuples of (speaker, interval, phones, word).

    Returns:
        Dict[str, IntervalTree]: Mapping speaker → interval tree.
    """
    trees = defaultdict(IntervalTree)
    for speaker, interval, phones, word in tqdm(gold_fragments, desc="Building trees"):
        trees[speaker].addi(interval.begin, interval.end, (phones, word))
    return trees


def convert_to_intervals(
    boundary_df: pd.DataFrame,
) -> Tuple[List[Tuple[str, Interval, List[str], str]], int]:
    """
    Convert alignment dataframe to gold fragments with intervals.

    Args:
        boundary_df (pd.DataFrame): Alignment dataframe with `word_start`,
            `word_end`, `phones`, `text`, and `filename`.

    Returns:
        Tuple[List[Tuple[str, Interval, List[str], str]], int]:
            - List of gold fragments (filename, interval, phones, word).
            - Count of non-silence fragments.
    """
    fragments = []
    dirty_fragments = []
    non_silence_fragments = 0

    for row in boundary_df.itertuples():
        start = float(row.word_start)
        end = float(row.word_end)
        interval = Interval(start, end)

        if isinstance(row.phones, str):
            non_silence_fragments += 1
            fragments.append(
                (row.filename, interval, list(row.phones.split(",")), row.text)
            )
        else:
            dirty_fragments.append((row.filename, interval, row.dirty_phones))

    return fragments, non_silence_fragments


def transcribe(
    discovered_fragments: List[Tuple[str, Interval, int]],
    trees: Dict[str, IntervalTree],
) -> List[Tuple[int, str, Interval, List[str], str]]:
    """
    Match discovered fragments to gold fragments using boundary checks.

    Args:
        discovered_fragments (List[Tuple[str, Interval, int]]):
            Discovered fragments as (speaker, interval, cluster).
        trees (Dict[str, IntervalTree]): Mapping speaker → gold interval tree.

    Returns:
        List[Tuple[int, str, Interval, List[str], str]]:
            Enriched fragments as (cluster, speaker, interval, phones, word).
    """
    enriched = []
    non_matched = 0

    for speaker, disc_interval, cluster in discovered_fragments:
        match_found = False
        tree = trees.get(speaker)
        if tree is None:
            print(f"⚠️ Speaker {speaker} not in trees.")
        else:
            for match in tree.overlap(disc_interval.begin, disc_interval.end):
                gold_interval = Interval(match.begin, match.end)
                phones, word = match.data

                if check_boundary(gold_interval, disc_interval):
                    enriched.append((cluster, speaker, disc_interval, phones, word))
                    match_found = True
                    break

        if not match_found:
            non_matched += 1
    if non_matched > 0:     
        print(f"Number of non-matched fragments: {non_matched}")
    return enriched


def discover(
    partition_file: str, trees: Dict[str, IntervalTree]
) -> List[Tuple[int, str, Interval, List[str], str]]:
    """
    Parse a partition file and align discovered fragments with gold trees.

    Args:
        partition_file (str): Path to partition file.
        trees (Dict[str, IntervalTree]): Mapping speaker → gold interval tree.

    Returns:
        List[Tuple[int, str, Interval, List[str], str]]:
            Enriched discovered fragments.
    """
    discovered_fragments = []
    cluster_id = None
    with open(partition_file, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("Class"):
                cluster_id = int(line.split()[1])
            else:
                parts = line.split(" ")
                
                if len(parts) < 3:
                    continue
                speaker = parts[0]
                start = parts[1]
                end = parts[2]

                interval = Interval(float(start.strip()), float(end.strip()))
                discovered_fragments.append(
                    (speaker.strip(), interval, int(cluster_id))
                )
            

    return transcribe(discovered_fragments, trees)


def ned_val(discovered_transcriptions: List[Tuple[int, str, Interval, List[str], str]]) -> float:
    """
    Compute Normalized Edit Distance (NED) across discovered clusters.

    Args:
        discovered_transcriptions (List[Tuple[int, str, Interval, List[str], str]]):
            Enriched discovered fragments.

    Returns:
        float: Mean NED across all clusters.
    """
    discovered_transcriptions.sort(key=lambda x: x[0])
    print(f"Example transcription: {discovered_transcriptions[0]}")
    overall_distances = []

    for _, group in itertools.groupby(discovered_transcriptions, key=lambda x: x[0]):
        group_list = list(group)
        group_size = len(group_list)

        if group_size < 2:
            continue

        cluster_distances = [
            distance(p[3], q[3]) for p, q in itertools.combinations(group_list, 2)
        ]
        overall_distances.extend(cluster_distances)

    if overall_distances:
        overall_avg = statistics.mean(overall_distances)
    else:
        overall_avg = 0.0

    return round(overall_avg, 5)


def purity(labels_true: List[str], labels_pred: List[int]) -> float:
    """
    Compute clustering purity.

    Args:
        labels_true (List[str]): Ground-truth labels.
        labels_pred (List[int]): Predicted cluster labels.

    Returns:
        float: Purity score in [0, 1].
    """
    labels_true = np.array(labels_true)
    labels_pred = np.array(labels_pred)

    purity = 0.0
    n_tokens = 0

    for k in set(labels_pred):
        labels_true_in_cluster = labels_true[labels_pred == k]
        counts = Counter(labels_true_in_cluster)
        if counts:
            purity += max(counts.values())
            n_tokens += sum(counts.values())

    return round(purity / n_tokens, 5)


def bitrate(labels_predicted: List[int], total_dur: float) -> float:
    """
    Compute bitrate of discovered units based on entropy of cluster assignments.

    Args:
        labels_predicted (List[int]): Predicted cluster labels.
        total_dur (float): Total duration of speech (seconds).

    Returns:
        float: Bitrate in bits per second.
    """
    labels_str = [str(label) for label in labels_predicted]

    label_counts = Counter(labels_str)
    total_counts = sum(label_counts.values())

    clus_probs = np.array([count / total_counts for count in label_counts.values()])

    entropy = -np.sum(clus_probs * np.log2(clus_probs))
    bitrate = (total_counts * entropy) / total_dur

    return bitrate


def evaluate_partition_file(
    partition_file: str,
    language: str,
    dataset: str,
    model_name: str,
    runtime: float,
    **kwargs: Any,
) -> None:
    """
    Evaluate a partition file against gold alignments.

    Args:
        partition_file (str): Path to discovered partition file.
        language (str): Language identifier.
        dataset (str): Dataset name.
        model_name (str): Model identifier.
        runtime (float): Total runtime of the experiment.
        **kwargs: Additional experiment parameters.

    Returns:
        None
    """
    boundary_df = pd.read_csv(f"Data/alignments/{language}/{dataset}_boundaries.csv")
    boundary_df['duration'] = boundary_df['word_end'] - boundary_df['word_start']
    total_dur = boundary_df['duration'].sum()

    gold_fragments, non_silence_fragments = convert_to_intervals(boundary_df)
    assert non_silence_fragments == len(gold_fragments), "Mismatch in non-silence fragments count."
    print(f"Number of gold fragments: {len(gold_fragments)}")

    trees = build_speaker_trees(gold_fragments)
    discovered_transcriptions = discover(partition_file, trees)
    print(f"{len(discovered_transcriptions)} == {len(gold_fragments)}")

    labels_true = [word for _, _, _, _, word in discovered_transcriptions]
    labels_predicted = [
        cluster_id for cluster_id, _, _, _, _ in discovered_transcriptions
    ]
    num_pred_clusters = len(set(labels_predicted))
    num_true_words = len(set(labels_true))
    diff = num_pred_clusters - num_true_words

    ned = ned_val(discovered_transcriptions)
    pur = purity(labels_true, labels_predicted)
    v_measure = metrics.v_measure_score(labels_true, labels_predicted)
    bitr = bitrate(labels_predicted, total_dur)

    print("\n===== Summary =====")
    print(f"Language         : {language}")
    print(f"Dataset          : {dataset}")
    print(f"Model            : {model_name}")

    print("\n--- Evaluation Metrics ---")
    print(f"NED              : {ned*100:.1f}%")
    print(f"Purity           : {pur*100:.1f}%")
    print(f"V-measure        : {v_measure*100:.1f}%")
    print(f"Bitrate          : {bitr:.1f} bits/s")

    print("\n--- Clustering ---")
    print(f"Predicted clusters: {num_pred_clusters}")
    print(f"True words        : {num_true_words}")
    print(f"Difference        : {diff}")

    print("\n--- Runtime ---")
    print(f"Total runtime (s): {runtime:.2f}")

    print("\n--- Parameters ---")
    for key, value in kwargs.items():
        print(f"{key:>16} : {value}")

