import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict


def extract_vad() -> Optional[pd.DataFrame]:
    """
    Extract voice activity detection (VAD) segments from word boundary alignments
    for a Mandarin dataset (hardcoded to `dev` for now).

    Reads word-level boundaries from the precomputed CSV, merges consecutive
    word intervals into continuous speech segments, and saves the resulting VAD
    segments to a new CSV.

    Returns:
        Optional[pd.DataFrame]:
            DataFrame with columns:
                - seg_id: Segment index within each file
                - filename: Audio file name
                - start: Start time of the VAD segment (in seconds)
                - end: End time of the VAD segment (in seconds)

            Returns `None` if no records are found.
    """
    language = "mandarin"
    dataset = "dev"

    align_dir = Path("Data/alignments") / language / dataset
    vad_path = align_dir.parent / f"{dataset}_vad.csv"
   
    records = get_records_from_df()

    if not records:
        print(f"⚠️ No valid alignments extracted from {align_dir}.")
        return

    vad_df = pd.DataFrame(records)
    vad_df.to_csv(vad_path, index=False)
    print(f"VAD saved to {vad_path}")
    return vad_df

def get_records_from_df() -> List[Dict]:
    """
    Merge word boundaries from alignment CSV into continuous VAD segments.

    Returns:
        List[Dict]: A list of segment records, each with keys:
            - seg_id: Segment index within the file
            - filename: Audio file name
            - start: Segment start time (in seconds)
            - end: Segment end time (in seconds)
    """
    align_df = pd.read_csv("Data/alignments/mandarin/dev_boundaries.csv")

    seg_id = 0
    seg_start, seg_end = None, None
    prev_filename = None

    records = []
    for _, row in align_df.iterrows():
        word_start = row["word_start"]
        word_end = row["word_end"]
        filename = row["filename"]

        if prev_filename is None or filename != prev_filename:
            if seg_start is not None and seg_end is not None:
                records.append({
                    "seg_id": seg_id,
                    "filename": prev_filename,
                    "start": seg_start,
                    "end": seg_end,
                })

            seg_id = 0
            seg_start = word_start
            seg_end = word_end
            prev_filename = filename
            continue

        if abs(seg_end - word_start) < 0.0001:
            seg_end = word_end  
        else:
            records.append({
                "seg_id": seg_id,
                "filename": filename,
                "start": seg_start,
                "end": seg_end,
            })
            seg_id += 1
            seg_start, seg_end = word_start, word_end

    if seg_start is not None and seg_end is not None:
        records.append({
            "seg_id": seg_id,
            "filename": filename,
            "start": seg_start,
            "end": seg_end,
        })
    
    return records

if __name__ == "__main__":
    extract_vad()