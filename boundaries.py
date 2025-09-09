from tqdm import tqdm
import textgrids
import pandas as pd
from pathlib import Path
import re
from typing import List, Optional


def clean_phones(phones: List[str], mandarin: bool = False) -> List[str]:
    """
    Clean a list of phonetic symbols by removing unwanted entries
    and (optionally) numeric tone markers.

    Args:
        phones (List[str]): A list of phone strings (phonetic symbols).
        mandarin (bool, optional): 
            If False (default): strip digits from phones (e.g., remove stress markers like "a1").
            If True: keep digits (tones are meaningful in Mandarin).
    
    Returns:
        List[str]: The cleaned list of phones.
    """
    if not mandarin:
        phones = [
            re.sub(r"\d+", "", p).lower() for p in phones if p not in {"sp", "sil", "spn", ""}
        ]
    else:
        phones = [p for p in phones if p not in {"sp", "sil", "spn", ""}]
    return phones


def extract_textgrid_boundaries(dataset: str) -> Optional[pd.DataFrame]:
    """
    Extract ground truth word boundaries and associated phones 
    from TextGrid alignment files.

    Args:
        dataset (str): The dataset name (used to locate alignment files).

    Returns:
        Optional[pd.DataFrame]:
            A DataFrame containing extracted alignments with columns:
                - word_id: Index of the word within the file
                - filename: Name of the audio file (without extension)
                - word_start: Word onset time (in seconds)
                - word_end: Word offset time (in seconds)
                - text: Orthographic word text
                - phones: Comma-separated list of cleaned phones for the word
            
            Returns `None` if no valid alignments are extracted.
    """
    align_dir = Path("Data/alignments/english") / dataset
    align_path = align_dir.parent / f"{dataset}_boundaries.csv"
        
    if align_path.exists():
        print(
            f"Ground truth boundaries already exist at {align_path}. Skipping extraction."
        )
        return pd.read_csv(align_path)
    textgrid_paths = list(align_dir.rglob("**/*.TextGrid"))

    records = []
    for textgrid_path in tqdm(
        textgrid_paths,
        desc="Processing TextGrid files",
    ):
        try:
            grid = textgrids.TextGrid(textgrid_path)
            filename = textgrid_path.stem

            words_tier = grid.get("words")
            phones_tier = grid.get("phones")

            if not words_tier or not phones_tier:
                print(f"⚠️ Missing 'words' or 'phones' tier in {filename}. Skipping.")
                continue

            for idx, word_interval in enumerate(words_tier, 1):
                word_text = word_interval.text.strip()
                word_start, word_end = word_interval.xmin, word_interval.xmax

                word_phones = [
                    phone_interval.text.strip()
                    for phone_interval in phones_tier
                    if phone_interval.xmax > word_start
                    and phone_interval.xmin < word_end
                ]
                phones = clean_phones(word_phones)
                if len(phones) > 0:
                    records.append(
                        {
                            "word_id": idx,
                            "filename": filename,
                            "word_start": word_start,
                            "word_end": word_end,
                            "text": word_text,
                            "phones": ",".join(phones),
                        }
                    )

        except Exception as e:
            print(f"⚠️ Error processing {textgrid_path}: {e}")

    if not records:
        print(f"⚠️ No valid alignments extracted from {align_dir}.")
        return None

    print(f"Total words processed: {len(records)}")
    alignments_df = pd.DataFrame(records)
    alignments_df.to_csv(align_path, index=False)
    print(f"Saved GT word boundaries with phones to {align_path}")
    return alignments_df


def extract_wrd_boundaries(dataset: str) -> Optional[pd.DataFrame]:
    """
    Extract ground truth word boundaries and associated phones 
    from WRD (word) and PHN (phone) alignment files for Mandarin datasets.

    Args:
        dataset (str): The dataset name (used to locate alignment files).

    Returns:
        Optional[pd.DataFrame]:
            A DataFrame containing extracted alignments with columns:
                - word_id: Index of the word within each file
                - filename: Name of the audio file
                - word_start: Word onset time (in seconds)
                - word_end: Word offset time (in seconds)
                - text: The orthographic word
                - phones: Comma-separated list of phones aligned to the word
            
            Returns `None` if no valid alignments are extracted.
    """
    align_dir = Path("Data/alignments/mandarin") / dataset
    align_path = align_dir.parent / f"{dataset}_boundaries.csv"

    if align_path.exists():
        print(
            f"Ground truth boundaries already exist at {align_path}. Skipping extraction."
        )
        return pd.read_csv(align_path)

    wrd_path = align_dir / f"mandarin.wrd"
    phn_path = align_dir / f"mandarin.phn"
    records = []

    with open(phn_path, "r") as f:
        phn_lines = [line.strip().split() for line in f.readlines()]
        phn_entries = [
            {
                "filename": fname,
                "start": float(start),
                "end": float(end),
                "phone": phone,
            }
            for fname, start, end, phone in phn_lines
        ]

    with open(wrd_path, "r") as f:
        word_lines = f.readlines()

    word_id = 1
    prev_filename = None
    for line in tqdm(word_lines, desc="Processing WRD file"):
        filename, start, end, word = line.strip().split()
        
        start = float(start)
        end = float(end)

        if prev_filename is None or filename != prev_filename:
            word_id = 0
            prev_filename = filename
            
        phones_in_word = [
            ph["phone"]
            for ph in phn_entries
            if ph["filename"] == filename and ph["start"] >= start and ph["end"] <= end
        ]
        phones_in_word = clean_phones(phones_in_word, mandarin=True)
        
        records.append(
            {
                "word_id": word_id,
                "filename": filename,
                "word_start": start,
                "word_end": end,
                "text": word,
                "phones": ",".join(phones_in_word),
            }
        )
        word_id += 1

    if not records:
        print(f"⚠️ No valid alignments extracted from {align_dir}.")
        return None

    align_df = pd.DataFrame(records)
    align_df.to_csv(align_path, index=False)
    print(f"Saved GT word boundaries with phones to {align_path}")
    return align_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract ground truth boundaries from TextGrid or WRD files."
    )
    parser.add_argument(
        "language",
        type=str,
        choices=["english", "mandarin", "wolof"],
        help="Language of the dataset (english or other).",
    )
    parser.add_argument(
        "dataset",
        type=str,
        help="Name of the dataset to process.",
    )

    args = parser.parse_args()

    if args.language == "english":
        extract_textgrid_boundaries(args.dataset)
    elif args.language == "mandarin":
        extract_wrd_boundaries(args.dataset)
    else:
        print(f"Language '{args.language}' not supported for ground-truth boundary extraction.")