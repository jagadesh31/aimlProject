"""Prepare RAVDESS speech files into train/dev/test wav folders with emotion labels CSV."""
from __future__ import annotations

import argparse
import random
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config

# RAVDESS filename: 03-01-05-01-02-01-12.wav
# modality-vocal-emotion-intensity-statement-repetition-actor
EMO_MAP = {
    "01": "neutral",
    "02": "neutral",  # calm -> neutral for 7-class alignment
    "03": "joy",  # happy
    "04": "sadness",
    "05": "anger",
    "06": "fear",
    "07": "disgust",
    "08": "surprise",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--zip",
        type=Path,
        default=config.DATA_DIR / "ravdess" / "Audio_Speech_Actors_01-24.zip",
    )
    args = parser.parse_args()
    if not args.zip.exists():
        raise SystemExit(f"Missing {args.zip}")

    extract_dir = config.DATA_DIR / "ravdess" / "raw"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip, "r") as zf:
        zf.extractall(extract_dir)

    rows = []
    wavs = list(extract_dir.rglob("*.wav"))
    for wav in wavs:
        parts = wav.stem.split("-")
        if len(parts) < 3:
            continue
        emo = EMO_MAP.get(parts[2])
        if emo is None or emo not in config.EMOTIONS:
            continue
        actor = int(parts[-1])
        rows.append({"path": str(wav.resolve()), "Emotion": emo, "actor": actor, "clip_id": wav.stem})

    df = pd.DataFrame(rows)
    # speaker-independent-ish split by actor id
    train_actors = set(range(1, 21))
    dev_actors = {21, 22}
    test_actors = {23, 24}

    def assign(a):
        if a in train_actors:
            return "train"
        if a in dev_actors:
            return "dev"
        return "test"

    df["split"] = df["actor"].map(assign)
    out_csv = config.DATA_DIR / "ravdess" / "manifest.csv"
    df.to_csv(out_csv, index=False)
    print(df["split"].value_counts().to_string())
    print("Saved", out_csv)

    # symlink/copy into AUDIO_DIR layout expected by optional loader
    import shutil

    for split, g in df.groupby("split"):
        dest = config.AUDIO_DIR / f"ravdess_{split}"
        dest.mkdir(parents=True, exist_ok=True)
        for _, row in g.iterrows():
            target = dest / f"{row['clip_id']}.wav"
            if not target.exists():
                shutil.copy2(row["path"], target)
    print("Copied wavs under", config.AUDIO_DIR)


if __name__ == "__main__":
    main()
