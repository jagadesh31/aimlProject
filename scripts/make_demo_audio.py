"""Build a continuous dialogue-like audio by concatenating RAVDESS clips of changing emotions."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emotions", type=str, default="neutral,anger,joy")
    parser.add_argument("--out", type=Path, default=config.OUTPUT_DIR / "demo_sequence.wav")
    args = parser.parse_args()

    manifest = config.DATA_DIR / "ravdess" / "manifest.csv"
    df = pd.read_csv(manifest)
    wanted = [e.strip().lower() for e in args.emotions.split(",")]
    chunks = []
    meta = []
    cursor = 0.0
    for emo in wanted:
        row = df[(df["Emotion"] == emo) & (df["split"] == "test")].head(1)
        if row.empty:
            row = df[df["Emotion"] == emo].head(1)
        path = Path(row.iloc[0]["path"])
        audio, sr = sf.read(str(path), always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != config.SAMPLE_RATE:
            import librosa

            audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=config.SAMPLE_RATE)
        audio = audio.astype(np.float32)
        dur = len(audio) / config.SAMPLE_RATE
        meta.append({"emotion": emo, "start": cursor, "end": cursor + dur, "file": path.name})
        chunks.append(audio)
        pause = np.zeros(int(0.2 * config.SAMPLE_RATE), dtype=np.float32)
        chunks.append(pause)
        cursor += dur + 0.2

    out_audio = np.concatenate(chunks)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(args.out), out_audio, config.SAMPLE_RATE)
    print("Wrote", args.out, f"duration={len(out_audio)/config.SAMPLE_RATE:.1f}s")
    for m in meta:
        print(f"  {m['start']:.1f}-{m['end']:.1f}s  {m['emotion']}  ({m['file']})")


if __name__ == "__main__":
    main()
