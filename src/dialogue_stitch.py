"""Stitch MELD utterance WAVs in dialogue order for continuous SER demos."""
from __future__ import annotations

import numpy as np

import config
from src.audio_utils import load_audio
from src.dataset import load_split_csv, wav_path_for


def stitch_dialogue(split: str, dialogue_id: int) -> tuple[np.ndarray, list[dict]]:
    df = load_split_csv(split)
    ddf = df[df["Dialogue_ID"] == dialogue_id].sort_values("Utterance_ID")
    if ddf.empty:
        raise ValueError(f"No utterances for dialogue_id={dialogue_id} in {split}")
    chunks = []
    meta = []
    cursor = 0.0
    for _, row in ddf.iterrows():
        wav = wav_path_for(split, row["clip_id"])
        if not wav.exists():
            continue
        a = load_audio(wav)
        chunks.append(a)
        dur = len(a) / config.SAMPLE_RATE
        meta.append(
            {
                "clip_id": row["clip_id"],
                "start": cursor,
                "end": cursor + dur,
                "gold_emotion": row["Emotion"],
                "utterance": str(row["Utterance"])[:120],
            }
        )
        cursor += dur
        pause = np.zeros(int(0.15 * config.SAMPLE_RATE), dtype=np.float32)
        chunks.append(pause)
        cursor += 0.15
    if not chunks:
        raise FileNotFoundError("No wav files found for this dialogue")
    audio = np.concatenate(chunks)
    return audio, meta
