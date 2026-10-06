"""MELD PyTorch dataset over cached mel spectrograms / wav files."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import config
from src.audio_utils import load_audio, pad_or_crop
from src.features import waveform_to_mel


def load_split_csv(split: str) -> pd.DataFrame:
    name = {"train": "train_sent_emo.csv", "dev": "dev_sent_emo.csv", "test": "test_sent_emo.csv"}[split]
    path = config.LABEL_DIR / name
    df = pd.read_csv(path)
    df["Emotion"] = df["Emotion"].str.lower().str.strip()
    df = df[df["Emotion"].isin(config.EMOTIONS)].reset_index(drop=True)
    df["clip_id"] = df.apply(lambda r: f"dia{int(r['Dialogue_ID'])}_utt{int(r['Utterance_ID'])}", axis=1)
    return df


def wav_path_for(split: str, clip_id: str) -> Path:
    return config.AUDIO_DIR / split / f"{clip_id}.wav"


def mel_cache_path(split: str, clip_id: str) -> Path:
    return config.CACHE_DIR / split / f"{clip_id}.npy"


class MeldMelDataset(Dataset):
    def __init__(self, split: str, use_cache: bool = True):
        self.split = split
        self.use_cache = use_cache
        self.df = load_split_csv(split)
        self.target_len = int(config.MAX_DURATION * config.SAMPLE_RATE)
        # Keep only rows with existing audio
        keep = []
        for i, row in self.df.iterrows():
            wav = wav_path_for(split, row["clip_id"])
            if wav.exists():
                keep.append(i)
        self.df = self.df.loc[keep].reset_index(drop=True)
        if len(self.df) == 0:
            raise FileNotFoundError(
                f"No audio found for split={split} under {config.AUDIO_DIR / split}. "
                "Run scripts/prepare_meld.py first."
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        clip_id = row["clip_id"]
        label = config.EMO2IDX[row["Emotion"]]
        cache = mel_cache_path(self.split, clip_id)
        if self.use_cache and cache.exists():
            mel = np.load(cache)
        else:
            audio = load_audio(wav_path_for(self.split, clip_id))
            audio = pad_or_crop(audio, self.target_len)
            mel = waveform_to_mel(audio)
            if self.use_cache:
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache, mel)
        # (1, n_mels, T)
        mel_t = torch.from_numpy(mel).unsqueeze(0)
        return mel_t, torch.tensor(label, dtype=torch.long), clip_id


def collate_pad(batch):
    mels, labels, ids = zip(*batch)
    # pad time dimension to max in batch
    max_t = max(m.shape[-1] for m in mels)
    n_mels = mels[0].shape[1]
    out = torch.zeros(len(mels), 1, n_mels, max_t)
    for i, m in enumerate(mels):
        out[i, :, :, : m.shape[-1]] = m
    labels_t = torch.stack(labels)
    return out, labels_t, list(ids)
