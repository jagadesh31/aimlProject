"""Generic emotion wav dataset from a manifest CSV (RAVDESS bootstrap)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import config
from src.audio_utils import load_audio, pad_or_crop
from src.features import waveform_to_mel


class ManifestMelDataset(Dataset):
    def __init__(self, manifest_csv: Path, split: str, use_cache: bool = True):
        df = pd.read_csv(manifest_csv)
        df["Emotion"] = df["Emotion"].str.lower().str.strip()
        self.df = df[(df["split"] == split) & (df["Emotion"].isin(config.EMOTIONS))].reset_index(drop=True)
        self.use_cache = use_cache
        self.split = split
        self.target_len = int(config.MAX_DURATION * config.SAMPLE_RATE)
        self.cache_root = config.CACHE_DIR / "ravdess" / split
        if len(self.df) == 0:
            raise FileNotFoundError(f"No rows for split={split} in {manifest_csv}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        clip_id = row["clip_id"]
        label = config.EMO2IDX[row["Emotion"]]
        cache = self.cache_root / f"{clip_id}.npy"
        if self.use_cache and cache.exists():
            mel = np.load(cache)
        else:
            audio = load_audio(Path(row["path"]))
            audio = pad_or_crop(audio, self.target_len)
            mel = waveform_to_mel(audio)
            if self.use_cache:
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache, mel)
        return torch.from_numpy(mel).unsqueeze(0), torch.tensor(label, dtype=torch.long), clip_id
