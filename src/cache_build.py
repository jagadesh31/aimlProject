"""Parallel feature-cache builder for MELD and RAVDESS."""
from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tqdm import tqdm

from src.dataset import mel_cache_path, wav_path_for
from src.features import save_feature_file


def _cache_job(job):
    wav, cache = job
    save_feature_file(wav, cache)


def build_cache(dataset, split_name: str):
    jobs = []
    for _, row in dataset.df.iterrows():
        clip_id = row["clip_id"]
        if split_name == "ravdess":
            cache = dataset.cache_root / f"{clip_id}.npy"
            wav = Path(row["path"])
        else:
            cache = mel_cache_path(dataset.split, clip_id)
            wav = wav_path_for(dataset.split, clip_id)
        if not cache.exists():
            jobs.append((str(wav), str(cache)))
    if not jobs:
        print(f"Feature cache ready for {split_name} ({len(dataset)} clips)")
        return
    print(f"Building feature cache for {split_name}: {len(jobs)} clips")
    workers = max(1, min(8, os.cpu_count() or 1))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for _ in tqdm(pool.map(_cache_job, jobs, chunksize=8), total=len(jobs), desc=f"cache {split_name}"):
            pass
