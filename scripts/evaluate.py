"""Evaluate checkpoint on MELD test split (utterance-level)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from src.cache_build import build_cache
from src.dataset import MeldMelDataset, collate_pad
from src.model import CNNBiLSTM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=str(config.CHECKPOINT_DIR / "best_model.pt"))
    parser.add_argument("--split", type=str, default="test", choices=["train", "dev", "test"])
    parser.add_argument("--source", type=str, default="meld", choices=["meld", "ravdess"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = CNNBiLSTM().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    if args.source == "meld":
        ds = MeldMelDataset(args.split)
        build_cache(ds, args.split)
    else:
        from src.manifest_dataset import ManifestMelDataset

        ds = ManifestMelDataset(config.DATA_DIR / "ravdess" / "manifest.csv", args.split)
        build_cache(ds, "ravdess")
    loader = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate_pad)
    ys, ps = [], []
    with torch.no_grad():
        for mel, y, _ in tqdm(loader, desc=f"eval-{args.split}"):
            logits = model(mel.to(device))
            pred = logits.argmax(1).cpu().tolist()
            ys.extend(y.tolist())
            ps.extend(pred)

    acc = accuracy_score(ys, ps)
    f1 = f1_score(ys, ps, average="weighted", zero_division=0)
    report = classification_report(ys, ps, target_names=config.EMOTIONS, zero_division=0)
    cm = confusion_matrix(ys, ps).tolist()
    out = {"split": args.split, "accuracy": acc, "weighted_f1": f1, "confusion_matrix": cm}
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (config.METRICS_DIR / f"{args.split}_metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (config.METRICS_DIR / f"{args.split}_classification_report.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"Accuracy={acc:.4f} Weighted-F1={f1:.4f}")


if __name__ == "__main__":
    main()
