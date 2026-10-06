"""Train CNN-BiLSTM SER model on MELD (or RAVDESS) audio."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

import config
from src.cache_build import build_cache
from src.dataset import MeldMelDataset, collate_pad
from src.model import CNNBiLSTM


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def class_weights_from_counts(counts: np.ndarray) -> np.ndarray:
    """Sqrt-inverse frequency. Full inverse frequency ignored joy and neutral."""
    weights = 1.0 / np.sqrt(counts.astype(np.float64) + 1e-6)
    weights = weights / weights.sum() * len(weights)
    return weights.astype(np.float32)


class FocalLoss(nn.Module):
    """Focal loss with a class weight on the true class. gamma=2."""

    def __init__(self, weight: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.register_buffer("weight", weight)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        nll = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = nll.neg().exp()
        alpha = self.weight[targets]
        return (alpha * (1.0 - pt).pow(self.gamma) * nll).mean()


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss = 0.0
    all_y, all_p = [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for mel, y, _ in tqdm(loader, leave=False, desc="train" if train else "eval"):
            mel = mel.to(device)
            y = y.to(device)
            logits = model(mel)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            total_loss += loss.item() * y.size(0)
            pred = logits.argmax(dim=1)
            all_y.extend(y.cpu().tolist())
            all_p.extend(pred.cpu().tolist())
    n = max(len(all_y), 1)
    acc = float(np.mean(np.array(all_y) == np.array(all_p)))
    f1 = f1_score(all_y, all_p, average="weighted", zero_division=0)
    return total_loss / n, acc, f1, all_y, all_p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--source", type=str, default="meld", choices=["meld", "ravdess"])
    parser.add_argument("--sampler", type=str, default="balanced", choices=["balanced", "none"])
    parser.add_argument("--loss", type=str, default="focal", choices=["focal", "ce"])
    args = parser.parse_args()

    set_seed(config.SEED)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("Device:", device)

    if args.source == "meld":
        train_ds = MeldMelDataset("train")
        dev_ds = MeldMelDataset("dev")
        emotion_counts = train_ds.df["Emotion"]
        build_cache(train_ds, "train")
        build_cache(dev_ds, "dev")
    else:
        from src.manifest_dataset import ManifestMelDataset

        manifest = config.DATA_DIR / "ravdess" / "manifest.csv"
        train_ds = ManifestMelDataset(manifest, "train")
        dev_ds = ManifestMelDataset(manifest, "dev")
        emotion_counts = train_ds.df["Emotion"]
        build_cache(train_ds, "ravdess")
        build_cache(dev_ds, "ravdess")
    print(f"Source={args.source} | Train clips: {len(train_ds)} | Dev clips: {len(dev_ds)}")

    counts = emotion_counts.value_counts().reindex(config.EMOTIONS).fillna(1).to_numpy()
    class_w = class_weights_from_counts(counts)
    if args.sampler == "balanced":
        sample_w = train_ds.df["Emotion"].map(lambda e: float(class_w[config.EMO2IDX[e]])).to_numpy()
        sampler = WeightedRandomSampler(
            torch.tensor(sample_w, dtype=torch.double),
            num_samples=len(train_ds),
            replacement=True,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=config.NUM_WORKERS,
            collate_fn=collate_pad,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            collate_fn=collate_pad,
        )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_pad,
    )

    model = CNNBiLSTM().to(device)
    if args.loss == "focal":
        criterion = FocalLoss(weight=torch.tensor(class_w, dtype=torch.float32, device=device), gamma=2.0)
    else:
        # Natural class mix. Label smoothing stops the model from collapsing onto one class.
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    stall = 0
    history = []
    ckpt_name = "best_model_meld.pt" if args.source == "meld" else "best_model_ravdess.pt"
    alias = config.CHECKPOINT_DIR / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, tr_f1, _, _ = run_epoch(model, train_loader, criterion, optimizer, device, True)
        dv_loss, dv_acc, dv_f1, y_true, y_pred = run_epoch(
            model, dev_loader, criterion, optimizer, device, False
        )
        scheduler.step()
        row = {
            "epoch": epoch,
            "source": args.source,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "train_f1": tr_f1,
            "dev_loss": dv_loss,
            "dev_acc": dv_acc,
            "dev_f1": dv_f1,
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d} | train loss {tr_loss:.4f} acc {tr_acc:.3f} f1 {tr_f1:.3f} "
            f"| dev loss {dv_loss:.4f} acc {dv_acc:.3f} f1 {dv_f1:.3f}"
        )
        score = dv_acc if args.loss == "ce" else dv_f1
        if score > best_f1:
            best_f1 = score
            stall = 0
            ckpt = {
                "model": model.state_dict(),
                "emotions": config.EMOTIONS,
                "epoch": epoch,
                "dev_f1": dv_f1,
                "dev_acc": dv_acc,
                "source": args.source,
                "in_channels": 3,
            }
            path = config.CHECKPOINT_DIR / ckpt_name
            torch.save(ckpt, path)
            torch.save(ckpt, alias)
            print(f"  saved {path}")
            report = classification_report(y_true, y_pred, target_names=config.EMOTIONS, zero_division=0)
            (config.METRICS_DIR / f"dev_classification_report_{args.source}.txt").write_text(report, encoding="utf-8")
        else:
            stall += 1
            if stall >= config.EARLY_STOP_PATIENCE:
                print(f"Early stop at epoch {epoch} (no dev F1 gain for {stall} epochs)")
                break

    with open(config.METRICS_DIR / f"train_history_{args.source}.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print("Best dev weighted-F1:", best_f1)


if __name__ == "__main__":
    main()
