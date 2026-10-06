"""Train CNN-BiLSTM SER model on MELD audio."""
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
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from src.dataset import MeldMelDataset, collate_pad
from src.model import CNNBiLSTM


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    args = parser.parse_args()

    set_seed(config.SEED)
    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print("Device:", device)

    if args.source == "meld":
        train_ds = MeldMelDataset("train")
        dev_ds = MeldMelDataset("dev")
        emotion_counts = train_ds.df["Emotion"]
    else:
        from src.manifest_dataset import ManifestMelDataset

        manifest = config.DATA_DIR / "ravdess" / "manifest.csv"
        train_ds = ManifestMelDataset(manifest, "train")
        dev_ds = ManifestMelDataset(manifest, "dev")
        emotion_counts = train_ds.df["Emotion"]
    print(f"Source={args.source} | Train clips: {len(train_ds)} | Dev clips: {len(dev_ds)}")

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

    # class weights for imbalance
    counts = emotion_counts.value_counts().reindex(config.EMOTIONS).fillna(1).values
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * len(weights)
    weight_t = torch.tensor(weights, dtype=torch.float32, device=device)

    # Focal Loss: focuses on hard/minority examples, gamma=2 is standard
    class FocalLoss(nn.Module):
        def __init__(self, weight=None, gamma=2.0):
            super().__init__()
            self.weight = weight
            self.gamma = gamma

        def forward(self, logits, targets):
            ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
            pt = torch.exp(-ce)
            return ((1 - pt) ** self.gamma * ce).mean()

    model = CNNBiLSTM().to(device)
    criterion = FocalLoss(weight=weight_t, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    history = []
    ckpt_name = "best_model_meld.pt" if args.source == "meld" else "best_model_ravdess.pt"
    # Keep a stable alias used by infer/evaluate
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
        if dv_f1 > best_f1:
            best_f1 = dv_f1
            ckpt = {
                "model": model.state_dict(),
                "emotions": config.EMOTIONS,
                "epoch": epoch,
                "dev_f1": dv_f1,
                "dev_acc": dv_acc,
                "source": args.source,
            }
            path = config.CHECKPOINT_DIR / ckpt_name
            torch.save(ckpt, path)
            torch.save(ckpt, alias)
            print(f"  saved {path}")
            report = classification_report(
                y_true, y_pred, target_names=config.EMOTIONS, zero_division=0
            )
            (config.OUTPUT_DIR / f"dev_classification_report_{args.source}.txt").write_text(
                report, encoding="utf-8"
            )

    with open(config.OUTPUT_DIR / f"train_history_{args.source}.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print("Best dev weighted-F1:", best_f1)


if __name__ == "__main__":
    main()
