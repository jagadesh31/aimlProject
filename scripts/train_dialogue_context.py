"""Dialogue-context SER: a GRU over per-utterance CNN-BiLSTM features.

Neighbouring turns in a MELD scene often share emotion, so reading the whole
dialogue corrects single-turn mistakes. Audio only; the CNN-BiLSTM is frozen.
Best epoch is chosen by dev accuracy; test is scored once at the end.
"""
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
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


def extract(split: str, checkpoint: Path, device: torch.device) -> dict:
    cache = config.CACHE_DIR / f"context_{checkpoint.stem}_{split}.pt"
    if cache.exists():
        return torch.load(cache, weights_only=False)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model = CNNBiLSTM().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    ds = MeldMelDataset(split)
    build_cache(ds, split)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_pad)
    embs, logits, labels, ids = [], [], [], []
    with torch.no_grad():
        for mel, y, cid in tqdm(loader, desc=f"features-{split}"):
            e = model.embed(mel.to(device))
            embs.append(e.squeeze(0).cpu())
            logits.append(model.fc(e).squeeze(0).cpu())
            labels.append(int(y.item()))
            ids.append(cid[0])
    df = ds.df.set_index("clip_id")
    out = {
        "emb": torch.stack(embs),
        "logits": torch.stack(logits),
        "labels": torch.tensor(labels),
        "clip_id": ids,
        "dialogue": [int(df.loc[c, "Dialogue_ID"]) for c in ids],
        "utt": [int(df.loc[c, "Utterance_ID"]) for c in ids],
        "speaker": [str(df.loc[c, "Speaker"]) for c in ids],
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, cache)
    return out


def to_dialogues(sources: list[dict]) -> list[dict]:
    """One entry per dialogue: features (n, d) and labels (n,) in turn order.

    Each source is one CNN-BiLSTM checkpoint's features over the same clips.
    """
    data = sources[0]
    for other in sources[1:]:
        assert other["clip_id"] == data["clip_id"]
    feats = torch.cat(
        [torch.cat([s["emb"], torch.softmax(s["logits"], dim=-1)], dim=1) for s in sources], dim=1
    )
    groups: dict[int, list[int]] = {}
    for i, d in enumerate(data["dialogue"]):
        groups.setdefault(d, []).append(i)
    dialogues = []
    for d, idx in groups.items():
        idx = sorted(idx, key=lambda i: data["utt"][i])
        speakers = [data["speaker"][i] for i in idx]
        same_prev = [0.0] + [float(a == b) for a, b in zip(speakers[1:], speakers[:-1])]
        x = torch.cat([feats[idx], torch.tensor(same_prev).unsqueeze(1)], dim=1)
        base = torch.stack([torch.softmax(s["logits"][idx], dim=-1) for s in sources]).mean(0)
        dialogues.append({"x": x, "y": data["labels"][idx], "base": base})
    return dialogues


def batches(dialogues: list[dict], size: int, shuffle: bool):
    order = list(range(len(dialogues)))
    if shuffle:
        random.shuffle(order)
    for s in range(0, len(order), size):
        chunk = [dialogues[i] for i in order[s : s + size]]
        n = max(len(c["y"]) for c in chunk)
        d = chunk[0]["x"].shape[1]
        x = torch.zeros(len(chunk), n, d)
        y = torch.full((len(chunk), n), -100, dtype=torch.long)
        for i, c in enumerate(chunk):
            x[i, : len(c["y"])] = c["x"]
            y[i, : len(c["y"])] = c["y"]
        yield x, y


class ContextGRU(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, causal: bool = False):
        super().__init__()
        self.proj = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_dim, 128), nn.ReLU(inplace=True))
        self.gru = nn.GRU(128, hidden, batch_first=True, bidirectional=not causal)
        out_dim = 128 + hidden * (1 if causal else 2)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(out_dim, config.NUM_CLASSES))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        ctx, _ = self.gru(h)
        return self.head(torch.cat([h, ctx], dim=-1))


def predict_probs(models, dialogues, device):
    """Labels, averaged context-model softmax, and CNN-BiLSTM softmax, one row per turn."""
    if not isinstance(models, (list, tuple)):
        models = [models]
    for m in models:
        m.eval()
    ys, probs, base = [], [], []
    with torch.no_grad():
        for c in dialogues:
            x = c["x"].unsqueeze(0).to(device)
            p = torch.stack([torch.softmax(m(x).squeeze(0), dim=-1) for m in models]).mean(0)
            probs.append(p.cpu())
            ys.extend(c["y"].tolist())
            base.append(c["base"])
    return ys, torch.cat(probs), torch.cat(base)


def predict(models, dialogues, device):
    ys, probs, base = predict_probs(models, dialogues, device)
    return ys, probs.argmax(1).tolist(), base.argmax(1).tolist()


def train_one(seed, splits, in_dim, args, device):
    set_seed(seed)
    model = ContextGRU(in_dim, causal=args.causal).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    best_acc, best_state, best_epoch, stall = -1.0, None, 0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in batches(splits["train"], 16, shuffle=True):
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x).reshape(-1, config.NUM_CLASSES), y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        ys, ps, _ = predict(model, splits["dev"], device)
        acc = accuracy_score(ys, ps)
        if acc > best_acc:
            best_acc, best_epoch, stall = acc, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stall += 1
            if stall >= 15:
                break
    model.load_state_dict(best_state)
    return model, best_state, best_epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        nargs="+",
        default=[str(config.CHECKPOINT_DIR / "best_model.pt")],
        help="one or more CNN-BiLSTM checkpoints",
    )
    parser.add_argument(
        "--combine",
        type=str,
        default="concat",
        choices=["concat", "separate"],
        help="concat: one context model on joined features; separate: context models per checkpoint, softmax averaged",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--n-models", type=int, default=1, help="seed ensemble size; softmax is averaged")
    parser.add_argument("--causal", action="store_true", help="past turns only (streaming)")
    parser.add_argument("--tag", type=str, default="context")
    parser.add_argument(
        "--tune-neutral",
        action="store_true",
        help="pick a neutral-class bias that maximises dev accuracy, then apply it to test",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_paths = [Path(p) for p in args.checkpoint]
    groups = [ckpt_paths] if args.combine == "concat" else [[p] for p in ckpt_paths]

    saved, best_epoch = [], []
    dev_probs, dev_base, test_probs, test_base = [], [], [], []
    for group in groups:
        splits = {
            s: to_dialogues([extract(s, p, device) for p in group]) for s in ("train", "dev", "test")
        }
        in_dim = splits["train"][0]["x"].shape[1]
        models = []
        for k in range(args.n_models):
            model, state, epoch = train_one(args.seed + k, splits, in_dim, args, device)
            models.append(model)
            saved.append({"features": [str(p) for p in group], "in_dim": in_dim, "state": state})
            best_epoch.append(epoch)
        dev_y, p, b = predict_probs(models, splits["dev"], device)
        dev_probs.append(p)
        dev_base.append(b)
        ys, p, b = predict_probs(models, splits["test"], device)
        test_probs.append(p)
        test_base.append(b)
        print(f"  {[q.name for q in group]}: dev {accuracy_score(dev_y, dev_probs[-1].argmax(1)):.4f} "
              f"test {accuracy_score(ys, test_probs[-1].argmax(1)):.4f}")

    dev_avg = torch.stack(dev_probs).mean(0).log()
    test_avg = torch.stack(test_probs).mean(0).log()
    neutral_shift = 0.0
    if args.tune_neutral:
        neutral = config.EMO2IDX["neutral"]
        best = -1.0
        for shift in np.arange(-1.0, 1.0001, 0.05):
            adj = dev_avg.clone()
            adj[:, neutral] += float(shift)
            a = accuracy_score(dev_y, adj.argmax(1).tolist())
            if a > best + 1e-9:
                best, neutral_shift = a, float(shift)
        dev_avg[:, neutral] += neutral_shift
        test_avg[:, neutral] += neutral_shift
        print(f"  neutral log-shift chosen on dev: {neutral_shift:+.2f}")
    dev_p = dev_avg.argmax(1).tolist()
    dev_base = torch.stack(dev_base).mean(0).argmax(1).tolist()
    ps = test_avg.argmax(1).tolist()
    base = torch.stack(test_base).mean(0).argmax(1).tolist()
    acc = accuracy_score(ys, ps)
    w_f1 = f1_score(ys, ps, average="weighted", zero_division=0)
    macro = f1_score(ys, ps, average="macro", zero_division=0)
    report = classification_report(ys, ps, labels=list(range(config.NUM_CLASSES)),
                                   target_names=config.EMOTIONS, zero_division=0)
    out = {
        "split": "test",
        "method": "dialogue-context GRU over frozen CNN-BiLSTM features",
        "feature_checkpoints": [str(p) for p in ckpt_paths],
        "combine": args.combine,
        "neutral_log_shift": neutral_shift,
        "seed": args.seed,
        "n_models": args.n_models,
        "causal": args.causal,
        "best_epoch": best_epoch,
        "dev_accuracy": accuracy_score(dev_y, dev_p),
        "dev_accuracy_without_context": accuracy_score(dev_y, dev_base),
        "test_accuracy_without_context": accuracy_score(ys, base),
        "accuracy": acc,
        "weighted_f1": w_f1,
        "macro_f1": macro,
        "confusion_matrix": confusion_matrix(ys, ps, labels=list(range(config.NUM_CLASSES))).tolist(),
    }
    torch.save(
        {"models": saved, "causal": args.causal, "combine": args.combine, "neutral_log_shift": neutral_shift},
        config.CHECKPOINT_DIR / f"{args.tag}_gru.pt",
    )
    (config.METRICS_DIR / f"test_metrics_{args.tag}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (config.METRICS_DIR / f"test_classification_report_{args.tag}.txt").write_text(report, encoding="utf-8")
    print(report)
    print(
        f"dev acc {out['dev_accuracy_without_context']:.4f} -> {out['dev_accuracy']:.4f} | "
        f"test acc {out['test_accuracy_without_context']:.4f} -> {acc:.4f} "
        f"(weighted F1 {w_f1:.4f}, macro F1 {macro:.4f}, models {args.n_models})"
    )


if __name__ == "__main__":
    main()
