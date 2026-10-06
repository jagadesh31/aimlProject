"""Evaluate continuous SER on many MELD dialogues (window majority vs gold turns)."""
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
from sklearn.metrics import accuracy_score, classification_report, f1_score

import config
from scripts.infer_timeline import load_model, predict_windows, stitch_dialogue
from src.continuous_eval import score_dialogue_continuous, score_transition_detection
from src.dataset import load_split_csv
from src.temporal_tracking import build_timeline, majority_smooth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=str(config.CHECKPOINT_DIR / "best_model.pt"))
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max-dialogues", type=int, default=40)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(Path(args.checkpoint), device)
    df = load_split_csv(args.split)
    dialogue_ids = sorted(df["Dialogue_ID"].unique().tolist())[: args.max_dialogues]

    all_y, all_p = [], []
    per_dia = []
    sum_tp = sum_fp = sum_fn = 0
    for dia in dialogue_ids:
        try:
            audio, gold = stitch_dialogue(args.split, int(dia))
        except Exception as e:
            print(f"skip dia {dia}: {e}")
            continue
        times, classes, confs, _ = predict_windows(model, audio, device)
        smooth = majority_smooth(classes, confs, k=config.SMOOTH_WINDOW, min_conf=config.MIN_CONF)
        segments = build_timeline(times, smooth)
        timeline = [s.as_dict() for s in segments]
        scored = score_dialogue_continuous(times, smooth, gold, config.EMO2IDX, hop=config.HOP_SEC)
        trans = score_transition_detection(gold, timeline, tolerance_sec=1.5)
        if scored["n_utterances_scored"] == 0:
            continue
        all_y.extend(scored["y_true"])
        all_p.extend(scored["y_pred"])
        sum_tp += trans["tp"]
        sum_fp += trans["fp"]
        sum_fn += trans["fn"]
        per_dia.append(
            {
                "dialogue_id": int(dia),
                "n": scored["n_utterances_scored"],
                "acc": scored["continuous_utterance_accuracy"],
                "transition_f1": trans["transition_f1"],
                "n_gold_changes": trans["n_gold_changes"],
            }
        )
        print(
            f"dia {dia:4d}  utt={scored['n_utterances_scored']:3d}  "
            f"cont_acc={scored['continuous_utterance_accuracy']:.3f}  "
            f"trans_f1={trans['transition_f1']:.3f}"
        )

    if not all_y:
        raise SystemExit("No dialogues scored")

    acc = accuracy_score(all_y, all_p)
    f1 = f1_score(all_y, all_p, average="weighted", zero_division=0)
    report = classification_report(all_y, all_p, target_names=config.EMOTIONS, zero_division=0)
    t_prec = sum_tp / max(sum_tp + sum_fp, 1)
    t_rec = sum_tp / max(sum_tp + sum_fn, 1)
    t_f1 = 0.0 if t_prec + t_rec == 0 else 2 * t_prec * t_rec / (t_prec + t_rec)
    out = {
        "mode": "continuous_SER_recognition_and_transition_detection",
        "split": args.split,
        "n_dialogues": len(per_dia),
        "n_utterances": len(all_y),
        "continuous_recognition_accuracy": float(acc),
        "continuous_recognition_weighted_f1": float(f1),
        "transition_detection": {
            "precision": t_prec,
            "recall": t_rec,
            "f1": t_f1,
            "tp": sum_tp,
            "fp": sum_fp,
            "fn": sum_fn,
        },
        "per_dialogue": per_dia,
    }
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (config.METRICS_DIR / "continuous_test_metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (config.METRICS_DIR / "continuous_test_classification_report.txt").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"Continuous recognition accuracy={acc:.4f}  Weighted-F1={f1:.4f}")
    print(f"Transition detection F1={t_f1:.4f}  (P={t_prec:.4f}, R={t_rec:.4f})")
    print("Saved:", config.METRICS_DIR / "continuous_test_metrics.json")


if __name__ == "__main__":
    main()
