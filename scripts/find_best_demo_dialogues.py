"""Find test dialogues where window-majority predictions match gold, including non-neutral."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

import config
from scripts.infer_timeline import load_model, predict_windows
from src.continuous_eval import score_dialogue_continuous
from src.dataset import load_split_csv
from src.dialogue_stitch import stitch_dialogue
from src.temporal_tracking import majority_smooth


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(config.CHECKPOINT_DIR / "best_model.pt", device)
    df = load_split_csv("test")
    ids = sorted(int(x) for x in df["Dialogue_ID"].unique().tolist())
    hits = []
    for dia in ids[:180]:
        try:
            audio, gold = stitch_dialogue("test", dia)
        except Exception:
            continue
        times, classes, confs, _ = predict_windows(model, audio, device)
        smooth = majority_smooth(classes, confs, k=config.SMOOTH_WINDOW, min_conf=config.MIN_CONF)
        scored = score_dialogue_continuous(times, smooth, gold, config.EMO2IDX, hop=config.HOP_SEC)
        details = scored["details"]
        good = [d for d in details if d["correct"] and d["gold"] != "neutral"]
        if not good:
            continue
        dur = len(audio) / config.SAMPLE_RATE
        acc = scored["continuous_utterance_accuracy"]
        print(f"dia={dia} acc={acc:.2f} dur={dur:.1f}s non_neu_hits={len(good)}/{len(details)}")
        for d in details:
            mark = "OK" if d["correct"] else "  "
            print(
                f"  {mark} {d['start']:.1f}-{d['end']:.1f} gold={d['gold']:10s} pred={d['pred_from_windows']}"
            )
        hits.append((len(good), acc, dur, dia))
    print("COUNT", len(hits))
    hits.sort(reverse=True)
    print("BEST", hits[:12])


if __name__ == "__main__":
    main()
