"""Print the MELD test results of every saved run side by side."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config

RUNS = [
    ("First model", "test_metrics_previous.json"),
    ("Balanced sampling", "test_metrics_balanced.json"),
    ("Original best_model.pt", "test_metrics.json"),
    ("Weighted loss", "test_metrics_wce.json"),
    ("+ dialogue context", "test_metrics_context_wce_ens.json"),
    ("+ neutral bias tuned on dev (final)", "test_metrics_context_wce_ens_tuned.json"),
    ("Dialogue context, past turns only", "test_metrics_context_wce_ens_causal.json"),
]


def macro_f1_from_cm(cm: list[list[int]]) -> float:
    f1s = []
    for i in range(len(cm)):
        tp = cm[i][i]
        pred = sum(row[i] for row in cm)
        true = sum(cm[i])
        p = tp / pred if pred else 0.0
        r = tp / true if true else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return sum(f1s) / len(f1s)


def main():
    neutral = config.EMO2IDX["neutral"]
    print(f"{'Run':38s} {'Accuracy':>9s} {'Weighted F1':>12s} {'Macro F1':>9s}")
    print("-" * 71)
    baseline = None
    for name, fname in RUNS:
        path = config.METRICS_DIR / fname
        if not path.exists():
            continue
        m = json.loads(path.read_text(encoding="utf-8"))
        cm = m.get("confusion_matrix")
        macro = m.get("macro_f1")
        if macro is None and cm:
            macro = macro_f1_from_cm(cm)
        if baseline is None and cm:
            baseline = sum(cm[neutral]) / sum(sum(r) for r in cm)
        print(f"{name:38s} {m['accuracy']:>8.1%} {m['weighted_f1']:>12.2f} {macro:>9.2f}")
    if baseline is not None:
        print("-" * 71)
        print(f"{'Always guessing neutral':38s} {baseline:>8.1%}")


if __name__ == "__main__":
    main()
