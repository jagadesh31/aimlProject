"""Continuous SER scoring utilities.

1) Window majority vs MELD utterance gold (continuous recognition accuracy)
2) Emotion-change / transition detection vs gold turn-to-turn shifts
3) Emotion-flow summary for a dialogue
"""
from __future__ import annotations

from typing import Dict, List, Sequence


def windows_in_span(times: Sequence[float], start: float, end: float, hop: float = 1.0) -> List[int]:
    """Return indices of windows that overlap [start, end)."""
    idxs = []
    for i, t in enumerate(times):
        w_end = t + hop
        if w_end > start and t < end:
            idxs.append(i)
    return idxs


def majority_label(labels: Sequence[int], idxs: Sequence[int]) -> int | None:
    if not idxs:
        return None
    vals = [int(labels[i]) for i in idxs]
    return max(set(vals), key=vals.count)


def score_dialogue_continuous(
    times: Sequence[float],
    smoothed_labels: Sequence[int],
    gold_utterances: List[dict],
    emo2idx: Dict[str, int],
    hop: float = 1.0,
) -> dict:
    """Compare continuous (window) predictions to each gold utterance via majority vote."""
    y_true, y_pred = [], []
    details = []
    for g in gold_utterances:
        gold = str(g["gold_emotion"]).lower().strip()
        if gold not in emo2idx:
            continue
        idxs = windows_in_span(times, float(g["start"]), float(g["end"]), hop=hop)
        pred = majority_label(smoothed_labels, idxs)
        if pred is None:
            continue
        gt = emo2idx[gold]
        y_true.append(gt)
        y_pred.append(pred)
        details.append(
            {
                "clip_id": g.get("clip_id"),
                "start": round(float(g["start"]), 2),
                "end": round(float(g["end"]), 2),
                "gold": gold,
                "pred_from_windows": [k for k, v in emo2idx.items() if v == pred][0],
                "n_windows": len(idxs),
                "correct": bool(pred == gt),
            }
        )
    n = max(len(y_true), 1)
    acc = sum(int(a == b) for a, b in zip(y_true, y_pred)) / n if y_true else 0.0
    return {
        "n_utterances_scored": len(y_true),
        "continuous_utterance_accuracy": acc,
        "details": details,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def gold_emotion_changes(gold_utterances: List[dict]) -> List[dict]:
    """Turn-to-turn emotion shifts in MELD gold (true emotion-change events)."""
    changes = []
    ordered = sorted(gold_utterances, key=lambda g: float(g["start"]))
    for prev, cur in zip(ordered, ordered[1:]):
        a = str(prev["gold_emotion"]).lower().strip()
        b = str(cur["gold_emotion"]).lower().strip()
        if a != b:
            changes.append({"time": float(cur["start"]), "from": a, "to": b})
    return changes


def predicted_emotion_changes(timeline: List[dict]) -> List[dict]:
    """Emotion-change events from our timeline transition segments."""
    changes = []
    for seg in timeline:
        if seg.get("type") == "transition":
            changes.append(
                {
                    "time": float(seg["start"]),
                    "from": seg.get("from"),
                    "to": seg.get("to"),
                }
            )
    return changes


def score_transition_detection(
    gold_utterances: List[dict],
    timeline: List[dict],
    tolerance_sec: float = 1.5,
) -> dict:
    """Match predicted transitions to gold turn-to-turn emotion changes."""
    gold = gold_emotion_changes(gold_utterances)
    pred = predicted_emotion_changes(timeline)
    matched_gold = set()
    matched_pred = set()
    for gi, g in enumerate(gold):
        best_pi, best_score = None, -1.0
        for pi, p in enumerate(pred):
            if pi in matched_pred:
                continue
            dt = abs(p["time"] - g["time"])
            if dt > tolerance_sec:
                continue
            direction_ok = p.get("from") == g["from"] and p.get("to") == g["to"]
            score = (2.0 if direction_ok else 1.0) - dt / max(tolerance_sec, 1e-6)
            if score > best_score:
                best_score, best_pi = score, pi
        if best_pi is not None:
            matched_gold.add(gi)
            matched_pred.add(best_pi)

    tp = len(matched_gold)
    fp = len(pred) - len(matched_pred)
    fn = len(gold) - len(matched_gold)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "n_gold_changes": len(gold),
        "n_pred_transitions": len(pred),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "transition_precision": precision,
        "transition_recall": recall,
        "transition_f1": f1,
        "gold_changes": gold,
        "pred_changes": pred,
    }


def emotion_flow_summary(gold_utterances: List[dict], timeline: List[dict]) -> dict:
    gold_labels = [
        str(g["gold_emotion"]).lower().strip()
        for g in sorted(gold_utterances, key=lambda x: float(x["start"]))
    ]
    stable = [s for s in timeline if s.get("type") == "stable"]
    return {
        "n_gold_turns": len(gold_labels),
        "n_gold_unique_emotions": len(set(gold_labels)),
        "n_gold_changes": sum(1 for a, b in zip(gold_labels, gold_labels[1:]) if a != b),
        "n_stable_segments": len(stable),
        "n_transitions": sum(1 for s in timeline if s.get("type") == "transition"),
        "pred_emotion_sequence": [s.get("emotion") for s in stable],
        "gold_emotion_sequence": gold_labels,
    }
