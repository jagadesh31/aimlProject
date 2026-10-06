"""Temporal emotion tracking: smooth + stable regions + transitions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

import config


@dataclass
class TimelineSegment:
    start: float
    end: float
    kind: str  # "stable" | "transition"
    emotion: str
    from_emotion: str | None = None
    to_emotion: str | None = None

    def as_dict(self):
        d = {
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "type": self.kind,
            "emotion": self.emotion,
        }
        if self.kind == "transition":
            d["from"] = self.from_emotion
            d["to"] = self.to_emotion
        return d


def majority_smooth(labels: Sequence[int], conf: Sequence[float], k: int = 3, min_conf: float = 0.35) -> List[int]:
    """Offline smooth: uses past and future neighbors (non-causal)."""
    labels = list(labels)
    conf = list(conf)
    n = len(labels)
    if n == 0:
        return []
    out = labels[:]
    half = k // 2
    for i in range(n):
        left = max(0, i - half)
        right = min(n, i + half + 1)
        window = labels[left:right]
        # if current is isolated low-confidence flip, replace by neighbors majority
        if 0 < i < n - 1 and labels[i] != labels[i - 1] and labels[i] != labels[i + 1]:
            if conf[i] < min_conf:
                out[i] = labels[i - 1]
                continue
        # majority in local window
        vals, counts = np.unique(window, return_counts=True)
        out[i] = int(vals[np.argmax(counts)])
    return out


def causal_smooth(labels: Sequence[int], conf: Sequence[float], k: int = 3, min_conf: float = 0.35) -> List[int]:
    """Online / streaming smooth: uses only past and current windows (causal).

    Needed for a real Continuous SER deployment where future audio is not available yet.
    """
    labels = list(labels)
    conf = list(conf)
    n = len(labels)
    if n == 0:
        return []
    out: List[int] = []
    for i in range(n):
        left = max(0, i - (k - 1))
        window = labels[left : i + 1]
        if i > 0 and labels[i] != labels[i - 1] and conf[i] < min_conf:
            out.append(out[-1])
            continue
        vals, counts = np.unique(window, return_counts=True)
        out.append(int(vals[np.argmax(counts)]))
    return out


def change_points(labels: Sequence[int]) -> List[int]:
    """Indices where the label changes relative to the previous window."""
    labels = list(labels)
    return [i for i in range(1, len(labels)) if labels[i] != labels[i - 1]]


def build_timeline(
    times: Sequence[float],
    labels: Sequence[int],
    hop_sec: float = config.HOP_SEC,
    min_stable_sec: float = config.MIN_STABLE_SEC,
    trans_sec: float = config.TRANS_SEC,
    idx2emo: dict | None = None,
) -> List[TimelineSegment]:
    idx2emo = idx2emo or config.IDX2EMO
    if not times:
        return []
    labels = list(labels)
    times = list(times)
    segments: List[TimelineSegment] = []

    start = float(times[0])
    cur = int(labels[0])

    def emit_stable(s: float, e: float, emo_idx: int):
        if e - s <= 0:
            return
        segments.append(
            TimelineSegment(start=s, end=e, kind="stable", emotion=idx2emo[emo_idx])
        )

    for i in range(1, len(labels)):
        if labels[i] == cur:
            continue
        change_t = float(times[i])
        # close previous stable region
        if change_t - start >= min_stable_sec:
            emit_stable(start, change_t, cur)
        else:
            # too short: absorb into previous if exists else keep
            emit_stable(start, change_t, cur)
        # transition interval
        trans_end = change_t + trans_sec
        segments.append(
            TimelineSegment(
                start=change_t,
                end=trans_end,
                kind="transition",
                emotion=f"{idx2emo[cur]}->{idx2emo[labels[i]]}",
                from_emotion=idx2emo[cur],
                to_emotion=idx2emo[labels[i]],
            )
        )
        start = trans_end
        cur = int(labels[i])

    end_t = float(times[-1]) + hop_sec
    emit_stable(start, max(end_t, start + hop_sec), cur)

    # merge consecutive stables with same emotion
    merged: List[TimelineSegment] = []
    for seg in segments:
        if (
            merged
            and seg.kind == "stable"
            and merged[-1].kind == "stable"
            and merged[-1].emotion == seg.emotion
        ):
            merged[-1].end = seg.end
        else:
            merged.append(seg)
    return merged


def format_timeline(segments: List[TimelineSegment]) -> str:
    lines = ["EMOTION TIMELINE", "-" * 40]
    for s in segments:
        if s.kind == "stable":
            lines.append(f"{s.start:>5.1f}-{s.end:<5.1f}s   {s.emotion.capitalize()} (stable)")
        else:
            lines.append(
                f"{s.start:>5.1f}-{s.end:<5.1f}s   Transition: {s.from_emotion} -> {s.to_emotion}"
            )
    return "\n".join(lines)
