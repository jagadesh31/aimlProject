"""
Continuous-in-time SER inference + emotion timeline.

Modes:
  1) --audio path.wav                 : arbitrary speech file
  2) --dialogue-id N --split test     : stitch MELD dialogue utterances in order
"""
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

import config
from src.audio_utils import load_audio
from src.dataset import load_split_csv, wav_path_for
from src.features import sliding_windows, waveform_to_mel
from src.model import CNNBiLSTM, predict_proba
from src.temporal_tracking import build_timeline, format_timeline, majority_smooth


def load_model(checkpoint: Path, device: torch.device) -> CNNBiLSTM:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model = CNNBiLSTM().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def predict_windows(model, audio: np.ndarray, device: torch.device):
    times, classes, confs, probs = [], [], [], []
    for t0, win in sliding_windows(audio, config.SAMPLE_RATE, config.WIN_SEC, config.HOP_SEC):
        mel = waveform_to_mel(win)
        mel_t = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).to(device)
        p = predict_proba(model, mel_t).squeeze(0).cpu().numpy()
        cls = int(p.argmax())
        times.append(t0)
        classes.append(cls)
        confs.append(float(p[cls]))
        probs.append(p)
    return times, classes, confs, np.stack(probs) if probs else np.zeros((0, config.NUM_CLASSES))


def stitch_dialogue(split: str, dialogue_id: int) -> tuple[np.ndarray, list[dict]]:
    df = load_split_csv(split)
    ddf = df[df["Dialogue_ID"] == dialogue_id].sort_values("Utterance_ID")
    if ddf.empty:
        raise ValueError(f"No utterances for dialogue_id={dialogue_id} in {split}")
    chunks = []
    meta = []
    cursor = 0.0
    for _, row in ddf.iterrows():
        wav = wav_path_for(split, row["clip_id"])
        if not wav.exists():
            continue
        a = load_audio(wav)
        chunks.append(a)
        dur = len(a) / config.SAMPLE_RATE
        meta.append(
            {
                "clip_id": row["clip_id"],
                "start": cursor,
                "end": cursor + dur,
                "gold_emotion": row["Emotion"],
                "utterance": str(row["Utterance"])[:120],
            }
        )
        cursor += dur
        # small pause between utterances
        pause = np.zeros(int(0.15 * config.SAMPLE_RATE), dtype=np.float32)
        chunks.append(pause)
        cursor += 0.15
    if not chunks:
        raise FileNotFoundError("No wav files found for this dialogue")
    audio = np.concatenate(chunks)
    return audio, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=str(config.CHECKPOINT_DIR / "best_model.pt"))
    parser.add_argument("--audio", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--dialogue-id", type=int, default=None)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(Path(args.checkpoint), device)

    gold_meta = None
    if args.audio:
        audio = load_audio(Path(args.audio))
        tag = Path(args.audio).stem
    elif args.dialogue_id is not None:
        audio, gold_meta = stitch_dialogue(args.split, args.dialogue_id)
        tag = f"{args.split}_dia{args.dialogue_id}"
    else:
        raise SystemExit("Provide --audio or --dialogue-id")

    times, classes, confs, probs = predict_windows(model, audio, device)
    smooth = majority_smooth(classes, confs, k=config.SMOOTH_WINDOW, min_conf=config.MIN_CONF)
    segments = build_timeline(times, smooth)
    text = format_timeline(segments)
    print(text)

    payload = {
        "tag": tag,
        "duration_sec": float(len(audio) / config.SAMPLE_RATE),
        "window_predictions": [
            {
                "time": round(t, 2),
                "emotion": config.IDX2EMO[c],
                "smoothed": config.IDX2EMO[s],
                "confidence": round(cf, 3),
            }
            for t, c, s, cf in zip(times, classes, smooth, confs)
        ],
        "timeline": [s.as_dict() for s in segments],
        "gold_utterances": gold_meta,
    }
    out = Path(args.out) if args.out else config.OUTPUT_DIR / f"timeline_{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\nSaved:", out)

    if gold_meta:
        print("\nMELD utterance gold labels (evaluation anchors):")
        for g in gold_meta:
            print(f"  {g['start']:.1f}–{g['end']:.1f}s  {g['gold_emotion']:10s}  {g['utterance']}")


if __name__ == "__main__":
    main()
