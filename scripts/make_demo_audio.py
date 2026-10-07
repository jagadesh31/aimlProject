"""Build demo WAVs: RAVDESS emotion sequence + stitched MELD test dialogue 0."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from src.dialogue_stitch import stitch_dialogue


def export_meld_dialogue(split: str, dialogue_id: int, out_wav: Path) -> list[dict]:
    audio, meta = stitch_dialogue(split, dialogue_id)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_wav), audio, config.SAMPLE_RATE)
    meta_path = out_wav.with_name(f"{out_wav.stem}_meta.json")
    payload = {
        "split": split,
        "dialogue_id": dialogue_id,
        "duration_sec": float(len(audio) / config.SAMPLE_RATE),
        "wav": str(out_wav),
        "gold_utterances": meta,
    }
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return meta


def build_ravdess_sequence(out: Path, emotions: str) -> None:
    manifest = config.DATA_DIR / "ravdess" / "manifest.csv"
    df = pd.read_csv(manifest)
    wanted = [e.strip().lower() for e in emotions.split(",")]
    chunks = []
    meta = []
    cursor = 0.0
    for emo in wanted:
        row = df[(df["Emotion"] == emo) & (df["split"] == "test")].head(1)
        if row.empty:
            row = df[df["Emotion"] == emo].head(1)
        path = Path(row.iloc[0]["path"])
        audio, sr = sf.read(str(path), always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != config.SAMPLE_RATE:
            import librosa

            audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=config.SAMPLE_RATE)
        audio = audio.astype(np.float32)
        dur = len(audio) / config.SAMPLE_RATE
        meta.append({"emotion": emo, "start": cursor, "end": cursor + dur, "file": path.name})
        chunks.append(audio)
        pause = np.zeros(int(0.2 * config.SAMPLE_RATE), dtype=np.float32)
        chunks.append(pause)
        cursor += dur + 0.2

    out_audio = np.concatenate(chunks)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), out_audio, config.SAMPLE_RATE)
    print("Wrote", out, f"duration={len(out_audio)/config.SAMPLE_RATE:.1f}s")
    for m in meta:
        print(f"  {m['start']:.1f}-{m['end']:.1f}s  {m['emotion']}  ({m['file']})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emotions", type=str, default="neutral,anger,joy")
    parser.add_argument("--out", type=Path, default=config.DEMO_DIR / "demo_sequence.wav")
    parser.add_argument("--skip-ravdess", action="store_true")
    parser.add_argument("--skip-meld-dia0", action="store_true")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--dialogue-id", type=int, default=0)
    parser.add_argument(
        "--meld-out",
        type=Path,
        default=None,
        help="default: outputs/demo/{split}_dia{id}_stitched.wav",
    )
    args = parser.parse_args()

    if not args.skip_ravdess:
        build_ravdess_sequence(args.out, args.emotions)

    if not args.skip_meld_dia0:
        meld_out = args.meld_out or (config.DEMO_DIR / f"{args.split}_dia{args.dialogue_id}_stitched.wav")
        try:
            meta = export_meld_dialogue(args.split, args.dialogue_id, meld_out)
        except (FileNotFoundError, ValueError) as e:
            print(f"Skip MELD stitch ({meld_out.name}): {e}")
        else:
            meta_path = meld_out.with_name(f"{meld_out.stem}_meta.json")
            dur = json.loads(meta_path.read_text(encoding="utf-8"))["duration_sec"]
            print("Wrote", meld_out, f"duration={dur:.1f}s  ({len(meta)} utterances)")
            for g in meta:
                print(
                    f"  {g['start']:.1f}-{g['end']:.1f}s  {g['gold_emotion']:10s}  "
                    f"{g['clip_id']}  {g['utterance'][:60]}"
                )
            print("Meta:", meta_path)


if __name__ == "__main__":
    main()
