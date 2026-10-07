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


# One scene (test dia 80) plus a few unique later turns from dia 72.
# Dia 72's first clips are the same WAV copied onto several CSV rows.
SHOWCASE_PLAN = (
    ("test", 80, None),
    ("test", 72, 4),  # start at utt4; earlier rows share one file
)


def _unique_turns(split: str, dialogue_id: int, start_utt: int | None) -> list[tuple[np.ndarray, dict]]:
    """Utterance clips in order, skipping rows that reuse the same WAV bytes."""
    import hashlib

    from src.audio_utils import load_audio
    from src.dataset import load_split_csv, wav_path_for

    df = load_split_csv(split)
    ddf = df[df["Dialogue_ID"] == dialogue_id].sort_values("Utterance_ID")
    turns = []
    seen = set()
    for _, row in ddf.iterrows():
        utt = int(row["Utterance_ID"])
        if start_utt is not None and utt < start_utt:
            continue
        wav = wav_path_for(split, row["clip_id"])
        if not wav.exists():
            continue
        digest = hashlib.md5(wav.read_bytes()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        audio = load_audio(wav)
        turns.append(
            (
                audio,
                {
                    "clip_id": row["clip_id"],
                    "dialogue_id": int(dialogue_id),
                    "gold_emotion": row["Emotion"],
                    "utterance": str(row["Utterance"])[:120],
                },
            )
        )
    return turns


def export_showcase(split: str, out_wav: Path) -> None:
    """One 30–40 s demo: unique clips only, no repeated WAV."""
    import torch

    from scripts.infer_timeline import load_model, predict_windows
    from src.continuous_eval import score_dialogue_continuous
    from src.temporal_tracking import majority_smooth

    pause = np.zeros(int(0.15 * config.SAMPLE_RATE), dtype=np.float32)
    selected: list[tuple[np.ndarray, dict]] = []
    seconds = 0.0
    target_min, target_max = 30.0, 40.0
    for plan_split, dia, start_utt in SHOWCASE_PLAN:
        for audio, meta in _unique_turns(plan_split, dia, start_utt):
            add = len(audio) / config.SAMPLE_RATE + (0.15 if selected else 0.0)
            if seconds >= target_min and seconds + add > target_max:
                break
            if seconds + add > target_max and seconds >= target_min:
                break
            selected.append((audio, meta))
            seconds += add
        if seconds >= target_min:
            break

    chunks = []
    gold = []
    cursor = 0.0
    for i, (clip, meta) in enumerate(selected):
        if i:
            chunks.append(pause)
            cursor += 0.15
        chunks.append(clip)
        dur = len(clip) / config.SAMPLE_RATE
        row = dict(meta)
        row["start"] = round(cursor, 2)
        row["end"] = round(cursor + dur, 2)
        gold.append(row)
        cursor += dur
    audio = np.concatenate(chunks)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_wav), audio, config.SAMPLE_RATE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = config.CHECKPOINT_DIR / "best_model.pt"
    model = load_model(ckpt, device)
    times, classes, confs, _ = predict_windows(model, audio, device)
    smooth = majority_smooth(classes, confs, k=config.SMOOTH_WINDOW, min_conf=config.MIN_CONF)
    scored = score_dialogue_continuous(times, smooth, gold, config.EMO2IDX, hop=config.HOP_SEC)
    payload = {
        "note": "30-40s demo. Duplicate MELD wav rows removed. Gold vs window-majority prediction.",
        "split": split,
        "dialogue_ids": sorted({int(g["dialogue_id"]) for g in gold}),
        "duration_sec": round(float(len(audio) / config.SAMPLE_RATE), 2),
        "wav": str(out_wav),
        "continuous_utterance_accuracy": scored["continuous_utterance_accuracy"],
        "n_utterances_scored": scored["n_utterances_scored"],
        "turns": scored["details"],
    }
    meta_path = out_wav.with_name(f"{out_wav.stem}_meta.json")
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        "Wrote",
        out_wav,
        f"duration={payload['duration_sec']:.1f}s  "
        f"match={payload['continuous_utterance_accuracy']:.0%} "
        f"({payload['n_utterances_scored']} turns)",
    )
    for d in scored["details"]:
        mark = "OK" if d["correct"] else "  "
        print(
            f"  {mark} {d.get('clip_id','')}  {d['start']:.1f}-{d['end']:.1f}s  "
            f"gold={d['gold']:10s} pred={d['pred_from_windows']}"
        )
    print("Meta:", meta_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emotions", type=str, default="neutral,anger,joy")
    parser.add_argument("--out", type=Path, default=config.DEMO_DIR / "demo_sequence.wav")
    parser.add_argument("--skip-ravdess", action="store_true")
    parser.add_argument("--skip-meld-dia0", action="store_true")
    parser.add_argument("--skip-showcase", action="store_true")
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

    if not args.skip_showcase:
        showcase = config.DEMO_DIR / "best_match_demo.wav"
        try:
            export_showcase(args.split, showcase)
        except (FileNotFoundError, ValueError) as e:
            print(f"Skip showcase: {e}")


if __name__ == "__main__":
    main()
