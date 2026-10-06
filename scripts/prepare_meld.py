"""Prepare MELD: unpack raw archive, extract wav audio from mp4 clips."""
from __future__ import annotations

import argparse
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tqdm import tqdm

import config
from src.audio_utils import extract_wav, find_ffmpeg


def unpack_raw(tar_path: Path, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {tar_path} -> {dest} ...")
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(dest)


def find_split_dirs(root: Path):
    """Return mapping split -> directory containing dia*_utt*.mp4"""
    candidates = {}
    # Common layouts after extracting MELD.Raw.tar.gz
    patterns = {
        "train": ["train", "train_splits", "output_repeated_splits_test/train"],
        "dev": ["dev", "dev_splits_complete", "output_repeated_splits_test/dev"],
        "test": ["test", "output_repeated_splits_test", "output_repeated_splits_test/test"],
    }
    # Search broadly
    mp4s = list(root.rglob("dia*_utt*.mp4"))
    by_parent = {}
    for p in mp4s:
        by_parent.setdefault(p.parent, 0)
        by_parent[p.parent] += 1
    # Heuristic: pick largest dirs and map by name
    ranked = sorted(by_parent.items(), key=lambda x: -x[1])
    for parent, count in ranked:
        name = parent.name.lower()
        parent_s = str(parent).lower().replace("\\", "/")
        if "train" in name or "/train" in parent_s:
            candidates.setdefault("train", parent)
        elif "dev" in name or "val" in name:
            candidates.setdefault("dev", parent)
        elif "test" in name:
            candidates.setdefault("test", parent)
    # Fallback: if only one big folder known as output_repeated_splits_test etc.
    if len(candidates) < 3 and ranked:
        print("Discovered mp4 parent folders:")
        for parent, count in ranked[:10]:
            print(f"  {parent} ({count} files)")
    return candidates


def extract_split_audio(split: str, video_dir: Path, limit: int | None = None):
    out_dir = config.AUDIO_DIR / split
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(video_dir.glob("dia*_utt*.mp4"))
    if limit:
        files = files[:limit]
    ok, fail = 0, 0
    for mp4 in tqdm(files, desc=f"extract {split}"):
        wav = out_dir / (mp4.stem + ".wav")
        if wav.exists() and wav.stat().st_size > 1000:
            ok += 1
            continue
        if extract_wav(mp4, wav):
            ok += 1
        else:
            fail += 1
    print(f"[{split}] extracted={ok} failed={fail} dir={out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tar", type=Path, default=config.RAW_DIR / "MELD.Raw.tar.gz")
    parser.add_argument("--skip-unpack", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Optional per-split file limit for smoke tests")
    args = parser.parse_args()

    if find_ffmpeg() is None:
        raise SystemExit("ffmpeg not available")

    extract_root = config.RAW_DIR / "MELD.Raw"
    if not args.skip_unpack:
        if not args.tar.exists():
            raise SystemExit(f"Missing archive: {args.tar}")
        unpack_raw(args.tar, extract_root)

    # Also handle nested tars inside MELD.Raw
    for nested in list(extract_root.rglob("*.tar.gz")) + list(extract_root.rglob("*.tar")):
        nested_dest = nested.with_suffix("").with_suffix("") if nested.suffixes[-2:] == [".tar", ".gz"] else nested.with_suffix("")
        # simpler: extract beside
        target = nested.parent / (nested.name.replace(".tar.gz", "").replace(".tar", ""))
        if not target.exists():
            print(f"Extracting nested {nested.name} ...")
            mode = "r:gz" if nested.suffix == ".gz" or nested.name.endswith(".tar.gz") else "r:"
            try:
                with tarfile.open(nested, mode) as tf:
                    tf.extractall(nested.parent)
            except Exception as e:
                print(f"  skip nested {nested}: {e}")

    splits = find_split_dirs(extract_root)
    if not splits:
        # try video dir already prepared
        for split in ["train", "dev", "test"]:
            d = config.VIDEO_DIR / split
            if d.exists():
                splits[split] = d
    if not splits:
        raise SystemExit("Could not locate dia*_utt*.mp4 files after unpack.")

    print("Using split dirs:")
    for k, v in splits.items():
        print(f"  {k}: {v}")

    for split, vdir in splits.items():
        extract_split_audio(split, vdir, limit=args.limit)

    print("Done. Audio at:", config.AUDIO_DIR)


if __name__ == "__main__":
    main()
