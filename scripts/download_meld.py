"""Download MELD.Raw.tar.gz if missing."""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config

URLS = [
    "https://huggingface.co/datasets/declare-lab/MELD/resolve/main/MELD.Raw.tar.gz",
    "https://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz",
]


def download(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading:\n  {url}\n-> {dest}")

    def hook(count, block, total):
        if total <= 0:
            return
        done = count * block
        pct = min(100.0, done * 100.0 / total)
        mb = done / (1024 * 1024)
        tot = total / (1024 * 1024)
        print(f"\r  {pct:5.1f}%  ({mb:.1f}/{tot:.1f} MB)", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=hook)
    print("\nDone.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    dest = config.RAW_DIR / "MELD.Raw.tar.gz"
    if dest.exists() and dest.stat().st_size > 1_000_000 and not args.force:
        print("Already present:", dest, f"({dest.stat().st_size/1e9:.2f} GB)")
        return
    last_err = None
    for url in URLS:
        try:
            download(url, dest)
            return
        except Exception as e:
            last_err = e
            print("Failed:", e)
    raise SystemExit(f"All download URLs failed: {last_err}")


if __name__ == "__main__":
    main()
