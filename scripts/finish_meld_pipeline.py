"""After MELD.Raw.tar.gz finishes downloading: prepare audio and train."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
tar = ROOT / "data" / "raw" / "MELD.Raw.tar.gz"


def main():
    if not tar.exists() or tar.stat().st_size < 10_000_000_000:
        size_gb = tar.stat().st_size / 1e9 if tar.exists() else 0
        raise SystemExit(
            f"MELD archive not ready yet ({size_gb:.2f} GB / ~10.9 GB). "
            "Wait for download, then re-run this script."
        )
    steps = [
        [sys.executable, str(ROOT / "scripts" / "prepare_meld.py")],
        [sys.executable, str(ROOT / "scripts" / "train.py"), "--source", "meld", "--epochs", "15", "--batch-size", "32"],
        [sys.executable, str(ROOT / "scripts" / "evaluate.py"), "--source", "meld", "--split", "test"],
        [sys.executable, str(ROOT / "scripts" / "infer_timeline.py"), "--split", "test", "--dialogue-id", "0"],
    ]
    for cmd in steps:
        print("\n>>>", " ".join(cmd))
        subprocess.check_call(cmd, cwd=str(ROOT))


if __name__ == "__main__":
    main()
