"""Audio extraction utilities for MELD mp4 clips."""
from __future__ import annotations

import subprocess
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

import config


def find_ffmpeg() -> Optional[str]:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def extract_wav(mp4_path: Path, wav_path: Path, sr: int = config.SAMPLE_RATE) -> bool:
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found. Install ffmpeg or imageio-ffmpeg.")
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(mp4_path),
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-vn",
        str(wav_path),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return proc.returncode == 0 and wav_path.exists()


def load_audio(path: Path, sr: int = config.SAMPLE_RATE) -> np.ndarray:
    audio, file_sr = sf.read(str(path), always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if file_sr != sr:
        import librosa

        audio = librosa.resample(audio.astype(np.float32), orig_sr=file_sr, target_sr=sr)
    return audio.astype(np.float32)


def pad_or_crop(audio: np.ndarray, target_len: int) -> np.ndarray:
    if len(audio) >= target_len:
        return audio[:target_len]
    out = np.zeros(target_len, dtype=np.float32)
    out[: len(audio)] = audio
    return out


def prepare_waveform(audio: np.ndarray, sr: int = config.SAMPLE_RATE) -> np.ndarray:
    """Crop long clips and pad only clips shorter than MIN_DURATION.

    Padding every utterance to 6 seconds of silence made the model learn
    the silence tail instead of the emotion in the speech.
    """
    audio = np.asarray(audio, dtype=np.float32)
    max_len = int(config.MAX_DURATION * sr)
    min_len = int(config.MIN_DURATION * sr)
    if audio.size == 0:
        return np.zeros(min_len, dtype=np.float32)
    if len(audio) > max_len:
        audio = audio[:max_len]
    if len(audio) < min_len:
        audio = np.pad(audio, (0, min_len - len(audio)))
    return audio.astype(np.float32)
