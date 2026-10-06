"""Mel-spectrogram feature extraction."""
from __future__ import annotations

import numpy as np
import librosa

import config


def waveform_to_mel(audio: np.ndarray, sr: int = config.SAMPLE_RATE) -> np.ndarray:
    """Return log-mel spectrogram shaped (n_mels, time)."""
    if audio.size == 0:
        return np.zeros((config.N_MELS, 1), dtype=np.float32)
    mel = librosa.feature.melspectrogram(
        y=audio.astype(np.float32),
        sr=sr,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        n_mels=config.N_MELS,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    # Normalize per-clip
    mean = log_mel.mean()
    std = log_mel.std() + 1e-6
    log_mel = (log_mel - mean) / std
    return log_mel.astype(np.float32)


def sliding_windows(audio: np.ndarray, sr: int, win_sec: float, hop_sec: float):
    win = int(win_sec * sr)
    hop = int(hop_sec * sr)
    if len(audio) < win:
        pad = np.zeros(win, dtype=np.float32)
        pad[: len(audio)] = audio
        yield 0.0, pad
        return
    for start in range(0, len(audio) - win + 1, hop):
        yield start / sr, audio[start : start + win]
    # ensure tail coverage
    last_start = len(audio) - win
    if last_start % hop != 0:
        yield last_start / sr, audio[last_start : last_start + win]
