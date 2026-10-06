"""Mel-spectrogram feature extraction."""
from __future__ import annotations

import numpy as np
import librosa

import config


def _norm(x: np.ndarray) -> np.ndarray:
    return ((x - x.mean()) / (x.std() + 1e-6)).astype(np.float32)


def waveform_to_features(audio: np.ndarray, sr: int = config.SAMPLE_RATE) -> np.ndarray:
    """Log-mel plus first and second deltas. Shape (3, n_mels, time).

    Deltas capture how the spectrum moves, which separates emotions that
    share a similar average pitch (joy vs neutral, surprise vs neutral).
    """
    if audio.size == 0:
        return np.zeros((3, config.N_MELS, 1), dtype=np.float32)
    mel = librosa.feature.melspectrogram(
        y=audio.astype(np.float32),
        sr=sr,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        n_mels=config.N_MELS,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    delta = librosa.feature.delta(log_mel)
    delta2 = librosa.feature.delta(log_mel, order=2)
    return np.stack([_norm(log_mel), _norm(delta), _norm(delta2)], axis=0)


def waveform_to_mel(audio: np.ndarray, sr: int = config.SAMPLE_RATE) -> np.ndarray:
    """Backward-compatible name. Returns the 3-channel feature stack."""
    return waveform_to_features(audio, sr)


def save_feature_file(wav_path, cache_path) -> None:
    """Build one cached feature file. Used by the parallel cache builder."""
    from pathlib import Path

    from src.audio_utils import load_audio, prepare_waveform

    cache_path = Path(cache_path)
    if cache_path.exists():
        return
    audio = prepare_waveform(load_audio(Path(wav_path)))
    feat = waveform_to_features(audio)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, feat)


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
