"""CNN + BiLSTM speech-emotion model.

Input is a 3-channel spectrogram: log-mel, delta, and delta-delta.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class SpecAugment(nn.Module):
    """Zero random frequency and time bands during training only."""

    def __init__(self, freq_mask_max: int = 8, time_mask_max: int = 12, num_masks: int = 2):
        super().__init__()
        self.freq_mask_max = freq_mask_max
        self.time_mask_max = time_mask_max
        self.num_masks = num_masks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        out = x.clone()
        _, _, n_freq, n_time = out.shape
        for _ in range(self.num_masks):
            f = int(torch.randint(0, self.freq_mask_max + 1, (1,)).item())
            f0 = int(torch.randint(0, max(1, n_freq - f), (1,)).item())
            out[:, :, f0 : f0 + f, :] = 0
            if n_time > 4:
                t_max = min(self.time_mask_max, max(1, n_time // 8))
                t = int(torch.randint(0, t_max + 1, (1,)).item())
                t0 = int(torch.randint(0, max(1, n_time - t), (1,)).item())
                out[:, :, :, t0 : t0 + t] = 0
        return out


class CNNBiLSTM(nn.Module):
    def __init__(self, n_mels: int = config.N_MELS, num_classes: int = config.NUM_CLASSES, in_channels: int = 3):
        super().__init__()
        self.augment = SpecAugment()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
        )
        self.freq_bins = n_mels // 8
        self.lstm_input = 128 * self.freq_bins
        self.lstm = nn.LSTM(
            input_size=self.lstm_input,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, n_mels, T)
        x = self.augment(x)
        h = self.conv(x)
        b, c, f, t = h.shape
        h = h.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
        out, _ = self.lstm(h)
        mean_pool = out.mean(dim=1)
        max_pool, _ = out.max(dim=1)
        pooled = 0.5 * (mean_pool + max_pool)
        return self.fc(pooled)


def predict_proba(model: nn.Module, mel: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        logits = model(mel)
        return F.softmax(logits, dim=-1)
