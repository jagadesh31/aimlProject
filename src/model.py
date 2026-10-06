"""CNN + BiLSTM SER model."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import config

class CNNBiLSTM(nn.Module):
    def __init__(self, n_mels: int = config.N_MELS, num_classes: int = config.NUM_CLASSES):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
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
        # After 3 pools, freq dim = n_mels // 8
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
        # x: (B, 1, n_mels, T)
        h = self.conv(x)
        b, c, f, t = h.shape
        h = h.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
        out, _ = self.lstm(h)
        # attention-ish: mean + max over time
        mean_pool = out.mean(dim=1)
        max_pool, _ = out.max(dim=1)
        pooled = 0.5 * (mean_pool + max_pool)
        return self.fc(pooled)


def predict_proba(model: nn.Module, mel: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        logits = model(mel)
        return F.softmax(logits, dim=-1)
