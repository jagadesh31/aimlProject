"""CNN + BiLSTM SER model with attention pooling and SpecAugment."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class SpecAugment(nn.Module):
    """SpecAugment: random freq and time masking during training."""
    def __init__(self, freq_mask_max: int = 12, time_mask_max: int = 20, num_masks: int = 2):
        super().__init__()
        self.freq_mask_max = freq_mask_max
        self.time_mask_max = time_mask_max
        self.num_masks = num_masks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, n_mels, T)
        if not self.training:
            return x
        out = x.clone()
        B, C, F, T = out.shape
        for _ in range(self.num_masks):
            # Frequency masking
            f = torch.randint(0, self.freq_mask_max + 1, (1,)).item()
            f0 = torch.randint(0, max(1, F - f), (1,)).item()
            out[:, :, f0:f0 + f, :] = 0
            # Time masking
            t = torch.randint(0, self.time_mask_max + 1, (1,)).item()
            t0 = torch.randint(0, max(1, T - t), (1,)).item()
            out[:, :, :, t0:t0 + t] = 0
        return out


class AttentionPool(nn.Module):
    """Learned attention pooling over time steps."""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H)
        weights = torch.softmax(self.attn(x), dim=1)  # (B, T, 1)
        return (x * weights).sum(dim=1)               # (B, H)


class CNNBiLSTM(nn.Module):
    def __init__(self, n_mels: int = config.N_MELS, num_classes: int = config.NUM_CLASSES):
        super().__init__()
        self.augment = SpecAugment(freq_mask_max=12, time_mask_max=20, num_masks=2)
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
            hidden_size=256,          # wider: 128 -> 256
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )
        lstm_out = 256 * 2  # bidirectional
        self.attn_pool = AttentionPool(lstm_out)
        self.fc = nn.Sequential(
            nn.LayerNorm(lstm_out),
            nn.Linear(lstm_out, 256),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, n_mels, T)
        x = self.augment(x)
        h = self.conv(x)
        b, c, f, t = h.shape
        h = h.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
        out, _ = self.lstm(h)
        pooled = self.attn_pool(out)   # learned attention instead of mean+max
        return self.fc(pooled)


def predict_proba(model: nn.Module, mel: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        logits = model(mel)
        return F.softmax(logits, dim=-1)
