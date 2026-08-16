"""Torch 1D U-Net surrogate (optional, lazy-imported by :mod:`fno_flow.train`)."""

from __future__ import annotations


class TorchUNet1D:
    """Compact 1D U-Net with skip connections, implemented in torch."""

    def __init__(self, width: int = 16, k: int = 7):
        from torch import nn

        self.width = width
        self.enc1 = nn.Sequential(nn.Conv1d(1, width, k, padding=k // 2), nn.ReLU())
        self.pool = nn.AvgPool1d(2)
        self.enc2 = nn.Sequential(nn.Conv1d(width, 2 * width, k, padding=k // 2), nn.ReLU())
        self.bot = nn.Sequential(nn.Conv1d(2 * width, 2 * width, k, padding=k // 2), nn.ReLU())
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec1 = nn.Sequential(
            nn.Conv1d(width + 2 * width, width, k, padding=k // 2), nn.ReLU()
        )
        self.head = nn.Conv1d(width, 1, k, padding=k // 2)

    def forward(self, x):
        import torch

        h1 = self.enc1(x)
        p1 = self.pool(h1)
        h2 = self.enc2(p1)
        p2 = self.pool(h2)
        bot = self.bot(p2)
        up = self.up(bot)
        cat = torch.cat([p1, up], dim=1)
        d1 = self.up(self.dec1(cat))
        return self.head(d1)[:, 0, :]

    def __call__(self, x):
        return self.forward(x)
