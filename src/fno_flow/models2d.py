"""Compact 2D Fourier Neural Operator implemented with torch."""

from __future__ import annotations

import torch
from torch import nn


class SpectralConv2d(nn.Module):
    def __init__(self, channels: int, modes: int):
        super().__init__()
        if channels <= 0 or modes <= 0:
            raise ValueError("channels and modes must be positive")
        self.channels = channels
        self.modes = modes
        scale = 1 / max(1, channels * channels)
        self.weight_pos = nn.Parameter(
            torch.randn(channels, channels, modes, modes, dtype=torch.cfloat) * scale
        )
        self.weight_neg = nn.Parameter(
            torch.randn(channels, channels, modes, modes, dtype=torch.cfloat) * scale
        )

    def forward(self, x):
        _batch, _channels, height, width = x.shape
        modes_h = min(self.modes, height // 2)
        modes_w = min(self.modes, width // 2 + 1)
        spectrum = torch.fft.rfft2(x, norm="ortho")
        out = torch.zeros_like(spectrum)
        low = spectrum[:, :, :modes_h, :modes_w]
        high = spectrum[:, :, -modes_h:, :modes_w]
        out[:, :, :modes_h, :modes_w] = torch.einsum(
            "bixy,oixy->boxy", low, self.weight_pos[:, :, :modes_h, :modes_w]
        )
        out[:, :, -modes_h:, :modes_w] = torch.einsum(
            "bixy,oixy->boxy", high, self.weight_neg[:, :, :modes_h, :modes_w]
        )
        return torch.fft.irfft2(out, s=(height, width), norm="ortho")


class FNO2D(nn.Module):
    """Field-to-field FNO with explicit coordinate channels in the caller."""

    def __init__(self, in_channels: int = 3, width: int = 16, modes: int = 12, blocks: int = 3):
        super().__init__()
        self.lift = nn.Conv2d(in_channels, width, 1)
        self.spectral = nn.ModuleList([SpectralConv2d(width, modes) for _ in range(blocks)])
        self.pointwise = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(blocks)])
        self.head = nn.Sequential(nn.Conv2d(width, width, 1), nn.GELU(), nn.Conv2d(width, 1, 1))

    def forward(self, x):
        h = self.lift(x)
        for spec, point in zip(self.spectral, self.pointwise):
            h = torch.nn.functional.gelu(spec(h) + point(h))
        return self.head(h)
