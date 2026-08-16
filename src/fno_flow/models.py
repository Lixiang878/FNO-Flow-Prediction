"""Neural-operator and convolutional surrogates for 1D Burgers' equation.

Both ``FNO1D`` and ``UNet1D`` implement their *forward pass* in pure numpy so the
architectures can be instantiated and run offline (e.g. in CI, or to sanity-check
shapes). The learnable weights are initialised randomly; real training is done by
the torch path in :mod:`fno_flow.train` (lazy-imported, optional). The mathematics
of ``FNO1D`` (spectral convolution in Fourier space) is the genuine Fourier Neural
Operator operation of Li et al., 2020.
"""

from __future__ import annotations

import numpy as np


class FNO1D:
    """Fourier Neural Operator for 1D sequences.

    Forward: lift -> K spectral-convolution blocks -> projection.
    Spectral convolution keeps the lowest ``modes`` Fourier modes and applies a
    learnable complex weight per mode, then adds a learnable pointwise branch.
    """

    def __init__(
        self,
        *,
        n_modes: int = 16,
        width: int = 32,
        n_blocks: int = 4,
        seed: int = 7,
    ) -> None:
        self.n_modes = n_modes
        self.width = width
        self.n_blocks = n_blocks
        rng = np.random.default_rng(seed)
        w = width
        # lift: 1 input channel -> width
        self.w_lift = (rng.standard_normal((w,)) * 0.1).astype(float)
        self.b_lift = np.zeros(w)
        # per-block spectral weights (complex) and pointwise weights
        self.spec_w = rng.standard_normal((n_blocks, w, w, n_modes)) \
            + 1j * rng.standard_normal((n_blocks, w, w, n_modes))
        self.spec_w = self.spec_w.astype(complex) * 0.1
        self.pt_w = (rng.standard_normal((n_blocks, w, w)) * 0.1).astype(float)
        self.pt_b = np.zeros((n_blocks, w))
        # projection: width -> 1
        self.w_proj = (rng.standard_normal((w,)) * 0.1).astype(float)
        self.b_proj = np.zeros(1)

    def _spectral_conv(self, x: np.ndarray, b: int) -> np.ndarray:
        # x: (B, w, N)
        _B, w, N = x.shape
        modes = min(self.n_modes, N // 2 + 1)
        X = np.fft.rfft(x, axis=-1)  # (B, w, N//2+1)
        out = np.zeros_like(X)
        W = self.spec_w[b, :, :, :modes]  # (w, w, modes)
        for cin in range(w):
            for cout in range(w):
                out[:, cout, :modes] += X[:, cin, :modes] * W[cout, cin]
        y = np.fft.irfft(out, n=N, axis=-1)  # (B, w, N)
        return y

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 2:
            x = x[:, None, :]  # (B, 1, N)
        B, _, N = x.shape
        w = self.width
        h = self.w_lift[None, :, None] * x[:, 0:1, :] + self.b_lift[None, :, None]
        h = np.broadcast_to(h, (B, w, N)).copy()
        for b in range(self.n_blocks):
            spec = self._spectral_conv(h, b)
            pt = np.einsum("bix,ji->bjx", h, self.pt_w[b]) + self.pt_b[b][None, :, None]
            h = np.maximum(spec + pt, 0.0)  # ReLU
        out = np.einsum("bwx,w->bx", h, self.w_proj) + self.b_proj[0]
        return out  # (B, N)


def _conv1d(x: np.ndarray, w: np.ndarray, b: np.ndarray, stride: int = 1,
            padding: str = "same") -> np.ndarray:
    """Simple 1D convolution. w: (Cout, Cin, k). x: (B, Cin, L)."""
    B, _Cin, L = x.shape
    Cout, _, k = w.shape
    if padding == "same":
        pad = k // 2
        x = np.pad(x, ((0, 0), (0, 0), (pad, pad)), mode="edge")
        L = x.shape[-1]
    olen = (L - k) // stride + 1
    out = np.zeros((B, Cout, olen), dtype=float)
    for i in range(olen):
        window = x[:, :, i * stride:i * stride + k]  # (B, Cin, k)
        out[:, :, i] = np.einsum("bck,ock->bo", window, w) + b[None, :]
    return out


def _pool(x: np.ndarray, stride: int = 2) -> np.ndarray:
    B, C, L = x.shape
    olen = L // stride
    return x[:, :, :olen * stride].reshape(B, C, olen, stride).mean(axis=-1)


def _upsample(x: np.ndarray, stride: int = 2) -> np.ndarray:
    return np.repeat(x, stride, axis=-1)


class UNet1D:
    """Compact 1D U-Net (encoder-decoder with skip connections)."""

    def __init__(self, *, width: int = 16, seed: int = 11, k: int = 7) -> None:
        self.width = width
        self.k = k
        rng = np.random.default_rng(seed)
        w = width
        self.e1 = (rng.standard_normal((w, 1, k)) * 0.1).astype(float)
        self.e1b = np.zeros(w)
        self.e2 = (rng.standard_normal((2 * w, w, k)) * 0.1).astype(float)
        self.e2b = np.zeros(2 * w)
        self.bot = (rng.standard_normal((2 * w, 2 * w, k)) * 0.1).astype(float)
        self.botb = np.zeros(2 * w)
        self.d1 = (rng.standard_normal((w, w + 2 * w, k)) * 0.1).astype(float)
        self.d1b = np.zeros(w)
        self.out = (rng.standard_normal((1, w, k)) * 0.1).astype(float)
        self.outb = np.zeros(1)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 2:
            x = x[:, None, :]
        # pad to multiple of 4 for clean pooling
        _B, _, L = x.shape
        pad = (4 - L % 4) % 4
        if pad:
            x = np.pad(x, ((0, 0), (0, 0), (0, pad)), mode="edge")
        h1 = np.maximum(_conv1d(x, self.e1, self.e1b), 0.0)      # (B, w, L)
        p1 = _pool(h1)                                            # (B, w, L/2)
        h2 = np.maximum(_conv1d(p1, self.e2, self.e2b), 0.0)      # (B, 2w, L/2)
        p2 = _pool(h2)                                            # (B, 2w, L/4)
        bot = np.maximum(_conv1d(p2, self.bot, self.botb), 0.0)   # (B, 2w, L/4)
        up = _upsample(bot)                                       # (B, 2w, L/2)
        cat = np.concatenate([p1, up], axis=1)                    # (B, w+2w, L/2)
        d1 = np.maximum(_conv1d(cat, self.d1, self.d1b), 0.0)     # (B, w, L/2)
        d1 = _upsample(d1)                                        # (B, w, L)
        if pad:
            d1 = d1[:, :, :L]
        out = _conv1d(d1, self.out, self.outb)                    # (B, 1, L)
        return out[:, 0, :]  # (B, L)
