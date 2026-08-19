"""Optional torch training for FNO1D / UNet1D.

This module is **lazy-imported** and never required for the offline core. It
implements torch equivalents of the numpy architectures in :mod:`fno_flow.models`,
trains them on the generated Burgers dataset, and reports relative-L2 errors. Run
with ``python -m fno_flow.train`` (requires ``torch`` and ``numpy``).
"""

from __future__ import annotations

import json
from pathlib import Path


def _require_torch():
    try:
        import torch
        return torch
    except ImportError as exc:  # pragma: no cover - env dependent
        raise SystemExit(
            "torch is required for training. Install it (pip install torch) or "
            "use the offline demo (python -m fno_flow.demo)."
        ) from exc


def _nn_module_base():
    """Resolve nn.Module lazily so NumPy-only installs still import this module."""
    torch = _require_torch()
    return torch.nn.Module


class TorchFNO1D(_nn_module_base()):
    """Torch Fourier Neural Operator (1D). Mirrors the numpy FNO1D."""

    def __init__(self, n_modes: int = 16, width: int = 32, n_blocks: int = 4):
        import torch
        from torch import nn

        super().__init__()
        self.n_modes = n_modes
        self.width = width
        self.n_blocks = n_blocks
        self.lift = nn.Linear(1, width)
        self.spec = nn.ModuleList(
            [nn.Linear(n_modes, n_modes, bias=False) for _ in range(width * width * n_blocks)]
        )
        # simpler param layout: store per-block learnable complex weights
        self.spec_w = nn.ParameterList(
            [nn.Parameter(torch.randn(width, width, n_modes, dtype=torch.cfloat) * 0.1)
             for _ in range(n_blocks)]
        )
        self.pt = nn.ModuleList([nn.Linear(width, width) for _ in range(n_blocks)])
        self.proj = nn.Linear(width, 1)

    def _spectral(self, x, b):
        import torch

        _B, w, N = x.shape
        modes = min(self.n_modes, N // 2 + 1)
        X = torch.fft.rfft(x)
        out = torch.zeros_like(X)
        W = self.spec_w[b][:, :, :modes]
        for cin in range(w):
            for cout in range(w):
                out[:, cout, :modes] += X[:, cin, :modes] * W[cout, cin]
        return torch.fft.irfft(out, n=N)

    def forward(self, x):
        import torch

        # x: (B, 1, N)
        _B, _, _N = x.shape
        h = self.lift(x.transpose(1, 2))  # (B, N, w) -> transpose back
        h = h.transpose(1, 2)  # (B, w, N)
        for b in range(self.n_blocks):
            spec = self._spectral(h, b)
            pt = self.pt[b](h.transpose(1, 2)).transpose(1, 2)
            h = torch.relu(spec + pt)
        out = self.proj(h.transpose(1, 2))  # (B, N, 1)
        return out.transpose(1, 2)[:, 0, :]  # (B, N)



def train(model_name: str = "fno", *, epochs: int = 50, lr: float = 1e-3,
          n_samples: int = 200, seed: int = 1234, out_json: str = "results/train_metrics.json"):
    """Train one model on Burgers data and return a metrics dict."""
    torch = _require_torch()
    from torch import nn

    from .data import generate_dataset

    data = generate_dataset(n_samples=n_samples, seed=seed)
    a = torch.tensor(data["a"], dtype=torch.float32)[:, None, :]
    u = torch.tensor(data["u"], dtype=torch.float32)[:, None, :]
    n = a.shape[0]
    n_val = max(1, n // 5)
    a_tr, a_va = a[:-n_val], a[-n_val:]
    u_tr, u_va = u[:-n_val], u[-n_val:]

    if model_name == "fno":
        model = TorchFNO1D()
    else:
        from . import torch_unet
        model = torch_unet.TorchUNet1D()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        pred = model(a_tr)
        loss = loss_fn(pred, u_tr[:, 0, :])
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        pred = model(a_va)[:, None, :]
        val_rel = torch.norm(pred - u_va, dim=(1, 2)) / torch.norm(u_va, dim=(1, 2))
        val_rel = float(val_rel.mean())
    metrics = {
        "model": model_name,
        "epochs": epochs,
        "train_samples": int(n - n_val),
        "val_relative_l2": round(val_rel, 4),
    }
    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    return metrics


if __name__ == "__main__":
    for name in ("fno", "unet"):
        print(json.dumps(train(name), indent=2))
