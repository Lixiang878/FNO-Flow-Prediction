"""Classical baselines used to contextualise the neural-operator results."""

from __future__ import annotations

import numpy as np

from .data import burgers_solver


def relative_l2(pred: np.ndarray, truth: np.ndarray) -> float:
    """Relative L2 error ``||pred - truth|| / ||truth||``."""
    pred = np.asarray(pred, dtype=float)
    truth = np.asarray(truth, dtype=float)
    denom = np.linalg.norm(truth)
    if denom == 0.0:
        return float(np.linalg.norm(pred - truth))
    return float(np.linalg.norm(pred - truth) / denom)


def _downsample(x: np.ndarray, n_coarse: int) -> np.ndarray:
    n = x.shape[-1]
    if n == n_coarse:
        return x.copy()
    # simple block average
    factor = n // n_coarse
    return x[..., : factor * n_coarse].reshape(x.shape[0], n_coarse, factor).mean(axis=-1)


def _upsample_linear(x: np.ndarray, n_fine: int) -> np.ndarray:
    n = x.shape[-1]
    xp = np.linspace(0.0, 1.0, n, endpoint=False)
    xq = np.linspace(0.0, 1.0, n_fine, endpoint=False)
    return np.array([np.interp(xq, xp, row) for row in x])


def lowres_solver_error(
    truth_u: np.ndarray,
    a: np.ndarray,
    *,
    n_coarse: int = 64,
    nu: float = 0.01,
    T: float = 1.0,
    dx: float = 1.0 / 256.0,
) -> float:
    """Relative-L2 error of an *under-resolved* classical FD solver.

    Solves Burgers' on a coarse grid (``n_coarse``) using the same scheme as the
    data generator, then upsamples to the fine grid for comparison. This is the
    honest "classical numerical method" baseline the learned surrogates should
    beat.
    """
    n_fine = truth_u.shape[-1]
    a_coarse = _downsample(a, n_coarse)
    dx_coarse = 1.0 / n_coarse
    dt = dx_coarse / 4.0  # CFL-safe for the Lax-Friedrichs step
    u_coarse = burgers_solver(a_coarse, nu=nu, T=T, dt=dt, dx=dx_coarse)
    u_up = _upsample_linear(u_coarse[None, :] if u_coarse.ndim == 1 else u_coarse,
                            n_fine)
    return relative_l2(u_up, truth_u)
