"""Data generation for 1D viscous Burgers' equation.

The equation is

    u_t + (u^2 / 2)_x = nu * u_xx,      periodic BC on [0, 1]

We use a Lax-Friedrichs finite-difference scheme (diffusive but unconditionally
stable for the chosen step sizes) to produce *ground-truth* full-resolution
solutions. The same solver at a coarser resolution serves as the "classical
numerical baseline" the neural operators are compared against.

Everything here is pure numpy so the dataset can be built and the baselines
evaluated with **no optional dependencies**.
"""

from __future__ import annotations

import numpy as np


def burgers_solver(
    u0: np.ndarray,
    *,
    nu: float = 0.01,
    T: float = 1.0,
    dt: float = 0.0005,
    dx: float = 1.0 / 256.0,
) -> np.ndarray:
    """Solve 1D viscous Burgers' from initial field ``u0`` to time ``T``.

    Uses a **Godunov (exact Riemann) flux** for the convective term
    (monotone and shock-capturing) plus central differences for diffusion,
    with periodic boundaries. Returns the final field of shape ``(n_grid,)``.
    """
    u0 = np.asarray(u0, dtype=float)
    n = u0.shape[0]
    n_steps = max(1, int(round(T / dt)))
    u = u0.copy()
    for _ in range(n_steps):
        uL = u
        uR = np.roll(u, -1)  # right neighbour
        # Exact Godunov flux for convex f(u)=u^2/2:
        F = 0.5 * np.maximum(uL, 0.0) ** 2 + 0.5 * np.minimum(uR, 0.0) ** 2
        F_left = np.roll(F, 1)  # flux at the i-1/2 interface
        divF = (F - F_left) / dx
        lap = (np.roll(u, -1) - 2.0 * u + np.roll(u, 1)) / (dx * dx)
        u = u - dt * divF + nu * dt * lap
    return u


def _initial_condition(rng: np.random.Generator, n: int) -> np.ndarray:
    """Smooth random initial condition: band-limited sum of sines."""
    x = np.linspace(0.0, 1.0, n, endpoint=False)
    u = np.zeros(n)
    n_modes = 3
    for k in range(1, n_modes + 1):
        amp = rng.uniform(0.4, 1.0) / k
        phase = rng.uniform(0.0, 2.0 * np.pi)
        u = u + amp * np.sin(2.0 * np.pi * k * x + phase)
    # scale into a stable regime for Burgers shocks
    u = u / max(1.0, np.max(np.abs(u))) * 1.2
    return u


def generate_dataset(
    n_samples: int = 64,
    *,
    n_grid: int = 256,
    nu: float = 0.01,
    T: float = 1.0,
    dx: float = 1.0 / 256.0,
    dt: float = 0.0005,
    seed: int = 1234,
    out_path=None,
) -> dict:
    """Generate ``(initial_condition, solution_at_T)`` pairs for Burgers' eq.

    Returns a dict with ``a`` (initial, shape ``(N, n_grid)``) and ``u`` (final,
    same shape). If ``out_path`` is given, also saves an ``.npz`` there.
    """
    rng = np.random.default_rng(seed)
    a = np.zeros((n_samples, n_grid), dtype=float)
    u = np.zeros((n_samples, n_grid), dtype=float)
    for i in range(n_samples):
        a0 = _initial_condition(rng, n_grid)
        a[i] = a0
        u[i] = burgers_solver(a0, nu=nu, T=T, dt=dt, dx=dx)
    data = {"a": a, "u": u, "nu": np.float64(nu), "T": np.float64(T),
            "dx": np.float64(dx), "n_grid": n_grid}
    if out_path is not None:
        np.savez(out_path, **data)
    return data
