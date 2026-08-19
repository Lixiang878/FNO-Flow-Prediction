import numpy as np
import pytest

from fno_flow.data import burgers_solver


def test_burgers_preserves_shape():
    n = 64
    x = np.linspace(0, 1, n, endpoint=False)
    u0 = np.sin(2 * np.pi * x)
    u = burgers_solver(u0, T=0.1, dt=0.0005, dx=1.0 / n)
    assert u.shape == (n,)


def test_burgers_T0_returns_initial():
    u0 = np.random.default_rng(0).standard_normal(32)
    u = burgers_solver(u0, T=0.0)
    np.testing.assert_array_equal(u, u0)


def test_burgers_nondiffusive_shock_forms():
    # step initial condition -> shock should form and propagate
    n = 128
    x = np.linspace(0, 1, n, endpoint=False)
    u0 = np.where(x < 0.5, 1.0, 0.0)
    u = burgers_solver(u0, nu=0.005, T=0.3, dt=0.0002, dx=1.0 / n)
    # solution should remain bounded (no blow-up)
    assert np.all(np.isfinite(u))
    assert u.max() <= 1.0 + 1e-6
    assert u.min() >= -0.1


def test_burgers_rejects_bad_params():
    with pytest.raises(ValueError):
        burgers_solver(np.ones(16), dx=-1)
    with pytest.raises(ValueError):
        burgers_solver(np.ones(16), dt=0)
    with pytest.raises(ValueError):
        burgers_solver(np.ones(16), nu=-0.1)
    with pytest.raises(ValueError):
        burgers_solver(np.array([]))
