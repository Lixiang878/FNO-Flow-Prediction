import numpy as np
import pytest

from fno_flow import (
    FNO1D,
    UNet1D,
    burgers_solver,
    generate_dataset,
    lowres_solver_error,
    relative_l2,
)


def test_burgers_solver_runs_and_shapes():
    u0 = np.sin(2 * np.pi * np.linspace(0, 1, 64, endpoint=False))
    u = burgers_solver(u0, nu=0.01, T=0.1, dt=0.0005, dx=1 / 64)
    assert u.shape == (64,)
    assert np.all(np.isfinite(u))
    # smooth initial condition should stay bounded
    assert np.max(np.abs(u)) < 5.0


def test_dataset_shapes_and_finite():
    data = generate_dataset(n_samples=4, n_grid=256, seed=1)
    assert data["a"].shape == (4, 256)
    assert data["u"].shape == (4, 256)
    assert np.all(np.isfinite(data["u"]))


def test_fno_forward_shape_and_finite():
    fno = FNO1D(n_modes=8, width=16, n_blocks=2)
    x = np.random.default_rng(0).standard_normal((3, 256))
    out = fno(x)
    assert out.shape == (3, 256)
    assert np.all(np.isfinite(out))


def test_unet_forward_shape_and_finite():
    unet = UNet1D(width=8)
    x = np.random.default_rng(0).standard_normal((3, 256))
    out = unet(x)
    assert out.shape == (3, 256)
    assert np.all(np.isfinite(out))


def test_fno_resolution_invariance():
    fno = FNO1D(n_modes=8, width=16, n_blocks=2)
    x256 = np.random.default_rng(1).standard_normal((2, 256))
    x128 = x256[:, ::2]
    assert fno(x256).shape == (2, 256)
    assert fno(x128).shape == (2, 128)


def test_relative_l2_basics():
    a = np.ones(10)
    assert relative_l2(a, a) == 0.0
    assert relative_l2(2 * a, a) == 1.0


def test_lowres_baseline_finite_positive():
    data = generate_dataset(n_samples=3, n_grid=256, seed=2)
    err = lowres_solver_error(data["u"][:1], data["a"][:1], n_coarse=64)
    assert err > 0.0
    assert np.isfinite(err)


def test_burgers_mass_conserved_periodic():
    rng = np.random.default_rng(7)
    for _ in range(5):
        n = 128
        u0 = rng.standard_normal(n)
        u = burgers_solver(u0, nu=0.02, T=0.3, dt=0.0005, dx=1.0 / n)
        # Periodic BC + telescoping flux => total mass is conserved.
        assert np.isclose(u.sum(), u0.sum(), rtol=1e-6)


def test_burgers_T0_returns_initial():
    u0 = np.sin(2 * np.pi * np.linspace(0, 1, 64, endpoint=False))
    u = burgers_solver(u0, T=0.0, dt=0.0005, dx=1.0 / 64)
    assert np.allclose(u, u0)


def test_burgers_rejects_invalid_params():
    u0 = np.ones(8)
    for kw in (dict(dx=0), dict(dt=0), dict(nu=-1), dict(T=-1)):
        with pytest.raises(ValueError):
            burgers_solver(u0, **kw)
    with pytest.raises(ValueError):
        burgers_solver(np.array([]))
