import numpy as np
import pytest

from fno_flow.fracture import (
    load_fracture_dataset,
    make_channels,
    pressure_physics_metrics,
    split_indices,
)


def test_fracture_loader_and_channels(tmp_path):
    path = tmp_path / "data.npz"
    n = 4
    b = np.ones((n, 9, 9))
    p = np.linspace(1, 0, 9)[None, None, :] * np.ones_like(b)
    np.savez(path, b=b, p=p, jrc=np.arange(n), seed=np.arange(n),
             split=np.array(["train", "train", "val", "test"]), Q=np.ones(n),
             residual=np.zeros(n), imbalance=np.zeros(n), dx=np.ones(n), dy=np.ones(n),
             dp=np.ones(n), mu=np.ones(n))
    data = load_fracture_dataset(path)
    splits = split_indices(data)
    x, y, stats = make_channels(data, splits["train"])
    assert x.shape == (2, 3, 9, 9)
    assert y.shape == (2, 1, 9, 9)
    assert stats["b_std"] > 0


def test_physics_metrics_are_zero_for_exact_field():
    b = np.ones((9, 9))
    p = np.tile(np.linspace(1, 0, 9), (9, 1))
    metrics = pressure_physics_metrics(b, p, p, 1.0, 1.0)
    assert metrics["relative_l2"] == pytest.approx(0)
    assert metrics["q_relative_error"] == pytest.approx(0)
    assert metrics["inlet_boundary_error"] == pytest.approx(0)


def test_physics_metrics_batch_uses_flow_axis_and_per_sample_spacing():
    b = np.ones((2, 5, 7))
    p = np.stack([
        np.tile(np.linspace(1, 0, 7), (5, 1)),
        np.tile(np.linspace(2, 0, 7), (5, 1)),
    ])
    metrics = pressure_physics_metrics(b, p, p, np.array([1.0, 2.0]), np.array([1.0, 3.0]))
    assert metrics["q_relative_error"] == pytest.approx(0)
    assert metrics["mass_imbalance_pred"] == pytest.approx(0)
