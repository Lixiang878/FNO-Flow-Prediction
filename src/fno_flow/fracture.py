"""Load and train small 2D FNO surrogates on fracture-flow NPZ exports."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_fracture_dataset(path: str | Path) -> dict:
    """Load the bench NPZ and validate its field/metadata contract."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        required = {"b", "p", "jrc", "seed", "split", "Q", "residual", "imbalance", "dx", "dy", "dp", "mu"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"dataset missing keys: {sorted(missing)}")
        out = {key: data[key] for key in data.files}
    if out["b"].ndim != 3 or out["p"].shape != out["b"].shape:
        raise ValueError("b and p must have matching (N,H,W) shapes")
    n = out["b"].shape[0]
    for key in ("jrc", "seed", "split", "Q", "residual", "imbalance", "dx", "dy"):
        if len(out[key]) != n:
            raise ValueError(f"metadata key {key!r} has wrong length")
    if out["b"].shape[1] < 3 or out["b"].shape[2] < 3:
        raise ValueError("fields need at least a 3x3 grid")
    if not np.all(np.isfinite(out["b"])) or np.any(out["b"] <= 0) or not np.all(np.isfinite(out["p"])):
        raise ValueError("b must be finite and positive; p must be finite")
    for key in ("Q", "residual", "imbalance", "dx", "dy"):
        if not np.all(np.isfinite(out[key])):
            raise ValueError(f"metadata key {key!r} contains non-finite values")
    if np.any(out["dx"] <= 0) or np.any(out["dy"] <= 0):
        raise ValueError("dx and dy must be positive")
    split_indices(out)
    manifest_path = path.with_suffix(".json")
    if manifest_path.exists():
        out["manifest"] = json.loads(manifest_path.read_text(encoding="utf-8"))
    return out


def split_indices(data: dict) -> dict[str, np.ndarray]:
    """Return condition-level split indices with bytes compatibility."""
    raw = np.asarray(data["split"])
    if raw.ndim != 1:
        raise ValueError("split must be a one-dimensional string array")
    split = np.array([
        value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in raw
    ])
    unknown = set(split) - {"train", "val", "test"}
    if unknown:
        raise ValueError(f"unknown split labels: {sorted(unknown)}")
    return {name: np.flatnonzero(split == name) for name in ("train", "val", "test")}


def make_channels(data: dict, indices: np.ndarray, *, include_coordinates: bool = True,
                  stats: dict | None = None):
    """Build normalized input/target using training statistics when supplied."""
    b = np.asarray(data["b"])[indices]
    p = np.asarray(data["p"])[indices]
    if b.shape[0] == 0:
        raise ValueError("cannot build channels for an empty split")
    if stats is None:
        mean_b, std_b = float(b.mean()), float(b.std() + 1e-8)
        mean_p, std_p = float(p.mean()), float(p.std() + 1e-8)
    else:
        mean_b, std_b = float(stats["b_mean"]), float(stats["b_std"])
        mean_p, std_p = float(stats["p_mean"]), float(stats["p_std"])
    x = (b - mean_b) / max(std_b, 1e-8)
    if include_coordinates:
        h, w = b.shape[-2:]
        yy, xx = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing="ij")
        x = np.stack([x, np.broadcast_to(xx, x.shape), np.broadcast_to(yy, x.shape)], axis=1)
    else:
        x = x[:, None]
    y = (p - mean_p) / std_p
    stats = {"b_mean": mean_b, "b_std": std_b, "p_mean": mean_p, "p_std": std_p}
    return x.astype(np.float32), y[:, None].astype(np.float32), stats


def pressure_physics_metrics(b: np.ndarray, p_pred: np.ndarray, p_true: np.ndarray,
                             dx: float | np.ndarray, dy: float | np.ndarray,
                             mu: float = 1.0) -> dict:
    """Compute field, discharge, conservation and pressure-boundary errors.

    The last axis is the flow direction (x); the penultimate axis is the
    transverse direction (y). Inputs may be one field or a batch of fields.
    """
    b = np.asarray(b, dtype=float)
    pred = np.asarray(p_pred, dtype=float)
    true = np.asarray(p_true, dtype=float)
    if b.shape != pred.shape or pred.shape != true.shape or b.ndim not in (2, 3):
        raise ValueError("b, p_pred and p_true must have matching 2D or 3D shapes")
    if not np.isfinite(mu) or mu <= 0:
        raise ValueError("mu must be finite and positive")
    batch = b.ndim == 3
    if not batch:
        b, pred, true = b[None], pred[None], true[None]
    n = b.shape[0]
    dx = np.broadcast_to(np.asarray(dx, dtype=float), (n,)).reshape(n, 1, 1)
    dy = np.broadcast_to(np.asarray(dy, dtype=float), (n,)).reshape(n, 1, 1)
    if not np.all(np.isfinite(dx)) or not np.all(np.isfinite(dy)) or np.any(dx <= 0) or np.any(dy <= 0):
        raise ValueError("dx and dy must be finite and positive")
    rel_l2 = float(np.linalg.norm(pred - true) / max(np.linalg.norm(true), 1e-12))
    b_face = ((b[..., 1:] + b[..., :-1]) * 0.5) ** 3
    q_pred_faces = -b_face * (pred[..., 1:] - pred[..., :-1]) / dx
    q_true_faces = -b_face * (true[..., 1:] - true[..., :-1]) / dx
    dy_line = dy[:, 0, 0][:, None]
    q_pred_cols = q_pred_faces.sum(axis=-2) * dy_line / (12 * mu)
    q_true_cols = q_true_faces.sum(axis=-2) * dy_line / (12 * mu)
    q_pred = q_pred_cols.mean(axis=-1)
    q_true = q_true_cols.mean(axis=-1)
    q_imbalance = np.max(np.abs(q_pred_cols - q_pred[:, None]), axis=1) / np.maximum(np.abs(q_pred), 1e-12)
    q_relative = np.abs(q_pred - q_true) / np.maximum(np.abs(q_true), 1e-12)
    return {
        "relative_l2": rel_l2,
        "q_pred_mean": float(np.mean(q_pred)),
        "q_true_mean": float(np.mean(q_true)),
        "q_relative_error": float(np.mean(q_relative)),
        "mass_imbalance_pred": float(np.mean(q_imbalance)),
        "inlet_boundary_error": float(np.mean(np.abs(pred[..., 0] - true[..., 0]))),
        "outlet_boundary_error": float(np.mean(np.abs(pred[..., -1] - true[..., -1]))),
    }


def _load_checkpoint(path: str | Path):
    """Load a CPU checkpoint and validate the fields required for inference."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("2D inference requires torch") from exc
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # older torch versions do not expose weights_only
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint must contain a mapping")
    required = {"model", "config", "stats"}
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"checkpoint missing keys: {sorted(missing)}")
    config = checkpoint["config"]
    for key in ("in_channels", "width", "modes"):
        if key not in config:
            raise ValueError(f"checkpoint config missing {key!r}")
    for key in ("b_mean", "b_std", "p_mean", "p_std"):
        if key not in checkpoint["stats"]:
            raise ValueError(f"checkpoint stats missing {key!r}")
    return checkpoint


def infer_2d(data_path: str | Path, checkpoint_path: str | Path, *, split: str = "test",
             output: str | Path | None = None) -> dict:
    """Run a saved 2D FNO on one split and optionally save predicted fields."""
    import torch

    from .models2d import FNO2D

    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val or test")
    checkpoint = _load_checkpoint(checkpoint_path)
    data = load_fracture_dataset(data_path)
    indices = split_indices(data)[split]
    if len(indices) == 0:
        raise ValueError(f"dataset has no {split} samples")
    stats = checkpoint["stats"]
    x, _, _ = make_channels(data, indices, stats=stats)
    config = checkpoint["config"]
    if x.shape[1] != int(config["in_channels"]):
        raise ValueError("checkpoint input channels do not match dataset")
    model = FNO2D(in_channels=int(config["in_channels"]), width=int(config["width"]),
                  modes=int(config["modes"]))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    with torch.no_grad():
        pred_norm = model(torch.from_numpy(x)).numpy()[:, 0]
    pred = pred_norm * float(stats["p_std"]) + float(stats["p_mean"])
    mu_values = np.asarray(data["mu"])[indices]
    if not np.allclose(mu_values, mu_values[0]):
        raise ValueError("per-sample mu values are not supported by one physics metric")
    physics = pressure_physics_metrics(
        data["b"][indices], pred, data["p"][indices], data["dx"][indices],
        data["dy"][indices], mu=float(mu_values[0]),
    )
    report = {
        "split": split,
        "samples": len(indices),
        "indices": indices.tolist(),
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "data": str(Path(data_path).resolve()),
        "physics": physics,
        "prediction_shape": list(pred.shape),
    }
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.suffix == ".npz":
            np.savez_compressed(output, p_pred=pred, p_true=data["p"][indices],
                                b=data["b"][indices], index=indices)
            report["fields"] = str(output)
        else:
            output.write_text(json.dumps(report, indent=2), encoding="utf-8")
            report["report"] = str(output)
    return report


def evaluate_2d(data_path: str | Path, checkpoint_path: str | Path, *, split: str = "test",
                output: str | Path | None = None) -> dict:
    """Evaluate a saved checkpoint and return a JSON-serializable report."""
    report = infer_2d(data_path, checkpoint_path, split=split, output=None)
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        report["report"] = str(output)
    return report


def train_2d(path: str | Path, *, epochs: int = 5, width: int = 16,
             modes: int = 12, lr: float = 1e-3, seed: int = 1234,
             checkpoint: str | Path = "results/fracture_fno.pt") -> dict:
    """Train a compact 2D FNO on the train split and save a checkpoint."""
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("2D training requires torch") from exc
    from .models2d import FNO2D

    if not isinstance(epochs, (int, np.integer)) or epochs <= 0:
        raise ValueError("epochs must be a positive integer")
    if not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    if width <= 0 or modes <= 0 or not np.isfinite(lr) or lr <= 0:
        raise ValueError("width, modes and lr must be positive")
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    data = load_fracture_dataset(path)
    splits = split_indices(data)
    if len(splits["train"]) == 0:
        raise ValueError("dataset has no train samples")
    train_idx = splits["train"]
    eval_idx = splits["val"] if len(splits["val"]) else train_idx
    test_idx = splits["test"]
    x, y, stats = make_channels(data, train_idx)
    xv, yv, _ = make_channels(data, eval_idx, stats=stats)
    model = FNO2D(in_channels=x.shape[1], width=width, modes=modes)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        loss = loss_fn(model(xt), yt); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(xv)).numpy()[:, 0]
    metrics = {"epochs": int(epochs), "train_samples": len(x), "val_samples": len(xv),
               "test_samples": len(test_idx), "normalized_mse": float(np.mean((pred - yv[:, 0]) ** 2)),
               "stats": stats}
    if len(test_idx):
        xtest, ytest, _ = make_channels(data, test_idx, stats=stats)
        with torch.no_grad():
            test_pred = model(torch.from_numpy(xtest)).numpy()[:, 0]
        metrics["test_normalized_mse"] = float(np.mean((test_pred - ytest[:, 0]) ** 2))
        if not np.allclose(data["mu"][test_idx], data["mu"][test_idx][0]):
            raise ValueError("per-sample mu values are not supported by one physics metric")
        metrics["test_physics"] = pressure_physics_metrics(
            data["b"][test_idx], test_pred * stats["p_std"] + stats["p_mean"],
            data["p"][test_idx], data["dx"][test_idx], data["dy"][test_idx],
            mu=float(data["mu"][test_idx][0]),
        )
    checkpoint = Path(checkpoint); checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "epoch": int(epochs),
                "config": {"in_channels": x.shape[1], "width": width, "modes": modes},
                "stats": stats, "seed": int(seed), "splits": {k: v.tolist() for k, v in splits.items()},
                "dataset": str(Path(path).resolve()), "metrics": metrics}, checkpoint)
    return metrics
