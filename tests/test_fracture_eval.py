import numpy as np
import pytest

from fno_flow.fracture import evaluate_2d, infer_2d, train_2d


def _write_dataset(path):
    n = 4
    b = np.ones((n, 9, 9))
    p = np.stack([np.tile(np.linspace(1, 0, 9), (9, 1)) * (i + 1) for i in range(n)])
    np.savez(path, b=b, p=p, jrc=np.arange(n), seed=np.arange(n),
             split=np.array(["train", "train", "val", "test"]), Q=np.ones(n),
             residual=np.zeros(n), imbalance=np.zeros(n), dx=np.ones(n), dy=np.ones(n),
             dp=np.ones(n), mu=np.ones(n))


def test_evaluate_and_infer_roundtrip(tmp_path):
    pytest.importorskip("torch")
    data_path = tmp_path / "data.npz"
    checkpoint = tmp_path / "model.pt"
    fields = tmp_path / "prediction.npz"
    report_path = tmp_path / "report.json"
    _write_dataset(data_path)
    train_2d(data_path, epochs=1, width=4, modes=3, seed=11, checkpoint=checkpoint)

    report = evaluate_2d(data_path, checkpoint, output=report_path)
    assert report["split"] == "test"
    assert report["samples"] == 1
    assert report["physics"]["relative_l2"] >= 0
    assert report_path.exists()

    inferred = infer_2d(data_path, checkpoint, output=fields)
    assert inferred["fields"] == str(fields)
    with np.load(fields) as saved:
        assert saved["p_pred"].shape == (1, 9, 9)
        assert saved["p_true"].shape == saved["p_pred"].shape


def test_infer_rejects_missing_split(tmp_path):
    pytest.importorskip("torch")
    data_path = tmp_path / "data.npz"
    checkpoint = tmp_path / "missing.pt"
    _write_dataset(data_path)
    with pytest.raises(FileNotFoundError):
        infer_2d(data_path, checkpoint)
