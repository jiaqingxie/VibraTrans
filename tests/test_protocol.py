import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from vibratrans.data import GlobalBatches, PairedArrays
from vibratrans.inference import evaluate, predict_file, load_checkpoint, normalize
from vibratrans.losses import objective
from vibratrans.metrics import per_sample_metrics
from vibratrans.model import build_model, predict
from vibratrans.runtime import configure_runtime, digest
from vibratrans.train import train


@pytest.fixture
def experiment(tmp_path):
    configure_runtime()
    rng = np.random.default_rng(381)
    data = tmp_path / "data/toy"
    data.mkdir(parents=True)
    for source in ("ir", "raman"):
        np.save(data / f"{source}.npy", rng.random((12, 83), dtype=np.float32))
    split = tmp_path / "split.npz"
    np.savez(split, train=np.arange(8), val=np.arange(8, 10), test=np.arange(10, 12))
    task = dict(
        name="toy",
        source="ir",
        seed=0,
        epochs=2,
        batch_size=4,
        validation_every=1,
        data_directory="toy",
        split_file=str(split),
        patch_size=20,
        data_contract=dict(length=83, split_sha256=digest(split)),
        model=dict(width=24, depth=2, heads=6),
        config=dict(
            architecture="vibradit",
            steps=1,
            dropout=0.0,
            lr=0.0002,
            terminal="raw",
            velocity_weight=0.1,
            shape_weight=0.02,
            r2_weight=0.005,
            ema_decay=0.9995,
            locked_dropout=True,
        ),
        selection_reference=dict(mae=0.3, r2=0.1),
    )
    return task, tmp_path / "data"


@pytest.mark.parametrize("architecture", ["vibradit", "pe", "cnn"])
def test_train_save_predict_evaluate(experiment, tmp_path, architecture):
    task, root = experiment
    task["config"]["architecture"] = architecture
    task["config"]["velocity_weight"] = 0.1 if architecture == "vibradit" else 0.0
    run = tmp_path / "run"
    train(task, root, run, device="cpu", workers=0)
    complete = json.loads((run / "complete.json").read_text())
    assert complete["updates"] == 4
    assert complete["best_checkpoint_sha256"] == digest(run / "best.pt")
    feature_path = tmp_path / "features.npy" if architecture != "cnn" else None
    predict_file(
        run / "best.pt",
        root / "toy/ir.npy",
        tmp_path / "prediction.npy",
        features_output=feature_path,
        batch_size=3,
    )
    prediction = np.load(tmp_path / "prediction.npy")
    assert prediction.shape == (12, 83)
    assert np.isfinite(prediction).all() and 0 <= prediction.min() <= prediction.max() <= 1
    if feature_path:
        features = np.load(feature_path)
        assert features.shape == (12, 24)
        model, saved_task = load_checkpoint(run / "best.pt", 83)
        captured = []
        module = model.velocity_field.blocks[-1] if architecture == "vibradit" else model.encoder
        handle = module.register_forward_hook(
            lambda m, a, value: captured.append(value.detach().mean(1))
        )
        x = torch.from_numpy(normalize(np.load(root / "toy/ir.npy")[:3]))[:, None, None]
        with torch.no_grad():
            predict(model, x, 2, saved_task["config"])
        handle.remove()
        np.testing.assert_allclose(features[:3], captured[0].numpy(), rtol=1e-5, atol=1e-6)
    evaluate(run / "best.pt", task, root, tmp_path / "evaluation")
    np.testing.assert_array_equal(np.load(tmp_path / "evaluation/row_indices.npy"), [10, 11])
    np.testing.assert_allclose(
        np.load(tmp_path / "evaluation/predictions.npy"), prediction[10:], atol=1e-6
    )
    with pytest.raises(FileExistsError):
        train(task, root, run, device="cpu", workers=0)
    wrong = copy.deepcopy(task)
    wrong["source"] = "raman"
    with pytest.raises(ValueError, match="source"):
        evaluate(run / "best.pt", wrong, root, tmp_path / "wrong")


def test_endpoint_loss_uses_unclipped_prediction(experiment):
    task, _ = experiment
    model = build_model(task, 83)
    with torch.no_grad():
        model.velocity_field.final_layer.linear.bias.fill_(1.5)
    x = torch.full((2, 1, 1, 83), 0.7)
    y = torch.rand_like(x)
    raw = predict(model, x, 2, task["config"], training=True)
    assert (raw > 1).all()
    assert predict(model, x, 2, task["config"]).eq(1).all()
    loss = objective(model, x, y, torch.tensor([0.2, 0.8]), 2, task["config"])
    loss.backward()
    assert torch.isfinite(loss)
    assert model.velocity_field.final_layer.linear.bias.grad.abs().sum() > 0


def test_batches_cover_tail_and_reproduce_times():
    sampler = GlobalBatches(13, 5, seed=42)
    first = list(sampler)
    assert first == list(sampler)
    assert sorted(item[0] for batch in first for item in batch) == list(range(13))
    assert [len(b) for b in first] == [5, 5, 3]
    sampler.set_epoch(1)
    assert first != list(sampler)


def test_normalization_and_undefined_metrics(experiment):
    task, root = experiment
    data = PairedArrays(root / "toy", "ir", [0])
    raw = np.load(root / "toy/ir.npy")[0]
    np.testing.assert_allclose(data[0][0].numpy().reshape(-1), normalize(raw))
    np.testing.assert_allclose(normalize(raw), normalize(7 * raw + 3), atol=2e-7)
    y = np.array([[0.0, 1.0, 2.0], [1.0, 1.0, 1.0]])
    metric = per_sample_metrics(y, y)
    assert metric["r2"][0] == 1 and metric["mae"].max() == 0
    assert np.isnan(metric["r2"][1]) and np.isnan(metric["pearson"][1])


def test_published_task_matrix_and_splits():
    root = Path(__file__).resolve().parents[1]
    tasks = json.loads((root / "configs/paper_tasks.json").read_text())
    assert len(tasks) == 120
    keys = {
        (t["dataset"], t["config"]["architecture"], t["source"], t["seed"]) for t in tasks.values()
    }
    assert len(keys) == 120
    for split_file in {t["split_file"] for t in tasks.values()}:
        with np.load(root / split_file) as split:
            for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
                assert not np.intersect1d(split[a], split[b]).size
        expected = next(
            t["data_contract"]["split_sha256"]
            for t in tasks.values()
            if t["split_file"] == split_file
        )
        assert digest(root / split_file) == expected


def test_peak_matching_maximizes_cardinality():
    pytest.importorskip("rdkit")
    from vibratrans.analysis import matches

    # Greedy nearest-first can lose a match; the assignment must recover both.
    assert matches(np.array([0.0, 2.0]), np.array([1.0, 3.0]), tolerance=1).all()
    assert not matches(np.array([0.0]), np.array([2.0]), tolerance=1).any()


def test_ridge_test_targets_do_not_select_alpha():
    pytest.importorskip("rdkit")
    from vibratrans.analysis import ridge_grid

    rng = np.random.default_rng(12)
    x = rng.normal(size=(60, 8))
    y = x[:, :2] + rng.normal(scale=0.1, size=(60, 2))
    split = np.array(["train"] * 30 + ["val"] * 15 + ["test"] * 15)
    p1, a1, grid1 = ridge_grid(x, y, split)
    y[split == "test"] += 1000
    p2, a2, grid2 = ridge_grid(x, y, split)
    np.testing.assert_array_equal(p1, p2)
    assert a1 == a2 and grid1 == grid2
