"""A small, offline CPU demonstration, not a paper benchmark."""
import json
import time
from pathlib import Path

import numpy as np
import torch

from .inference import evaluate, predict_file
from .runtime import digest, write
from .train import train


def run_demo(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    data = output / "data/synthetic"
    data.mkdir(parents=True)
    rng = np.random.default_rng(381)
    axis = np.linspace(0, 1, 83)
    centers = rng.uniform(0.1, 0.9, (16, 3, 1))
    heights = rng.uniform(0.2, 1, (16, 3, 1))
    widths = rng.uniform(0.02, 0.07, (16, 3, 1))
    ir = (heights * np.exp(-0.5 * ((axis - centers) / widths) ** 2)).sum(1)
    raman = (heights[:, ::-1] * np.exp(-0.5 * ((axis - centers - 0.02) / widths) ** 2)).sum(1)
    for name, values in [("ir", ir), ("raman", raman)]:
        np.save(data / f"{name}.npy", values.astype(np.float32))
    split = output / "split.npz"
    np.savez(split, train=np.arange(10), val=np.arange(10, 13), test=np.arange(13, 16))
    task = dict(
        name="synthetic_demo", dataset="synthetic", source="ir", seed=0,
        epochs=2, batch_size=5, validation_every=1, data_directory="synthetic",
        split_file=str(split), patch_size=20,
        data_contract=dict(length=83, split_sha256=digest(split)),
        model=dict(width=24, depth=2, heads=6),
        config=dict(architecture="vibradit", steps=1, dropout=0.0, lr=0.0002,
                    terminal="raw", velocity_weight=0.1, shape_weight=0.02,
                    r2_weight=0.005, ema_decay=0.9995, locked_dropout=True),
        selection_reference=dict(mae=0.3, r2=0.1),
    )
    write(output / "task.json", task)
    train(task, output / "data", output / "run", device="cpu", workers=0)
    checkpoint = output / "run/best.pt"
    predict_file(checkpoint, data / "ir.npy", output / "predictions.npy",
                 features_output=output / "features.npy", device="cpu", batch_size=4)
    evaluate(checkpoint, task, output / "data", output / "evaluation", device="cpu")
    predictions, features = np.load(output / "predictions.npy"), np.load(output / "features.npy")
    assert predictions.shape == (16, 83) and features.shape == (16, 24)
    assert np.isfinite(predictions).all() and np.isfinite(features).all()
    assert 0 <= predictions.min() <= predictions.max() <= 1
    report = dict(
        passed=True, scope="Synthetic CPU workflow demonstration; not paper accuracy",
        predictions_shape=list(predictions.shape), features_shape=list(features.shape),
        test_rows=3, optimizer_updates=4, torch_version=str(torch.__version__),
        numpy_version=np.__version__, elapsed_seconds=time.monotonic() - started,
    )
    write(output / "demo_report.json", report)
    print(json.dumps(report, indent=2))
