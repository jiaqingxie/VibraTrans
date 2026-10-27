"""Batched prediction and benchmark evaluation from selected checkpoints."""

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import PairedArrays
from .metrics import per_sample_metrics
from .model import build_model, predict, feature_module
from .runtime import configure_runtime, digest, write


def load_checkpoint(path, length, device="cpu"):
    configure_runtime()
    # State dictionaries and plain metadata only; no arbitrary pickle objects.
    saved = torch.load(path, map_location="cpu", weights_only=True)
    task = saved["task"]
    model = build_model(task, length).to(device).eval()
    model.load_state_dict(saved["model_state_dict"], strict=True)
    return model, task


def normalize(values):
    x = np.asarray(values, dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError("Input spectra must be finite.")
    lo = x.min(axis=-1, keepdims=True)
    return (x - lo) / (x.max(axis=-1, keepdims=True) - lo + 1e-8)


def new_array(path, shape):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(path, mode="w+", dtype="float32", shape=shape)


@torch.no_grad()
def predict_file(
    checkpoint, source_file, output, device="cpu", batch_size=192, features_output=None
):
    if batch_size < 1:
        raise ValueError("Batch size must be positive.")
    source = np.load(source_file, mmap_mode="r", allow_pickle=False)
    if source.ndim != 2 or min(source.shape) < 1:
        raise ValueError("Input must be a nonempty (spectra, ordered spectral bins) .npy array.")
    model, task = load_checkpoint(checkpoint, source.shape[1], device)
    mode = 2 if task["source"] == "ir" else 0
    cfg = task["config"]
    captured = []
    handle = None
    features = None
    if features_output:
        module = feature_module(model, cfg["architecture"])
        handle = module.register_forward_hook(lambda m, args, value: captured.append(value.mean(1)))
    prediction = new_array(output, source.shape)
    try:
        for start in range(0, len(source), batch_size):
            x = torch.from_numpy(normalize(source[start : start + batch_size]).copy()).to(device)
            x = x[:, None, None, :]
            captured.clear()
            y = predict(model, x, mode, cfg).flatten(1).cpu().numpy()
            if not np.isfinite(y).all():
                raise FloatingPointError("Model produced non-finite predictions.")
            prediction[start : start + len(y)] = y
            if features_output:
                if len(captured) != 1:
                    raise RuntimeError("Expected one feature capture per network evaluation.")
                h = captured[0].cpu().numpy()
                if features is None:
                    features = new_array(features_output, (len(source), h.shape[1]))
                features[start : start + len(y)] = h
    finally:
        if handle is not None:
            handle.remove()
    prediction.flush()
    if features is not None:
        features.flush()
    write(
        Path(output).with_suffix(".json"),
        dict(
            source=task["source"],
            target="raman" if task["source"] == "ir" else "ir",
            architecture=cfg["architecture"],
            rows=len(source),
            bins=source.shape[1],
            normalization="Independent min-max; predictions clipped to [0, 1]",
            network_evaluations_per_spectrum=1,
            checkpoint_sha256=digest(checkpoint),
            features=Path(features_output).name if features_output else None,
        ),
    )


@torch.no_grad()
def evaluate(checkpoint, task, data_root, output, device="cpu", batch_size=192):
    output = Path(output)
    with np.load(task["split_file"], allow_pickle=False) as parts:
        rows = parts["test"].copy()
        for other in ("train", "val"):
            if np.intersect1d(rows, parts[other]).size:
                raise ValueError("Test rows overlap with training or validation.")
    if digest(task["split_file"]) != task["data_contract"]["split_sha256"]:
        raise ValueError("Split hash does not match the selected task.")
    data = PairedArrays(Path(data_root) / task["data_directory"], task["source"], rows)
    model, saved_task = load_checkpoint(checkpoint, data.length, device)
    for field in ("source", "seed"):
        if saved_task[field] != task[field]:
            raise ValueError(f"Checkpoint {field} differs from requested task.")
    if saved_task["config"]["architecture"] != task["config"]["architecture"]:
        raise ValueError("Checkpoint architecture differs from requested task.")
    if saved_task.get("dataset", "qm9s") != task.get("dataset", "qm9s"):
        raise ValueError("Checkpoint belongs to a different dataset protocol.")
    saved_hash = saved_task.get("data_contract", {}).get("split_sha256")
    if saved_hash and saved_hash != task["data_contract"]["split_sha256"]:
        raise ValueError("Checkpoint belongs to a different split.")
    prediction = new_array(output / "predictions.npy", (len(data), data.length))
    mode = 2 if task["source"] == "ir" else 0
    values, cursor = [], 0
    for batch in DataLoader(data, batch_size=batch_size, num_workers=0):
        x, y = batch[:2]
        p = predict(model, x.to(device), mode, task["config"]).flatten(1).cpu().numpy()
        if not np.isfinite(p).all():
            raise FloatingPointError("Model produced non-finite predictions.")
        prediction[cursor : cursor + len(p)] = p
        metric = per_sample_metrics(p, y.flatten(1).numpy())
        metric["elastic"] = 0.6 * metric["mae"] + 0.4 * metric["rmse"] ** 2
        values.append(metric)
        cursor += len(p)
    prediction.flush()
    np.save(output / "row_indices.npy", rows)
    columns = {k: np.concatenate([v[k] for v in values]) for k in values[0]}
    with (output / "per_sample_metrics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["row", *columns])
        writer.writerows(zip(rows, *columns.values()))

    def finite_mean(x):
        finite = x[np.isfinite(x)]
        return float(finite.mean()) if len(finite) else None

    write(
        output / "summary.json",
        dict(
            task=task["name"],
            n=len(data),
            checkpoint_sha256=digest(checkpoint),
            metrics={k: finite_mean(v) for k, v in columns.items()},
            nonfinite_counts={k: int((~np.isfinite(v)).sum()) for k, v in columns.items()},
            aggregation="Arithmetic mean of defined per-spectrum metrics",
        ),
    )
