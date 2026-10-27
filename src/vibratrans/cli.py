"""Command-line entry points for the current paper protocols."""

import argparse
import copy
import json
from pathlib import Path


def selected_task(config, name):
    path = Path(config).resolve()
    tasks = json.loads(path.read_text())
    if name not in tasks:
        raise ValueError(f"Unknown task {name!r}; use list-tasks.")
    task = copy.deepcopy(tasks[name])
    split = Path(task["split_file"])
    task["split_file"] = str(split if split.is_absolute() else path.parent.parent / split)
    return task


def main():
    parser = argparse.ArgumentParser(description="VibraTrans: one-step IR–Raman translation")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("demo", help="Offline CPU training, inference and evaluation example")
    p.add_argument("--output-dir", type=Path, required=True)
    for command in ("list-tasks", "train", "evaluate", "verify-data"):
        p = sub.add_parser(command)
        p.add_argument("--task-config", default="configs/paper_tasks.json")
        if command == "list-tasks":
            continue
        p.add_argument("--task", required=True)
        p.add_argument("--data-root", type=Path, required=True)
        if command == "verify-data":
            p.add_argument("--full-hash", action="store_true")
            continue
        p.add_argument("--output-dir", type=Path, required=True)
        p.add_argument("--device", default="cuda" if command == "train" else "cpu")
        if command == "train":
            p.add_argument("--workers", type=int, default=2)
        else:
            p.add_argument("--checkpoint", type=Path, required=True)
            p.add_argument("--batch-size", type=int, default=192)
    p = sub.add_parser("predict")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--features-output", type=Path)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=192)
    args = parser.parse_args()
    if args.command == "demo":
        from .demo import run_demo

        run_demo(args.output_dir)
        return
    if args.command == "list-tasks":
        for name in json.loads(Path(args.task_config).read_text()):
            print(name)
        return
    if args.command == "predict":
        from .inference import predict_file

        predict_file(
            args.checkpoint,
            args.input,
            args.output,
            args.device,
            args.batch_size,
            args.features_output,
        )
        return
    task = selected_task(args.task_config, args.task)
    if args.command == "train":
        from .train import train

        train(task, args.data_root, args.output_dir, args.device, args.workers)
    elif args.command == "evaluate":
        from .inference import evaluate

        evaluate(
            args.checkpoint, task, args.data_root, args.output_dir, args.device, args.batch_size
        )
    else:
        import numpy as np
        from .runtime import digest

        contract = task["data_contract"]
        if digest(task["split_file"]) != contract["split_sha256"]:
            raise ValueError("Split checksum differs from the recorded task.")
        with np.load(task["split_file"], allow_pickle=False) as split:
            rows = {k: split[k].copy() for k in split.files}
        for a in ("train", "val", "test"):
            if rows[a].ndim != 1 or not len(rows[a]) or len(np.unique(rows[a])) != len(rows[a]):
                raise ValueError(f"Invalid or repeated rows in {a}.")
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            if np.intersect1d(rows[a], rows[b]).size:
                raise ValueError(f"Overlapping partitions: {a}, {b}")
        shapes = []
        for source in ("ir", "raman"):
            path = args.data_root / task["data_directory"] / f"{source}.npy"
            x = np.load(path, mmap_mode="r", allow_pickle=False)
            if x.ndim != 2 or x.shape[1] != contract["length"]:
                raise ValueError(f"Unexpected shape for {source}: {x.shape}")
            if any(v.min() < 0 or v.max() >= len(x) for v in rows.values() if len(v)):
                raise ValueError("Split index outside the data array.")
            if args.full_hash and digest(path) != contract[f"{source}_sha256"]:
                raise ValueError(f"Array checksum mismatch for {source}.")
            shapes.append(x.shape)
        if shapes[0] != shapes[1]:
            raise ValueError("Paired array shapes differ.")
        print(
            json.dumps(
                dict(
                    task=task["name"],
                    shape=shapes[0],
                    counts={k: len(v) for k, v in rows.items()},
                    full_hash=args.full_hash,
                )
            )
        )


if __name__ == "__main__":
    main()
