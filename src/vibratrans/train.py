"""From-scratch training with validation selection over raw and EMA weights."""

import copy
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import PairedArrays, GlobalBatches
from .model import build_model, predict
from .losses import objective, elastic
from .runtime import configure_runtime, digest, write, selection_score


@torch.no_grad()
def validate(model, dataset, mode, cfg, device="cuda", workers=2):
    model.eval()
    sums = torch.zeros(4, dtype=torch.float64, device=device)
    for batch in DataLoader(
        dataset, batch_size=192, num_workers=workers, pin_memory=str(device).startswith("cuda")
    ):
        source, target = (v.to(device, non_blocking=True) for v in batch[:2])
        p = predict(model, source, mode, cfg).flatten(1).double()
        y = target.flatten(1).double()
        error = p - y
        variance = (y - y.mean(1, keepdim=True)).square().sum(1)
        assert (variance > 0).all()
        sums[0] += elastic(p, y).sum()
        sums[1] += error.abs().mean(1).sum()
        sums[2] += (1 - error.square().sum(1) / variance).sum()
        sums[3] += len(source)
    return dict(
        elastic=(sums[0] / sums[3]).item(),
        mae=(sums[1] / sums[3]).item(),
        r2=(sums[2] / sums[3]).item(),
        n=int(sums[3].item()),
    )


def state_digest(model):
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def train(task, data_root, output, device="cuda", workers=2):
    configure_runtime()
    torch.manual_seed(task["seed"])
    np.random.seed(task["seed"])
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu for a small smoke test.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "run.json").exists():
        raise FileExistsError("This output already contains a run; choose a new directory.")
    cfg = task["config"]
    source, mode = task["source"], 2 if task["source"] == "ir" else 0
    splitfile = Path(task["split_file"])
    with np.load(splitfile) as split:
        train_rows, val_rows = split["train"].copy(), split["val"].copy()
    assert not np.intersect1d(train_rows, val_rows).size
    if task.get("data_contract", {}).get("split_sha256"):
        if digest(splitfile) != task["data_contract"]["split_sha256"]:
            raise ValueError("Split hash differs from the selected paper task.")
    directory = Path(data_root) / task["data_directory"]
    train_data = PairedArrays(directory, source, train_rows)
    val_data = PairedArrays(directory, source, val_rows)
    model = build_model(task, val_data.length).to(device)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    initial = validate(model, val_data, mode, cfg, device, workers)
    ref = task["selection_reference"]
    best_score = selection_score(initial, ref)
    record = dict(
        task=task,
        initial_validation=initial,
        initial_score=best_score,
        validation_reference=ref,
        pretrained_checkpoint=None,
        initial_state_sha256=state_digest(model),
        split_sha256=digest(splitfile),
        script_sha256=digest(Path(__file__)),
        architecture_sha256=digest(Path(__file__).with_name("backbone.py")),
        parameters=sum(p.numel() for p in model.parameters()),
        torch_version=torch.__version__,
        gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        created_unix=time.time(),
        test_evaluated=False,
        precision="BF16 training; FP32 validation; TF32 disabled"
        if device.type == "cuda"
        else "FP32 CPU",
        initialization="From scratch: programmatic random/zero initialization only; no model checkpoint loaded.",
    )
    write(output / "run.json", record)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)

    def learning_rate_factor(epoch):
        if epoch < 5:
            return (epoch + 1) / 5
        progress = min(1.0, (epoch - 5) / max(1, task["epochs"] - 5))
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_factor)
    sampler = GlobalBatches(len(train_data), task["batch_size"], seed=task["seed"] + 20261020)
    loader = DataLoader(
        train_data,
        batch_sampler=sampler,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    history, active_seconds, updates = [], 0.0, 0
    best_epoch, best_kind, best_validation = 0, "initial", initial

    def save(name, selected_model, epoch, kind, validation):
        payload = dict(
            model_state_dict=selected_model.state_dict(),
            epoch=epoch,
            weights=kind,
            validation=validation,
            task=task,
            history=history,
            optimizer_state_dict=optimizer.state_dict(),
            scheduler_state_dict=scheduler.state_dict(),
            ema_state_dict=ema.state_dict(),
        )
        torch.save(payload, output / (name + ".tmp"))
        (output / (name + ".tmp")).replace(output / name)

    save("best.pt", model, 0, "initial", initial)
    save("initial.pt", model, 0, "initial", initial)
    print(json.dumps(dict(initial_validation=initial, initial_score=best_score)), flush=True)
    model_params, ema_params = list(model.parameters()), list(ema.parameters())
    for epoch in range(1, task["epochs"] + 1):
        started = time.monotonic()
        sampler.set_epoch(epoch - 1)
        model.train()
        total, seen = 0.0, 0
        for batch_index, batch in enumerate(loader, 1):
            if (output / "STOP").exists():
                raise RuntimeError("Run STOP requested.")
            x, y = (v.to(device, non_blocking=True) for v in batch[:2])
            times = batch[4].to(device, non_blocking=True).float()
            torch.cuda.synchronize(device) if device.type == "cuda" else None
            timer = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            loss = objective(model, x, y, times, mode, cfg)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model_params, 1.0)
            if not torch.isfinite(loss) or not torch.isfinite(norm):
                raise FloatingPointError("Non-finite training loss or gradient")
            optimizer.step()
            with torch.no_grad():
                torch._foreach_mul_(ema_params, cfg["ema_decay"])
                torch._foreach_add_(ema_params, model_params, alpha=1 - cfg["ema_decay"])
                for destination, value in zip(ema.buffers(), model.buffers()):
                    destination.copy_(value)
            torch.cuda.synchronize(device) if device.type == "cuda" else None
            active_seconds += time.monotonic() - timer
            total += loss.item() * len(x)
            seen += len(x)
            updates += 1
            if batch_index % 100 == 0:
                print(
                    json.dumps(
                        dict(epoch=epoch, batch=batch_index, batches=len(loader), loss=loss.item())
                    ),
                    flush=True,
                )
        current = {}
        evaluate_now = (
            epoch == 1 or epoch % task["validation_every"] == 0 or epoch == task["epochs"]
        )
        for kind, candidate in (("raw", model), ("ema", ema)) if evaluate_now else []:
            metrics = validate(candidate, val_data, mode, cfg, device, workers)
            score = selection_score(metrics, ref)
            current[kind] = dict(validation=metrics, score=score)
            if score < best_score:
                best_score, best_epoch, best_kind, best_validation = score, epoch, kind, metrics
                save("best.pt", candidate, epoch, kind, metrics)
        scheduler.step()
        history.append(
            dict(
                epoch=epoch,
                train_loss=total / seen,
                candidates=current,
                best_score=best_score,
                best_epoch=best_epoch,
                best_kind=best_kind,
                active_train_seconds=active_seconds,
                seconds=time.monotonic() - started,
                updates=updates,
                learning_rate=optimizer.param_groups[0]["lr"],
            )
        )
        save("last.pt", model, epoch, "raw", current.get("raw", {}).get("validation"))
        # Preserve sparse snapshots plus atomic best/last and full metric history.
        for kind, candidate in (("raw", model), ("ema", ema)) if epoch % 20 == 0 else []:
            torch.save(
                dict(
                    model_state_dict=candidate.state_dict(),
                    task=task,
                    epoch=epoch,
                    validation=current[kind]["validation"],
                    weights=kind,
                ),
                output / f"{kind}_{epoch:02d}.pt",
            )
        write(output / "history.json", history)
        print(json.dumps(history[-1]), flush=True)
    write(
        output / "complete.json",
        dict(
            **record,
            best_score=best_score,
            best_epoch=best_epoch,
            best_kind=best_kind,
            best_validation=best_validation,
            active_train_seconds=active_seconds,
            updates=updates,
            best_checkpoint_sha256=digest(output / "best.pt"),
        ),
    )
