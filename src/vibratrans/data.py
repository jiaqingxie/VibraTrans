"""Memory-mapped spectra and deterministic global minibatches for torchrun."""

import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


class PairedArrays(Dataset):
    def __init__(self, directory, source, indices=None, spectral_order="physical"):
        self.directory = Path(directory)
        self.source = np.load(self.directory / f"{source}.npy", mmap_mode="r")
        target = "raman" if source == "ir" else "ir"
        self.target = np.load(self.directory / f"{target}.npy", mmap_mode="r")
        if self.source.shape != self.target.shape or self.source.ndim != 2:
            raise ValueError("Paired spectra must have matching (N,L) shapes")
        self.length = self.source.shape[1]
        self.side = math.isqrt(self.length)
        self.storage_shape = (
            (self.side, self.side) if self.side**2 == self.length else (1, self.length)
        )
        if spectral_order == "legacy" and self.side**2 != self.length:
            raise ValueError("Legacy tiling requires a square length")
        self.indices = (
            np.arange(len(self.source)) if indices is None else np.asarray(indices, dtype=np.int64)
        )
        if (
            self.indices.ndim != 1
            or np.any(self.indices < 0)
            or np.any(self.indices >= len(self.source))
        ):
            raise ValueError("Invalid dataset row indices")
        self.spectral_order = spectral_order

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            index, valid, flow_time, global_count = key
        else:
            index, valid, flow_time, global_count = key, 1.0, 0.5, 1
        row = int(self.indices[index])
        values = []
        extrema = []
        for matrix in (self.source, self.target):
            raw = np.array(matrix[row], dtype=np.float32)
            lo, hi = raw.min(), raw.max()
            if not np.isfinite(raw).all():
                raise ValueError(f"Non-finite spectrum at row {row}")
            norm = (raw - lo) / (hi - lo + 1e-8)
            if self.spectral_order == "legacy":
                patch = max(p for p in [10, 8, 5, 4, 2, 1] if self.side % p == 0)
                count = self.side // patch
                norm = norm.reshape(count, count, patch, patch).transpose(0, 2, 1, 3)
            values.append(torch.from_numpy(norm.reshape(1, *self.storage_shape).copy()))
            extrema.extend((float(lo), float(hi)))
        return (*values, row, float(valid), float(flow_time), int(global_count), *extrema)


class GlobalBatches(Sampler):
    """Identical global samples/times for any world size; zero-weight tail padding.

    Every real example occurs exactly once per epoch. Each rank's loss is
    world_size * sum(valid * per_example_loss) / global_count, compensating
    for DDP's gradient average even for an uneven final minibatch.
    """

    def __init__(self, size, batch_size, rank=0, world_size=1, seed=1):
        if batch_size < world_size or batch_size % world_size:
            raise ValueError("Global batch size must be a positive multiple of world size")
        self.size, self.batch_size = size, batch_size
        self.rank, self.world_size, self.seed = rank, world_size, seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return math.ceil(self.size / self.batch_size)

    def __iter__(self):
        order_gen = torch.Generator().manual_seed(self.seed + 1000003 * self.epoch)
        time_gen = torch.Generator().manual_seed(self.seed + 2000003 * self.epoch + 17)
        order = torch.randperm(self.size, generator=order_gen).tolist()
        for start in range(0, self.size, self.batch_size):
            chunk = order[start : start + self.batch_size]
            count = len(chunk)
            times = torch.rand(count, generator=time_gen).tolist()
            items = [(idx, 1.0, t, count) for idx, t in zip(chunk, times)]
            while len(items) % self.world_size:
                items.append((chunk[0], 0.0, times[0], count))
            yield items[self.rank :: self.world_size]


def resolve_indices(directory, split, split_file=None):
    if split_file:
        with np.load(split_file) as data:
            return data[split].copy()
    return None
