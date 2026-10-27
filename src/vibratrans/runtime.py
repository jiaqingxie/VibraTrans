"""Recorded numerical settings and atomic result writing."""

import hashlib
import json
import uuid
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fast_conv(module, x):
    b, c, length = x.shape
    kernel, dilation, stride, pad = (
        module.kernel_size[0],
        module.dilation[0],
        module.stride[0],
        module.padding[0],
    )
    padded = F.pad(x, (pad, pad))
    n = (length + 2 * pad - dilation * (kernel - 1) - 1) // stride + 1
    windows = torch.stack(
        [padded[:, :, i * dilation : i * dilation + stride * n : stride] for i in range(kernel)],
        dim=-1,
    )
    windows = windows.permute(0, 2, 1, 3).reshape(b * n, c * kernel)
    return (
        F.linear(windows, module.weight.reshape(module.out_channels, c * kernel), module.bias)
        .reshape(b, n, module.out_channels)
        .transpose(1, 2)
    )


def selection_score(metrics, ref):
    return max(metrics["mae"] / ref["mae"], (1 - metrics["r2"]) / max(1e-10, 1 - ref["r2"]))


def configure_runtime():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    # Match the GEMM implementation used for kernel-three convolutions in training.
    if not getattr(nn.Conv1d.forward, "_vibradit_kernel", False):
        original = nn.Conv1d.forward

        def forward(module, x):
            if module.kernel_size == (3,) and module.groups == 1 and module.padding_mode == "zeros":
                return fast_conv(module, x)
            return original(module, x)

        forward._vibradit_kernel = True
        nn.Conv1d.forward = forward
