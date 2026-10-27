"""The shared endpoint objective and auxiliary interpolated-state supervision."""

import torch
from torch.nn import functional as F
from .model import predict


def derivatives(x):
    return F.pad(x[:, 1:] - x[:, :-1], (1, 0)), F.pad(x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2], (1, 1))


def importance(target):
    gradient, curvature = derivatives(target)

    def norm(x):
        return x.abs() / (x.abs().amax(1, keepdim=True) + 1e-8)

    return 1 + norm(target) + 0.5 * norm(gradient) + 0.5 * norm(curvature)


def elastic(pred, target, weights=None):
    error = pred - target
    values = 0.6 * error.abs() + 0.4 * error.square()
    return (values if weights is None else values * weights).flatten(1).mean(1)


def objective(model, source, target, times, mode, cfg):
    pred = predict(model, source, mode, cfg, training=True)
    p, y = pred.flatten(1), target.float().flatten(1)
    total = elastic(p, y)
    if cfg["shape_weight"]:
        dp, d2p = derivatives(p)
        dy, d2y = derivatives(y)
        total += cfg["shape_weight"] * ((dp - dy).abs().mean(1) + 0.5 * (d2p - d2y).abs().mean(1))
    if cfg["r2_weight"]:
        total += (
            cfg["r2_weight"] * (p - y).square().mean(1) / y.var(1, unbiased=False).clamp_min(1e-6)
        )
    if cfg["velocity_weight"]:
        state = (1 - times[:, None, None, None]) * source + times[:, None, None, None] * target
        with torch.autocast(source.device.type, dtype=torch.bfloat16, enabled=source.is_cuda):
            velocity = model.velocity_field(state, times, condition=source, target_mode=mode)
        total += cfg["velocity_weight"] * elastic(
            velocity.float().flatten(1), (target - source).float().flatten(1), importance(y)
        )
    return total.mean()
