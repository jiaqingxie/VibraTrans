"""One-evaluation prediction with checkpoint-compatible parameter names."""

import torch
from torch import nn

from .backbone import VibraTransVelocityFieldNetwork
from .baselines import PatchTransformer, SpectralResNet1D


class VibraTrans(nn.Module):
    def __init__(self, length, width=384, depth=8, heads=6, patch_size=20):
        super().__init__()
        self.velocity_field = VibraTransVelocityFieldNetwork(
            in_channels=1,
            num_modes=3,
            image_size=(1, length),
            patch_size=patch_size,
            hidden_size=width,
            depth=depth,
            num_heads=heads,
        )

    def forward(self, source, target_mode=2):
        time = torch.zeros(len(source), device=source.device)
        velocity = self.velocity_field(source, time, condition=source, target_mode=target_mode)
        return (source.float() + velocity.float().clamp(-2, 2)).clamp(0, 1)


def build_model(task, length):
    if task.get("checkpoint") is not None:
        raise ValueError("Training tasks must initialize from scratch.")
    cfg = task["config"]
    if cfg.get("steps", 1) != 1:
        raise ValueError("This release exposes the current one-step formulation.")
    expected = task.get("data_contract", {}).get("length")
    if expected is not None and length != expected:
        raise ValueError(f"Expected {expected} spectral bins, received {length}.")
    constructors = {"vibradit": VibraTrans, "pe": PatchTransformer, "cnn": SpectralResNet1D}
    if cfg["architecture"] not in constructors:
        raise ValueError(f"Unknown architecture: {cfg['architecture']}")
    dims = dict(width=384, depth=8, heads=6, patch_size=task.get("patch_size", 20))
    dims.update(task.get("model", {}))
    model = constructors[cfg["architecture"]](length, **dims)
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = cfg["dropout"]
        if isinstance(module, nn.MultiheadAttention):
            module.dropout = cfg["dropout"]
    return model


def predict(model, source, mode, cfg, training=False):
    with torch.autocast(
        source.device.type, dtype=torch.bfloat16, enabled=training and source.is_cuda
    ):
        if cfg["architecture"] in ("pe", "cnn"):
            return model(source, target_mode=mode).float()
        time = torch.zeros(len(source), device=source.device)
        velocity = model.velocity_field(source.float(), time, condition=source, target_mode=mode)
    state = source.float() + velocity.float().clamp(-2, 2)
    # Endpoint supervision precedes the output clipping in the paper recipe.
    return state if training else state.clamp(0, 1)


def feature_module(model, architecture):
    if architecture == "vibradit":
        return model.velocity_field.blocks[-1]
    if architecture == "pe":
        return model.encoder
    raise ValueError(
        "The paper's frozen-token feature export supports VibraTrans and PE Transformer."
    )
