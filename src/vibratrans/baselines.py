"""Position-aware Transformer and residual 1D CNN paper baselines."""

import math
import torch
from torch import nn
from torch.nn import functional as F


class PatchTransformer(nn.Module):
    """A non-autoregressive 1D Transformer with ordered spectral patches."""

    def __init__(self, length, width=256, depth=6, heads=4, patch_size=20):
        super().__init__()
        self.length, self.patch_size = length, patch_size
        tokens = math.ceil(length / patch_size)
        self.embed = nn.Conv1d(1, width, patch_size, stride=patch_size)
        self.mode_embed = nn.Embedding(3, width)
        position = torch.arange(tokens).float().unsqueeze(1)
        frequency = torch.exp(torch.arange(0, width, 2).float() * (-math.log(10000.0) / width))
        enc = torch.zeros(tokens, width)
        enc[:, 0::2], enc[:, 1::2] = (
            torch.sin(position * frequency),
            torch.cos(position * frequency),
        )
        self.register_buffer("position", enc, persistent=False)
        layer = nn.TransformerEncoderLayer(
            width,
            heads,
            4 * width,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, patch_size)
        # Independent initialization for every cloned encoder layer.
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for block in self.encoder.layers:
            nn.init.xavier_uniform_(block.self_attn.in_proj_weight)

    def forward(self, source, target_mode):
        shape = source.shape
        seq = source.flatten(2)
        seq = F.pad(seq, (0, (-self.length) % self.patch_size))
        tokens = self.embed(seq).transpose(1, 2)
        mode = torch.full((len(source),), target_mode, device=source.device, dtype=torch.long)
        tokens = tokens + self.position.to(tokens.dtype) + self.mode_embed(mode).unsqueeze(1)
        out = self.head(self.norm(self.encoder(tokens))).flatten(1)[:, : self.length]
        return out.sigmoid().reshape(shape)


class ResidualBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, width),
            nn.SiLU(),
            nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(8, width),
            nn.SiLU(),
            nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation),
        )

    def forward(self, x):
        return x + self.net(x)


class SpectralResNet1D(nn.Module):
    def __init__(self, length, width=384, depth=8, heads=6, patch_size=20):
        super().__init__()
        self.length = length
        self.patch_size = patch_size
        self.embed = nn.Conv1d(1, width, patch_size, stride=patch_size)
        self.mode_embed = nn.Embedding(3, width)
        # Receptive field spans the full molecular patch sequence at depth 8.
        self.encoder = nn.Sequential(*[ResidualBlock(width, 2 ** (i % 6)) for i in range(depth)])
        self.norm = nn.GroupNorm(8, width)
        self.head = nn.Conv1d(width, patch_size, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
        for block in self.encoder:
            nn.init.zeros_(block.net[-1].weight)

    def tokens(self, source, target_mode):
        x = source.reshape(len(source), 1, -1)
        x = F.pad(x, (0, (-self.length) % self.patch_size))
        x = self.embed(x)
        mode = torch.full((len(source),), target_mode, dtype=torch.long, device=source.device)
        return self.encoder(x + self.mode_embed(mode).unsqueeze(-1))

    def forward(self, source, target_mode):
        x = self.head(F.silu(self.norm(self.tokens(source, target_mode))))
        return x.transpose(1, 2).flatten(1)[:, : self.length].sigmoid().reshape(source.shape)

    def features(self, source, target_mode):
        return self.tokens(source, target_mode).mean(-1)
