"""Ordered one-dimensional adaptive Transformer used by the paper checkpoints."""

import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class SinusoidalPositionalEmbedding(nn.Module):
    """时间步嵌入"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = time[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


def get_1d_sincos_pos_embed(num_patches: int, embed_dim: int) -> torch.Tensor:
    """Fixed 1D sin/cos position embedding, returned as (1, N, D)."""
    pos = torch.arange(num_patches, dtype=torch.float32)
    half = embed_dim // 2
    omega = torch.arange(half, dtype=torch.float32) / max(half, 1)
    omega = 1.0 / (10000**omega)
    out = torch.einsum("n,d->nd", pos, omega)
    emb = torch.cat([torch.sin(out), torch.cos(out)], dim=1)
    if embed_dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb.unsqueeze(0)


class SourceEncoder1D(nn.Module):
    """Global source-spectrum encoder for VibraTrans conditioning."""

    def __init__(self, hidden_size: int):
        super().__init__()
        mid = max(hidden_size // 2, 64)
        self.net = nn.Sequential(
            nn.Conv1d(1, mid, kernel_size=7, padding=3),
            nn.SiLU(),
            nn.Conv1d(mid, hidden_size, kernel_size=7, padding=3),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x_source: torch.Tensor) -> torch.Tensor:
        h = self.net(x_source.unsqueeze(1))
        return self.proj(h)


class PatchEmbed1D(nn.Module):
    """Patchify a flattened 1D spectrum with Conv1d projection."""

    def __init__(self, seq_len: int, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.num_patches = math.ceil(seq_len / patch_size)
        self.padded_len = self.num_patches * patch_size
        self.pad_len = self.padded_len - seq_len
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad_len > 0:
            x = F.pad(x, (0, self.pad_len), mode="constant", value=0.0)
        x = self.proj(x)
        return x.transpose(1, 2)


def modulate_tokens(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale[:, None, :]) + shift[:, None, :]


class VibraTransBlock(nn.Module):
    """1D DiT block with adaLN-Zero conditioning."""

    def __init__(
        self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_size),
            nn.Dropout(dropout),
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            c
        ).chunk(6, dim=1)
        h = modulate_tokens(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate_msa[:, None, :] * attn_out
        h = modulate_tokens(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp[:, None, :] * self.mlp(h)
        return x


class VibraTransFinalLayer(nn.Module):
    """Final 1D DiT layer: tokens to velocity patches."""

    def __init__(self, hidden_size: int, patch_size: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
        self.linear = nn.Linear(hidden_size, patch_size)
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate_tokens(self.norm_final(x), shift, scale)
        return self.linear(x)


class VibraTransVelocityFieldNetwork(nn.Module):
    """
    Ordered one-dimensional spectral DiT with adaptive source conditioning.

    Inputs retain the checkpoint-compatible (B, 1, H, W) storage interface.
    All spectral operations flatten H and W in physical order and use 1D
    patches; this storage shape does not introduce 2D spectral convolutions.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_modes: int = 3,
        image_size: tuple = (60, 60),
        patch_size: int = 16,
        hidden_size: int = 384,
        depth: int = 8,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if in_channels != 1:
            raise ValueError("VibraTrans backbone currently expects in_channels=1.")

        self.in_channels = in_channels
        self.image_size = image_size
        self.seq_len = int(image_size[0] * image_size[1])
        self.patch_size = patch_size

        self.x_embedder = PatchEmbed1D(self.seq_len, patch_size, in_chans=2, embed_dim=hidden_size)
        self.num_patches = self.x_embedder.num_patches
        self.register_buffer(
            "pos_embed",
            get_1d_sincos_pos_embed(self.num_patches, hidden_size),
            persistent=False,
        )

        self.time_embed = nn.Sequential(
            SinusoidalPositionalEmbedding(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.mode_embed = nn.Embedding(num_modes, hidden_size)
        self.source_encoder = SourceEncoder1D(hidden_size)
        self.cond_fuse = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, hidden_size))

        self.blocks = nn.ModuleList(
            [
                VibraTransBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
                for _ in range(depth)
            ]
        )
        self.final_layer = VibraTransFinalLayer(hidden_size, patch_size)

    def _target_mode_tensor(self, x: torch.Tensor, target_mode) -> torch.Tensor:
        if target_mode is None:
            return torch.zeros((x.size(0),), dtype=torch.long, device=x.device)
        if isinstance(target_mode, int):
            return torch.full((x.size(0),), target_mode, dtype=torch.long, device=x.device)
        if isinstance(target_mode, torch.Tensor):
            return target_mode.long().to(x.device)
        return torch.tensor(target_mode, dtype=torch.long, device=x.device)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, condition=None, target_mode=None
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("VibraTrans backbone expects x with shape (B, C, H, W).")
        if condition is None:
            condition = x

        batch_size = x.size(0)
        original_shape = x.shape
        x_tau = x.reshape(batch_size, -1)
        x_source = condition.reshape(batch_size, -1)
        target_mode_tensor = self._target_mode_tensor(x, target_mode)

        x_in = torch.stack([x_tau, x_source], dim=1)
        h = self.x_embedder(x_in) + self.pos_embed.to(dtype=x.dtype, device=x.device)
        c = self.time_embed(t) + self.mode_embed(target_mode_tensor) + self.source_encoder(x_source)
        c = self.cond_fuse(c)

        for block in self.blocks:
            h = block(h, c)

        patches = self.final_layer(h, c)
        velocity = patches.reshape(batch_size, -1)[:, : self.seq_len]
        return velocity.reshape(original_shape)
