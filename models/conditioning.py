import torch
import torch.nn as nn
import numpy as np
from einops import repeat


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size, max_period):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period

    def timestep_embedding(self, t, dim):
        # Standard sinusoidal embedding
        half = dim // 2
        freqs = torch.exp(
            -np.log(self.max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """Standard DiT positional embedding logic."""
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])

    # Helper logic to create sin-cos waves based on coordinates
    half_dim = embed_dim // 2
    emb_h = get_1d_sincos_pos_embed_from_grid(half_dim, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(half_dim, grid[1])
    return torch.from_numpy(np.concatenate([emb_h, emb_w], axis=1)).float().unsqueeze(0)


def get_1d_sincos_pos_embed_from_grid(dim, grid):
    freqs = np.exp(-np.log(10000) * np.arange(dim // 2) / (dim // 2))
    args = grid.flatten()[:, None] * freqs[None, :]
    return np.concatenate([np.cos(args), np.sin(args)], axis=-1)


class SinCosPosEmbed2D(nn.Module):
    def __init__(self, hidden_size, grid_size):
        super().__init__()
        # Logic from previous response moved here
        pos_embed = get_2d_sincos_pos_embed(hidden_size, grid_size)
        self.register_buffer("pos_embed", pos_embed)

    def forward(self, x):
        # x shape: (B, N, D)
        # Explicitly expand to match batch size for stability
        return self.pos_embed.expand(x.shape[0], -1, -1)


# conditioning.py - Update SinCosPosEmbed3D
class SinCosPosEmbed3D(nn.Module):
    def __init__(self, hidden_size, grid_size, max_frames=512):
        super().__init__()
        self.grid_size = grid_size
        # Spatial 2D
        self.register_buffer("pos_embed_spatial", get_2d_sincos_pos_embed(hidden_size, grid_size))
        # Temporal 1D
        self.register_buffer("pos_embed_temporal",
                             get_1d_sincos_pos_embed_from_grid(hidden_size, np.arange(max_frames)))

    def forward(self, x):
        num_spatial_patches = self.grid_size ** 2
        f = x.shape[1] // num_spatial_patches

        # spatial: [1, HW, D] -> [1, 1, HW, D]
        spatial = self.pos_embed_spatial.unsqueeze(1)

        # temporal: [F, D] -> [1, F, 1, D]
        temporal = self.pos_embed_temporal[:f, :].unsqueeze(0).unsqueeze(2)

        # Resulting broadcasted shape: [1, F, HW, D]
        combined = spatial + temporal
        return combined.view(1, -1, x.shape[-1])
