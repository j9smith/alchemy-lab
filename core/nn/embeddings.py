import torch
import torch.nn as nn
import math

from dataclasses import dataclass

def sinusoidal_embedding(
        timesteps: torch.Tensor, 
        embedding_dim: int, 
        max_period: float = 10000.0
    ) -> torch.Tensor:
    """
    Creates sinusoidal timestep embeddings. 

    Args:
        timesteps: Tensor of shape (B,) or (B, 1)
        dim: Embedding dimension (must be even).
    
    Returns:
        Tensor of shape (B, embedding_dim).
    """
    if embedding_dim % 2 != 0:
        raise ValueError("Embedding dimension must be even.")

    if timesteps.ndim == 2 and timesteps.shape[1] == 1:
        timesteps = timesteps.squeeze(1)
    if timesteps.ndim != 1:
        raise ValueError("timesteps must have shape (B,) or (B,1)")

    device = timesteps.device
    dtype = timesteps.dtype if timesteps.is_floating_point() else torch.float32

    timesteps = timesteps.to(dtype)

    half_dim = embedding_dim // 2

    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half_dim, device=device, dtype=dtype)
        / (embedding_dim // 2)
    )

    args = timesteps[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return emb

@dataclass(frozen=True)
class TimeEmbeddingConfig:
    base_dim: int # Sinusoidal embeddings dim
    time_dim: int # Output dim after MLP
    hidden_multiplier: int = 4 # MLP hidden dim multiplier over base
    max_period: float = 10000.0

class TimeEmbedding(nn.Module):
    """
    Timestep embedding module.

    Input: timesteps (B,) or (B, 1)
    Output: (B, time_dim)
    """
    def __init__(self, cfg: TimeEmbeddingConfig):
        if cfg.base_dim <= 0 or cfg.time_dim <= 0:
            raise ValueError("base_dim/time_dim must be positive.")
        if cfg.base_dim % 2 != 0:
            raise ValueError("base_dim must be even for sinusoidal embeddings.")
        if cfg.hidden_multiplier <= 0:
            raise ValueError("hidden_multiplier must be positive.")
        
        super().__init__()
        
        act = nn.SiLU()
        hidden_dims = cfg.base_dim * cfg.hidden_multiplier

        self.cfg = cfg

        self.mlp = nn.Sequential(
            nn.Linear(cfg.base_dim, hidden_dims),
            act,
            nn.Linear(hidden_dims, cfg.time_dim)
        )

    def forward(self, timesteps:torch.Tensor) -> torch.Tensor:
        embs = sinusoidal_embedding(
            timesteps=timesteps,
            embedding_dim=self.cfg.base_dim,
            max_period=self.cfg.max_period
        )
        return self.mlp(embs)