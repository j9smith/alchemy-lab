import torch
import torch.nn as nn

from typing import Optional
from dataclasses import dataclass

@dataclass(frozen=True)
class ResBlock2DConfig:
    in_channels: int
    out_channels: int
    time_embed_dim: int
    norm_groups: int = 32
    conv_bias: bool = False
    dropout: float = 0.0

class ResBlock2D(nn.Module):
    """ 
    Residual block inspired by Denoising Diffusion Probabilistic Models (Ho et al, 2020).

    Input: (B, in_channels, H, W)
    Output: (B, out_channels, H, W)
    """
    def __init__(self, cfg: ResBlock2DConfig):
        super().__init__()
        Cin, Cout = cfg.in_channels, cfg.out_channels

        if Cin <= 0 or Cout <= 0:
            raise ValueError("in_channels/out_channels must be positive.")
        if cfg.norm_groups <= 0:
            raise ValueError("norm_groups must be positive.")
        if Cin % cfg.norm_groups != 0:
            raise ValueError("in_channels must be divisible by norm_groups.")
        if Cout % cfg.norm_groups != 0:
            raise ValueError("out_channels must be divisible by norm_groups.")

        self.norm1 = nn.GroupNorm(
            num_groups=cfg.norm_groups,
            num_channels=Cin,
        )

        self.act1 = nn.SiLU()

        self.conv1 = nn.Conv2d(
            in_channels=Cin,
            out_channels=Cout,
            kernel_size=3,
            padding=1,
            bias=cfg.conv_bias
        )

        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                in_features=cfg.time_embed_dim,
                out_features=Cout
            )
        )

        self.norm2 = nn.GroupNorm(
            num_groups=cfg.norm_groups,
            num_channels=Cout,
        )

        self.act2 = nn.SiLU()

        self.conv2 = nn.Conv2d(
            in_channels=Cout,
            out_channels=Cout,
            kernel_size=3,
            padding=1,
            bias=cfg.conv_bias
        )

        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()
        
        self.skip = (
            nn.Identity() if Cin == Cout else
            nn.Conv2d(Cin, Cout, kernel_size=1)
        )

    def forward(self, x:torch.Tensor, t_embedding: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)

        h = h + self.time_proj(t_embedding)[:, :, None, None]

        h = self.norm2(h)
        h = self.act2(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.skip(x)