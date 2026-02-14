import torch
import torch.nn as nn

from typing import Literal
from dataclasses import dataclass

DownsampleMode = Literal["convstride"]
DOWNSAMPLE_MODES = ("convstride",)

@dataclass(frozen=True)
class DownsampleConfig():
    in_channels: int
    out_channels: int
    scale: int = 2
    mode: DownsampleMode = "convstride"

    # Conv options
    kernel_size: int = 3
    conv_bias: bool = False

class Downsample(nn.Module):
    """
    Downsamples spatially by specified scale and maps channels from in_channels -> out_channels.

    Input: (B, C_in, H, W)
    Output: (B, C_out, H/scale, W/scale)
    """
    def __init__(self, cfg: DownsampleConfig):
        super().__init__()
        if cfg.mode not in DOWNSAMPLE_MODES:
            raise ValueError(
                f"Unknown downsample mode: {cfg.mode}. "
                f"Must be one of {DOWNSAMPLE_MODES}."
            )
        if cfg.in_channels <= 0 or cfg.out_channels <= 0:
            raise ValueError("in_channels/out_channels must be positive.")
        
        self.cfg = cfg

        if cfg.mode == "convstride":
            k = cfg.kernel_size
            if k % 2 == 0:
                raise ValueError("kernel_size should be odd for symmetric padding.")
            p = k // 2
            s = 2

            self.down = nn.Conv2d(
                in_channels=cfg.in_channels,
                out_channels=cfg.out_channels,
                kernel_size=k,
                padding=p,
                stride=s,
                bias=cfg.conv_bias
            )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.down(x)