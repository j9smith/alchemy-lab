import torch
import torch.nn as nn

from typing import Literal
from dataclasses import dataclass

UpsampleMode = Literal["convtranspose"]
UPSAMPLE_MODES = ("convtranspose")

@dataclass(frozen=True)
class UpsampleConfig:
    in_channels: int
    out_channels: int
    scale: int = 2
    mode: UpsampleMode = "convtranspose"

    # ConvTranspose options
    conv_bias: bool = False

class Upsample(nn.Module):
    """
    Upsamples spatially by specified scale and maps channels from in_channels -> out_channels.

    Input: (B, C_in, H, W)
    Output: (B, C_out, H*scale, W*scale)
    """
    def __init__(self, cfg: UpsampleConfig):
        super().__init__()
        if cfg.mode not in UPSAMPLE_MODES:
                raise ValueError(
                    f"Unknown upsample mode: {cfg.mode}. "
                    f"Must be one of {UPSAMPLE_MODES}."
                )
        if cfg.in_channels <= 0 or cfg.out_channels <= 0:
            raise ValueError("in_channels/out_channels must be positive.")
        if cfg.scale != 2:
            raise ValueError("Upsample currently only supports scale=2.")

        self.cfg = cfg

        if cfg.mode == "convtranspose":
            if cfg.scale == 2:
                k, s, p, op = 4, 2, 1, 0

            self.up = nn.ConvTranspose2d(
                in_channels=cfg.in_channels,
                out_channels=cfg.out_channels,
                kernel_size=k,
                stride=s,
                padding=p,
                output_padding=op,
                bias=cfg.conv_bias
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg

        if cfg.mode == "convtranspose":
            return self.up(x)