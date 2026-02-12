import torch
import torch.nn as nn
from dataclasses import dataclass

from core.nn.attention import SelfAttention2D, SelfAttention2DConfig
from core.nn.downsample import Downsample, DownsampleConfig
from core.nn.upsample import Upsample, UpsampleConfig
from core.nn.embeddings import TimeEmbedding, TimeEmbeddingConfig
from core.nn.residual import ResBlock2D, ResBlock2DConfig

@dataclass(frozen=True)
class UNet2DConfig:
    in_channels: int = 3
    out_channels: int = 3
    base_channels: int = 64
    channel_multipliers: tuple[int, ...] = (1, 2, 4, 8)
    attn_levels: tuple[int, ...] = (1,)
    use_mid_attn: bool = False
    attn_num_heads: int = 8
    num_res_blocks: int = 2
    time_embed_dim: int = 256
    norm_groups: int = 32
    dropout: float = 0.0
    conv_bias: bool = False

class UNet2D(nn.Module):
    """
    Composes a denoising UNet.
    """
    def __init__(self, cfg: UNet2DConfig):
        super().__init__()
        self.cfg = cfg


        self.time_embed = TimeEmbedding(TimeEmbeddingConfig(
            base_dim=256,
            time_dim=cfg.time_embed_dim,
            hidden_multiplier=4
        ))

        self.in_conv = nn.Conv2d(
            in_channels=cfg.in_channels,
            out_channels=cfg.base_channels,
            kernel_size=3,
            padding=1
        )

        skip_channels: list[int] = []

        # Build down path
        self.down = nn.ModuleList()
        self.downsample = nn.ModuleList()

        current_channels = cfg.base_channels

        for level, multiplier in enumerate(cfg.channel_multipliers):
            out_channels = cfg.base_channels * multiplier

            blocks = nn.ModuleList()

            for _ in range(cfg.num_res_blocks):
                blocks.append(
                    ResBlock2D(ResBlock2DConfig(
                        in_channels=current_channels,
                        out_channels=out_channels,
                        time_embed_dim=cfg.time_embed_dim,
                        norm_groups=cfg.norm_groups,
                        conv_bias=cfg.conv_bias,
                        dropout=cfg.dropout,
                    ))
                )
                current_channels = out_channels
                skip_channels.append(current_channels)

            if level in cfg.attn_levels:
                blocks.append(
                    SelfAttention2D(SelfAttention2DConfig(
                        channels=current_channels,
                        num_heads=cfg.attn_num_heads,
                        norm_groups=cfg.norm_groups,
                        dropout=cfg.dropout
                    ))
                )

            self.down.append(blocks)

            if level != len(cfg.channel_multipliers) - 1:
                self.downsample.append(
                        Downsample(DownsampleConfig(
                        in_channels=current_channels,
                        out_channels=current_channels,
                        scale=2,
                        mode="convstride",
                        kernel_size=3,
                        conv_bias=cfg.conv_bias,
                    ))
                )
            else: self.downsample.append(nn.Identity())

        # Bottleneck
        mid_resblock_cfg = ResBlock2DConfig(
            in_channels=current_channels,
            out_channels=current_channels,
            time_embed_dim=cfg.time_embed_dim,
            norm_groups=cfg.norm_groups,
            conv_bias=cfg.conv_bias,
            dropout=cfg.dropout
        )

        mid_attn_cfg = SelfAttention2DConfig(
            channels=current_channels,
            num_heads=cfg.attn_num_heads,
            norm_groups=cfg.norm_groups,
            dropout=cfg.dropout
        )

        self.mid1 = ResBlock2D(mid_resblock_cfg)
        self.mid_attn = SelfAttention2D(mid_attn_cfg) if cfg.use_mid_attn else nn.Identity()
        self.mid2 = ResBlock2D(mid_resblock_cfg)

        # Build up path
        self.up = nn.ModuleList()
        self.upsample = nn.ModuleList()

        for level, multiplier in reversed(list(enumerate(cfg.channel_multipliers))):
            out_channels = cfg.base_channels * multiplier

            blocks = nn.ModuleList()
            for _ in range(cfg.num_res_blocks):
                skip_ch = skip_channels.pop()
                blocks.append(
                    ResBlock2D(ResBlock2DConfig(
                        in_channels=current_channels + skip_ch,
                        out_channels=out_channels,
                        time_embed_dim=cfg.time_embed_dim,
                        norm_groups=cfg.norm_groups,
                        conv_bias=cfg.conv_bias,
                        dropout=cfg.dropout
                    ))
                )
                current_channels = out_channels

            if level in cfg.attn_levels:
                blocks.append(
                    SelfAttention2D(SelfAttention2DConfig(
                        channels=current_channels,
                        num_heads=cfg.attn_num_heads,
                        norm_groups=cfg.norm_groups,
                        dropout=cfg.dropout
                    ))
                )

            self.up.append(blocks)

            if level != 0:
                self.upsample.append(
                    Upsample(UpsampleConfig(
                        in_channels=current_channels,
                        out_channels=current_channels,
                        scale=2,
                        mode="convtranspose",
                        conv_bias=cfg.conv_bias
                    ))
                )
            else: self.upsample.append(nn.Identity())

        assert len(skip_channels) ==  0, "Not all skip channels consumed"

        self.out_norm = nn.GroupNorm(
            num_groups=cfg.norm_groups,
            num_channels=current_channels
        )

        self.out_conv = nn.Conv2d(
            in_channels=current_channels,
            out_channels=cfg.out_channels,
            kernel_size=3,
            padding=1
        )

        self.out_act = nn.SiLU()

    def forward(self, x:torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)
        h = self.in_conv(x)

        skips: list[torch.Tensor] = []

        for blocks, downsample in zip(self.down, self.downsample):
            for mod in blocks:
                if isinstance(mod, ResBlock2D):
                    h = mod(h, t_emb)
                    skips.append(h)
                else: h = mod(h)
            h = downsample(h)

        h = self.mid1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid2(h, t_emb)

        for blocks, upsample in zip(self.up, self.upsample):
            for mod in blocks:
                if isinstance(mod, ResBlock2D):
                    skip = skips.pop()
                    h = torch.cat([h, skip], dim=1)
                    h = mod(h, t_emb)
                else: h = mod(h)
            h = upsample(h)

        h = self.out_norm(h)
        h = self.out_act(h)
        h = self.out_conv(h)

        return h