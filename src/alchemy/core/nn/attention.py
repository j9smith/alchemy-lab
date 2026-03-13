import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Literal
from dataclasses import dataclass

AttentionBackend = Literal["alchemy", "torch"]
ATTENTION_BACKENDS = ("alchemy", "torch")

@dataclass(frozen=True)
class SelfAttention2DConfig:
    channels: int
    backend: AttentionBackend = "torch"
    num_heads: int = 8
    head_dim: Optional[int] = None
    norm_groups: int = 32
    qkv_bias: bool = True
    out_bias: bool = True
    dropout: float = 0.0

class SelfAttention2D(nn.Module):
    """
    Spatial multi-head self-attention across H*W tokens. 

    Input: (B, C, H, W)
    Output: (B, C, H, W)
    """
    def __init__(self, cfg: SelfAttention2DConfig):
        super().__init__()
        if cfg.num_heads <= 0:
            raise ValueError("num_heads must be positive.")
        if cfg.channels <= 0:
            raise ValueError("channels must be positive.")
        if cfg.channels % cfg.num_heads != 0:
            raise ValueError("channels must be divisible by num_heads.")
        if cfg.backend not in ATTENTION_BACKENDS:
            raise ValueError(f"Attn backend must be one of {ATTENTION_BACKENDS}.")
        
        self.cfg = cfg
        self.C = cfg.channels
        self.num_heads = cfg.num_heads
        
        if cfg.head_dim is None:
            self.head_dim = self.C // self.num_heads
        else: self.head_dim = cfg.head_dim

        self.norm = nn.GroupNorm(
            num_groups=cfg.norm_groups,
            num_channels=self.C
        )

        self.qkv = nn.Conv1d(
            in_channels=self.C,
            out_channels=self.C * 3,
            kernel_size=1,
            bias=cfg.qkv_bias
        )

        self.proj = nn.Conv1d(
            in_channels=self.C,
            out_channels=self.C,
            kernel_size=1,
            bias=cfg.out_bias
        )

        self.drop = nn.Dropout(cfg.dropout) if cfg.dropout > 0  else nn.Identity()

        self._attn_fn = self._alchemy_attn if cfg.backend == "alchemy" else self._torch_attn

    def _alchemy_attn(self, q, k, v) -> torch.Tensor:
        """
        Manual scaled dot-product attention.
        """
        # q, k, v each (B, heads, HW, head_dim)
        attn = q @ k.transpose(-2, -1) 
        attn = attn * (self.head_dim ** -0.5)
        attn = torch.softmax(attn, dim=-1)
        attn = self.drop(attn)
        out = attn @ v # (B, heads, HW, head_dim)

        return out

    def _torch_attn(self, q, k , v) -> torch.Tensor:
        """
        Use Torch's SDPA to leverage performant kernels.

        SDPA expects and returns (B, heads, HW, head_dim) shaped input.
        """
        out = F.scaled_dot_product_attention(
        query=q,
        key=k,
        value=v,
        dropout_p=self.cfg.dropout if self.training else 0.0
        )
        return out

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        HW = H*W

        x_flat = self.norm(x).reshape(B, C, HW)

        qkv = self.qkv(x_flat)

        q, k, v = torch.chunk(
            input=qkv,
            chunks=3,
            dim=1
        )

        q = q.view(B, self.num_heads, self.head_dim, HW).transpose(-2, -1).contiguous()
        k = k.view(B, self.num_heads, self.head_dim, HW).transpose(-2, -1).contiguous()
        v = v.view(B, self.num_heads, self.head_dim, HW).transpose(-2, -1).contiguous()

        out = self._attn_fn(q, k, v) # (B, heads, HW, head_dim)
        
        out = out.transpose(-1, -2) # (B, heads, head_dim, HW)
        out = out.reshape(B, C, HW) 

        out = self.proj(out) # Mix channels
        out = out.reshape(B, C, H, W)

        return x + out

# import torch
# import torch.nn as nn

# from typing import Optional
# from dataclasses import dataclass

# @dataclass(frozen=True)
# class SelfAttention2DConfig:
#     channels: int
#     num_heads: int = 8
#     head_dim: Optional[int] = None
#     norm_groups: int = 32
#     qkv_bias: bool = True
#     out_bias: bool = True
#     dropout: float = 0.0

# class SelfAttention2D(nn.Module):
#     """
#     Spatial multi-head self-attention across H*W tokens. 

#     Input: (B, C, H, W)
#     Output: (B, C, H, W)
#     """
#     def __init__(self, cfg: SelfAttention2DConfig):
#         super().__init__()
#         if cfg.num_heads <= 0:
#             raise ValueError("num_heads must be positive.")
#         if cfg.channels <= 0:
#             raise ValueError("channels must be positive.")
#         if cfg.channels % cfg.num_heads != 0:
#             raise ValueError("channels must be divisible by num_heads.")
        
#         self.cfg = cfg
#         self.C = cfg.channels
#         self.num_heads = cfg.num_heads
        
#         if cfg.head_dim is None:
#             self.head_dim = self.C // self.num_heads
#         else: self.head_dim = cfg.head_dim

#         self.norm = nn.GroupNorm(
#             num_groups=cfg.norm_groups,
#             num_channels=self.C
#         )

#         self.qkv = nn.Conv1d(
#             in_channels=self.C,
#             out_channels=self.C * 3,
#             kernel_size=1,
#             bias=cfg.qkv_bias
#         )

#         self.proj = nn.Conv1d(
#             in_channels=self.C,
#             out_channels=self.C,
#             kernel_size=1,
#             bias=cfg.out_bias
#         )

#         self.drop = nn.Dropout(cfg.dropout) if cfg.dropout > 0  else nn.Identity()

#     def forward(self, x:torch.Tensor) -> torch.Tensor:
#         B, C, H, W = x.shape
#         HW = H*W

#         x_flat = self.norm(x).reshape(B, C, HW)

#         qkv = self.qkv(x_flat)

#         q, k, v = torch.chunk(
#             input=qkv,
#             chunks=3,
#             dim=1
#         )

#         q = q.view(B, self.num_heads, self.head_dim, HW)
#         k = k.view(B, self.num_heads, self.head_dim, HW)
#         v = v.view(B, self.num_heads, self.head_dim, HW)

#         attn = q.transpose(-2, -1) @ k # (B, heads, HW, HW)
#         attn = attn * (self.head_dim ** -0.5)
#         attn = torch.softmax(attn, dim=-1)
#         attn = self.drop(attn)

#         out = attn @ v.transpose(-2, -1) # (B, heads, HW, head_dim)
#         out = out.transpose(-1, -2) # (B, heads, head_dim, HW)
#         out = out.reshape(B, C, HW) 

#         out = self.proj(out) # Mix channels
#         out = out.reshape(B, C, H, W)

#         return x + out