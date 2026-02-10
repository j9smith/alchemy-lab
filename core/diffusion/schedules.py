import torch
from dataclasses import dataclass
from typing import Literal, Optional

BetaScheduleType = Literal["linear", "cosine"]

@dataclass(frozen=True)
class BetaScheduleConfig:
    type: BetaScheduleType
    T: int
    device: Optional[torch.device] = None
    dtype: torch.dtype = torch.float32

    # Linear params
    beta_start: float = 1e-4
    beta_end: float = 2e-2

def linear_beta_schedule(
        T: int,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32
):
    if T <= 0:
        raise ValueError("T must be positive.")
    if not (0.0 < beta_start < 1.0) or not (0.0 < beta_end < 1.0):
        raise ValueError("beta_start/beta_end must be in (0, 1)")
    if beta_end <= beta_start:
        raise ValueError("beta_end must not be greater than beta_start.")
    
    return torch.linspace(
        start=beta_start,
        end=beta_end,
        steps=T,
        device=device,
        dtype=dtype
    )

def make_beta_schedule(cfg: BetaScheduleConfig) -> torch.Tensor:
    if cfg.type == "linear":
        return linear_beta_schedule(
            T=cfg.T,
            beta_start=cfg.beta_start,
            beta_end=cfg.beta_end,
            device=cfg.device,
            dtype=cfg.dtype
        )
    raise ValueError(f"Unknown schedule type: {cfg.type}")