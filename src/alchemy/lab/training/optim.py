import torch
from dataclasses import dataclass
from typing import Literal, Optional

@dataclass(frozen=True)
class OptimiserConfig:
    name: Literal["adamw", "adam"] = "adamw"
    lr: float = 2e-4

@dataclass(frozen=True)
class SchedulerConfig:
    pass

def build_optimiser(model: torch.nn.Module, cfg: OptimiserConfig) -> torch.optim.Optimizer:
    """
    Builds and returns an optimiser from configuration.

    :param model: The model to optimise.
    :type model: torch.nn.Module
    :param cfg: Configruation for the optimiser.
    :type cfg: OptimiserConfig
    :return: Loaded optimiser object.
    :rtype: Optimizer
    """
    if cfg.name == "adamw":
        return torch.optim.AdamW(
            params=model.parameters(),
            lr=cfg.lr
        )
    elif cfg.name == "adam":
        return torch.optim.Adam(
            params=model.parameters(),
            lr=cfg.lr
        )
    else: raise ValueError(f"Unknown or unsupported optimiser: {cfg.name}.")


def build_scheduler(optimiser: torch.optim.Optimizer, cfg: SchedulerConfig) -> torch.optim.lr_scheduler.LRScheduler:
    # TODO: implement scheduler logic
    pass