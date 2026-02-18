from typing import Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path
from omegaconf import OmegaConf
import torch
import time
import os

@dataclass(frozen=True)
class CheckpointManagerConfig:
    save_every_n_steps: int
    path: str
    prefix: str

class CheckpointManager():
    def __init__(self, cfg: CheckpointManagerConfig):
        self.save_every_n_steps = cfg.save_every_n_steps
        self.path = Path(cfg.path).expanduser()
        self.prefix = cfg.prefix

    def save(
            self,
            model: torch.nn.Module,
            global_step: int,
            epoch: int,
            ema: Optional[torch.nn.Module] = None,
            optimiser: Optional[torch.optim.Optimizer] = None,
            scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
            logging: Optional[Dict[str, Any]] = None,
            run_config: Optional[Dict[str, Any]] = None
    ):
        # TODO: Include logging checkpoints so we can resume loggers
        if global_step % self.save_every_n_steps != 0:
            return
        
        if run_config is not None:
            run_config = OmegaConf.to_container(run_config, resolve=True)

        checkpoint = {
            "time": time.time(),
            "progress": {
                "global_step": global_step,
                "epoch": epoch
            },
            "state": {
                "model": model.state_dict(),
                "ema": ema.state_dict() if ema is not None else None,
                "optimiser": optimiser.state_dict() if optimiser is not None else None,
                "scheduler": scheduler.state_dict() if scheduler is not None else None
            },
            "logging": {},
            "run_config": run_config
        }

        path_name = self.path / f"{self.prefix}_step{global_step:08d}.pt"
        path_name.parent.mkdir(parents=True, exist_ok=True)

        tmp = path_name.with_suffix(path_name.suffix + ".tmp")

        torch.save(checkpoint, tmp)
        os.replace(tmp, path_name)

    @staticmethod
    def load(
            path,
            model: torch.nn.Module,
            ema: Optional[torch.nn.Module],
            optimiser: Optional[torch.optim.Optimizer],
            scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
            strict_model: bool = True,
            load_optimiser: bool = True,
            load_scheduler: bool = True,
            load_ema: bool = True
    ):
        path = Path(path)
        checkpoint = torch.load(path, map_location="cpu")

        state = checkpoint["state"]

        model.load_state_dict(state["model"], strict=strict_model)

        if load_optimiser and optimiser is not None:
            if state.get("optimiser") is None:
                raise RuntimeError("Checkpoint has no optimiser state.")
            else: optimiser.load_state_dict(state["optimiser"])
        
        if load_ema and ema is not None:
            if state.get("ema") is None:
                raise RuntimeError("Checkpoint has no EMA state.")
            else: ema.load_state_dict(state["ema"], strict=True)

        if load_scheduler and scheduler is not None:
            if state.get("scheduler") is None:
                raise RuntimeError("Checkpoint has no scheduler state.")
            else: scheduler.load_state_dict(state["scheduler"])