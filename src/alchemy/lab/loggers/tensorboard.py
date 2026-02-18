from pathlib import Path
from alchemy.lab.loggers.base import Logger
from torch.utils.tensorboard import SummaryWriter
from dataclasses import dataclass

@dataclass(frozen=True)
class TensorBoardLoggerConfig:
    log_dir: str = "runs"
    experiment_name: str = "default"
    enabled: bool = True
    flush_secs: int = 20

class TensorBoardLogger(Logger):
    def __init__(self, cfg: TensorBoardLoggerConfig):
        self.cfg = cfg

        if not cfg.enabled:
            return
        
        self.enabled = True
        
        log_dir = Path(cfg.log_dir).expanduser() / cfg.experiment_name
        log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir, flush_secs=cfg.flush_secs)

    def log_scalar(self, name: str, value: float, step: int) -> None:
        if not self.enabled: 
            return
        self.writer.add_scalar(name, value, step)

    def close(self):
        if not self.enabled:
            return
        self.writer.close()