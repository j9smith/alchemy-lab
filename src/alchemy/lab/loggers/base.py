from abc import ABC, abstractmethod
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

class Logger(ABC):
    """
    Base class for loggers.
    """
    @abstractmethod
    def log_scalar(self):
        raise NotImplementedError()
    
    @abstractmethod
    def close(self):
        raise NotImplementedError
    
class CompositeLogger(Logger):
    def __init__(self):
        super().__init__()
        self.loggers = []

    def add(self, logger: Logger):
        self.loggers.append(logger)
        
    def log_scalar(self, name: str, value: float, step: int):
        for logger in self.loggers:
            logger.log_scalar(name, value, step)

    def close(self):
        for logger in self.loggers:
            logger.close()

def build_logger(cfg: DictConfig) -> CompositeLogger:
    composite = CompositeLogger()
    loggers_loaded = []

    for name, logger_cfg in cfg.logging.items():
        if logger_cfg is None:
            continue
        enabled = bool(logger_cfg.cfg.enabled)

        if not enabled:
            continue

        logger = instantiate(logger_cfg)
        composite.add(logger)
        loggers_loaded.append(name)

    return composite
    