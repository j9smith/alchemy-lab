from abc import ABC, abstractmethod

class Logger(ABC):
    """
    Base class for loggers.
    """
    @abstractmethod
    def log_scalar(self):
        raise NotImplementedError()
    
class CompositeLogger(Logger):
    def __init__(self):
        super().__init__()
    def log_scalar(self, str, scalar, index):
        print(f"Loss: {scalar}")
        return