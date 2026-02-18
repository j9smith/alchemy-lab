from dataclasses import dataclass
from typing import Optional

from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

@dataclass
class TerminalLoggerConfig:
    total_steps: Optional[int] = None
    print_every_n_steps: int = 10
    enabled: bool = True

class TerminalLogger:
    def __init__(self, cfg: TerminalLoggerConfig):
        self.cfg = cfg
        self.enabled = cfg.enabled

        if not self.enabled:
            return
        
        self.latest = {}

        self.console = Console()
        self.progress = Progress(
            TextColumn("Train"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}" if cfg.total_steps else "{task.completed}"),
            TextColumn(" | Elapsed:"),
            TimeElapsedColumn(),
            TextColumn(" | ETA:"),
            TimeRemainingColumn() if cfg.total_steps else TextColumn(""),
            TextColumn("| {task.fields[metrics]}"),
        )
        self.progress.start()

        self.task_id = self.progress.add_task(
            description="",
            total=cfg.total_steps,
            metrics="",
        )

        self._last_step = 0

    def log_scalar(self, name: str, value: float, step: int):
        if not self.enabled:
            return
        
        self.latest[name] = value

        if step % self.cfg.print_every_n_steps != 0:
            return
        
        metrics = " | ".join(f"{k}: {v:.4f}" for k, v in self.latest.items())

        advance = step - self._last_step

        self.progress.update(
            self.task_id,
            advance=advance,
            metrics=metrics
        )

        self._last_step = step

    def close(self):
        if not self.enabled:
            return
        self.progress.stop()
