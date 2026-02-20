import torch
from torch.utils.data import DistributedSampler
from typing import Optional 

import alchemy.lab.training.distributed as dist
from alchemy.lab.loggers.base import Logger
from alchemy.lab.training.checkpoints import CheckpointManager
from alchemy.lab.training.ema import update_ema_model

class TrainingRunner():
    def __init__(
            self,
            model: torch.nn.Module,
            optimiser: torch.optim.Optimizer,
            loss_fn,
            logger: Logger,
            cfg,
            device: torch.device,
            checkpoint_manager: CheckpointManager,
            ema: Optional[torch.nn.Module] = None,
            scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
    ):
        self.model = model
        self.optimiser = optimiser
        self.loss_fn = loss_fn
        self.logger = logger
        self.cfg = cfg
        self.device = device
        self.checkpoint_manager = checkpoint_manager
        self.ema = ema
        self.scheduler = scheduler

        self.step = 0
        self.epoch = 0

    def train(self, dataloader):
        self.model.train()
        sampler = getattr(dataloader, "sampler", None)

        it = iter(dataloader)

        while self.step < self.cfg.train.max_steps:
            try:
                batch = next(it)
            except StopIteration:
                self.epoch += 1
                if isinstance(sampler, DistributedSampler):
                    sampler.set_epoch(self.epoch) # Re-seed the sampler
                it = iter(dataloader) # Re-create iterator
                batch = next(it)

            loss, metrics = self._train_step(batch)
            # TODO: use metrics for logging
            self.step += 1

            if dist.is_main_process():
                self.logger.log_scalar("train/loss", loss.item(), self.step)
                self.logger.log_scalar("lr", self.optimiser.param_groups[0]["lr"], self.step)

                if self.checkpoint_manager is not None:
                    self.checkpoint_manager.save(
                        model=self.model,
                        global_step=self.step,
                        epoch=self.epoch,
                        ema=self.ema,
                        optimiser=self.optimiser,
                        scheduler=self.scheduler,
                        logging=None, # TODO: add logging identifiers in checkpointing
                        run_config=self.cfg
                    )

        # TODO: Save on final step
        self.logger.close()

    def _train_step(self, batch) -> tuple[torch.Tensor, dict]:
        self.optimiser.zero_grad(set_to_none=True)
        loss, metrics = self.loss_fn(self.model, batch)
        loss.backward()
        self.optimiser.step()

        if self.ema is not None:
            update_ema_model(self.model, self.ema, decay=self.cfg.train.ema_decay)

        mean_loss = dist.reduce_mean(loss.detach())
        return mean_loss, metrics
