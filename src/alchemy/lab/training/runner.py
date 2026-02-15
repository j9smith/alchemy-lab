import torch
from torch.utils.data import DistributedSampler
import alchemy.lab.training.distributed as dist
from alchemy.lab.loggers.base import Logger

class TrainingRunner():
    def __init__(
            self,
            model: torch.nn.Module,
            optimiser: torch.optim.Optimizer,
            loss_fn,
            logger: Logger,
            cfg,
            device: torch.device
    ):
        self.model = model
        self.optimiser = optimiser
        self.loss_fn = loss_fn
        self.logger = logger
        self.cfg = cfg
        self.device = device

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

            loss = self._train_step(batch)
            self.step += 1

            if dist.is_main_process() and (self.step % self.cfg.train.log_every_n_steps == 0):
                self.logger.log_scalar("train/loss", loss, self.step)

    def _train_step(self, batch) -> torch.Tensor:
        self.optimiser.zero_grad(set_to_none=True)
        loss = self.loss_fn(self.model, batch)
        loss.backward()
        self.optimiser.step()

        loss_detached = loss.detach()
        mean_loss = dist.reduce_mean(loss_detached)
        return mean_loss
