import torch
import torch.cuda.nvtx as nvtx
import torch.cuda.profiler as profiler
from torch.utils.data import DistributedSampler
from typing import Optional 

import alchemy.lab.training.distributed as dist
from alchemy.core.nn.precision import PRECISION_MAP
from alchemy.lab.loggers.base import Logger
from alchemy.lab.training.checkpoints import CheckpointManager
from alchemy.lab.training.ema import update_ema_model

class TrainingRunner():
    def __init__(
            self,
            model: torch.nn.Module,
            vae: torch.nn.Module,
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
        self.vae = vae
        self.optimiser = optimiser
        self.loss_fn = loss_fn
        self.logger = logger
        self.cfg = cfg
        self.device = device
        self.checkpoint_manager = checkpoint_manager
        self.ema = ema
        self.scheduler = scheduler

        self.step = 0
        self.start_step = 0 # If we're loading from checkpoint
        self.epoch = 0

        self.dtype = PRECISION_MAP[cfg.train.precision]

    def train(self, dataloader, warmup_steps: int = 0, profile_steps: int = 0):
        profiling = profile_steps > 0
        profile_end = warmup_steps + profile_steps

        self.model.train()
        sampler = getattr(dataloader, "sampler", None)

        it = iter(dataloader)

        total_steps = (profile_end + 1) if profiling else (self.cfg.train.max_steps + self.start_step)

        while self.step < total_steps:
            if profiling and self.step == warmup_steps:
                if dist.is_main_process():
                    print(f"Starting capture at step {self.step}.")
                torch.cuda.synchronize()
                profiler.start()

            with nvtx.range(f"iter_{self.step}"):
                try:
                    with nvtx.range("batch_data_fetch"):
                        batch = next(it)
                except StopIteration:
                    self.epoch += 1
                    if isinstance(sampler, DistributedSampler):
                        sampler.set_epoch(self.epoch) # Re-seed the sampler
                    it = iter(dataloader) # Re-create iterator
                    with nvtx.range("batch_data_fetch"):
                        batch = next(it)                    

                loss, metrics = self._train_step(batch)
                # TODO: use metrics for logging
                self.step += 1

                if dist.is_main_process() and self.logger is not None:
                    self.logger.log_scalar("train/loss", loss.item(), self.step)
                    self.logger.log_scalar("lr", self.optimiser.param_groups[0]["lr"], self.step)

                if dist.is_main_process():
                    with nvtx.range("checkpoint"):
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

            if profiling and self.step == profile_end:
                profiler.stop()
                if dist.is_main_process():
                    print(f"Capture complete at step {self.step}.")
                break

        # TODO: Save on final step

        if self.logger is not None:
            self.logger.add_text("Job complete.")
            self.logger.close()

    def _train_step(self, batch) -> tuple[torch.Tensor, dict]:
        self.optimiser.zero_grad(set_to_none=True)

        if self.vae is not None:
            with nvtx.range("vae_encode"):
                with torch.no_grad():
                    images = batch[0].to(self.device, non_blocking=True)
                    latents = self.vae.encode(images)
                    batch = (latents, batch[1])
        
        with nvtx.range("forward"):
            with torch.autocast(device_type="cuda", dtype=self.dtype):
                loss, metrics = self.loss_fn(self.model, batch)

        with nvtx.range("backward"):
            loss.backward()

        with nvtx.range("optimiser_step"):
            self.optimiser.step()

        with nvtx.range("ema_update"):
            if self.ema is not None:
                update_ema_model(self.model, self.ema, decay=self.cfg.train.ema_decay)

        with nvtx.range("reduce_mean"):
            mean_loss = dist.reduce_mean(loss.detach())

        return mean_loss, metrics
