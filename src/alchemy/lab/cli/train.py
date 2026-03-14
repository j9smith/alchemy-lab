import copy
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

import alchemy.lab.training.distributed as dist
from alchemy.lab.training.runner import TrainingRunner
from alchemy.lab.training.checkpoints import CheckpointManager
from alchemy.lab.loggers.base import build_logger

@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    dist.init_distributed(backend=cfg.dist.backend)

    if dist.is_main_process():
        print("\n\n"
            " █████╗ ██╗      ██████╗██╗  ██╗███████╗███╗   ███╗██╗   ██╗\n"
            "██╔══██╗██║     ██╔════╝██║  ██║██╔════╝████╗ ████║╚██╗ ██╔╝\n"
            "███████║██║     ██║     ███████║█████╗  ██╔████╔██║ ╚████╔╝ \n"
            "██╔══██║██║     ██║     ██╔══██║██╔══╝  ██║╚██╔╝██║  ╚██╔╝  \n"
            "██║  ██║███████╗╚██████╗██║  ██║███████╗██║ ╚═╝ ██║   ██║   \n"
            "╚═╝  ╚═╝╚══════╝ ╚═════╝╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝   ╚═╝   \n"
            "                            L A B\n\n"

        )

    device = dist.get_device()
    logger = build_logger(cfg)

    model = instantiate(cfg.model).to(device)
    ema  = copy.deepcopy(model).eval().to(device)
    for p in ema.parameters():
        p.requires_grad_(False)

    if dist.get_world_size() > 1:
        model = DDP(model, device_ids=[torch.cuda.current_device()]) if torch.cuda.is_available() else DDP(model)

    vae = instantiate(cfg.vae).to(device) if "vae" in cfg else None

    optimiser = instantiate(cfg.optim, model=model)
    loss_fn = instantiate(cfg.loss, device=device)

    dataset = instantiate(cfg.data.dataset)
    dataloader = instantiate(cfg.data.loader, dataset=dataset)

    checkpoint_manager = instantiate(cfg.checkpoints)

    runner = TrainingRunner(
        model=model,
        vae=vae,
        optimiser=optimiser,
        loss_fn=loss_fn,
        logger=logger,
        cfg=cfg,
        ema=ema,
        device=device,
        checkpoint_manager=checkpoint_manager
    )

    if cfg.train.resume is not None:
        logger.add_text(f"Loading checkpoint from {cfg.train.resume}")
        progress = CheckpointManager.load(
            path=cfg.train.resume,
            model=model,
            ema=ema,
            optimiser=optimiser,
            scheduler=None,
            strict_model=True,
            load_optimiser=True,
            load_scheduler=False,
            load_ema=True
        )
        runner.step = progress["global_step"]
        runner.epoch = progress["epoch"]
        runner.start_step = progress["global_step"]
        logger.set_start_step(progress["global_step"])

    dist.barrier()

    runner.train(dataloader=dataloader)

if __name__ == "__main__":
    main()