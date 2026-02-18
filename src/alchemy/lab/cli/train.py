import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

import alchemy.lab.training.distributed as dist
from alchemy.lab.training.runner import TrainingRunner
from alchemy.lab.loggers.base import CompositeLogger
from alchemy.lab.loggers.terminal import TerminalLogger, TerminalLoggerConfig

@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    print("\n\n"
        " █████╗ ██╗      ██████╗██╗  ██╗███████╗███╗   ███╗██╗   ██╗\n"
        "██╔══██╗██║     ██╔════╝██║  ██║██╔════╝████╗ ████║╚██╗ ██╔╝\n"
        "███████║██║     ██║     ███████║█████╗  ██╔████╔██║ ╚████╔╝ \n"
        "██╔══██║██║     ██║     ██╔══██║██╔══╝  ██║╚██╔╝██║  ╚██╔╝  \n"
        "██║  ██║███████╗╚██████╗██║  ██║███████╗██║ ╚═╝ ██║   ██║   \n"
        "╚═╝  ╚═╝╚══════╝ ╚═════╝╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝   ╚═╝   \n"
        "                            L A B\n\n"

    )
    dist.init_distributed(backend=cfg.dist.backend)
    device = dist.get_device()
    dtype = torch.float32

    model = instantiate(cfg.model).to(device)

    if dist.get_world_size() > 1:
        model = DDP(model, device_ids=[torch.cuda.current_device()]) if torch.cuda.is_available() else DDP(model)

    optimiser = instantiate(cfg.optim, model=model)
    loss_fn = instantiate(cfg.loss, device=device)

    #logger = CompositeLogger()
    logger = TerminalLogger(TerminalLoggerConfig(
        total_steps=cfg.train.max_steps
    ))

    dataset = instantiate(cfg.data.dataset)
    dataloader = instantiate(cfg.data.loader, dataset=dataset)

    checkpoint_manager = instantiate(cfg.checkpoints)

    runner = TrainingRunner(
        model=model,
        optimiser=optimiser,
        loss_fn=loss_fn,
        logger=logger,
        cfg=cfg,
        device=device,
        checkpoint_manager=checkpoint_manager
    )

    runner.train(dataloader=dataloader)

if __name__ == "__main__":
    main()