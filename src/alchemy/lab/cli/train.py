import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

import alchemy.lab.training.distributed as dist
from alchemy.lab.training.runner import TrainingRunner
from alchemy.lab.loggers.base import CompositeLogger

@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    dist.init_distributed(backend=cfg.dist.backend)
    device = dist.get_device()
    dtype = torch.float32

    model = instantiate(cfg.model).to(device)

    if dist.get_world_size() > 1:
        model = DDP(model, device_ids=[torch.cuda.current_device()]) if torch.cuda.is_available() else DDP(model)

    optimiser = instantiate(cfg.optim, model=model)
    loss_fn = instantiate(cfg.loss, device=device)

    logger = CompositeLogger()

    #dataset = build_dataset()
    dataset = instantiate(cfg.data.dataset)
    dataloader = instantiate(cfg.data.loader, dataset=dataset)

    runner = TrainingRunner(
        model=model,
        optimiser=optimiser,
        loss_fn=loss_fn,
        logger=logger,
        cfg=cfg,
        device=device
    )

    runner.train(dataloader=dataloader)

if __name__ == "__main__":
    main()