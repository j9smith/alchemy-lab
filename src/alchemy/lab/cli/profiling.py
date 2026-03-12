"""
nsys profile \
  --output ~/reports/alchemy_$(date +%Y%m%d_%H%M%S) \
  --trace cuda,nvtx,osrt,cublas,cudnn \
  --capture-range cudaProfilerApi \
  --capture-range-end stop \
  --stats true \
  --gpu-metrics-devices all \
  uv run python profiling.py
"""

import copy
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

import alchemy.lab.training.distributed as dist
from alchemy.lab.training.runner import TrainingRunner
from alchemy.lab.loggers.base import build_logger

WARMUP_STEPS = 5
PROFILE_STEPS = 20

@hydra.main(version_base=None, config_path="../configs", config_name="profile")
def main(cfg: DictConfig):
    print("\n\n"
        " █████╗ ██╗      ██████╗██╗  ██╗███████╗███╗   ███╗██╗   ██╗\n"
        "██╔══██╗██║     ██╔════╝██║  ██║██╔════╝████╗ ████║╚██╗ ██╔╝\n"
        "███████║██║     ██║     ███████║█████╗  ██╔████╔██║ ╚████╔╝ \n"
        "██╔══██║██║     ██║     ██╔══██║██╔══╝  ██║╚██╔╝██║  ╚██╔╝  \n"
        "██║  ██║███████╗╚██████╗██║  ██║███████╗██║ ╚═╝ ██║   ██║   \n"
        "╚═╝  ╚═╝╚══════╝ ╚═════╝╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝   ╚═╝   \n"
        "                            L A B                           \n"
        "                           Profiler                         \n"

    )
    dist.init_distributed(backend=cfg.dist.backend)
    device = dist.get_device()
    dtype = torch.float32

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

    runner = TrainingRunner(
        model=model,
        vae=vae,
        optimiser=optimiser,
        loss_fn=loss_fn,
        logger=None,
        cfg=cfg,
        ema=ema,
        device=device,
        checkpoint_manager=None
    )

    runner.train(
        dataloader=dataloader,
        warmup_steps=WARMUP_STEPS,
        profile_steps=PROFILE_STEPS
    )

if __name__ == "__main__":
    main()