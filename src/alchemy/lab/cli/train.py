import hydra
from omegaconf import DictConfig

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from alchemy.core.diffusion.schedules import linear_beta_schedule
from alchemy.core.diffusion.coeffs import make_diffusion_coefficients
from alchemy.core.models.unet.unet2d import UNet2D, UNet2DConfig

import alchemy.lab.training.distributed as dist
from alchemy.lab.training.losses import DiffusionLossFn, DiffusionLossWrapperConfig
from alchemy.lab.training.optim import build_optimiser, OptimiserConfig
from alchemy.lab.training.runner import TrainingRunner
from alchemy.lab.data.loader import build_dataloader, LoaderConfig
from alchemy.lab.data.dataset import build_dataset, DatasetConfig
from alchemy.lab.loggers.base import CompositeLogger

@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    dist.init_distributed(backend=cfg.dist.backend)
    device = dist.get_device()

    betas = linear_beta_schedule(
        T=1000,
        beta_start=0.0001,
        beta_end=0.02,
        device=device,
        dtype=torch.float32
    )

    coeffs = make_diffusion_coefficients(betas=betas)

    model = UNet2D(
        UNet2DConfig(
            in_channels=3,
            out_channels=3,
            base_channels=64,
            channel_multipliers=(1, 2, 4, 8),
            attn_levels=(1,),
            use_mid_attn=False,
            attn_num_heads=8,
            num_res_blocks=2,
            time_embed_dim=256,
            norm_groups=32,
            dropout=0.1,
            conv_bias=False
        )
    ).to(device)

    if dist.get_world_size() > 1:
        model = DDP(model, device_ids=[torch.cuda.current_device()]) if torch.cuda.is_available() else DDP(model)

    optimiser = build_optimiser(model, OptimiserConfig(
        name="adamw",
        lr=0.0002
    ))

    loss_fn = DiffusionLossFn(
        coeffs=coeffs,
        cfg=DiffusionLossWrapperConfig(
            objective="eps",
            num_timesteps=betas.shape[0]
        )
    )

    logger = CompositeLogger()

    dataset = build_dataset()
    dataloader = build_dataloader(dataset=dataset, cfg=LoaderConfig(
        batch_size=128,
        num_workers=8,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        persistent_workers=True
    ))

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