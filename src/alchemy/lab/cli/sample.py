import argparse
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torchvision.utils import make_grid, save_image

from alchemy.lab.training.checkpoints import CheckpointManager
from alchemy.core.diffusion.schedules import make_beta_schedule
from alchemy.core.diffusion.coeffs import make_diffusion_coefficients
from alchemy.core.diffusion.samplers import ddpm_sample

def main():
    """
    Will load config from the specified checkpoint and then sample 
    n samples from the model.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--use_ema", action="store_true")
    p.add_argument("--n", type=int, default=24)
    args = p.parse_args()

    checkpoint = torch.load(Path(args.ckpt), map_location="cpu")
    cfg = OmegaConf.create(checkpoint["run_config"])

    device = torch.device(args.device)

    # TODO: Feels a bit hacky loading both models?
    # Is there another way around this?
    model: torch.nn.Module = instantiate(cfg.model).to(device)
    ema: torch.nn.Module | None = None

    # TODO: Load EMA model
    if args.use_ema and checkpoint["state"].get("ema") is not None:
        ema = instantiate(cfg.model).to(device)

    CheckpointManager.load(
        path=args.ckpt,
        model=model,
        ema=ema,
        optimiser=None,
        scheduler=None,
        strict_model=True,
        load_optimiser=False,
        load_scheduler=False,
        load_ema=args.use_ema,
    )

    denoiser = ema if (args.use_ema and ema is not None) else model
    denoiser.eval()

    beta_cfg = cfg.loss.loss_cfg.beta_schedule_cfg
    betas = make_beta_schedule(cfg=beta_cfg, device=device)
    coeffs = make_diffusion_coefficients(betas=betas)

    img_cfg = cfg.data.image

    samples = ddpm_sample(
        denoiser=denoiser,
        coeffs=coeffs,
        shape=(args.n, img_cfg.channels, img_cfg.resolution, img_cfg.resolution),
        device=device
    )

    grid = make_grid(samples, nrow=6, normalize=True, value_range=(-1, 1))
    out_path = Path("output/samples.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, out_path)

if __name__ == "__main__":
    main()