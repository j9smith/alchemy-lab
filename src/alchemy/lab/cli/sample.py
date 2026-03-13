import argparse
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torchvision.utils import make_grid, save_image

from alchemy.lab.training.checkpoints import CheckpointManager
from alchemy.core.nn.precision import PRECISION_MAP
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
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--precision", type=str, default=None)
    args = p.parse_args()

    checkpoint = torch.load(Path(args.ckpt), map_location="cpu")
    cfg = OmegaConf.create(checkpoint["run_config"])
    img_cfg = cfg.data.image

    device = torch.device(args.device)

    if args.precision is not None:
        cfg.train.precision = args.precision
        
    dtype = PRECISION_MAP[cfg.train.precision]

    # TODO: Feels a bit hacky loading both models?
    # Is there another way around this?
    model: torch.nn.Module = instantiate(cfg.model).to(device)
    ema: torch.nn.Module | None = None

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

    vae = None
    if "vae" in cfg:
        vae = instantiate(cfg.vae).to(device)
        vae.eval()

    if vae is not None:
        with torch.no_grad():
            # Discover the latent shape for noise init
            # TODO: Can we make this a property of the VAE class rather than dummy pass?
            dummy = torch.zeros(
                1, 
                img_cfg.channels, 
                img_cfg.resolution, 
                img_cfg.resolution,
                device=device
            )
            dummy_latent = vae.encode(dummy)
        _, C, H, W = dummy_latent.shape
        shape = (args.n, C, H, W)
    else:
        shape = (args.n, img_cfg.channels, img_cfg.resolution, img_cfg.resolution)

    beta_cfg = cfg.loss.loss_cfg.beta_schedule_cfg
    betas = make_beta_schedule(cfg=beta_cfg, device=device)
    coeffs = make_diffusion_coefficients(betas=betas)

    with torch.autocast(device_type=device.type, dtype=dtype):
        samples = ddpm_sample(
            denoiser=denoiser,
            coeffs=coeffs,
            shape=shape,
            device=device,
            dtype=dtype,
            decoder=vae.decode if vae is not None else None
        )

    grid = make_grid(samples, nrow=6, normalize=True, value_range=(-1, 1))
    out_path = Path("output/samples.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, out_path)

if __name__ == "__main__":
    main()