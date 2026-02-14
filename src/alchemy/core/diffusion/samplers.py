import torch
import torch.nn as nn

from dataclasses import dataclass
from typing import Optional, Literal

from alchemy.core.diffusion.coeffs import DiffusionCoefficients
from alchemy.core.diffusion.parameterisation import make_x0_from_eps, make_x0_from_v

PredictionMode = Literal["x0", "eps", "v"]
PREDICTION_MODE = ("x0", "eps", "v")

# TODO: Add more sampling modes (DDIM etc.)

@dataclass(frozen=True)
class DDPMSampleConfig:
    prediction_mode: PredictionMode = "eps"
    dtype: torch.dtype = torch.float32

@torch.no_grad()
def ddpm_sample(
        denoiser: torch.nn.Module,
        coeffs: DiffusionCoefficients,
        shape: tuple[int, int, int, int], # [B, C, H, W]
        device: torch.device,
        decoder: Optional[nn.Module] = None,
        cfg: DDPMSampleConfig = DDPMSampleConfig(),
) -> torch.Tensor:
    """
    Samples images from the denoiser via ancestral sampling according to the
    algoritmh proposed in DDPM (Ho et al, 2020).
    
    :param denoiser: A pre-trained denoiser model.
    :type denoiser: torch.nn.Module
    :param coeffs: Diffusion coefficients.
    :type coeffs: DiffusionCoefficients
    :param shape: Desired output shape (n_samples, C, H, W)
    :type shape: tuple[int, int, int, int]
    :param device: The sampling device.
    :type device: torch.device
    :param decoder: Decoder module to project latents.
    :type decoder: Optional[nn.Module]
    :param cfg: Sampling config.
    :type cfg: DDPMSampleConfig
    :return: Sampled images.
    :rtype: Tensor
    """
    if cfg.prediction_mode not in PREDICTION_MODE:
        raise ValueError(f"Unknown prediction mode: {cfg.prediction_mode}")
    
    denoiser.eval()
    T = coeffs.betas.shape[0]
    B = shape[0]

    xt = torch.randn(size=shape, device=device, dtype=cfg.dtype)

    for t in reversed(range(T)):
        t_batch = torch.full((B,), t, device=device, dtype=torch.long)
        c1 = coeffs.posterior_mean_coef1[t]
        c2 = coeffs.posterior_mean_coef2[t]

        eps_theta = denoiser(xt, t_batch)

        # TODO: add different prediction modes
        if cfg.prediction_mode == "eps":
            x0_pred = make_x0_from_eps(
                xt=xt,
                eps=eps_theta,
                t=t_batch,
                coeffs=coeffs
            )
        else: raise ValueError(f"Unsupported prediction mode: {cfg.prediction_mode}.")

        x0_pred = x0_pred.clamp(-1.0, 1.0)
        
        mu = c1 * x0_pred + c2 * xt

        if t == 0:
            xt = mu
        else:
            sigma = torch.sqrt(coeffs.posterior_variance[t])
            z = torch.randn_like(xt)
            xt = mu + (sigma * z)
    return xt