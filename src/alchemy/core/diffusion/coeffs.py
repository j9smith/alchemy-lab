import torch
from dataclasses import dataclass

@dataclass(frozen=True)
class DiffusionCoefficients:
    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_bars: torch.Tensor
    alpha_bars_prev: torch.Tensor
    sqrt_alpha_bars: torch.Tensor
    sqrt_one_minus_alpha_bars: torch.Tensor
    posterior_mean_coef1: torch.Tensor
    posterior_mean_coef2: torch.Tensor
    posterior_variance: torch.Tensor

def make_diffusion_coefficients(betas: torch.Tensor) -> DiffusionCoefficients:
    """
    Pre-computes all commonly used coefficients for diffusion training and sampling.
    
    :param betas: Beta schedule.
    :type betas: torch.Tensor
    :return: Coefficients for diffusion.
    :rtype: DiffusionCoefficients
    """
    if betas.ndim != 1:
        raise ValueError("betas should have shape (T, ).")
    
    device, dtype = betas.device, betas.dtype

    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0) 
    alpha_bars_prev = torch.cat(
        [torch.ones(1, device=device, dtype=dtype), alpha_bars[:-1]], 
        dim=0
    )
    sqrt_alpha_bars = torch.sqrt(alpha_bars)
    sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)

    # Equation 7 from DDPM (Ho et al, 2020) - x0 formulation
    posterior_mean_coef1 = (betas * torch.sqrt(alpha_bars_prev)) / (1.0 - alpha_bars)
    posterior_mean_coef2 = (torch.sqrt(alphas) * (1.0 - alpha_bars_prev)) / (1.0 - alpha_bars)
    posterior_variance = ((1.0 - alpha_bars_prev) / (1.0 - alpha_bars)) * betas

    return DiffusionCoefficients(
        betas=betas,
        alphas=alphas,
        alpha_bars=alpha_bars,
        alpha_bars_prev=alpha_bars_prev,
        sqrt_alpha_bars=sqrt_alpha_bars,
        sqrt_one_minus_alpha_bars=sqrt_one_minus_alpha_bars,
        posterior_mean_coef1=posterior_mean_coef1,
        posterior_mean_coef2=posterior_mean_coef2,
        posterior_variance=posterior_variance
    )

