import torch

from dataclasses import dataclass
from typing import Literal, Optional

from alchemy.core.diffusion.coeffs import DiffusionCoefficients

ObjectiveType = Literal["eps", "x0", "v"]

@dataclass(frozen=True)
class LossConfig:
    objective: ObjectiveType = "eps"

def _generate_xt(
        x0: torch.Tensor, 
        t: torch.Tensor, 
        eps: torch.Tensor, 
        coeffs: DiffusionCoefficients
) -> torch.Tensor:
    """
    Generates and returns x_t from x_0 and t according to:
    x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * eps
    
    :param x0: The original image tensor.
    :type x0: torch.Tensor
    :param t: Un-embedded timesteps.
    :type t: torch.Tensor
    :param eps: Noise tensor, if it has already been generated.
    :type eps: torch.Tensor
    :param coeffs: Pre-compute diffusion coefficients object.
    :type coeffs: DiffusionCoefficients
    :return: x_t, noised x0 according to t.
    :rtype: Tensor
    """
    B = x0.shape[0]
    shape = (B,) + (1, ) * (x0.ndim - 1) # Handles cases where ndim != 4 (e.g., high-dim latents)

    sqrt_alpha_bars = coeffs.sqrt_alpha_bars[t].view(shape)
    sqrt_one_minus_alpha_bars = coeffs.sqrt_one_minus_alpha_bars[t].view(shape)

    return sqrt_alpha_bars * x0 + sqrt_one_minus_alpha_bars * eps

def _pointwise_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Returns the point-wise loss.
    
    :param pred: Model predictions.
    :type pred: torch.Tensor
    :param target: Desired target.
    :type target: torch.Tensor
    :return: Returns the MSE loss between prediction and target.
    :rtype: Tensor
    """
    return (pred - target) ** 2

def compute_diffusion_loss(
        model,
        x0: torch.Tensor,
        t: torch.Tensor,
        coeffs: DiffusionCoefficients,
        noise: Optional[torch.Tensor] = None,
        conditioning=None,
        cfg: LossConfig = LossConfig()
):
    """
    Computes and returns the mean loss across the batch.
    
    :param model: The diffusion model.
    :param x0: The original clean datapoints.
    :type x0: torch.Tensor
    :param t: Timesteps.
    :type t: torch.Tensor
    :param coeffs: Pre-computed diffusion coefficients.
    :type coeffs: DiffusionCoefficients
    :param noise: Optional pre-computed noise tensor.
    :type noise: Optional[torch.Tensor]
    :param cfg: Loss configuration object. 
    :type cfg: LossConfig
    """
    B = x0.shape[0]

    if noise is None:
        noise = torch.randn_like(x0)

    x_t = _generate_xt(x0, t, noise, coeffs)
    pred = model(x_t, t, conditioning=None)

    # TODO: Support v-prediction
    if cfg.objective == "eps":
        target = noise
    elif cfg.objective == "x0":
        target = x0
    else: raise ValueError(f"Unknown or unsupported objective: {cfg.objective}.")

    point_loss = _pointwise_loss(pred=pred, target=target)
    mean_example_loss = point_loss.view(B, -1).mean(dim=1) # Average pointwise loss across individual example
    mean_batch_loss = mean_example_loss.mean() # Average loss across batch

    return mean_batch_loss