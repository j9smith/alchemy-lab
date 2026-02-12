import torch
from dataclasses import dataclass
from core.diffusion.coeffs import DiffusionCoefficients


@dataclass(frozen=True)
class DDPMSampleConfig:
    dtype: torch.dtype = torch.float32

def ddpm_sample(
        model: torch.nn.Module,
        coeffs: DiffusionCoefficients,
        shape: tuple[int, int, int, int], # [B, C, H, W]
        device: torch.device,
        cfg: DDPMSampleConfig = DDPMSampleConfig(),
):
    model.eval()
    T = coeffs.betas.shape[0]
    B = shape[0]

    x_T = torch.randn(size=shape, device=device, dtype=cfg.dtype)

    # TODO: implement DDPM sampling logic