from dataclasses import dataclass

import torch
import torch.nn as nn
from diffusers.models import AutoencoderKL

@dataclass(frozen=True)
class PretrainedVAEConfig:
    pretrained_model: str = "stabilityai/sd-vae-ft-mse"
    scaling_factor: float = 0.18215

class PretrainedVAE(nn.Module):
    def __init__(self, cfg: PretrainedVAEConfig):
        super().__init__()
        self.cfg = cfg
        self.vae: AutoencoderKL = AutoencoderKL.from_pretrained(cfg.pretrained_model)
        self.scaling_factor = cfg.scaling_factor

        for p in self.vae.parameters():
            p.requires_grad_(False)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        dist = self.vae.encode(x).latent_dist
        z = dist.sample() * self.scaling_factor
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(z / self.scaling_factor).sample