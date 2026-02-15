import torch
from dataclasses import dataclass
from alchemy.core.diffusion.coeffs import DiffusionCoefficients
from alchemy.core.diffusion.objectives import compute_diffusion_loss, LossConfig

@dataclass(frozen=True)
class DiffusionLossWrapperConfig:
    objective: str = "eps"
    num_timesteps: int = 1000

class DiffusionLossFn:
    def __init__(
            self,
            coeffs: DiffusionCoefficients,
            cfg: DiffusionLossWrapperConfig
    ):
        self.coeffs = coeffs
        self.loss_cfg = LossConfig(objective=cfg.objective)
        self.T = cfg.num_timesteps

    def __call__(self, model, batch):
        # TODO: Will later enforce batch to be a tuple or dict for conditioning
        # Make sure this is updated to reflect the new structure
        device = next(model.parameters()).device
        x0 = batch[0].to(device, non_blocking=True)

        B = x0.shape[0]
        t = torch.randint(
            low=0,
            high=self.T,
            size=(B,),
            device=x0.device,
            dtype=torch.long
        )

        return compute_diffusion_loss(
            model=model,
            x0=x0,
            t=t,
            coeffs=self.coeffs,
            noise=None,
            cfg=self.loss_cfg,
            conditioning=None
        )