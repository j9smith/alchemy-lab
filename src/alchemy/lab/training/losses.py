import torch
from dataclasses import dataclass
from alchemy.core.diffusion.schedules import make_beta_schedule, BetaScheduleConfig
from alchemy.core.diffusion.coeffs import make_diffusion_coefficients
from alchemy.core.diffusion.objectives import compute_diffusion_loss, LossConfig

@dataclass(frozen=True)
class DiffusionLossWrapperConfig:
    beta_schedule_cfg: BetaScheduleConfig
    objective: str = "eps"

class DiffusionLossFn:
    def __init__(
            self,
            loss_cfg: DiffusionLossWrapperConfig,
            device: torch.device
    ):
        self.betas = make_beta_schedule(cfg=loss_cfg.beta_schedule_cfg, device=device)
        self.coeffs = make_diffusion_coefficients(betas=self.betas)
        self.loss_cfg = LossConfig(objective=loss_cfg.objective)
        self.T = loss_cfg.beta_schedule_cfg.T
        self.device = device

    def __call__(self, model, batch):
        # TODO: Will later enforce batch to be a tuple or dict for conditioning
        # Make sure this is updated to reflect the new structure
        x0 = batch[0].to(self.device, non_blocking=True)

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