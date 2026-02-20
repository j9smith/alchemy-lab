import torch

@torch.no_grad()
def update_ema_model(model: torch.nn.Module, ema_model: torch.nn.Module, decay: float):
    for params, ema_params in zip(model.parameters(), ema_model.parameters()):
        ema_params.mul_(decay).add_(params, alpha=1.0 - decay)

    for buffer, ema_buffers in zip(model.buffers(), ema_model.buffers()):
        ema_buffers.copy_(buffer)