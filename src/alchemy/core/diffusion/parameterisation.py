import torch

def _extract(a: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    B = x.shape[0]
    shape = (B,) + (1,) * (x.ndim - 1)

    return a[t].view(shape)

def make_x0_from_eps(
        xt: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor,
        coeffs
) -> torch.Tensor:
    sqrt_alpha_bars = _extract(coeffs.sqrt_alpha_bars, t, xt)
    sqrt_one_minus_alpha_bars = _extract(coeffs.sqrt_one_minus_alpha_bars, t, xt)

    x0 = (xt - sqrt_one_minus_alpha_bars * eps) / sqrt_alpha_bars
    return x0

def make_x0_from_v(x_t, v, t, coeffs):
    # TODO: implement x_0 from v logic
    pass