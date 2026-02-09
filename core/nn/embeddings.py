import torch

def sinusoidal_embedding(
        timesteps: torch.Tensor, 
        embedding_dim: int, 
        max_period: float = 10000.0
    ) -> torch.Tensor:
    """
    Creates sinusoidal timestep embeddings. 

    Args:
        timesteps: Tensor of shape (B,) or (B, 1)
        dim: Embedding dimension (must be even).
    
    Returns:
        Tensor of shape (B, embedding_dim).
    """
    if embedding_dim % 2 != 0:
        raise ValueError("Embedding dimension must be even.")

    if timesteps.ndim == 2 and timesteps.shape[1] == 1:
        timesteps = timesteps.squeeze(1)
    if timesteps.ndim != 1:
        raise ValueError("timesteps must have shape (B,) or (B,1)")

    device = timesteps.device
    dtype = timesteps.dtype if timesteps.is_floating_point() else torch.float32

    timesteps = timesteps.to(dtype)

    half_dim = embedding_dim // 2

    freqs = torch.exp(
        -torch.log(max_period)
        * torch.arange(half_dim, device=device, dtype=dtype)
        / (embedding_dim // 2)
    )

    args = timesteps[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return emb