import os
import torch
import torch.distributed as dist

def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()

def get_device() -> torch.device:
    return "cuda" if torch.cuda.is_available() else "cpu"

def get_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0 

def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1

def is_main_process() -> bool:
    return get_rank() == 0

def barrier() -> None:
    if is_distributed():
        dist.barrier()

def init_distributed(backend: str = 'nccl'):
    if not dist.is_available() and not dist.is_initialized():
        return

    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return
    
    dist.init_process_group(backend=backend)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

def reduce_mean(x: torch.Tensor) -> torch.Tensor:
    if not is_distributed():
        return x
    y = x.clone()
    dist.all_reduce(y, op=dist.ReduceOp.AVG)
    return y