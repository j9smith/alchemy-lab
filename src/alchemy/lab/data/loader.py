from dataclasses import dataclass
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
import alchemy.lab.training.distributed as dist

@dataclass(frozen=True)
class LoaderConfig:
    batch_size: int = 128
    num_workers: int = 8
    shuffle: bool = True
    drop_last: bool = True
    pin_memory: bool = True
    persistent_workers: bool = True

def build_dataloader(dataset, cfg:LoaderConfig) -> DataLoader:
    """
    Builds dataloader from config and returns.
    
    :param dataset: Dataset underlying the dataloader.
    :param cfg: Configuration for the dataloader.
    :type cfg: LoaderConfig
    :return: Dataloader loaded with the desired config.
    :rtype: DataLoader
    """
    distributed = dist.get_world_size() > 1
    sampler = DistributedSampler(
        dataset=dataset,
        shuffle=cfg.shuffle
    ) if distributed else None

    shuffle = cfg.shuffle and sampler is None

    return DataLoader(
        dataset=dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=cfg.drop_last
    )