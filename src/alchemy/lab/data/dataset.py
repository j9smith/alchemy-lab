from dataclasses import dataclass

from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms

@dataclass(frozen=True)
class DatasetConfig:
    path: str

def build_dataset(cfg: DatasetConfig) -> Dataset:
    train_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_ds = datasets.CIFAR10(
        './datasets',
        train=True,
        transform=train_tf,
        download=True
    )

    dog_idxs = [i for i, target in enumerate(train_ds.targets) if target == 5] 
    dog_idxs = dog_idxs[:128]
    dog_ds = Subset(train_ds, dog_idxs)

    return dog_ds