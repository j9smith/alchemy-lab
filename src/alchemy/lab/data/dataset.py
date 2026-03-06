from dataclasses import dataclass

from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms
from datasets import load_dataset
from PIL import Image


@dataclass(frozen=True)
class DatasetConfig:
    path: str = "mattymchen/celeba-hq"
    resolution: int = 256

# TODO: Abstract into generic 'Huggingface' dataset interface
class CelebAHQDataset(Dataset):
    def __init__(self, hf_dataset, transform):
        self.data = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        img = img.convert("RGB")
        return self.transform(img), 0


def build_dataset(cfg: DatasetConfig) -> Dataset:
    train_tf = transforms.Compose([
        transforms.Resize(cfg.resolution),
        transforms.CenterCrop(cfg.resolution),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    hf_ds = load_dataset(cfg.path, split="train")

    return CelebAHQDataset(hf_ds, train_tf)

# def build_dataset(cfg: DatasetConfig) -> Dataset:
#     train_tf = transforms.Compose([
#         transforms.RandomHorizontalFlip(p=0.5),
#         transforms.RandomCrop(32, padding=4),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ])

#     train_ds = datasets.CIFAR10(
#         './datasets',
#         train=True,
#         transform=train_tf,
#         download=True
#     )

#     dog_idxs = [i for i, target in enumerate(train_ds.targets) if target == 5] 
#     #dog_idxs = dog_idxs[:128]
#     dog_ds = Subset(train_ds, dog_idxs)

#     return dog_ds