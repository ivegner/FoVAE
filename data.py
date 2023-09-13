from typing import Literal
from pytorch_lightning import LightningDataModule
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np


class ImageDataModule(LightningDataModule):
    def __init__(
        self,
        dataset: Literal["mnist", "omniglot", "cifar10", "imagenet"] = "cifar10",
        data_dir: str = "data",
        batch_size: int = 16,
        num_workers: int = 4,
        persistent_workers: bool = True,
        resize: int = None,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        _transforms = [transforms.ToTensor()]

        self.dataset_name = dataset
        self.resize = resize

        if resize:
            _transforms.insert(0, transforms.Resize(resize, antialias=True))
        if dataset == "mnist":
            _transforms.append(transforms.Normalize(0.5, 0.5))
            self.dataset_full = torchvision.datasets.MNIST(
                root=data_dir,
                transform=transforms.Compose(_transforms),
                download=True,
            )
        if dataset == "omniglot":
            _transforms.insert(0, transforms.functional.invert)
            _transforms.append(transforms.Normalize(0.5, 0.5))
            self.dataset_full = torchvision.datasets.Omniglot(
                root=data_dir,
                transform=transforms.Compose(_transforms),
                download=True,
            )
        elif dataset == "cifar10":
            _transforms.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
            self.dataset_full = torchvision.datasets.CIFAR10(
                root=data_dir,
                transform=transforms.Compose(_transforms),
                download=True,
            )
        elif dataset == "imagenet":
            _transforms.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
            self.dataset_train = torchvision.datasets.ImageNet(
                root=data_dir,
                transform=transforms.Compose(_transforms),
                split="train",
            )
            self.dataset_val = torchvision.datasets.ImageNet(
                root=data_dir,
                transform=transforms.Compose(_transforms),
                split="val",
            )

    def setup(self, stage: str) -> None:
        if stage in ("fit", "validate") and not (
            hasattr(self, "dataset_train") and hasattr(self, "dataset_val")
        ):
            n_all = len(self.dataset_full)
            n_train = int(0.8 * n_all)
            n_val = n_all - n_train
            indices = np.arange(n_all)
            np.random.shuffle(indices)

            self.dataset_train = Subset(self.dataset_full, indices=indices[:n_train])
            self.dataset_val = Subset(self.dataset_full, indices=indices[n_train:])

            assert len(self.dataset_train) == n_train
            assert len(self.dataset_val) == n_val

        # if stage == "test":
        #     self.dataset_test = Subset(self.dataset_full, indices=range(64 * 2, 64 * 3))

        # if stage == "predict":
        #     self.dataset_predict = Subset(self.dataset_full, indices=range(64 * 3, 64 * 4))

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.persistent_workers,
        )

    # def test_dataloader(self) -> DataLoader:
    #     return DataLoader(self.random_test)

    # def predict_dataloader(self) -> DataLoader:
    #     return DataLoader(self.random_predict)
