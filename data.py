from typing import Literal
from pytorch_lightning import LightningDataModule
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np


class ImageDataModule(LightningDataModule):
    def __init__(
        self,
        dataset: Literal["mnist", "cifar10", "imagenet"] = "cifar10",
        batch_size: int = 16,
        n_workers: int = 4,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.n_workers = n_workers
        if dataset == "mnist":
            self.dataset_full = torchvision.datasets.MNIST(
                root="data",
                transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]
                ),
                download=True,
            )
        elif dataset == "cifar10":
            self.dataset_full = torchvision.datasets.CIFAR10(
                root="data",
                transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
                ),
            )
        elif dataset == "imagenet":
            self.dataset_train = torchvision.datasets.ImageNet(
                root="data",
                transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
                ),
                split="train",
            )
            self.dataset_val = torchvision.datasets.ImageNet(
                root="data",
                transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
                ),
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
            num_workers=self.n_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    # def test_dataloader(self) -> DataLoader:
    #     return DataLoader(self.random_test)

    # def predict_dataloader(self) -> DataLoader:
    #     return DataLoader(self.random_predict)
