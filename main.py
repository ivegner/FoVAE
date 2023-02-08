# main.py
from pytorch_lightning.cli import LightningCLI
import pytorch_lightning as ptl

from models.foveated_vae import FoVAE
from data import ImageDataModule
from pytorch_lightning.demos.boring_classes import DemoModel, BoringDataModule


def cli_main():
    cli = LightningCLI(FoVAE, ImageDataModule)

if __name__ == "__main__":
    ptl.seed_everything(42)
    cli_main()