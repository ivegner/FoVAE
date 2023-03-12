# main.py
from pytorch_lightning.cli import LightningCLI
import pytorch_lightning as ptl

from models.foveated_vae import FoVAE
from data import ImageDataModule


def cli_main():
    _cli = LightningCLI(FoVAE, ImageDataModule)

if __name__ == "__main__":
    ptl.seed_everything(42)
    cli_main()