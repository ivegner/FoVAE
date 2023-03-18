# main.py
from pytorch_lightning.cli import LightningCLI
import pytorch_lightning as ptl

from models.foveated_vae import FoVAE
from data import ImageDataModule

default_config_dir = "default_configs"


def cli_main():
    _cli = LightningCLI(
        FoVAE,
        ImageDataModule,
        parser_kwargs={
            "fit": {
                "default_config_files": [
                    f"{default_config_dir}/trainer.yaml",
                    f"{default_config_dir}/model_mnist.yaml",
                    f"{default_config_dir}/data_mnist.yaml",
                ]
            },
        },
    )


if __name__ == "__main__":
    ptl.seed_everything(42)
    cli_main()
