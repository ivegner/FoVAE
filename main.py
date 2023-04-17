# main.py
from pathlib import Path
from pytorch_lightning.cli import LightningCLI, SaveConfigCallback
import pytorch_lightning as ptl

from models.foveated_vae import FoVAE
from data import ImageDataModule
import wandb
from pprint import pprint

default_config_dir = "default_configs"


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        # parser.add_lightning_class_args(WandbLogger, "wandb_logger")
        parser.add_argument(
            "--resume_run_id", default="", type=str, help="W&B run ID to resume from"
        )
        # parser.link_arguments("resume_run_id", "trainer.logger.init_args.resume", compute_fn=lambda x: "must" if x else "never")
        # parser.link_arguments("resume_run_id", "trainer.logger.init_args.id")

    def before_instantiate_classes(self):
        # pprint(self.config.as_dict())
        # print("resume_run_id" in self.config)
        subcommand = self.config.subcommand
        c = self.config[subcommand]

        def make_checkpoint_dir(run_id):
            return f"checkpoints/{run_id}"

        run_id = None
        if c.resume_run_id and subcommand == "fit":
            run_id = c.resume_run_id
            c.trainer.logger.init_args.resume = "must"
        else:
            run_id = wandb.util.generate_id()
            c.trainer.logger.init_args.resume = "never"

        if c.resume_run_id:
            api = wandb.Api()
            artifact = api.artifact(f'{c.trainer.logger.init_args.project}/model-{c.resume_run_id}:latest', type="model")
            # artifact_dir = artifact.download(make_checkpoint_dir(c.resume_run_id))
            artifact_dir = artifact.download()
            c.ckpt_path=str(Path(artifact_dir) / "model.ckpt")

        c.trainer.logger.init_args.id = run_id
        for callback in c.trainer.callbacks:
            if callback.class_path == "pytorch_lightning.callbacks.ModelCheckpoint":
                callback.init_args.dirpath = f"checkpoints/{run_id}"

def cli_main():
    _cli = MyLightningCLI(
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
            "validate": {
                "default_config_files": [
                    f"{default_config_dir}/trainer.yaml",
                    f"{default_config_dir}/model_mnist.yaml",
                    f"{default_config_dir}/data_mnist.yaml",
                ]
            },
        },
        save_config_callback=None,
    )


if __name__ == "__main__":
    ptl.seed_everything(42)
    cli_main()
