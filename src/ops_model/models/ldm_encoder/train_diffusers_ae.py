import yaml
from datetime import datetime

import pytorch_lightning as L
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from diffusers import AutoencoderKL

from ops_model.data import data_loader
from ops_model.models.ldm_encoder.diffusers_ae import LitAe


def train(config_path):

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    lit_model = LitAe(
        AutoencoderKL,
        model_kwargs=config["model"],
        lightning_config=config["lightning_config"],
    )
    logger = TensorBoardLogger(
        save_dir=f"/hpc/projects/intracellular_dashboard/ops/models/logs/{config['model_type']}",
        name=f"{config['run_name']}",
    )
    timestamp = datetime.now().strftime("%Y-%m-%d")
    checkpoint_callback = ModelCheckpoint(
        monitor=config["callbacks"]["monitor"],
        dirpath=f"/hpc/projects/intracellular_dashboard/ops/models/model_checkpoints/{config['model_type']}/",
        filename=f"{config['run_name']}-{timestamp}-{{val/loss:.2f}}",
        save_top_k=3,
        mode="min",
    )

    trainer = L.Trainer(
        max_epochs=config["trainer"]["max_epochs"],
        accelerator="gpu",
        devices=1,
        limit_train_batches=config["trainer"]["limit_train_batches"],
        limit_val_batches=config["trainer"]["limit_val_batches"],
        logger=logger,
        log_every_n_steps=10,
        callbacks=[checkpoint_callback],
    )

    data_manager = data_loader.OpsDataManager(
        experiments=config["data_manager"]["experiments"],
        batch_size=config["data_manager"]["batch_size"],
        data_split=tuple(config["data_manager"]["data_split"]),
        out_channels=config["data_manager"]["out_channels"],
        initial_yx_patch_size=tuple(config["data_manager"]["initial_yx_patch_size"]),
    )
    data_manager.construct_dataloaders(
        num_workers=config["data_manager"]["num_workers"],
        dataset_type=config["dataset_type"],
    )

    train_loader = data_manager.train_loader
    val_loader = data_manager.val_loader
    test_loader = data_manager.test_loader

    trainer.fit(
        model=lit_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    return


if __name__ == "__main__":
    config_path = "/hpc/mydata/alexander.hillsley/ops/ops_model/src/ops_model/configs/diffusers_ae_phase.yml"
    train(config_path)
# %%
