from datetime import datetime

import lightning as L
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from ops_analysis.model.models import byol
from ops_analysis.model.vesuvius import vesuvius_dataloader


def train(config_path):

    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_type = config["model_type"]
    run_name = model_type + "_" + config["run_name"]

    data_manager = vesuvius_dataloader.VesuviusDataManager(
        batch_size=config["data_manager"]["batch_size"],
        data_split=tuple(config["data_manager"]["data_split"]),
        in_channels=config["data_manager"]["in_channels"],
        final_yx_patch_size=tuple(config["data_manager"]["final_yx_patch_size"]),
    )
    data_manager.construct_dataloaders(
        num_workers=config["data_manager"]["num_workers"],
        dataset_kwargs=config["data_manager"].get("dataset_kwargs"),
    )

    train_loader = data_manager.train_loader
    val_loader = data_manager.val_loader
    test_loader = data_manager.test_loader

    lit_model = byol.LitBYOL(
        image_size=config["model"]["image_size"],
        projection_size=config["model"]["projection_size"],
        projection_hidden_size=config["model"]["projection_hidden_size"],
        lr=config["model"]["lr"],
        in_channels=len(train_loader.dataset.in_channels),
    )

    timestamp = datetime.now().strftime("%Y-%m-%d")
    checkpoint_callback = ModelCheckpoint(
        monitor=config["callbacks"]["monitor"],
        dirpath=f"/hpc/projects/intracellular_dashboard/ops/models/model_checkpoints/vesuvius/{model_type}/{run_name}/",
        filename=f"{run_name}-{timestamp}-{{global_step:2f}}-{{val_loss:.2f}}",
        save_top_k=3,
        mode="min",
    )

    trainer = L.Trainer(
        max_epochs=config["trainer"]["max_epochs"],
        accelerator="gpu",
        devices=1,
        limit_train_batches=config["trainer"]["limit_train_batches"],
        limit_val_batches=config["trainer"]["limit_val_batches"],
        logger=TensorBoardLogger(
            save_dir=f"/hpc/projects/intracellular_dashboard/ops/models/logs/{model_type}/{run_name}",
            name=run_name,
        ),
        callbacks=[checkpoint_callback],
    )
    trainer.fit(
        model=lit_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    return


if __name__ == "__main__":
    train(
        "/hpc/mydata/alexander.hillsley/ops/ops_analysis/ops_analysis/model/vesuvius/configs/byol/byol_20250901_4-Actin.yml"
    )
