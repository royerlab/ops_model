import yaml
from datetime import datetime
import argparse

import torch
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from ops_model.data import data_loader
from ops_model.models import cytoself_model, dynaclr


def train(config_path):

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_type = config["model_type"]
    run_name = model_type + "_" + str(config["run_name"])
    dataset_type = config["dataset_type"]

    data_manager = data_loader.OpsDataManager(
        experiments=config["data_manager"]["experiments"],
        batch_size=config["data_manager"]["batch_size"],
        data_split=tuple(config["data_manager"]["data_split"]),
        out_channels=config["data_manager"]["out_channels"],
        initial_yx_patch_size=tuple(config["data_manager"]["initial_yx_patch_size"]),
        final_yx_patch_size=tuple(config["data_manager"]["final_yx_patch_size"]),
    )
    data_manager.construct_dataloaders(
        num_workers=config["data_manager"]["num_workers"],
        dataset_type=dataset_type,
        basic_kwargs=config["data_manager"].get("basic_kwargs"),
        contrastive_kwargs=config["data_manager"].get("contrastive_kwargs"),
        balanced_sampling=config["data_manager"].get("balanced_sampling", False),
    )

    train_loader = data_manager.train_loader
    val_loader = data_manager.val_loader
    test_loader = data_manager.test_loader

    torch.set_float32_matmul_precision("medium")  # huge boost in speed

    if model_type == "dynaclr":
        lit_model = dynaclr.LitDynaClr(
            lr=config["model"]["lr"],
            schedule=config["model"]["schedule"],
            log_batches_per_epoch=config["model"]["log_batches_per_epoch"],
            log_samples_per_batch=config["model"]["log_samples_per_batch"],
            log_embeddings=config["model"]["log_embeddings"],
            example_input_array_shape=(1, 1)
            + tuple(config["data_manager"]["final_yx_patch_size"]),
            **config["model"].get("encoder", {}),
        )
        logger = TensorBoardLogger(
            save_dir=f"/hpc/projects/intracellular_dashboard/ops/models/logs/dynaclr",
            name=run_name,
        )

    if model_type == "cytoself":
        mc = config["model"]
        lit_model = cytoself_model.LitCytoSelf(
            emb_shapes=(
                tuple(mc["embedding_shapes"][0]),
                tuple(mc["embedding_shapes"][1]),
            ),
            vq_args=mc["vq_args"],
            num_class=mc["num_classes"],
            input_shape=tuple(mc["input_shape"]),
            output_shape=tuple(mc["input_shape"]),
            fc_input_type=mc["fc_input_type"],
            fc_output_idx=[mc["fc_output_index"]],
            vq_coeff=mc.get("vq_coeff", 1.0),
            fc_coeff=mc.get("fc_coeff", 1.0),
        )

        logger = TensorBoardLogger(
            save_dir=f"/hpc/projects/intracellular_dashboard/ops/models/logs/cytoself",
            name=run_name,
        )

    timestamp = datetime.now().strftime("%Y-%m-%d")
    checkpoint_callback = ModelCheckpoint(
        monitor=config["callbacks"]["monitor"],
        dirpath=f"/hpc/projects/intracellular_dashboard/ops/models/logs/{model_type}/{run_name}",  # TODO: add path to dataset
        filename=f"{run_name}-{timestamp}-{{global_step:2f}}-{{val/total_loss:.2f}}",
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
        callbacks=[checkpoint_callback],
    )
    trainer.fit(
        model=lit_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model with a given config file."
    )
    parser.add_argument("--config_path", type=str, help="Path to the config file.")
    args = parser.parse_args()

    train(args.config_path)
