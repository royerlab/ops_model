import argparse
from datetime import datetime

import lightning as L
import numpy as np
import pandas as pd
import torch
import yaml
from lightning import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_metric_learning.losses import NTXentLoss

from ops_model.data import data_loader
from ops_model.models import dynaclr


def train(config_path):

    seed_everything(42)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    labels_df = pd.read_csv(config["data_manager"]["labels_df_path"])
    # labels_df["total_index"] = np.arange(len(labels_df))
    sample_weights = labels_df.pop("sample_weight").to_numpy()
    experiments = labels_df["store_key"].unique().tolist()
    experiment_dict = {a: [] for a in experiments}

    model_type = config["model_type"]
    run_name = model_type + "_" + str(config["run_name"])
    dataset_type = config["dataset_type"]

    data_manager = data_loader.OpsDataManager(
        experiments=experiment_dict,
        batch_size=config["data_manager"]["batch_size"],
        data_split=tuple(config["data_manager"]["data_split"]),
        out_channels=None,
        initial_yx_patch_size=tuple(config["data_manager"]["initial_yx_patch_size"]),
        final_yx_patch_size=tuple(config["data_manager"]["final_yx_patch_size"]),
    )
    data_manager.construct_dataloaders(
        num_workers=config["data_manager"]["num_workers"],
        pin_memory=config["data_manager"].get("pin_memory", True),
        prefetch_factor=config["data_manager"].get("prefetch_factor", 2),
        labels_df=labels_df,
        balanced_sampling=config["data_manager"].get("balanced_sampling", True),
        balanced_sampling_val=config["data_manager"].get("balanced_sampling_val", False),
        balance_col=config["data_manager"].get("balance_col", "reporter"),
        dataset_type=dataset_type,
        basic_kwargs=config["data_manager"].get("basic_kwargs"),
        contrastive_kwargs=config["data_manager"].get("contrastive_kwargs"),
    )

    train_loader = data_manager.train_loader
    val_loader = data_manager.val_loader
    test_loader = data_manager.test_loader

    torch.set_float32_matmul_precision("medium")  # huge boost in speed

    # Initialize model from config
    model_config = config["model"].copy()
    encoder_config = model_config.pop("encoder")
    temperature = model_config.pop("temperature")

    # Check for checkpoint to load
    ckpt_path = config.get("ckpt_path", None)

    if ckpt_path is not None:
        print(f"Loading model from checkpoint: {ckpt_path}")
        lit_model = dynaclr.LitDynaClr.load_from_checkpoint(
            ckpt_path,
            loss_function=NTXentLoss(temperature=temperature),
            example_input_array_shape=(1, 1)
            + tuple(config["data_manager"]["final_yx_patch_size"]),
            **model_config,
            **encoder_config,
        )
    else:
        lit_model = dynaclr.LitDynaClr(
            loss_function=NTXentLoss(temperature=temperature),
            example_input_array_shape=(1, 1)
            + tuple(config["data_manager"]["final_yx_patch_size"]),
            **model_config,
            **encoder_config,
        )
    # Prepare trainer config
    trainer_config = config["trainer"].copy()

    # Create logger first (so we know the version directory)
    logger_config = trainer_config.pop("logger", None)
    if logger_config:
        from lightning.pytorch.cli import instantiate_class

        logger = instantiate_class(tuple(), logger_config)
    else:
        logger = TensorBoardLogger(
            save_dir=f"/hpc/projects/intracellular_dashboard/ops/models/logs/dynaclr",
            name=run_name,
        )

    # Instantiate callbacks from config if present in trainer section
    callbacks = []
    if "callbacks" in trainer_config:
        from lightning.pytorch.cli import instantiate_class

        callback_configs = trainer_config.pop("callbacks")
        for cb_config in callback_configs:
            # Add dirpath and filename to checkpoint callback
            if "ModelCheckpoint" in cb_config["class_path"]:
                # Save checkpoints in the same directory as TensorBoard logs
                # This will be: .../dynaclr/{run_name}/version_X/checkpoints/
                if "dirpath" not in cb_config["init_args"]:
                    cb_config["init_args"]["dirpath"] = None  # Use logger's default
                if "filename" not in cb_config["init_args"]:
                    cb_config["init_args"]["filename"] = (
                        "epoch={epoch:03d}-loss={loss/val:.3f}"
                    )
            callbacks.append(instantiate_class(tuple(), cb_config))

    trainer = L.Trainer(
        **trainer_config,
        logger=logger,
        callbacks=callbacks if callbacks else None,
    )

    # Save a copy of the config file to the log directory for reproducibility
    import shutil
    from pathlib import Path

    log_dir = Path(logger.log_dir)
    config_backup_path = log_dir / "config.yml"
    shutil.copy2(config_path, config_backup_path)
    print(f"Config saved to: {config_backup_path}")

    trainer.fit(
        model=lit_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=ckpt_path,  # Resume training from checkpoint if provided
    )

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model with a given config file."
    )
    parser.add_argument("--config_path", type=str, help="Path to the config file.")
    args = parser.parse_args()

    train(args.config_path)
