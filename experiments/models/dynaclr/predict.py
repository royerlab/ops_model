import argparse
from pathlib import Path

import lightning as L
import pandas as pd
import torch
import yaml
from lightning import seed_everything
from pytorch_metric_learning.losses import NTXentLoss

from ops_model.data import data_loader
from ops_model.models import dynaclr


def predict(config_path):
    """
    Run DynaCLR prediction from config file.

    Parameters
    ----------
    config_path : str
        Path to prediction configuration YAML file

    Returns
    -------
    Path
        Path to the generated AnnData Zarr file
    """

    seed_everything(42)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Load labels dataframe
    labels_path = config["data_manager"]["labels_df_path"]
    if labels_path.endswith(".parquet"):
        labels_df = pd.read_parquet(labels_path)
    else:
        labels_df = pd.read_csv(labels_path, low_memory=False)
    labels_df.set_index("total_index", inplace=True, drop=False)

    # Create data manager with test-only split
    experiments = labels_df["store_key"].unique().tolist()
    experiment_dict = {exp: [] for exp in experiments}

    data_manager = data_loader.OpsDataManager(
        experiments=experiment_dict,
        batch_size=config["data_manager"]["batch_size"],
        data_split=tuple(config["data_manager"]["data_split"]),
        out_channels=None,
        initial_yx_patch_size=tuple(config["data_manager"]["initial_yx_patch_size"]),
        final_yx_patch_size=tuple(config["data_manager"]["final_yx_patch_size"]),
    )

    # For prediction, use BasicDataset (no augmentations, no positive pairs)
    basic_kwargs = {"cell_masks": config["data_manager"].get("cell_masks", True)}

    data_manager.construct_dataloaders(
        num_workers=config["data_manager"]["num_workers"],
        pin_memory=config["data_manager"].get("pin_memory", True),
        prefetch_factor=config["data_manager"].get("prefetch_factor", 2),
        labels_df=labels_df,
        dataset_type="basic",
        basic_kwargs=basic_kwargs,
    )

    test_loader = data_manager.test_loader

    # Load model from checkpoint
    model_config = config["model"].copy()
    encoder_config = model_config.pop("encoder")
    temperature = model_config.pop("temperature")

    lit_model = dynaclr.LitDynaClr.load_from_checkpoint(
        config["ckpt_path"],
        loss_function=NTXentLoss(temperature=temperature),
        **model_config,
        **encoder_config,
    )

    # Set model to eval mode
    lit_model.eval()
    lit_model.freeze()

    # Create prediction writer
    prediction_config = config.get("prediction", {})
    embedding_type = f"{config['model_type']}_{config['run_name']}"
    writer = dynaclr.DynaClrAnnDataWriter(
        output_dir=prediction_config["output_dir"],
        run_name=config["run_name"],
        labels_df=labels_df,
        save_features=prediction_config.get("save_features", True),
        save_projections=prediction_config.get("save_projections", False),
        cell_type=config.get("cell_type"),
        embedding_type=embedding_type,
    )

    # Create trainer for prediction
    trainer_config = config.get("trainer", {})
    trainer = L.Trainer(
        accelerator=trainer_config.get("accelerator", "gpu"),
        devices=trainer_config.get("devices", 1),
        precision=trainer_config.get("precision", "32-true"),
        callbacks=[writer],
        logger=False,
    )

    # Run prediction
    print(f"Running prediction on {len(test_loader.dataset)} samples...")
    predictions = trainer.predict(lit_model, dataloaders=test_loader)

    output_path = (
        Path(prediction_config["output_dir"])
        / f"dynaclr_embeddings_{config['run_name']}.zarr"
    )
    print(f"\nPrediction complete! Output saved to:\n{output_path}")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run DynaCLR prediction with a given config file."
    )
    parser.add_argument(
        "--config_path", type=str, help="Path to the prediction config file."
    )
    args = parser.parse_args()

    predict(args.config_path)
