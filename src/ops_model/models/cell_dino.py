import yaml
from tqdm import tqdm

import torch
import numpy as np
import pandas as pd
import os
from pathlib import Path
from torchvision.transforms import v2
from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    SpatialPadd,
    ToTensord,
)

from ops_model.data import data_loader

REPO_DIR = "/hpc/projects/icd.ops/models/model_checkpoints/cell_dino/dinov2"
CHECKPOINT = "/hpc/projects/icd.ops/models/model_checkpoints/cell_dino/channel_adaptive_dino_vitl16_pretrain_cells-ef7c17ff.pth"


class CellDinoModel:

    def __init__(self):
        super().__init__()
        self.load_model()
        self.model.cuda()

    def load_model(self):
        self.model = torch.hub.load(
            REPO_DIR,
            "channel_adaptive_dino_vitl16",
            source="local",
            pretrained_path=CHECKPOINT,
            in_channels=1,
        )
        return

    def extract_features(self, batch: dict) -> torch.Tensor:

        x = batch["data"]
        inputs = self.preprocess(x)

        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.float32):
                output = self.model(inputs.cuda())
        return output

    def preprocess(self, x: torch.Tensor, resize_size: int = 224) -> torch.Tensor:
        """Per-image z-score normalization (Cell-DINO training convention).

        Unlike DINOv3, Cell-DINO uses per-image z-score rather than global ImageNet
        statistics. Input is expected to be a float tensor (our dataloader already
        produces floats), so no uint8->float conversion is needed.
        """
        resize = v2.Resize((resize_size, resize_size), antialias=True)
        x = resize(x.cuda().float())
        mean = x.mean(dim=(-2, -1), keepdim=True)
        std = x.std(dim=(-2, -1), keepdim=True)
        return (x - mean) / (std + 1e-7)

    def _reshape_to_spatial(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Reshape (B, N, D) patches to (B, D, H', W') spatial map.
        * Only need if running model.get_intermediate_layers, just calling
        model.forward will return a single embedding per image

        """
        batch_size, n_patches, embed_dim = patches.shape
        h_patches = w_patches = int(np.sqrt(n_patches))

        if h_patches * w_patches != n_patches:
            raise ValueError(
                f"Number of patches ({n_patches}) is not a perfect square. "
                f"Expected {h_patches}*{h_patches} = {h_patches * h_patches}."
            )

        # (B, N, D) -> (B, H', W', D) -> (B, D, H', W')
        spatial = patches.reshape(batch_size, h_patches, w_patches, embed_dim)
        return spatial.permute(0, 3, 1, 2)


def extract_cell_dino_features(
    config: dict = None,
):

    print(
        f"Extracting Cell-DINO features for {list(config['data_manager']['experiments'].keys())}"
    )
    dm = data_loader.OpsDataManager(
        experiments=config["data_manager"]["experiments"],
        batch_size=config["data_manager"]["batch_size"],
        data_split=config["data_manager"]["data_split"],
        out_channels=config["data_manager"]["out_channels"],
        initial_yx_patch_size=config["data_manager"]["initial_yx_patch_size"],
        final_yx_patch_size=config["data_manager"]["final_yx_patch_size"],
        verbose=False,
    )
    dm.construct_dataloaders(
        num_workers=config["data_manager"]["num_workers"],
        dataset_type=config["dataset_type"],
        basic_kwargs={
            "cell_masks": True,
            "transform": Compose(
                [
                    SpatialPadd(
                        keys=["data", "mask"],
                        spatial_size=dm.initial_yx_patch_size,
                    ),
                    CenterSpatialCropd(
                        keys=["data", "mask"], roi_size=(dm.final_yx_patch_size)
                    ),
                    ToTensord(
                        keys=["data", "mask"],
                    ),
                ]
            ),
        },
    )
    test_loader = dm.test_loader
    print(f"Created dataset with {len(test_loader.dataset)} crops")

    cell_dino = CellDinoModel()
    print("Cell-DINO model loaded")

    # Setup output directory
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    chunk_subdir = output_dir / f"chunks_{config['data_manager']['out_channels'][0]}"
    chunk_subdir.mkdir(parents=True, exist_ok=True)

    save_every = 100  # Save every N batches
    all_features = []
    chunk_idx = 0

    # Extract features from all batches
    for batch_idx, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
        # Extract features using the Cell-DINO model
        features = cell_dino.extract_features(batch)

        # Convert to numpy and create dataframe with metadata
        features_np = features.cpu().numpy()

        features_db = pd.DataFrame(features_np)
        features_db["label_int"] = batch["gene_label"].numpy()
        features_db["label_str"] = [
            dm.int_label_lut[label] for label in batch["gene_label"].numpy()
        ]
        features_db["sgRNA"] = [a["sgRNA"] for a in batch["crop_info"]]
        features_db["experiment"] = [a["store_key"] for a in batch["crop_info"]]
        features_db["x_position"] = [a["x_pheno"] for a in batch["crop_info"]]
        features_db["y_position"] = [a["y_pheno"] for a in batch["crop_info"]]

        features_db["well"] = [
            a["well"] + "_" + a["store_key"] for a in batch["crop_info"]
        ]
        all_features.append(features_db)

        if batch_idx % save_every == 0 and batch_idx > 0:
            df_chunk = pd.concat(all_features, ignore_index=True)
            csv_path = chunk_subdir / f"features_chunk_{chunk_idx}.csv"
            df_chunk.to_csv(csv_path, index=False)
            print(
                f"Saved chunk {chunk_idx} with {len(all_features)} batches to {csv_path}"
            )

            # Reset for next chunk
            all_features = []
            chunk_idx += 1

    # Save any remaining features
    if all_features:
        df_chunk = pd.concat(all_features, ignore_index=True)
        csv_path = chunk_subdir / f"features_chunk_{chunk_idx}.csv"
        df_chunk.to_csv(csv_path, index=False)
        print(
            f"Saved final chunk {chunk_idx} with {len(all_features)} batches to {csv_path}"
        )
        chunk_idx += 1

    # Load and concatenate all CSV files
    print(f"\nLoading and concatenating {chunk_idx} chunks...")
    csv_files = sorted(chunk_subdir.glob("features_chunk_*.csv"))

    if not csv_files:
        print("No feature files found!")
        return None

    df_list = [pd.read_csv(csv_file) for csv_file in csv_files]
    final_df = pd.concat(df_list, ignore_index=True)

    # Save the final concatenated dataframe
    final_path = (
        output_dir
        / f"cell_dino_features_{config['data_manager']['out_channels'][0]}.csv"
    )
    final_df.to_csv(final_path, index=False)
    print(f"Saved final concatenated features to {final_path}")
    print(f"Final dataframe shape: {final_df.shape}")

    return final_df


import argparse
import submitit
import copy


def cell_dino_main(config_path: str):
    """
    Main orchestrator function for Cell-DINO feature extraction.

    Spawns one SLURM job per channel specified in the config.
    Each job runs extract_cell_dino_features() independently.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        List of submitit Job objects
    """

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Extract channels and experiment info
    out_channels = config["data_manager"]["out_channels"]
    experiments = list(config["data_manager"]["experiments"].keys())
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get SLURM parameters from config (with defaults)
    slurm_config = config.get("slurm", {})
    partition = slurm_config.get("partition", "gpu")
    gres = slurm_config.get("gres", "gpu:1")
    cpus_per_task = slurm_config.get("cpus_per_task", 20)
    mem = slurm_config.get("mem", "64G")
    time_limit = slurm_config.get("time", "4:00:00")
    constraint = slurm_config.get("constraint", "h100|h200|a100|a40")

    # Parse memory (remove 'G' suffix if present and convert to int)
    if isinstance(mem, str):
        mem_gb = int(mem.rstrip("G"))
    else:
        mem_gb = int(mem)

    # Parse time limit to minutes
    if isinstance(time_limit, int):
        # Already in minutes
        timeout_min = time_limit
    elif isinstance(time_limit, str):
        time_parts = time_limit.split(":")
        if len(time_parts) == 3:  # HH:MM:SS
            timeout_min = int(time_parts[0]) * 60 + int(time_parts[1])
        elif len(time_parts) == 2:  # MM:SS
            timeout_min = int(time_parts[0])
        else:
            timeout_min = 240  # Default 4 hours
    else:
        timeout_min = 240  # Default 4 hours

    print(
        f"Spawning {len(out_channels)} Cell-DINO jobs for experiment(s): {experiments}"
    )
    print(f"Channels: {out_channels}")
    print(f"Output directory: {output_dir}")

    # Setup submitit executor
    log_dir = Path(
        "/hpc/projects/intracellular_dashboard/ops/models/logs/cell_dino/slurm_logs"
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    executor = submitit.AutoExecutor(folder=log_dir)

    jobs = []

    for channel in out_channels:
        # Create a modified config for this channel
        channel_config = copy.deepcopy(config)
        channel_config["data_manager"]["out_channels"] = [channel]

        # Update executor parameters for each job
        executor.update_parameters(
            timeout_min=timeout_min,
            slurm_partition=partition,
            slurm_gres=gres,
            cpus_per_task=cpus_per_task,
            mem_gb=mem_gb,
            slurm_constraint=constraint,
            slurm_job_name=f"cell_dino_{experiments[0].split('_')[0]}_{channel}",
        )

        # Submit job
        job = executor.submit(extract_cell_dino_features, config=channel_config)
        jobs.append(job)

        print(f"Submitted job {job.job_id} for channel {channel}")

    print(f"\nAll {len(jobs)} jobs submitted. Check status with 'squeue -u $USER'")
    print(f"Job IDs: {[job.job_id for job in jobs]}")

    return jobs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract Cell-DINO features from OPS dataset based on config"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the YAML config file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cell_dino_main(config_path=args.config_path)
