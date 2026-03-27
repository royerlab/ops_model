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
import copy


def cell_dino_main(config_paths: list[str]):
    """Orchestrate Cell-DINO feature extraction via SLURM.

    For each config, reads out_channels and spawns one SLURM job per channel,
    using submit_parallel_jobs.

    Args:
        config_paths: List of paths to YAML configuration files.

    Returns:
        Result dict from submit_parallel_jobs.
    """
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs

    def _parse_time(t) -> int:
        if isinstance(t, int):
            return t
        parts = t.split(":")
        return int(parts[0]) * 60 + int(parts[1])  # HH:MM[:SS] → minutes

    jobs_to_submit = []
    slurm_params = None

    for config_path in config_paths:
        config_stem = Path(config_path).stem
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        out_channels = config["data_manager"]["out_channels"]
        experiments = list(config["data_manager"]["experiments"].keys())

        print(
            f"Config {config_stem}: {len(out_channels)} channel(s) for experiment(s) {experiments}"
        )
        print(f"  Channels: {out_channels}")

        # Extract slurm params from the first config encountered
        if slurm_params is None:
            slurm_config = config.get("slurm", {})
            mem = slurm_config.get("mem", "64G")
            time_limit = slurm_config.get("time", "4:00:00")

            mem_gb = int(mem.rstrip("G")) if isinstance(mem, str) else int(mem)
            timeout_min = _parse_time(time_limit)

            slurm_params = {
                "slurm_partition": slurm_config.get("partition", "gpu"),
                "slurm_gres": slurm_config.get("gres", "gpu:1"),
                "cpus_per_task": slurm_config.get("cpus_per_task", 20),
                "mem_gb": mem_gb,
                "timeout_min": timeout_min,
            }
            constraint = slurm_config.get("constraint")
            if constraint:
                slurm_params["slurm_constraint"] = constraint

        for channel in out_channels:
            channel_config = copy.deepcopy(config)
            channel_config["data_manager"]["out_channels"] = [channel]
            jobs_to_submit.append(
                {
                    "name": f"{config_stem}_{channel}",
                    "func": extract_cell_dino_features,
                    "kwargs": {"config": channel_config},
                    "metadata": {
                        "config": config_path,
                        "channel": channel,
                        "experiments": experiments,
                    },
                }
            )

    log_dir = "/hpc/projects/intracellular_dashboard/ops/models/logs/cell_dino/slurm_logs"

    print(f"\nSubmitting {len(jobs_to_submit)} Cell-DINO job(s) via submit_parallel_jobs")
    result = submit_parallel_jobs(
        jobs_to_submit=jobs_to_submit,
        experiment="cell_dino",
        slurm_params=slurm_params,
        log_dir=log_dir,
        wait_for_completion=True,
    )
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract Cell-DINO features from OPS dataset based on config"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--config_path",
        type=str,
        help="Path to a single YAML config file",
    )
    group.add_argument(
        "--config_list",
        type=str,
        help="Path to .txt file with one config path per line",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.config_list:
        with open(args.config_list) as f:
            config_paths = [line.strip() for line in f if line.strip()]
    else:
        config_paths = [args.config_path]
    cell_dino_main(config_paths)
