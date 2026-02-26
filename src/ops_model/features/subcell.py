"""SubCell bg feature extraction pipeline for OPS data.

Follows the same pattern as dinov3.py / cell_dino.py:
  - extract_subcell_features(config)  — single-job extraction loop
  - subcell_main(config_path)         — SLURM orchestrator (one job per protein channel)
  - CLI entry point via __main__

Config structure (YAML):
    data_manager:
      experiments:
        <experiment_key>: <zarr_path>
      dna_channel: DAPI          # nuclear/Blue channel for SubCell bg model
      batch_size: 32
      data_split: [0, 0, 1]
      out_channels: ["Phase2D", "GFP"]  # imaging channels; subcell_main pairs each with dna_channel
      initial_yx_patch_size: [256, 256]
      final_yx_patch_size: [128, 128]
      num_workers: 10
    dataset_type: basic
    output_dir: /path/to/output
    slurm:
      partition: gpu
      gres: "gpu:1"
      cpus_per_task: 10
      mem: "36G"
      time: "4:00:00"
      constraint: "h100|h200|a100|a40"

Note: subcell_main reads dna_channel and spawns one SLURM job per channel in
out_channels, passing [dna_channel, channel] as out_channels to each job.
"""

import argparse
import copy
from pathlib import Path

import pandas as pd
import yaml
from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    SpatialPadd,
    ToTensord,
)
from tqdm import tqdm

from ops_model.data import data_loader
from ops_model.models.subcell import SubCellModel


def build_subcell_dataloader(config: dict, out_channels: list):
    """Construct and return an OpsDataManager with dataloaders for SubCell inference.

    Args:
        config: Parsed YAML config dict.
        out_channels: Two-element list [dna_channel, imaging_channel].

    Returns:
        OpsDataManager with test_loader populated.
    """
    dm = data_loader.OpsDataManager(
        experiments=config["data_manager"]["experiments"],
        batch_size=config["data_manager"]["batch_size"],
        data_split=config["data_manager"]["data_split"],
        out_channels=out_channels,
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
    return dm


def extract_subcell_features(config: dict = None):
    """Extract SubCell embeddings for a single two-channel configuration.

    Args:
        config: Parsed YAML config dict. data_manager.out_channels must have
                exactly 2 entries: [dna_channel, imaging_channel].

    Returns:
        DataFrame with 1536 feature columns plus metadata columns.
    """
    out_channels = config["data_manager"]["out_channels"]
    assert (
        len(out_channels) == 2
    ), f"SubCell requires exactly 2 channels [DNA, imaging], got {out_channels}"

    print(
        f"Extracting SubCell features for "
        f"{list(config['data_manager']['experiments'].keys())} "
        f"channels={out_channels}"
    )

    dm = build_subcell_dataloader(config, out_channels=out_channels)
    test_loader = dm.test_loader
    print(f"Created dataset with {len(test_loader.dataset)} crops")

    model = SubCellModel()
    print("SubCell model loaded")

    # Setup output directory and chunk subdirectory
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    protein_channel = out_channels[1]
    chunk_subdir = output_dir / f"chunks_{protein_channel}"
    chunk_subdir.mkdir(parents=True, exist_ok=True)

    save_every = 100
    all_features = []
    chunk_idx = 0

    for batch_idx, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
        features = model.extract_features(batch)
        features_np = features.numpy()

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
            all_features = []
            chunk_idx += 1

    if all_features:
        df_chunk = pd.concat(all_features, ignore_index=True)
        csv_path = chunk_subdir / f"features_chunk_{chunk_idx}.csv"
        df_chunk.to_csv(csv_path, index=False)
        print(
            f"Saved final chunk {chunk_idx} with {len(all_features)} batches to {csv_path}"
        )
        chunk_idx += 1

    print(f"\nLoading and concatenating {chunk_idx} chunks...")
    csv_files = sorted(chunk_subdir.glob("features_chunk_*.csv"))

    if not csv_files:
        print("No feature files found!")
        return None

    df_list = [pd.read_csv(csv_file) for csv_file in csv_files]
    final_df = pd.concat(df_list, ignore_index=True)

    final_path = output_dir / f"subcell_features_{protein_channel}.csv"
    final_df.to_csv(final_path, index=False)
    print(f"Saved final concatenated features to {final_path}")
    print(f"Final dataframe shape: {final_df.shape}")

    return final_df


def subcell_main(config_path: str):
    """Orchestrate SubCell feature extraction via SLURM.

    Reads out_channels from config. Channel 0 is the DNA/DAPI channel; each
    remaining channel is a protein channel. Spawns one SLURM job per protein
    channel, pairing it with the DNA channel.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        List of submitit Job objects.
    """
    import submitit

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    dna_channel = config["data_manager"]["dna_channel"]
    protein_channels = config["data_manager"]["out_channels"]
    experiments = list(config["data_manager"]["experiments"].keys())
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    slurm_config = config.get("slurm", {})
    partition = slurm_config.get("partition", "gpu")
    gres = slurm_config.get("gres", "gpu:1")
    cpus_per_task = slurm_config.get("cpus_per_task", 10)
    mem = slurm_config.get("mem", "64G")
    time_limit = slurm_config.get("time", "4:00:00")
    constraint = slurm_config.get("constraint", "h100|h200|a100|a40")

    if isinstance(mem, str):
        mem_gb = int(mem.rstrip("G"))
    else:
        mem_gb = int(mem)

    if isinstance(time_limit, int):
        timeout_min = time_limit
    elif isinstance(time_limit, str):
        time_parts = time_limit.split(":")
        if len(time_parts) == 3:
            timeout_min = int(time_parts[0]) * 60 + int(time_parts[1])
        elif len(time_parts) == 2:
            timeout_min = int(time_parts[0])
        else:
            timeout_min = 240
    else:
        timeout_min = 240

    print(
        f"Spawning {len(protein_channels)} SubCell jobs for experiment(s): {experiments}"
    )
    print(f"DNA channel: {dna_channel}, Imaging channels: {protein_channels}")
    print(f"Output directory: {output_dir}")

    log_dir = Path(
        "/hpc/projects/intracellular_dashboard/ops/models/logs/subcell/slurm_logs"
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    executor = submitit.AutoExecutor(folder=log_dir)

    jobs = []
    for protein_channel in protein_channels:
        channel_config = copy.deepcopy(config)
        channel_config["data_manager"]["out_channels"] = [dna_channel, protein_channel]

        executor.update_parameters(
            timeout_min=timeout_min,
            slurm_partition=partition,
            slurm_gres=gres,
            cpus_per_task=cpus_per_task,
            mem_gb=mem_gb,
            slurm_constraint=constraint,
            slurm_job_name=f"subcell_{experiments[0].split('_')[0]}_{protein_channel}",
        )

        job = executor.submit(extract_subcell_features, config=channel_config)
        jobs.append(job)
        print(
            f"Submitted job {job.job_id} for channel pair [{dna_channel}, {protein_channel}]"
        )

    print(f"\nAll {len(jobs)} jobs submitted. Check status with 'squeue -u $USER'")
    print(f"Job IDs: {[job.job_id for job in jobs]}")

    return jobs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract SubCell features from OPS dataset based on config"
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
    subcell_main(config_path=args.config_path)
