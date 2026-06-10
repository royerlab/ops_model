"""Shared building blocks for the pretrained feature extractors.

The Cell-DINO, DINOv3, and SubCell extractors all follow the same shape:
build a dataloader, run a model over every batch while assembling a metadata
dataframe, write the results in chunks, concatenate, and (when orchestrating)
fan jobs out over SLURM followed by an AnnData conversion.

Only three things genuinely differ per model — checkpoint loading, the
``preprocess`` step, and the output embedding — so those live in each model's
own ``*Model`` class. Everything else is the free functions below, which each
extractor composes. Nothing here is model-specific; the model is always passed
in already constructed.
"""

from pathlib import Path

import pandas as pd
from tqdm import tqdm
from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    SpatialPadd,
    ToTensord,
)

from ops_model.data import data_loader
from ops_model.data.labels import load_immunostaining_labels, SOURCE_FILENAME_TEMPLATES

SAVE_EVERY = 100  # flush a CSV chunk every N batches


def maybe_build_labels_df(config: dict):
    """Build a labels dataframe from immunostaining-style CSVs, if configured.

    Returns None for the default ``csv_source == "standard"`` case (labels come
    from the dataloader itself). Used by the Cell-DINO and DINOv3 extractors;
    SubCell configs have no ``csv_source`` and don't call this.
    """
    csv_source = config.get("csv_source", "standard")
    if csv_source not in ("cell_painting", "four_i", "immunostaining"):
        return None
    filename_template = config.get("filename_template") or SOURCE_FILENAME_TEMPLATES.get(
        csv_source
    )
    return load_immunostaining_labels(
        experiments=config["data_manager"]["experiments"],
        filename_template=filename_template,
        base_path=config.get("base_path"),
    )


def build_dataloader(config: dict, out_channels: list, *, labels_df=None):
    """Construct an OpsDataManager with a populated ``test_loader`` for inference.

    Args:
        config: Parsed YAML config dict.
        out_channels: Channels to load (a single-element list for Cell-DINO /
            DINOv3, ``[dna_channel, protein_channel]`` for SubCell).
        labels_df: Optional precomputed labels dataframe (see
            ``maybe_build_labels_df``).

    Returns:
        The constructed OpsDataManager.
    """
    dm = data_loader.OpsDataManager(
        experiments=config["data_manager"]["experiments"],
        batch_size=config["data_manager"]["batch_size"],
        data_split=config["data_manager"]["data_split"],
        out_channels=out_channels,
        initial_yx_patch_size=config["data_manager"]["initial_yx_patch_size"],
        final_yx_patch_size=config["data_manager"]["final_yx_patch_size"],
        link_csv_dir=config["data_manager"].get("link_csv_dir"),
        verbose=False,
        guide_col=config.get("guide_col", "sgRNA"),
    )
    dm.construct_dataloaders(
        labels_df=labels_df,
        num_workers=config["data_manager"]["num_workers"],
        dataset_type=config["dataset_type"],
        basic_kwargs={
            "cell_masks": config["data_manager"].get("cell_masks", True),
            "dataloader_normalization": config["data_manager"].get(
                "dataloader_normalization", "log"
            ),
            "transform": Compose(
                [
                    SpatialPadd(
                        keys=["data", "mask"],
                        spatial_size=dm.initial_yx_patch_size,
                    ),
                    CenterSpatialCropd(
                        keys=["data", "mask"], roi_size=(dm.final_yx_patch_size)
                    ),
                    ToTensord(keys=["data", "mask"]),
                ]
            ),
        },
    )
    return dm


def run_extraction(config: dict, dm, model, *, model_prefix: str, name_channel: str):
    """Run ``model`` over every batch and write features to CSV.

    Iterates the manager's ``test_loader``, builds a per-batch dataframe of
    features plus metadata, flushes chunks to ``chunks_{name_channel}/`` every
    ``SAVE_EVERY`` batches, then concatenates them into
    ``{model_prefix}_features_{name_channel}.csv``.

    Args:
        config: Parsed YAML config dict (uses ``output_dir``).
        dm: A constructed OpsDataManager (from ``build_dataloader``).
        model: An object exposing ``extract_features(batch) -> torch.Tensor``.
        model_prefix: Output filename prefix, e.g. ``"cell_dino"``.
        name_channel: Channel name used in the chunk dir and final filename.
    """
    test_loader = dm.test_loader
    print(f"Created dataset with {len(test_loader.dataset)} crops")
    print(f"{model_prefix} model loaded")

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    chunk_subdir = output_dir / f"chunks_{name_channel}"
    chunk_subdir.mkdir(parents=True, exist_ok=True)

    guide_col = dm.guide_col
    all_features = []
    chunk_idx = 0

    for batch_idx, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
        features = model.extract_features(batch)
        features_np = features.cpu().numpy()

        features_db = pd.DataFrame(features_np)
        features_db["label_int"] = batch["gene_label"].numpy()
        features_db["label_str"] = [
            dm.int_label_lut[label] for label in batch["gene_label"].numpy()
        ]
        features_db[guide_col] = [a[guide_col] for a in batch["crop_info"]]
        features_db["experiment"] = [a["store_key"] for a in batch["crop_info"]]
        features_db["x_position"] = [a["x_pheno"] for a in batch["crop_info"]]
        features_db["y_position"] = [a["y_pheno"] for a in batch["crop_info"]]
        features_db["well"] = [
            a["well"] + "_" + a["store_key"] for a in batch["crop_info"]
        ]
        all_features.append(features_db)

        if batch_idx % SAVE_EVERY == 0 and batch_idx > 0:
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

    final_df = pd.concat(
        [pd.read_csv(csv_file) for csv_file in csv_files], ignore_index=True
    )
    final_path = output_dir / f"{model_prefix}_features_{name_channel}.csv"
    final_df.to_csv(final_path, index=False)
    print(f"Saved final concatenated features to {final_path}")
    print(f"Final dataframe shape: {final_df.shape}")


def build_slurm_params(config: dict, *, default_cpus: int = 20) -> dict:
    """Translate a config's ``slurm:`` block into submit_parallel_jobs kwargs."""

    def _parse_time(t) -> int:
        if isinstance(t, int):
            return t
        parts = t.split(":")
        return int(parts[0]) * 60 + int(parts[1])  # HH:MM[:SS] -> minutes

    slurm_config = config.get("slurm", {})
    mem = slurm_config.get("mem", "64G")
    mem_gb = int(mem.rstrip("G")) if isinstance(mem, str) else int(mem)

    slurm_params = {
        "slurm_partition": slurm_config.get("partition", "gpu"),
        "slurm_gres": slurm_config.get("gres", "gpu:1"),
        "cpus_per_task": slurm_config.get("cpus_per_task", default_cpus),
        "mem_gb": mem_gb,
        "timeout_min": _parse_time(slurm_config.get("time", "4:00:00")),
    }
    constraint = slurm_config.get("constraint")
    if constraint:
        slurm_params["slurm_constraint"] = constraint
    return slurm_params


def run_anndata_followon(config_paths: list, result: dict):
    """Submit the AnnData-conversion SLURM batch after extraction completes.

    The conversion has a different resource profile (CPU + more memory) than GPU
    extraction, so it runs as a separate batch rather than sharing the GPU jobs.
    Skipped (with a warning) if any extraction job failed.
    """
    from ops_model.features.batch_process_embeddings import batch_process_slurm

    if not result.get("all_completed", True):
        print(
            "\n⚠ Skipping AnnData conversion: some extraction jobs failed. "
            "Fix and re-run, or convert manually via batch_process_embeddings.py."
        )
        return
    print("\nExtraction complete — submitting AnnData conversion jobs...")
    batch_process_slurm(config_paths=config_paths)
