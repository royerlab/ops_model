"""
DinoV3 Feature Evaluation Pipeline

This module processes DinoV3 embeddings (1024-dimensional features from ViT-L/16)
into AnnData objects for downstream analysis. Unlike CellProfiler features,
DinoV3 embeddings are pre-computed and require minimal preprocessing.

Key differences from CellProfiler pipeline:
- No array string conversion (features already numeric)
- Minimal QC (embeddings are clean)
- No cell-level PCA/UMAP (only at guide/gene aggregation level)
- Works directly with 1024-dim embeddings

Usage:
    python evaluate_dinov3.py --save_path /path/to/dinov3_features_Phase2D.csv
"""

from tqdm import tqdm
from pathlib import Path
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd
import scanpy as sc
import scanpy.external as sce
import anndata as ad

# Import shared utilities from CellProfiler evaluation
from ops_model.features.evaluate_cp import (
    # center_scale_fast,
    timer,
    NONFEATURE_COLUMNS,
)
from ops_model.features.anndata_utils import create_aggregated_embeddings, pca_embed

# DinoV3-specific feature count
DINOV3_FEATURE_DIM = 1024


def create_adata_object_dinov3(
    save_path: str, config: dict = None, channel: str = None, experiment: str = None
) -> ad.AnnData:
    """
    Create AnnData object from DinoV3 features CSV.

    This is a simplified version of create_adata_object() tailored for
    DinoV3 embeddings which require minimal preprocessing.

    Args:
        save_path: Path to DinoV3 features CSV file
        config: Configuration dictionary (optional)
        channel: Channel name (e.g., "Phase2D", "GFP")
        experiment: Experiment name (e.g., "ops0089")

    Returns:
        AnnData object with normalized embeddings and metadata
    """
    with timer("Reading CSV"):
        # Read CSV - DinoV3 features are already numeric
        features = pd.read_csv(save_path, low_memory=False)

    print(f"Dataset shape: {features.shape}")

    with timer("Extracting labels and experiment info"):
        # Extract metadata columns
        gene_strs = np.asarray(features["label_str"].values)
        gene_ints = np.asarray(features["label_int"].values)
        sgRNA_ids = np.asarray(features["sgRNA"].values)
        well_id = np.asarray(features["well"].values)

        # Read experiment from CSV if not provided
        if experiment is None and "experiment" in features.columns:
            unique_experiments = features["experiment"].unique()
            if len(unique_experiments) != 1:
                raise ValueError(
                    f"Multi-experiment datasets not supported. "
                    f"Found experiments: {unique_experiments}"
                )
            experiment = unique_experiments[0]
            print(f"Detected experiment from CSV: {experiment}")

        try:
            x_pos = np.asarray(features["x_position"].values)
            y_pos = np.asarray(features["y_position"].values)
            has_positions = True
        except KeyError:
            print("x_position/y_position columns not found, skipping.")
            has_positions = False

        # Drop metadata columns to get only features
        metadata_cols = ["label_str", "label_int", "sgRNA", "well", "experiment"]
        if has_positions:
            metadata_cols.extend(["x_position", "y_position"])

        features = features.drop(columns=metadata_cols, errors="ignore")

    # DinoV3 features are already numeric - no array string conversion needed
    print(f"DinoV3 features: {features.shape[1]} dimensions")

    with timer("Converting to float32"):
        # Convert to float32 for memory efficiency
        features = features.astype("float32")
        print(f"Converted {len(features.columns)} feature columns to float32")

    with timer("QC filtering"):
        # Minimal QC for DinoV3 embeddings

        # Check for NaN values (shouldn't be any in embeddings)
        n_nans = features.isna().sum().sum()
        if n_nans > 0:
            print(f"WARNING: Found {n_nans} NaN values in embeddings")
            # Drop rows with any NaN
            features = features.dropna()
            gene_strs = gene_strs[features.index]
            gene_ints = gene_ints[features.index]
            sgRNA_ids = sgRNA_ids[features.index]
            well_id = well_id[features.index]
            if has_positions:
                x_pos = x_pos[features.index]
                y_pos = y_pos[features.index]
            print(f"Dropped rows with NaN, remaining: {features.shape[0]}")

        # Check for infinity values (shouldn't be any in neural network embeddings)
        n_infs = np.isinf(features.values).sum()
        if n_infs > 0:
            print(f"WARNING: Found {n_infs} infinity values in embeddings")
            # Replace inf with NaN and drop rows
            features = features.replace([np.inf, -np.inf], np.nan)
            features = features.dropna()
            gene_strs = gene_strs[features.index]
            gene_ints = gene_ints[features.index]
            sgRNA_ids = sgRNA_ids[features.index]
            well_id = well_id[features.index]
            if has_positions:
                x_pos = x_pos[features.index]
                y_pos = y_pos[features.index]
            print(f"Dropped rows with inf, remaining: {features.shape[0]}")

        # Check for constant columns (very rare in embeddings)
        constant_cols = features.columns[features.nunique(dropna=False) == 1]
        if len(constant_cols) > 0:
            print(f"WARNING: Found {len(constant_cols)} constant columns in embeddings")
            features = features.drop(columns=constant_cols)
            print(f"Dropped constant columns, remaining features: {features.shape[1]}")

    # with timer("Center-scaling features"):
    #     # Normalize embeddings (mean=0, std=1)
    #     features['label_str'] = gene_strs
    #     features_norm = center_scale_fast(
    #         features,
    #         on_controls=False,  # Normalize on all data by default
    #         control_column="label_str",
    #         control_gene="NTC"
    #     )

    with timer("Creating AnnData object"):
        # Use features directly (no normalization applied)
        features_norm = features.copy()

        # Remove label_str from features before creating AnnData (if present)
        if "label_str" in features_norm.columns:
            features_norm = features_norm.drop(columns=["label_str"])

        # Create AnnData object
        adata = ad.AnnData(features_norm)

        # Add metadata to .obs
        adata.obs["label_str"] = gene_strs
        adata.obs["label_int"] = gene_ints
        adata.obs["sgRNA"] = sgRNA_ids
        adata.obs["well"] = well_id

        if has_positions:
            adata.obs["x_position"] = x_pos
            adata.obs["y_position"] = y_pos

        adata.var_names = features_norm.columns

        # Store channel/reporter metadata if provided
        if channel and experiment:
            use_reporter_names = False
            if config and "processing" in config:
                use_reporter_names = config["processing"].get(
                    "use_reporter_names", False
                )

            if use_reporter_names:
                from ops_model.data.feature_metadata import FeatureMetadata

                meta = FeatureMetadata()

                # Get short experiment name for metadata lookup
                exp_short = experiment.split("_")[0]
                reporter = meta.get_short_label(exp_short, channel)

                # Store metadata
                adata.uns["channel"] = channel
                adata.uns["reporter"] = reporter
                adata.uns["channel_mapping"] = {channel: reporter}
                adata.uns["experiment"] = experiment
                print(f"Stored metadata: channel={channel}, reporter={reporter}")
            else:
                # Store channel only (legacy mode)
                adata.uns["channel"] = channel
                adata.uns["experiment"] = experiment

    return adata


def process_dinov3(
    save_path: str,
    config_path: str = None,
):
    """
    Process DinoV3 features through the full pipeline.

    Unlike CellProfiler pipeline, this does NOT compute cell-level PCA/UMAP.
    PCA and UMAP are only computed at guide/gene aggregation level.

    If reporter names are enabled (use_reporter_names=True in config), files will be
    saved with reporter suffixes (e.g., features_processed_EEA1.h5ad) instead of
    channel names (e.g., features_processed_GFP.h5ad).

    Args:
        save_path: Path to DinoV3 features CSV
        config_path: Path to configuration YAML file

    Returns:
        Cell-level AnnData object (without PCA/UMAP)
    """

    config = {}
    if config_path is not None:
        import yaml

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        print(f"Loaded configuration from {config_path}")

    print("\n" + "=" * 60)
    print("Starting DinoV3 feature processing pipeline")
    print("=" * 60 + "\n")

    total_start = time.time()

    save_path = Path(save_path)
    save_dir = save_path.parent / "anndata_objects"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Extract channel from CSV filename (e.g., dinov3_features_Phase2D.csv -> Phase2D)
    csv_stem = save_path.stem  # e.g., "dinov3_features_Phase2D"
    if "_" in csv_stem:
        # Split on underscore and take the last part as the channel
        channel = csv_stem.split("_")[-1]
    else:
        # Fallback if no underscore found (shouldn't happen with standard naming)
        channel = "unknown"

    print(f"Detected channel: {channel}")

    # Read experiment from CSV to enable reporter name lookup
    with timer("Reading experiment from CSV"):
        df_sample = pd.read_csv(save_path, nrows=1)
        if "experiment" in df_sample.columns:
            experiment = df_sample["experiment"].iloc[0]
            print(f"Detected experiment: {experiment}")
        else:
            experiment = None
            print("Warning: experiment column not found in CSV")

    # Determine if using reporter names
    use_reporter_names = False
    filename_suffix = channel  # Default to channel
    if config and "processing" in config:
        use_reporter_names = config["processing"].get("use_reporter_names", False)

    if use_reporter_names and experiment:
        from ops_model.data.feature_metadata import FeatureMetadata

        meta = FeatureMetadata()
        exp_short = experiment.split("_")[0]
        reporter = meta.get_short_label(exp_short, channel)
        filename_suffix = reporter
        print(f"Using reporter name for files: {reporter} (channel: {channel})")
    else:
        print(f"Using channel name for files: {channel}")

    # Define checkpoint paths with appropriate suffix
    checkpoint_path = save_dir / f"features_processed_{filename_suffix}.h5ad"
    guide_avg_path = save_dir / f"guide_bulked_umap_{filename_suffix}.h5ad"
    gene_avg_path = save_dir / f"gene_bulked_umap_{filename_suffix}.h5ad"

    # Create AnnData object from DinoV3 embeddings
    with timer("TOTAL: Create AnnData object"):
        features_adata = create_adata_object_dinov3(
            str(save_path), config=config, channel=channel, experiment=experiment
        )

    print(
        f"Created AnnData with {features_adata.shape[0]} cells and {features_adata.shape[1]} features"
    )

    # Save cell-level data WITHOUT PCA/UMAP
    features_adata.write_h5ad(checkpoint_path)
    print(f"Saved cell-level AnnData (no PCA/UMAP) to {checkpoint_path}")

    # Guide-level aggregation and analysis
    with timer("TOTAL: Guide-level aggregation and UMAP"):
        embeddings_guide_bulk_ad = create_aggregated_embeddings(
            features_adata,
            level="guide",
            n_pca_components=128,
            n_neighbors=15,
        )
        # Propagate metadata to aggregated objects
        if "channel" in features_adata.uns:
            embeddings_guide_bulk_ad.uns["channel"] = features_adata.uns["channel"]
        if "reporter" in features_adata.uns:
            embeddings_guide_bulk_ad.uns["reporter"] = features_adata.uns["reporter"]
        if "channel_mapping" in features_adata.uns:
            embeddings_guide_bulk_ad.uns["channel_mapping"] = features_adata.uns[
                "channel_mapping"
            ]
        if "experiment" in features_adata.uns:
            embeddings_guide_bulk_ad.uns["experiment"] = features_adata.uns[
                "experiment"
            ]

        embeddings_guide_bulk_ad.write_h5ad(guide_avg_path)
        print(f"Saved guide-bulked analysis to {guide_avg_path}")

        # Optional plotting
        if (
            config.get("plot_guide_umap", False)
            and "X_umap" in embeddings_guide_bulk_ad.obsm.keys()
        ):
            plot_path = save_dir / "guide_umap.png"
            sc.pl.umap(embeddings_guide_bulk_ad, color="sgRNA", save=str(plot_path))

    # Gene-level aggregation and analysis
    with timer("TOTAL: Gene-level aggregation and UMAP"):
        embeddings_gene_avg_ad = create_aggregated_embeddings(
            features_adata,
            level="gene",
            n_pca_components=128,
            n_neighbors=15,
        )
        # Propagate metadata to aggregated objects
        if "channel" in features_adata.uns:
            embeddings_gene_avg_ad.uns["channel"] = features_adata.uns["channel"]
        if "reporter" in features_adata.uns:
            embeddings_gene_avg_ad.uns["reporter"] = features_adata.uns["reporter"]
        if "channel_mapping" in features_adata.uns:
            embeddings_gene_avg_ad.uns["channel_mapping"] = features_adata.uns[
                "channel_mapping"
            ]
        if "experiment" in features_adata.uns:
            embeddings_gene_avg_ad.uns["experiment"] = features_adata.uns["experiment"]

        embeddings_gene_avg_ad.write_h5ad(gene_avg_path)
        print(f"Saved gene-bulked analysis to {gene_avg_path}")

        # Optional plotting
        if (
            config.get("plot_gene_umap", False)
            and "X_umap" in embeddings_gene_avg_ad.obsm.keys()
        ):
            plot_path = save_dir / "gene_umap.png"
            sc.pl.umap(embeddings_gene_avg_ad, color="label_str", save=str(plot_path))

    total_time = time.time() - total_start
    print("\n" + "=" * 60)
    print(
        f"Pipeline completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)"
    )
    print(f"Cell-level output: {checkpoint_path} (raw embeddings, no PCA/UMAP)")
    print(f"Guide-bulked output: {guide_avg_path} (with PCA/UMAP)")
    print(f"Gene-bulked output: {gene_avg_path} (with PCA/UMAP)")
    if use_reporter_names:
        print(f"Files saved with reporter suffix: {filename_suffix}")
    print("=" * 60 + "\n")

    return features_adata


def _build_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="Process DinoV3 features into AnnData objects."
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Path to the CSV file containing DinoV3 features.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to configuration YAML file.",
    )
    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    process_dinov3(
        args.save_path,
        config_path=args.config_path,
    )


if __name__ == "__main__":
    main()
