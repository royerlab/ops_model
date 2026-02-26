"""
Embedding Feature Evaluation Pipeline

This module processes neural-network embeddings (e.g. DinoV3, Cell-DINO)
into AnnData objects for downstream analysis. Unlike CellProfiler features,
these embeddings are pre-computed and require minimal preprocessing.

Key differences from CellProfiler pipeline:
- No array string conversion (features already numeric)
- Minimal QC (embeddings are clean)
- No cell-level PCA/UMAP (only at guide/gene aggregation level)
- Works directly with high-dimensional embeddings

Usage:
    python evaluate_embeddings.py --save_path /path/to/features_Phase2D.csv
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
from ops_model.post_process.anndata_processing.anndata_validator import (
    AnndataValidator,
    IssueLevel,
)


def validate_and_save(
    adata: ad.AnnData,
    path: Path,
    level: str,
) -> None:
    """
    Validate AnnData object and save to h5ad file.

    Validation is always enforced with hard constraints - errors will raise exceptions.

    Args:
        adata: AnnData object to validate and save
        path: Path to save h5ad file
        level: Schema level ("cell", "guide", "gene")

    Raises:
        ValueError: If validation fails
    """
    print(f"\nValidating {level}-level AnnData before saving...")

    # Initialize validator
    validator = AnndataValidator()

    # Run validation (returns ValidationReport object)
    report = validator.validate(adata, level=level)

    # Access errors and warnings from report
    errors = report.errors
    warnings = report.warnings

    # Report results
    if report.is_valid:
        print(f"✓ Validation passed: {level}-level AnnData is compliant")
    else:
        print(f"Validation found {len(errors)} errors, {len(warnings)} warnings")

        # Show errors
        if errors:
            print("\nERROR-level issues:")
            for issue in errors[:10]:  # Show first 10
                print(f"  - {issue.field}: {issue.message}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more errors")

        # Show warnings
        if warnings:
            print("\nWARNING-level issues:")
            for issue in warnings[:5]:  # Show first 5
                print(f"  - {issue.field}: {issue.message}")
            if len(warnings) > 5:
                print(f"  ... and {len(warnings) - 5} more warnings")

    # Fail fast if errors found (always enforced)
    if errors:
        error_summary = (
            f"{len(errors)} validation error(s) found in {level}-level AnnData"
        )
        print(f"\n✗ Validation FAILED: {error_summary}")
        print(f"Fix these issues before saving. First error: {errors[0].message}")
        raise ValueError(error_summary)

    # Save file
    print(f"Saving {level}-level AnnData to {path}")
    adata.write_h5ad(path)
    print(f"✓ Saved successfully: {path}")


def create_adata_object_embedding(
    save_path: str,
    config: dict = None,
    channel: str = None,
    experiment: str = None,
    cell_type: str = None,
    embedding_type: str = "dinov3",
) -> ad.AnnData:
    """
    Create AnnData object from an embeddings CSV.

    Tailored for neural-network embeddings (DinoV3, Cell-DINO, etc.)
    which are pre-computed numeric features requiring minimal preprocessing.

    Args:
        save_path: Path to embeddings CSV file
        config: Configuration dictionary (optional)
        channel: Channel name (e.g., "Phase2D", "GFP")
        experiment: Experiment name (e.g., "ops0089")
        cell_type: Cell type used in experiment (e.g., "A549", "HeLa") - required for validator
        embedding_type: Method used to extract embeddings (e.g., "dinov3", "cell_dino")

    Returns:
        AnnData object with embeddings and metadata
    """
    with timer("Reading CSV"):
        # Read CSV - embedding features are already numeric
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
            # Position fields are required by validator
            print("WARNING: x_position/y_position not found. Setting to NaN.")
            x_pos = np.full(len(features), np.nan, dtype=np.float32)
            y_pos = np.full(len(features), np.nan, dtype=np.float32)
            has_positions = False  # Track for warnings

        # Drop metadata columns to get only features
        metadata_cols = ["label_str", "label_int", "sgRNA", "well", "experiment"]
        if has_positions:
            metadata_cols.extend(["x_position", "y_position"])

        features = features.drop(columns=metadata_cols, errors="ignore")

    # Embeddings are already numeric - no array string conversion needed
    print(f"{embedding_type} features: {features.shape[1]} dimensions")

    with timer("Converting to float32"):
        # Convert to float32 for memory efficiency
        features = features.astype("float32")
        print(f"Converted {len(features.columns)} feature columns to float32")

    with timer("QC filtering"):
        # Minimal QC for neural-network embeddings

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
        # Use perturbation instead of label_str (validator requirement)
        adata.obs["perturbation"] = gene_strs
        adata.obs["label_int"] = gene_ints
        adata.obs["sgRNA"] = sgRNA_ids
        adata.obs["well"] = well_id

        # Position fields are required by validator (always add)
        adata.obs["x_position"] = x_pos
        adata.obs["y_position"] = y_pos
        if not has_positions:
            print("WARNING: Position data missing - validator may flag this")

        # Add experiment field (required by validator)
        adata.obs["experiment"] = experiment if experiment else "unknown"

        # Add reporter field to .obs (required by validator)
        # Always use FeatureMetadata to convert channel names to reporter names
        if channel and experiment:
            from ops_model.data.feature_metadata import FeatureMetadata

            meta = FeatureMetadata()
            exp_short = experiment.split("_")[0]
            reporter = meta.get_biological_signal(exp_short, channel)
            adata.obs["reporter"] = reporter
            print(f"Mapped channel '{channel}' to reporter '{reporter}'")

            # Store channel_mapping for .uns (will be added in next step)
            channel_mapping = {channel: reporter}
        else:
            # Fallback if channel not provided
            print(
                "WARNING: channel or experiment not provided, using 'unknown' as reporter"
            )
            adata.obs["reporter"] = "unknown"
            channel_mapping = None

        adata.var_names = features_norm.columns

        # Add required validator .uns fields
        if cell_type:
            adata.uns["cell_type"] = cell_type
        else:
            raise ValueError(
                "cell_type must be provided for validator compliance. "
                "Add cell_type to your config file."
            )

        adata.uns["embedding_type"] = embedding_type

        # Add channel metadata (optional but useful)
        if channel:
            adata.uns["channel"] = channel

        # Add channel_mapping dict (always, if we have mapping)
        if channel_mapping is not None:
            adata.uns["channel_mapping"] = channel_mapping

        # Note: experiment is in .obs, no need to duplicate in .uns
        # Note: reporter is in .obs, no need to duplicate in .uns

    return adata


def process_embedding_csv(
    save_path: str,
    config_path: str = None,
):
    """
    Process neural-network embedding features through the full pipeline.

    Unlike CellProfiler pipeline, this does NOT compute cell-level PCA/UMAP.
    PCA and UMAP are only computed at guide/gene aggregation level.

    If reporter names are enabled (use_reporter_names=True in config), files will be
    saved with reporter suffixes (e.g., features_processed_EEA1.h5ad) instead of
    channel names (e.g., features_processed_GFP.h5ad).

    Args:
        save_path: Path to embeddings CSV (e.g. dinov3_features_Phase2D.csv or
                   cell_dino_features_Phase2D.csv)
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

    # Extract required config parameters for validator compliance
    cell_type = config.get("cell_type", None)
    embedding_type = config.get("embedding_type", "dinov3")

    if not cell_type:
        raise ValueError(
            "cell_type must be specified in config for validator compliance.\n"
            "Add to your config file:\n"
            "  cell_type: 'A549'  # or your cell type"
        )

    print("\n" + "=" * 60)
    print(f"Starting {embedding_type} feature processing pipeline")
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

    # Always use FeatureMetadata to get reporter names for file naming
    filename_suffix = channel  # Default to channel
    if channel and experiment:
        from ops_model.data.feature_metadata import FeatureMetadata

        meta = FeatureMetadata()
        exp_short = experiment.split("_")[0]
        reporter = meta.get_biological_signal(exp_short, channel)
        filename_suffix = reporter
        print(f"Using reporter name for files: {reporter} (channel: {channel})")
    else:
        print(f"Using channel name for files: {channel}")

    # Define checkpoint paths with appropriate suffix
    checkpoint_path = save_dir / f"features_processed_{filename_suffix}.h5ad"
    guide_avg_path = save_dir / f"guide_bulked_{filename_suffix}.h5ad"
    gene_avg_path = save_dir / f"gene_bulked_{filename_suffix}.h5ad"

    # Create AnnData object from embeddings
    with timer("TOTAL: Create AnnData object"):
        features_adata = create_adata_object_embedding(
            str(save_path),
            config=config,
            channel=channel,
            experiment=experiment,
            cell_type=cell_type,
            embedding_type=embedding_type,
        )

    print(
        f"Created AnnData with {features_adata.shape[0]} cells and {features_adata.shape[1]} features"
    )

    # Validate and save cell-level data
    validate_and_save(
        features_adata,
        checkpoint_path,
        level="cell",
    )

    # Read aggregation configuration
    agg_config = config.get("aggregation", {})
    guide_config = agg_config.get("guide_level", {})
    gene_config = agg_config.get("gene_level", {})

    # Guide-level aggregation and analysis (configurable)
    if guide_config.get("enabled", True):  # Default True for backwards compatibility
        print("\n" + "=" * 60)
        print("Guide-level aggregation")
        print("=" * 60)

        # Get embedding settings
        guide_embeddings = guide_config.get("embeddings", {})
        compute_embeddings = guide_config.get("compute_embeddings", True)

        # Get embedding parameters
        n_pca = guide_embeddings.get("n_pca_components", 128)
        n_neighbors = guide_embeddings.get("n_neighbors", 15)
        compute_pca = guide_embeddings.get("pca", True) if compute_embeddings else False
        compute_umap = (
            guide_embeddings.get("umap", True) if compute_embeddings else False
        )
        compute_phate = (
            guide_embeddings.get("phate", True) if compute_embeddings else False
        )

        # Validate: UMAP requires PCA
        if compute_umap and not compute_pca:
            print("WARNING: UMAP requires PCA. Enabling PCA.")
            compute_pca = True

        with timer("TOTAL: Guide-level processing"):
            embeddings_guide_bulk_ad = create_aggregated_embeddings(
                features_adata,
                level="guide",
                n_pca_components=n_pca,
                n_neighbors=n_neighbors,
                compute_pca=compute_pca,
                compute_umap=compute_umap,
                compute_phate=compute_phate,
            )

            # Validate and save guide-level output
            if guide_config.get("save_output", True):
                validate_and_save(
                    embeddings_guide_bulk_ad,
                    guide_avg_path,
                    level="guide",
                )

            # Optional plotting
            if (
                guide_config.get("plot_umap", False)
                and "X_umap" in embeddings_guide_bulk_ad.obsm.keys()
            ):
                plot_path = save_dir / "guide_umap.png"
                sc.pl.umap(embeddings_guide_bulk_ad, color="sgRNA", save=str(plot_path))
                print(f"Saved guide UMAP plot to {plot_path}")
    else:
        print("\nSkipping guide-level aggregation (disabled in config)")
        embeddings_guide_bulk_ad = None

    # Gene-level aggregation and analysis (configurable)
    if gene_config.get("enabled", True):  # Default True for backwards compatibility
        print("\n" + "=" * 60)
        print("Gene-level aggregation")
        print("=" * 60)

        # Get embedding settings
        gene_embeddings = gene_config.get("embeddings", {})
        compute_embeddings = gene_config.get("compute_embeddings", True)

        # Get embedding parameters
        n_pca = gene_embeddings.get("n_pca_components", 128)
        n_neighbors = gene_embeddings.get("n_neighbors", 15)
        compute_pca = gene_embeddings.get("pca", True) if compute_embeddings else False
        compute_umap = (
            gene_embeddings.get("umap", True) if compute_embeddings else False
        )
        compute_phate = (
            gene_embeddings.get("phate", True) if compute_embeddings else False
        )

        # Validate: UMAP requires PCA
        if compute_umap and not compute_pca:
            print("WARNING: UMAP requires PCA. Enabling PCA.")
            compute_pca = True

        with timer("TOTAL: Gene-level processing"):
            embeddings_gene_avg_ad = create_aggregated_embeddings(
                features_adata,
                level="gene",
                n_pca_components=n_pca,
                n_neighbors=n_neighbors,
                compute_pca=compute_pca,
                compute_umap=compute_umap,
                compute_phate=compute_phate,
            )

            # Validate and save gene-level output
            if gene_config.get("save_output", True):
                validate_and_save(
                    embeddings_gene_avg_ad,
                    gene_avg_path,
                    level="gene",
                )

            # Optional plotting
            if (
                gene_config.get("plot_umap", False)
                and "X_umap" in embeddings_gene_avg_ad.obsm.keys()
            ):
                plot_path = save_dir / "gene_umap.png"
                sc.pl.umap(
                    embeddings_gene_avg_ad, color="perturbation", save=str(plot_path)
                )
                print(f"Saved gene UMAP plot to {plot_path}")
    else:
        print("\nSkipping gene-level aggregation (disabled in config)")
        embeddings_gene_avg_ad = None

    total_time = time.time() - total_start
    print("\n" + "=" * 60)
    print(
        f"Pipeline completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)"
    )
    print(f"Cell-level output: {checkpoint_path} (raw embeddings, no PCA/UMAP)")

    # Report guide-level output
    if guide_config.get("enabled", True) and guide_config.get("save_output", True):
        guide_embeddings = guide_config.get("embeddings", {})
        compute_guide_embeddings = guide_config.get("compute_embeddings", True)
        if compute_guide_embeddings:
            embeddings_list = []
            if guide_embeddings.get("pca", True):
                embeddings_list.append("PCA")
            if guide_embeddings.get("umap", True):
                embeddings_list.append("UMAP")
            if guide_embeddings.get("phate", True):
                embeddings_list.append("PHATE")
            embeddings_str = ", ".join(embeddings_list) if embeddings_list else "none"
            print(f"Guide-bulked output: {guide_avg_path} (with {embeddings_str})")
        else:
            print(f"Guide-bulked output: {guide_avg_path} (aggregated, no embeddings)")

    # Report gene-level output
    if gene_config.get("enabled", True) and gene_config.get("save_output", True):
        gene_embeddings = gene_config.get("embeddings", {})
        compute_gene_embeddings = gene_config.get("compute_embeddings", True)
        if compute_gene_embeddings:
            embeddings_list = []
            if gene_embeddings.get("pca", True):
                embeddings_list.append("PCA")
            if gene_embeddings.get("umap", True):
                embeddings_list.append("UMAP")
            if gene_embeddings.get("phate", True):
                embeddings_list.append("PHATE")
            embeddings_str = ", ".join(embeddings_list) if embeddings_list else "none"
            print(f"Gene-bulked output: {gene_avg_path} (with {embeddings_str})")
        else:
            print(f"Gene-bulked output: {gene_avg_path} (aggregated, no embeddings)")

    print(f"Files saved with reporter suffix: {filename_suffix}")
    print("=" * 60 + "\n")

    return features_adata


def _build_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="Process neural-network embedding features into AnnData objects."
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Path to the CSV file containing embedding features.",
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

    process_embedding_csv(
        args.save_path,
        config_path=args.config_path,
    )


if __name__ == "__main__":
    main()
