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

from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad

from ops_model.features.evaluate_cp import timer, DEFAULT_GUIDE_COL


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
    guide_col = (config or {}).get("guide_col", DEFAULT_GUIDE_COL)

    with timer("Reading CSV"):
        # Read CSV - embedding features are already numeric
        features = pd.read_csv(save_path, low_memory=False)

    print(f"Dataset shape: {features.shape}")

    with timer("Extracting labels and experiment info"):
        # Extract metadata columns
        gene_strs = np.asarray(features["label_str"].values)
        gene_ints = np.asarray(features["label_int"].values)
        guide_ids = np.asarray(features[guide_col].values)
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

        # Drop all non-numeric columns (catches known string metadata + any unexpected label columns)
        non_numeric_cols = features.select_dtypes(exclude="number").columns.tolist()
        known_str_metadata = {"label_str", guide_col, "well", "experiment"}
        unexpected = [c for c in non_numeric_cols if c not in known_str_metadata]
        if unexpected:
            print(
                f"WARNING: Dropping {len(unexpected)} unexpected non-numeric column(s): {unexpected}"
            )
        features = features.drop(columns=non_numeric_cols)

        # Drop known numeric metadata columns
        numeric_metadata = [
            c
            for c in ["label_int", "x_position", "y_position"]
            if c in features.columns
        ]
        features = features.drop(columns=numeric_metadata)

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
            guide_ids = guide_ids[features.index]
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
            guide_ids = guide_ids[features.index]
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
        adata.obs[guide_col] = guide_ids
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
            from ops_utils.data.feature_metadata import FeatureMetadata

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
        adata.uns["guide_col"] = guide_col

        # Add channel metadata (optional but useful)
        if channel:
            adata.uns["channel"] = channel

        # Add channel_mapping dict (always, if we have mapping)
        if channel_mapping is not None:
            adata.uns["channel_mapping"] = channel_mapping

        # Note: experiment is in .obs, no need to duplicate in .uns
        # Note: reporter is in .obs, no need to duplicate in .uns

    return adata
