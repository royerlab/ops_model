# %%
from pathlib import Path
import re
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd
import anndata as ad

DEFAULT_GUIDE_COL = "sgRNA"


def _nonfeature_columns(guide_col: str = DEFAULT_GUIDE_COL) -> list[str]:
    return [
        "label_str",
        "label_int",
        guide_col,
        "well",
        "experiment",
        "x_position",
        "y_position",
    ]


NONFEATURE_COLUMNS = _nonfeature_columns()


# Profiling context manager
@contextmanager
def timer(name: str):
    """Context manager to time code blocks"""
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"[TIMING] {name}: {elapsed:.2f} seconds")


def create_adata_object(
    save_path: str,
    config: dict = None,
    cell_type: str = None,
    embedding_type: str = "cellprofiler",
) -> ad.AnnData:
    """
    Create AnnData object from CellProfiler features CSV

    Args:
        save_path: Path to CSV file
        config: Configuration dictionary
                - 'cell-profiler': bool
                - 'normalize_features': bool
                - 'cell_type': str - Cell type used in experiment (required for validator)
                - 'embedding_type': str - Embedding type (default: 'cellprofiler')
        cell_type: Cell type used in experiment (e.g., 'A549', 'HeLa')
        embedding_type: Embedding type (default: 'cellprofiler')
    """
    with timer("Reading CSV"):
        # Read CSV - let pandas infer dtypes initially
        features = pd.read_csv(save_path, low_memory=False)

    print(f"Dataset shape: {features.shape}")

    # Extract required validator fields from config
    if config:
        cell_type = config.get("cell_type", cell_type)
        embedding_type = config.get("embedding_type", embedding_type)
    guide_col = (config or {}).get("guide_col", DEFAULT_GUIDE_COL)
    nonfeature_cols = _nonfeature_columns(guide_col)

    # Validate required fields
    if not cell_type:
        raise ValueError(
            "cell_type must be specified in config or as parameter. "
            "Add to config: cell_type: 'A549'  # or your cell line"
        )

    # Always map channel names to reporter names using FeatureMetadata
    with timer("Mapping channel names to reporter names"):
        from ops_utils.data.feature_metadata import FeatureMetadata

        # Check which experiments are in the dataset
        unique_experiments = features["experiment"].unique()

        if len(unique_experiments) != 1:
            raise ValueError(
                f"Multi-experiment datasets not yet supported for reporter name mapping. "
                f"Found experiments: {unique_experiments}"
            )

        experiment = unique_experiments[0]
        feature_meta = FeatureMetadata()

        # Identify channels from feature columns
        # CellProfiler features follow pattern: single_object_{channel}_{feature}
        feature_cols = [
            col for col in features.columns if col not in nonfeature_cols
        ]

        # Extract unique channels from feature column names.
        # Anchor on the compartment mask to capture multi-part channel names like
        # CP1_nuclei_Hoechst rather than just CP1.
        channels_in_data = set()
        for col in feature_cols:
            if col.startswith("single_object_"):
                m = re.match(r"single_object_(.+?)_(?:cell|nucleus|cytoplasm)_", col)
                if m:
                    channels_in_data.add(m.group(1))

        print(f"Detected channels: {sorted(channels_in_data)}")

        # Create channel-to-reporter mapping
        channel_mapping = {}
        for channel in channels_in_data:
            reporter = feature_meta.get_biological_signal(experiment, channel)
            channel_mapping[channel] = reporter
            print(f"  {channel} -> {reporter}")

        # Rename feature columns with reporter names
        renamed_cols = {}
        for col in feature_cols:
            new_name = feature_meta.replace_channel_in_feature_name(col, experiment)
            if new_name != col:
                renamed_cols[col] = new_name

        if renamed_cols:
            features = features.rename(columns=renamed_cols)
            print(f"Renamed {len(renamed_cols)} feature columns with reporter names")
            example = list(renamed_cols.items())[0]
            print(f"Example: '{example[0]}' -> '{example[1]}'")

    with timer("Extracting labels and metadata"):
        gene_strs = np.asarray(features["label_str"].values)
        gene_ints = np.asarray(features["label_int"].values)
        guide_ids = np.asarray(features[guide_col].values)
        well_id = np.asarray(features["well"].values)

        # Extract experiment field (required by validator)
        if "experiment" in features.columns:
            experiment_ids = np.asarray(features["experiment"].values)
        else:
            print("WARNING: 'experiment' column not found, using 'unknown'")
            experiment_ids = np.full(len(features), "unknown")

        # Handle position fields (required by validator, but may be missing)
        has_positions = True
        try:
            x_pos = np.asarray(features["x_position"].values)
            y_pos = np.asarray(features["y_position"].values)
        except KeyError:
            print(
                "WARNING: x_position/y_position not found. Setting to NaN (validator requires these fields)"
            )
            x_pos = np.full(len(features), np.nan)
            y_pos = np.full(len(features), np.nan)
            has_positions = False

        # Drop non-feature columns
        cols_to_drop = [col for col in nonfeature_cols if col in features.columns]
        features = features.drop(columns=cols_to_drop)

    with timer("Dropping constant columns and nans"):
        if config is not None and config["processing"].get("cell-profiler", False):
            features = features.dropna(subset=["cell_Area"])

        # if more than 5% of cells have NaN for a feature, drop that feature
        threshold = (
            int(
                config["processing"].get("max_nan_fraction_per_feature", 0.05)
                * features.shape[0]
            )
            if config
            else int(0.05 * features.shape[0])
        )
        cols_to_drop = features.columns[features.isna().sum(axis=0) > threshold]
        features = features.drop(columns=cols_to_drop)
        print(
            f"Dropped {len(cols_to_drop)} columns with >{threshold/features.shape[0]:.2%} cells NaN values"
        )

        # Filter rows with too many NaNs and track which rows are kept
        num_nan_features_per_row = features.isna().sum(axis=1)
        good_rows_mask = num_nan_features_per_row <= 0
        features = features[good_rows_mask]

        # Update metadata arrays to match filtered rows
        gene_strs = gene_strs[good_rows_mask]
        gene_ints = gene_ints[good_rows_mask]
        guide_ids = guide_ids[good_rows_mask]
        well_id = well_id[good_rows_mask]
        experiment_ids = experiment_ids[good_rows_mask]
        x_pos = x_pos[good_rows_mask]
        y_pos = y_pos[good_rows_mask]

        print(f"Kept {features.shape[0]} rows after filtering NaN rows")

        # Check for and handle infinity values (from division by zero in ratio features)
        # Replace inf with NaN, then drop columns containing them
        features = features.replace([np.inf, -np.inf], np.nan)
        cols_with_inf = features.columns[features.isna().any()]
        if len(cols_with_inf) > 0:
            features = features.drop(columns=cols_with_inf)
            print(f"Dropped {len(cols_with_inf)} columns with inf values")

        # Drop constant columns
        constant_cols = features.columns[features.nunique(dropna=False) == 1]
        features = features.drop(columns=constant_cols)
        print(f"Dropped {len(constant_cols)} constant columns")

    # Keep features in original units (no normalization at cell level)
    # Normalization happens later during multi-experiment concatenation
    with timer("Preserving original feature units"):
        features_norm = features.copy()
        print("Features kept in original units (no normalization)")

    with timer("Creating AnnData object"):
        # Remove label_str from features if present
        if "label_str" in features_norm.columns:
            features_norm = features_norm.drop(columns=["label_str"])

        # Create AnnData
        adata = ad.AnnData(features_norm)

        # Add required .obs fields (base schema)
        adata.obs["perturbation"] = (
            gene_strs  # Validator requires 'perturbation', not 'label_str'
        )
        adata.obs["label_int"] = gene_ints  # Keep for backwards compatibility

        # Map first channel to reporter for base reporter field
        # (In split_by_reporter, each subset will have its specific reporter)
        if channel_mapping:
            # Use the first channel's reporter as the primary reporter
            # (will be overridden in split_adata_by_reporter for each subset)
            primary_channel = sorted(channels_in_data)[0]
            primary_reporter = channel_mapping[primary_channel]
            adata.obs["reporter"] = primary_reporter
            print(
                f"Set primary reporter to '{primary_reporter}' (from channel '{primary_channel}')"
            )
        else:
            print("WARNING: No channel mapping available, using 'unknown' as reporter")
            adata.obs["reporter"] = "unknown"

        # Add required .obs fields (cell schema)
        adata.obs[guide_col] = guide_ids
        adata.obs["well"] = well_id
        adata.obs["experiment"] = experiment_ids
        adata.obs["x_position"] = x_pos  # Always add (may be NaN)
        adata.obs["y_position"] = y_pos  # Always add (may be NaN)

        if not has_positions:
            print("WARNING: Position data is NaN - validator may flag this")

        adata.var_names = features_norm.columns

        # Add required .uns fields (base schema)
        adata.uns["cell_type"] = cell_type
        adata.uns["embedding_type"] = embedding_type
        adata.uns["guide_col"] = guide_col

        # Add optional .uns fields (useful metadata)
        adata.uns["channel_mapping"] = channel_mapping  # Always add
        # NOTE: Do NOT add experiment to .uns (only in .obs per validator spec)

        print(f"Added .uns metadata:")
        print(f"  cell_type: {cell_type}")
        print(f"  embedding_type: {embedding_type}")
        print(f"  guide_col: {guide_col}")
        print(f"  channel_mapping: {channel_mapping}")

    return adata


def split_adata_by_reporter(adata: ad.AnnData, verbose: bool = True) -> dict:
    """
    Split AnnData object by reporter/biological signal, return by reporter name.

    Extracts channel_mapping from adata.uns and creates separate AnnData
    objects for each reporter. Features are assigned by pattern matching:
    - Features containing _{reporter}_ pattern → assigned to that reporter
    - Compartment features (cell_, nucleus_, cytoplasm_) → duplicated across all reporters
      (these are channel-agnostic morphology features)
    - Colocalization features (containing multiple reporters with _ delimiters) → duplicated in each

    Returns dictionary keyed by REPORTER NAME (e.g., 'SEC61B', '5xUPRE', 'ChromaLive561emission').
    Files will be saved with reporter names to ensure feature name consistency during
    vertical pooling across experiments.

    Args:
        adata: Combined AnnData object with all features
        verbose: Print splitting information

    Returns:
        Dictionary mapping reporter name to AnnData subset
        Example: {'SEC61B': adata_sec61b, '5xUPRE': adata_5xupre, 'Phase': adata_phase}

    Example:
        >>> adata = ad.read_h5ad("features_processed.h5ad")
        >>> reporter_adatas = split_adata_by_reporter(adata)
        >>> for reporter, adata_sub in reporter_adatas.items():
        ...     print(f"{reporter}: {adata_sub.shape}")
    """
    if "channel_mapping" not in adata.uns:
        raise ValueError(
            "AnnData object missing channel_mapping in .uns. Cannot split by reporter."
        )

    channel_mapping = adata.uns["channel_mapping"]
    reporters = list(channel_mapping.values())  # e.g., ['SEC61B', '5xUPRE', 'Phase']

    # Create mapping: reporter -> channel
    reporter_to_channel = {v: k for k, v in channel_mapping.items()}

    if verbose:
        print(f"\nSplitting AnnData by reporter (returning by reporter name)...")
        print(f"Found {len(reporters)} reporters: {reporters}")
        print(f"Total features: {adata.shape[1]}")

    var_names = adata.var_names.tolist()

    # Identify compartment-level features (shared across all reporters)
    # These are channel-agnostic morphology features measured once per cell
    cell_features = [f for f in var_names if f.startswith("cell_")]
    nucleus_features = [f for f in var_names if f.startswith("nucleus_")]
    cytoplasm_features = [f for f in var_names if f.startswith("cytoplasm_")]
    shared_features = cell_features + nucleus_features + cytoplasm_features

    if verbose and shared_features:
        print(f"Shared compartment features (duplicated across all reporters):")
        print(f"  Cell: {len(cell_features)}")
        print(f"  Nucleus: {len(nucleus_features)}")
        print(f"  Cytoplasm: {len(cytoplasm_features)}")
        print(f"  Total shared: {len(shared_features)}")

    reporter_adatas = {}  # Key by reporter name for file naming consistency

    for reporter in reporters:
        # Find features containing this reporter name
        # Use _{reporter}_ pattern to avoid false matches (e.g., "Phase" in "ZernikePhase")
        reporter_features = [f for f in var_names if f"_{reporter}_" in f]

        # Combine reporter-specific features with shared compartment features
        all_features = sorted(set(reporter_features + shared_features))

        if len(all_features) == 0:
            print(f"  WARNING: No features found for reporter '{reporter}', skipping")
            continue

        # Create subset AnnData
        feature_indices = [var_names.index(f) for f in all_features]
        adata_subset = adata[:, feature_indices].copy()

        # Store metadata - reporter is primary, channel kept for reference
        channel_name = reporter_to_channel[reporter]
        # Update .obs reporter field to match this subset's reporter
        adata_subset.obs["reporter"] = reporter  # All cells get this reporter
        # Store channel reference in .uns (optional metadata)
        adata_subset.uns["channel"] = channel_name
        # Keep the full channel_mapping for reference
        adata_subset.uns["channel_mapping"] = channel_mapping

        # Key by reporter name (ensures consistent feature names during pooling)
        reporter_adatas[reporter] = adata_subset

        if verbose:
            reporter_only = len(reporter_features)
            shared_count = len([f for f in all_features if f in shared_features])
            print(
                f"  {reporter} (channel: {channel_name}): {len(all_features)} features "
                f"({reporter_only} reporter-specific, {shared_count} shared compartment)"
            )

    return reporter_adatas
