# %%
from tqdm import tqdm
from pathlib import Path
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd
import scanpy as sc
import scanpy.external as sce
import anndata as ad
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from ops_model.data.paths import OpsPaths
from ops_model.features.anndata_utils import create_aggregated_embeddings, pca_embed

NONFEATURE_COLUMNS = [
    "label_str",
    "label_int",
    "sgRNA",
    "well",
    "experiment",
    "x_position",
    "y_position",
]


# Profiling context manager
@contextmanager
def timer(name: str):
    """Context manager to time code blocks"""
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"[TIMING] {name}: {elapsed:.2f} seconds")


def pca_embed(
    adata: ad.AnnData,
    n_components: int = 128,
    variance_plot=False,
) -> ad.AnnData:

    sc.tl.pca(adata, n_comps=n_components)
    if variance_plot:
        sc.pl.pca_variance_ratio(adata, n_pcs=100, log=False, save=False)
        plt.figure()

    return adata


def pca_fit_manual(df: pd.DataFrame, n_components: int = 128):
    pca = PCA(n_components=n_components)
    pca.fit_transform(df)
    return pca


def cell_size(save_path: str):
    features = pd.read_csv(save_path)

    df_sorted = (
        features.groupby("label_str", observed=False)
        .mean()
        .sort_values(by="cell_mask_sizeshape_Area", ascending=False)
    )

    return df_sorted["cell_mask_sizeshape_Area"]


def _convert_array_strings_to_float(value):
    """
    Convert string representations of arrays to float values.
    Examples: '[0.2875]' -> 0.2875, '[1.0, 2.0]' -> 1.5 (mean)
    """
    if isinstance(value, str):
        # Check if it looks like an array string
        if value.startswith("[") and value.endswith("]"):
            try:
                # Remove brackets and split by comma
                inner = value[1:-1].strip()
                if inner:
                    values = [float(x.strip()) for x in inner.split(",")]
                    # If single value, return it; otherwise raise an error
                    if len(values) == 1:
                        return values[0]
                    else:
                        raise ValueError(f"Too many values in array string: {value}")
                else:
                    return 0.0
            except (ValueError, AttributeError):
                return np.nan
    return value


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

    # Validate required fields
    if not cell_type:
        raise ValueError(
            "cell_type must be specified in config or as parameter. "
            "Add to config: cell_type: 'A549'  # or your cell line"
        )

    # Always map channel names to reporter names using FeatureMetadata
    with timer("Mapping channel names to reporter names"):
        from ops_model.data.feature_metadata import FeatureMetadata

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
            col for col in features.columns if col not in NONFEATURE_COLUMNS
        ]

        # Extract unique channels from feature column names
        channels_in_data = set()
        for col in feature_cols:
            if col.startswith("single_object_"):
                parts = col.split("_", 3)
                if len(parts) >= 3:
                    channel = parts[2]
                    channels_in_data.add(channel)

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
        sgRNA_ids = np.asarray(features["sgRNA"].values)
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
        cols_to_drop = [col for col in NONFEATURE_COLUMNS if col in features.columns]
        features = features.drop(columns=cols_to_drop)

    with timer("Converting array strings to floats"):
        pass
        # Convert any string representations of arrays to float values
        # for col in features.columns:
        #     if col == "label_str":
        #         continue
        #     # Check if column contains string array representations
        #     if features[col].dtype == "object":
        #         features[col] = features[col].apply(_convert_array_strings_to_float)
        #         # Convert to numeric, coercing errors to NaN
        #         features[col] = pd.to_numeric(features[col], errors="coerce")

    with timer("Converting numeric columns to float32"):
        pass
        # Convert numeric columns to float32 for memory efficiency
        # This can halve memory usage compared to float64
        # numeric_cols = features.select_dtypes(include=[np.number]).columns
        # numeric_cols = [col for col in numeric_cols if col != "label_str"]
        # features[numeric_cols] = features[numeric_cols].astype("float32")
        # print(f"Converted {len(numeric_cols)} numeric columns to float32")

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
        good_rows_mask = num_nan_features_per_row <= 0 if config else 0
        features = features[good_rows_mask]

        # Update metadata arrays to match filtered rows
        gene_strs = gene_strs[good_rows_mask]
        gene_ints = gene_ints[good_rows_mask]
        sgRNA_ids = sgRNA_ids[good_rows_mask]
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
        adata.obs["sgRNA"] = sgRNA_ids
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

        # Add optional .uns fields (useful metadata)
        adata.uns["channel_mapping"] = channel_mapping  # Always add
        # NOTE: Do NOT add experiment to .uns (only in .obs per validator spec)

        print(f"Added .uns metadata:")
        print(f"  cell_type: {cell_type}")
        print(f"  embedding_type: {embedding_type}")
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


def process(save_path: str, config_path: str = None):
    """
    Process CellProfiler features through the full pipeline

    Reporter names are always derived from FeatureMetadata. If multiple channels are present:
    1. Create combined AnnData object
    2. Split by reporter/biological signal
    3. Save separate files for each reporter (keyed by reporter name for feature consistency):
       - features_processed_{reporter}.h5ad (cell-level)
       - guide_bulked_{reporter}.h5ad
       - gene_bulked_{reporter}.h5ad

    Args:
        save_path: Path to features CSV
        config_path: Path to configuration YAML file (must include 'cell_type' field)
    """
    print("\n" + "=" * 60)
    print("Starting feature processing pipeline")
    print("=" * 60 + "\n")

    total_start = time.time()

    # Load config if provided
    config = {}
    if config_path is not None:
        import yaml

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        print(f"Loaded configuration from {config_path}")
    else:
        print("No configuration file provided, using default settings.")

    save_path = Path(save_path)
    save_dir = save_path.parent / "anndata_objects"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Define single checkpoint path
    checkpoint_path = save_dir / "features_processed.h5ad"

    # Extract cell_type and embedding_type from config
    cell_type = config.get("cell_type", None) if config else None
    embedding_type = (
        config.get("embedding_type", "cellprofiler") if config else "cellprofiler"
    )

    if not cell_type:
        raise ValueError(
            "cell_type must be specified in config. "
            "Add to config YAML:\n"
            "  cell_type: 'A549'  # or your cell line (e.g., 'HeLa', 'RPE1')"
        )

    # Create anndata object with all features combined
    with timer("TOTAL: Create AnnData object"):
        features_adata = create_adata_object(
            save_path,
            config=config,
            cell_type=cell_type,
            embedding_type=embedding_type,
        )

    # Check if we have multiple channels/reporters to split by
    has_multiple_channels = (
        "channel_mapping" in features_adata.uns
        and len(features_adata.uns["channel_mapping"]) > 1
    )

    # Read aggregation configuration (same as DinoV3)
    agg_config = config.get("aggregation", {})
    guide_config = agg_config.get("guide_level", {})
    gene_config = agg_config.get("gene_level", {})

    if has_multiple_channels:
        # Split by reporter and save separate files for each reporter
        print("\n" + "=" * 60)
        print("SPLITTING BY REPORTER (SAVING BY REPORTER NAME)")
        print("=" * 60)

        with timer("TOTAL: Split AnnData by reporter"):
            reporter_adatas = split_adata_by_reporter(features_adata, verbose=True)

        # Save cell-level, guide-level, and gene-level for each reporter
        for reporter, adata_cell in reporter_adatas.items():
            channel_name = adata_cell.uns["channel"]
            print(
                f"\n--- Processing reporter: {reporter} (channel: {channel_name}) ---"
            )

            # Save cell-level with REPORTER NAME
            cell_path = save_dir / f"features_processed_{reporter}.h5ad"
            with timer(f"Save cell-level for {reporter}"):
                adata_cell.write_h5ad(cell_path)
                print(f"  Saved: {cell_path}")

            # Guide-level aggregation (configurable)
            if guide_config.get(
                "enabled", True
            ):  # Default True for backwards compatibility
                with timer(f"Guide-level aggregation for {reporter}"):
                    # Get embedding settings from config
                    guide_embeddings = guide_config.get("embeddings", {})
                    compute_embeddings = guide_config.get("compute_embeddings", True)

                    # Get embedding parameters
                    n_pca = guide_embeddings.get("n_pca_components", 128)
                    n_neighbors = guide_embeddings.get("n_neighbors", 15)
                    compute_pca = (
                        guide_embeddings.get("pca", True)
                        if compute_embeddings
                        else False
                    )
                    compute_umap = (
                        guide_embeddings.get("umap", True)
                        if compute_embeddings
                        else False
                    )
                    compute_phate = (
                        guide_embeddings.get("phate", True)
                        if compute_embeddings
                        else False
                    )

                    # Validate: UMAP requires PCA
                    if compute_umap and not compute_pca:
                        print(
                            f"  WARNING: UMAP requires PCA. Enabling PCA for {reporter}."
                        )
                        compute_pca = True

                    adata_guide = create_aggregated_embeddings(
                        adata_cell,
                        level="guide",
                        n_pca_components=n_pca,
                        n_neighbors=n_neighbors,
                        compute_pca=compute_pca,
                        compute_umap=compute_umap,
                        compute_phate=compute_phate,
                    )

                    # Save output if enabled
                    if guide_config.get("save_output", True):
                        guide_path = save_dir / f"guide_bulked_{reporter}.h5ad"
                        adata_guide.write_h5ad(guide_path)
                        print(f"  Saved: {guide_path}")
                    else:
                        print(f"  Skipped saving guide-level (save_output=False)")
            else:
                print(f"  Skipped guide-level aggregation (enabled=False)")

            # Gene-level aggregation (configurable)
            if gene_config.get(
                "enabled", True
            ):  # Default True for backwards compatibility
                with timer(f"Gene-level aggregation for {reporter}"):
                    # Get embedding settings from config
                    gene_embeddings = gene_config.get("embeddings", {})
                    compute_embeddings = gene_config.get("compute_embeddings", True)

                    # Get embedding parameters
                    n_pca = gene_embeddings.get("n_pca_components", 128)
                    n_neighbors = gene_embeddings.get("n_neighbors", 15)
                    compute_pca = (
                        gene_embeddings.get("pca", True)
                        if compute_embeddings
                        else False
                    )
                    compute_umap = (
                        gene_embeddings.get("umap", True)
                        if compute_embeddings
                        else False
                    )
                    compute_phate = (
                        gene_embeddings.get("phate", True)
                        if compute_embeddings
                        else False
                    )

                    # Validate: UMAP requires PCA
                    if compute_umap and not compute_pca:
                        print(
                            f"  WARNING: UMAP requires PCA. Enabling PCA for {reporter}."
                        )
                        compute_pca = True

                    adata_gene = create_aggregated_embeddings(
                        adata_cell,
                        level="gene",
                        n_pca_components=n_pca,
                        n_neighbors=n_neighbors,
                        compute_pca=compute_pca,
                        compute_umap=compute_umap,
                        compute_phate=compute_phate,
                    )

                    # Save output if enabled
                    if gene_config.get("save_output", True):
                        gene_path = save_dir / f"gene_bulked_{reporter}.h5ad"
                        adata_gene.write_h5ad(gene_path)
                        print(f"  Saved: {gene_path}")
                    else:
                        print(f"  Skipped saving gene-level (save_output=False)")
            else:
                print(f"  Skipped gene-level aggregation (enabled=False)")

        total_time = time.time() - total_start
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE (SPLIT BY REPORTER, SAVED BY REPORTER)")
        print("=" * 60)
        print(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        print(f"Reporters processed: {len(reporter_adatas)}")
        for reporter, adata in reporter_adatas.items():
            channel_name = adata.uns["channel"]
            print(f"  - {reporter} (channel: {channel_name})")
            print(f"      Cell:  {save_dir}/features_processed_{reporter}.h5ad")
            print(f"      Guide: {save_dir}/guide_bulked_{reporter}.h5ad")
            print(f"      Gene:  {save_dir}/gene_bulked_{reporter}.h5ad")
        print("=" * 60 + "\n")

    else:
        # Original behavior: save combined file
        print("\n(No channel_mapping found - saving combined file)")

        # Save cell-level
        features_adata.write_h5ad(checkpoint_path)
        print(f"Saved initial AnnData object to {checkpoint_path}")

        # Guide-level averaged analysis (configurable)
        if guide_config.get(
            "enabled", True
        ):  # Default True for backwards compatibility
            with timer("TOTAL: Guide-level processing"):
                # Get embedding settings from config
                guide_embeddings = guide_config.get("embeddings", {})
                compute_embeddings = guide_config.get("compute_embeddings", True)

                # Get embedding parameters
                n_pca = guide_embeddings.get("n_pca_components", 128)
                n_neighbors = guide_embeddings.get("n_neighbors", 15)
                compute_pca = (
                    guide_embeddings.get("pca", True) if compute_embeddings else False
                )
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

                embeddings_guide_bulk_ad = create_aggregated_embeddings(
                    features_adata,
                    level="guide",
                    n_pca_components=n_pca,
                    n_neighbors=n_neighbors,
                    compute_pca=compute_pca,
                    compute_umap=compute_umap,
                    compute_phate=compute_phate,
                )

                # Save output if enabled
                if guide_config.get("save_output", True):
                    guide_avg_path = save_dir / "guide_bulked.h5ad"
                    embeddings_guide_bulk_ad.write_h5ad(guide_avg_path)

                    # Build embedding info for display
                    embeddings_computed = []
                    if compute_pca:
                        embeddings_computed.append("PCA")
                    if compute_umap:
                        embeddings_computed.append("UMAP")
                    if compute_phate:
                        embeddings_computed.append("PHATE")
                    embeddings_str = (
                        "+".join(embeddings_computed) if embeddings_computed else "none"
                    )

                    print(
                        f"Saved guide-bulked analysis to {guide_avg_path} (embeddings: {embeddings_str})"
                    )
                else:
                    print("Skipped saving guide-level (save_output=False)")
        else:
            print("Skipped guide-level aggregation (enabled=False)")

        # Gene-level averaged analysis (configurable)
        if gene_config.get("enabled", True):  # Default True for backwards compatibility
            with timer("TOTAL: Gene-level processing"):
                # Get embedding settings from config
                gene_embeddings = gene_config.get("embeddings", {})
                compute_embeddings = gene_config.get("compute_embeddings", True)

                # Get embedding parameters
                n_pca = gene_embeddings.get("n_pca_components", 128)
                n_neighbors = gene_embeddings.get("n_neighbors", 15)
                compute_pca = (
                    gene_embeddings.get("pca", True) if compute_embeddings else False
                )
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

                embeddings_gene_avg_ad = create_aggregated_embeddings(
                    features_adata,
                    level="gene",
                    n_pca_components=n_pca,
                    n_neighbors=n_neighbors,
                    compute_pca=compute_pca,
                    compute_umap=compute_umap,
                    compute_phate=compute_phate,
                )

                # Save output if enabled
                if gene_config.get("save_output", True):
                    gene_avg_path = save_dir / "gene_bulked.h5ad"
                    embeddings_gene_avg_ad.write_h5ad(gene_avg_path)

                    # Build embedding info for display
                    embeddings_computed = []
                    if compute_pca:
                        embeddings_computed.append("PCA")
                    if compute_umap:
                        embeddings_computed.append("UMAP")
                    if compute_phate:
                        embeddings_computed.append("PHATE")
                    embeddings_str = (
                        "+".join(embeddings_computed) if embeddings_computed else "none"
                    )

                    print(
                        f"Saved gene-bulked analysis to {gene_avg_path} (embeddings: {embeddings_str})"
                    )
                else:
                    print("Skipped saving gene-level (save_output=False)")
        else:
            print("Skipped gene-level aggregation (enabled=False)")

        total_time = time.time() - total_start
        print("\n" + "=" * 60)
        print(
            f"Pipeline completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)"
        )
        print(f"Cell-level output: {checkpoint_path} (contains raw features)")

        # Show guide/gene outputs only if they were created
        if guide_config.get("enabled", True) and guide_config.get("save_output", True):
            print(f"Guide-bulked output: {save_dir / 'guide_bulked.h5ad'}")
        if gene_config.get("enabled", True) and gene_config.get("save_output", True):
            print(f"Gene-bulked output: {save_dir / 'gene_bulked.h5ad'}")

        print("=" * 60 + "\n")

    return features_adata


def _build_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Process features.")
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Path to the CSV file containing CellProfiler features.",
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

    process(
        args.save_path,
        config_path=args.config_path,
    )


if __name__ == "__main__":
    main()
