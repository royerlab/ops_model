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


def create_adata_object(save_path: str, config: dict = None) -> ad.AnnData:
    """
    Create AnnData object from CellProfiler features CSV

    Args:
        save_path: Path to CSV file
        config: Configuration dictionary
                - 'cell-profiler': bool
                - 'normalize_features': bool
                - 'use_reporter_names': bool - Replace channel names with reporter names
    """
    with timer("Reading CSV"):
        # Read CSV - let pandas infer dtypes initially
        features = pd.read_csv(save_path, low_memory=False)

    print(f"Dataset shape: {features.shape}")

    # Extract config settings
    use_reporter_names = (
        config["processing"].get("use_reporter_names", False) if config else False
    )

    # Replace channel names with reporter names if requested
    if use_reporter_names:
        with timer("Replacing channel names with reporter names"):
            from ops_model.data.feature_metadata import FeatureMetadata

            # Check which experiments are in the dataset
            unique_experiments = features["experiment"].unique()

            if len(unique_experiments) != 1:
                raise ValueError(
                    f"Multi-experiment datasets not yet supported for reporter name replacement. "
                    f"Found experiments: {unique_experiments}"
                )

            experiment = unique_experiments[0]
            feature_meta = FeatureMetadata()

            # Rename feature columns (excluding metadata columns)
            feature_cols = [
                col for col in features.columns if col not in NONFEATURE_COLUMNS
            ]

            renamed_cols = {}
            channel_mapping = {}
            for col in feature_cols:
                new_name = feature_meta.replace_channel_in_feature_name(col, experiment)
                if new_name != col:  # Only track if actually changed
                    renamed_cols[col] = new_name

                    # Extract channel names for metadata tracking
                    # Parse out channels from old column name for mapping
                    if col.startswith("single_object_"):
                        parts = col.split("_", 3)
                        if len(parts) >= 3:
                            old_channel = parts[2]
                            # Get reporter from new name
                            new_parts = new_name.split("_", 3)
                            if len(new_parts) >= 3:
                                reporter = new_parts[2]
                                if old_channel not in channel_mapping:
                                    channel_mapping[old_channel] = reporter

            features = features.rename(columns=renamed_cols)
            print(f"Renamed {len(renamed_cols)} feature columns with reporter names")
            if renamed_cols:
                example = list(renamed_cols.items())[0]
                print(f"Example: '{example[0]}' -> '{example[1]}'")

    with timer("Extracting labels"):
        gene_strs = np.asarray(features["label_str"].values)
        gene_ints = np.asarray(features["label_int"].values)
        sgRNA_ids = np.asarray(features["sgRNA"].values)
        well_id = np.asarray(features["well"].values)
        try:
            x_pos = np.asarray(features["x_position"].values)
            y_pos = np.asarray(features["y_position"].values)
        except KeyError:
            print("x_position/y_position columns not found, skipping.")
            NONFEATURE_COLUMNS.remove("x_position")
            NONFEATURE_COLUMNS.remove("y_position")
        features = features.drop(columns=NONFEATURE_COLUMNS)

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

        # Filter rows with too many NaNs and track which rows are kept
        num_nan_features_per_row = features.isna().sum(axis=1)
        good_rows_mask = (
            num_nan_features_per_row
            <= config["processing"].get("max_nan_features_per_cell", 0)
            if config
            else 0
        )
        features = features[good_rows_mask]

        # Update metadata arrays to match filtered rows
        gene_strs = gene_strs[good_rows_mask]
        gene_ints = gene_ints[good_rows_mask]
        sgRNA_ids = sgRNA_ids[good_rows_mask]
        well_id = well_id[good_rows_mask]
        try:
            x_pos = x_pos[good_rows_mask]
            y_pos = y_pos[good_rows_mask]
        except NameError:
            pass  # x_pos/y_pos not defined if columns weren't in CSV

        print(f"Kept {features.shape[0]} rows after filtering NaN rows")

        # Drop columns with any NaN values
        cols_with_nans = features.columns[features.isna().any()]
        features = features.drop(columns=cols_with_nans)
        print(f"Dropped {len(cols_with_nans)} columns with NaN values")

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
        if "label_str" in features_norm.columns:
            features_norm = features_norm.drop(columns=["label_str"])
        adata = ad.AnnData(features_norm)
        adata.obs["label_str"] = gene_strs
        adata.obs["label_int"] = gene_ints
        adata.obs["sgRNA"] = sgRNA_ids
        adata.obs["well"] = well_id
        try:
            adata.obs["x_position"] = x_pos
            adata.obs["y_position"] = y_pos
        except NameError:
            pass
        adata.var_names = features_norm.columns

        # Store channel mapping metadata if reporter names were used
        if use_reporter_names:
            adata.uns["channel_mapping"] = channel_mapping
            adata.uns["experiment"] = experiment
            print(f"Stored channel mapping in adata.uns: {channel_mapping}")

    return adata


def split_adata_by_reporter(adata: ad.AnnData, verbose: bool = True) -> dict:
    """
    Split AnnData object by reporter/biological signal, return by reporter name.

    Extracts channel_mapping from adata.uns and creates separate AnnData
    objects for each reporter. Features are assigned by string matching:
    - Features containing reporter name → assigned to that reporter
    - Cell-level features (starting with 'cell_') → duplicated across all reporters
    - Colocalization features (containing multiple reporters) → duplicated in each

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

    # Identify cell-level features (shared across all channels)
    cell_features = [f for f in var_names if f.startswith("cell_")]

    if verbose and cell_features:
        print(f"Cell-level features (duplicated across all): {len(cell_features)}")

    reporter_adatas = {}  # Key by reporter name for file naming consistency

    for reporter in reporters:
        # Find features containing this reporter name
        reporter_features = [f for f in var_names if reporter in f]

        # Combine with cell-level features
        all_features = sorted(set(reporter_features + cell_features))

        if len(all_features) == 0:
            print(f"  WARNING: No features found for reporter '{reporter}', skipping")
            continue

        # Create subset AnnData
        feature_indices = [var_names.index(f) for f in all_features]
        adata_subset = adata[:, feature_indices].copy()

        # Store metadata - reporter is primary, channel kept for reference
        channel_name = reporter_to_channel[reporter]
        adata_subset.uns["reporter"] = reporter  # Primary: reporter/marker name
        adata_subset.uns["channel"] = channel_name  # Reference: original channel name
        adata_subset.uns["channel_mapping"] = channel_mapping

        # Key by reporter name (ensures consistent feature names during pooling)
        reporter_adatas[reporter] = adata_subset

        if verbose:
            coloc_count = sum(1 for f in all_features if f.startswith("coloc_"))
            print(
                f"  {reporter} (channel: {channel_name}): {len(all_features)} features ({coloc_count} colocalization)"
            )

    return reporter_adatas


def process(save_path: str, config_path: str = None):
    """
    Process CellProfiler features through the full pipeline

    If reporter names are used (use_reporter_names=True in config), this will:
    1. Create combined AnnData object
    2. Split by reporter/biological signal
    3. Save separate files for each reporter (keyed by reporter name for feature consistency):
       - features_processed_{reporter}.h5ad (cell-level)
       - guide_bulked_umap_{reporter}.h5ad
       - gene_bulked_umap_{reporter}.h5ad

    Args:
        save_path: Path to features CSV
        config_path: Path to configuration YAML file
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

    # Create anndata object with all features combined
    with timer("TOTAL: Create AnnData object"):
        features_adata = create_adata_object(save_path, config=config)

    # Check if reporter names were used (enables splitting)
    use_reporter_names = "channel_mapping" in features_adata.uns

    if use_reporter_names:
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

            # Guide-level aggregation
            with timer(f"Guide-level aggregation for {reporter}"):
                adata_guide = create_aggregated_embeddings(
                    adata_cell,
                    level="guide",
                    n_pca_components=128,
                    n_neighbors=15,
                )
                guide_path = save_dir / f"guide_bulked_umap_{reporter}.h5ad"
                adata_guide.write_h5ad(guide_path)
                print(f"  Saved: {guide_path}")

            # Gene-level aggregation
            with timer(f"Gene-level aggregation for {reporter}"):
                adata_gene = create_aggregated_embeddings(
                    adata_cell,
                    level="gene",
                    n_pca_components=128,
                    n_neighbors=15,
                )
                gene_path = save_dir / f"gene_bulked_umap_{reporter}.h5ad"
                adata_gene.write_h5ad(gene_path)
                print(f"  Saved: {gene_path}")

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
            print(f"      Guide: {save_dir}/guide_bulked_umap_{reporter}.h5ad")
            print(f"      Gene:  {save_dir}/gene_bulked_umap_{reporter}.h5ad")
        print("=" * 60 + "\n")

    else:
        # Original behavior: save combined file
        print("\n(No channel_mapping found - saving combined file)")

        # Save cell-level
        features_adata.write_h5ad(checkpoint_path)
        print(f"Saved initial AnnData object to {checkpoint_path}")

        # Guide-level averaged analysis
        with timer("TOTAL: Guide-averaged UMAP analysis"):
            embeddings_guide_bulk_ad = create_aggregated_embeddings(
                features_adata,
                level="guide",
                n_pca_components=128,
                n_neighbors=15,
            )
            guide_avg_path = save_dir / "guide_bulked_umap.h5ad"
            embeddings_guide_bulk_ad.write_h5ad(guide_avg_path)
            print(f"Saved guide-bulked UMAP analysis to {guide_avg_path}")

        # Gene-level averaged analysis
        with timer("TOTAL: Gene-averaged UMAP analysis"):
            embeddings_gene_avg_ad = create_aggregated_embeddings(
                features_adata,
                level="gene",
                n_pca_components=128,
                n_neighbors=15,
            )
            gene_avg_path = save_dir / "gene_bulked_umap.h5ad"
            embeddings_gene_avg_ad.write_h5ad(gene_avg_path)
            print(f"Saved gene-bulked UMAP analysis to {gene_avg_path}")

        total_time = time.time() - total_start
        print("\n" + "=" * 60)
        print(
            f"Pipeline completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)"
        )
        print(f"Cell-level output: {checkpoint_path} (contains raw features)")
        print(
            f"Guide-bulked output: {guide_avg_path} (averaged full features with PCA/UMAP)"
        )
        print(
            f"Gene-bulked output: {gene_avg_path} (averaged full features with PCA/UMAP)"
        )
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
