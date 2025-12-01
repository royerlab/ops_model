# %%
from tqdm import tqdm
from pathlib import Path
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from ops_model.data.paths import OpsPaths

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


def center_scale_fast(
    df: pd.DataFrame,
    on_controls: bool = False,
    control_column: str = "label_str",
    control_gene: str = "NTC",
) -> pd.DataFrame:
    """
    Optimized version using numpy operations and avoiding copies
    Can be 5-10x faster for large datasets
    """
    # Extract labels without copy
    gene_labels = df[control_column].values

    # Get feature columns (all except control_column)
    feature_cols = [col for col in df.columns if col != control_column]

    # Work with numpy arrays directly for speed
    features_array = df[feature_cols].values

    if on_controls:
        # Boolean mask for control genes
        control_mask = gene_labels == control_gene
        ref_array = features_array[control_mask]
    else:
        ref_array = features_array

    # Compute mean and std on numpy arrays (much faster)
    # Using float64 for precision in statistics
    means = np.mean(ref_array, axis=0, dtype=np.float64)
    stds = np.std(
        ref_array, axis=0, dtype=np.float64, ddof=1
    )  # ddof=1 for sample std like pandas

    # Avoid division by zero
    stds[stds == 0] = 1.0

    # Normalize in-place on numpy array
    normalized = (features_array - means) / stds

    # Create result DataFrame efficiently
    result = pd.DataFrame(normalized, columns=feature_cols, index=df.index)
    result.insert(0, control_column, gene_labels)

    return result


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
        features.groupby("label_str")
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


def create_adata_object(save_path: str) -> ad.AnnData:
    """
    Create AnnData object from CellProfiler features CSV

    Args:
        save_path: Path to CSV file
    """
    with timer("Reading CSV"):
        # Read CSV - let pandas infer dtypes initially
        features = pd.read_csv(save_path, low_memory=False)

    print(f"Dataset shape: {features.shape}")

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
        # Convert any string representations of arrays to float values
        for col in features.columns:
            if col == "label_str":
                continue
            # Check if column contains string array representations
            if features[col].dtype == "object":
                features[col] = features[col].apply(_convert_array_strings_to_float)
                # Convert to numeric, coercing errors to NaN
                features[col] = pd.to_numeric(features[col], errors="coerce")

    with timer("Converting numeric columns to float32"):
        # Convert numeric columns to float32 for memory efficiency
        # This can halve memory usage compared to float64
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != "label_str"]
        features[numeric_cols] = features[numeric_cols].astype("float32")
        print(f"Converted {len(numeric_cols)} numeric columns to float32")

    with timer("Dropping constant columns and nans"):
        features = features.dropna(subset=["single_object_Phase2D_cell_Area"])
        good_rows = features.index[features.isna().sum(axis=1) < 19]

        gene_strs = gene_strs[good_rows]
        gene_ints = gene_ints[good_rows]
        sgRNA_ids = sgRNA_ids[good_rows]
        well_id = well_id[good_rows]

        features = features[features.isna().sum(axis=1) <= 19]
        cols_with_nans = features.columns[features.isna().any()]
        features = features.drop(columns=cols_with_nans)
        print(f"Dropped {len(cols_with_nans)} columns with NaN values")

        constant_cols = features.columns[features.nunique(dropna=False) == 1]
        features = features.drop(columns=constant_cols)
        print(f"Dropped {len(constant_cols)} constant columns")

    with timer("Center-scaling features"):
        features_norm = center_scale_fast(features)

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

    return adata


def process(
    save_path: str, experiment: str, plot_all_genes: bool = False, config: dict = None
):
    """
    Process CellProfiler features through the full pipeline

    Args:
        save_path: Path to features CSV
        plot_all_genes: Whether to plot individual gene UMAPs
        config: Configuration dictionary
    """
    print("\n" + "=" * 60)
    print("Starting feature processing pipeline")
    print("=" * 60 + "\n")

    total_start = time.time()

    save_path = Path(save_path)
    save_dir = save_path.parent / "anndata_objects"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Define single checkpoint path
    checkpoint_path = save_dir / "features_processed.h5ad"

    # Create anndata object and center-scale features
    with timer("TOTAL: Create AnnData object"):
        features_adata = create_adata_object(save_path)

    # Save immediately after creation
    features_adata.write_h5ad(checkpoint_path)
    print(f"Saved initial AnnData object to {checkpoint_path}")

    # PCA embedding of features
    with timer("TOTAL: PCA embedding"):
        features_adata = pca_embed(features_adata, n_components=128, variance_plot=True)

    # Update checkpoint with PCA
    features_adata.write_h5ad(checkpoint_path)
    print(f"Updated AnnData object with PCA to {checkpoint_path}")

    # Compute UMAP on individual cells using PCA embeddings
    with timer("TOTAL: Compute neighbors and UMAP on cell-level data"):
        sc.pp.neighbors(features_adata, n_pcs=128, n_neighbors=15, metric="cosine")
        sc.tl.umap(features_adata, min_dist=0.1)

    # Update checkpoint with UMAP
    features_adata.write_h5ad(checkpoint_path)
    print(f"Updated AnnData object with UMAP to {checkpoint_path}")

    # Optional: Guide-level averaged analysis for visualization
    with timer("TOTAL: Guide-averaged UMAP analysis"):
        # Center-scale the PCA embeddings
        embeddings_df = pd.DataFrame(features_adata.obsm["X_pca"])
        # embeddings_df['label_str'] = features_adata.obs['label_str'].values
        embeddings_df["sgRNA"] = features_adata.obs["sgRNA"].values
        if config["normalize_PCA_embeddings"]:
            embeddings_norm_df = center_scale_fast(embeddings_df, on_controls=True)
        else:
            embeddings_norm_df = embeddings_df

        # bulk features per guide
        embeddings_norm_df_bulk = embeddings_norm_df.groupby("sgRNA").mean()
        embeddings_guide_bulk_ad = ad.AnnData(embeddings_norm_df_bulk)
        embeddings_guide_bulk_ad.obs["sgRNA"] = embeddings_guide_bulk_ad.obs_names
        sc.pp.neighbors(
            embeddings_guide_bulk_ad, n_pcs=0, n_neighbors=15, metric="cosine"
        )
        sc.tl.umap(embeddings_guide_bulk_ad, min_dist=0.1)

        guide_avg_path = save_dir / "guide_bulked_umap.h5ad"
        embeddings_guide_bulk_ad.write_h5ad(guide_avg_path)
        print(f"Saved guide-bulked UMAP analysis to {guide_avg_path}")

    total_time = time.time() - total_start
    print("\n" + "=" * 60)
    print(
        f"Pipeline completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)"
    )
    print(f"Main output: {checkpoint_path} (contains raw features + PCA + UMAP)")
    print(f"Guide-bulked output: {guide_avg_path} (for visualization)")
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
        "--plot_all_genes",
        action="store_true",
        help="Whether to plot UMAPs for all genes individually.",
    )

    parser.add_argument(
        "--experiment", type=str, required=True, help="Experiment name."
    )
    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    config = {
        "normalize_PCA_embeddings": False,
    }

    process(
        args.save_path,
        plot_all_genes=args.plot_all_genes,
        config=config,
        experiment=args.experiment,
    )


if __name__ == "__main__":
    main()
