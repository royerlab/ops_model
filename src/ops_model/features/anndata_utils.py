"""
Utilities for combining and managing multiple AnnData objects.

This module provides functions for concatenating AnnData objects from different
experiments, feature types, or datasets while preserving metadata and enabling
cross-dataset analysis.

Key functions:
- concatenate_anndata_objects: Simple concatenation with batch tracking
- recompute_embeddings: Compute shared PCA/UMAP on combined data
- aggregate_to_level: Aggregate cell-level data to guide/gene level
- compute_embeddings: Compute PCA/UMAP/PHATE on any AnnData object
- create_aggregated_embeddings: Complete pipeline (aggregate + compute embeddings)
- load_multiple_experiments: Batch loading helper
"""

from pathlib import Path
from typing import List, Union, Optional, Literal, Tuple, Dict, Any
import warnings

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import scanpy.external as sce
import matplotlib.pyplot as plt

from ops_model.data.feature_metadata import FeatureMetadata


def normalize_adata_zscore(
    adata: ad.AnnData,
    normalize_on_controls: bool = False,
    control_column: str = "label_str",
    control_gene: str = "NTC",
) -> ad.AnnData:
    """
    Z-score normalize AnnData features in-place: (x - mean) / std

    Optimized version using numpy operations for speed.
    Can normalize using statistics from all cells or control cells only.

    Args:
        adata: AnnData object with raw features in .X
        normalize_on_controls: If True, compute mean/std from control cells only,
                               then apply to all cells. If False, compute from all cells.
        control_column: Column in .obs containing gene labels (default: "label_str")
        control_gene: Gene name for control cells (default: "NTC")

    Returns:
        AnnData object with normalized features (modifies .X in place)

    Example:
        >>> adata = ad.read_h5ad("features_processed.h5ad")  # Raw features
        >>> adata = normalize_adata_zscore(adata)  # Normalize on all cells
        >>> print(adata.X.mean(axis=0))  # Should be ~0
        >>> print(adata.X.std(axis=0))   # Should be ~1

        >>> # Normalize using control statistics
        >>> adata = normalize_adata_zscore(
        ...     adata,
        ...     normalize_on_controls=True,
        ...     control_gene="NTC"
        ... )

    Note:
        This modifies adata.X in place and also returns the object.
        Adapted from evaluate_cp.center_scale_fast() for use with AnnData objects.
    """
    # Get feature matrix
    features_array = adata.X

    if normalize_on_controls:
        # Extract control gene labels
        if control_column not in adata.obs.columns:
            raise ValueError(f"Column '{control_column}' not found in adata.obs")

        gene_labels = adata.obs[control_column].values
        control_mask = gene_labels == control_gene

        if not control_mask.any():
            raise ValueError(
                f"No control cells found with {control_column}='{control_gene}'. "
                f"Available values: {sorted(adata.obs[control_column].unique())}"
            )

        ref_array = features_array[control_mask]
        print(
            f"  Normalizing using {control_mask.sum()} control cells ({control_gene})"
        )
    else:
        ref_array = features_array
        print(f"  Normalizing using all {len(features_array)} cells")

    # Compute mean and std on numpy arrays (using float64 for precision in statistics)
    means = np.mean(ref_array, axis=0, dtype=np.float64)
    stds = np.std(ref_array, axis=0, dtype=np.float64, ddof=1)  # ddof=1 for sample std

    # Avoid division by zero
    stds[stds == 0] = 1.0

    # Normalize in-place
    adata.X = (features_array - means) / stds

    print(f"  Features normalized to z-scores (mean=0, std=1)")

    return adata


def pca_embed(
    adata: ad.AnnData,
    n_components: int = 128,
    variance_plot: bool = False,
) -> ad.AnnData:
    """
    Compute PCA on AnnData object.

    Args:
        adata: AnnData object
        n_components: Number of PCA components
        variance_plot: Whether to plot variance ratio

    Returns:
        AnnData object with PCA in .obsm['X_pca']
    """
    sc.tl.pca(adata, n_comps=n_components)
    if variance_plot:
        sc.pl.pca_variance_ratio(adata, n_pcs=100, log=False, save=False)
        plt.figure()

    return adata


def concatenate_anndata_objects(
    adata_paths: List[Union[str, Path]],
    batch_key: str = "batch",
    join: Literal["inner", "outer"] = "inner",
    index_unique: Optional[str] = "-",
) -> ad.AnnData:
    """
    Concatenate multiple AnnData objects with proper metadata tracking.

    This function combines AnnData objects from different sources (experiments,
    channels, feature types) while preserving batch information for downstream
    analysis. It does NOT recompute embeddings - use recompute_embeddings()
    for that.

    Args:
        adata_paths: List of paths to .h5ad files
        batch_key: Column name in .obs to track dataset origin
                   Default: "batch"
        join: How to handle different feature sets
              - "inner": Keep only common features (default)
              - "outer": Keep all features, fill missing with 0
        index_unique: Separator for making obs_names unique across batches
                      Default: "-" (e.g., "cell1" becomes "cell1-0", "cell1-1")
                      Set to None to keep original indices (may cause conflicts)

    Returns:
        Combined AnnData object with batch metadata in .obs[batch_key]

    Example:
        >>> adata_combined = concatenate_anndata_objects([
        ...     "ops0089/anndata_objects/features_processed.h5ad",
        ...     "ops0084/anndata_objects/features_processed.h5ad",
        ... ], batch_key="experiment")
        >>> print(adata_combined.obs["experiment"].value_counts())

    Notes:
        - Original .obsm (PCA/UMAP) are preserved but represent different spaces
        - Use recompute_embeddings() to get shared embedding space
        - .var_names must be compatible (same features or use join="outer")
    """
    # Load all AnnData objects
    print(f"Loading {len(adata_paths)} AnnData objects...")
    adata_objects = []

    for path in adata_paths:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"AnnData file not found: {path}")

        adata = ad.read_h5ad(path)
        adata_objects.append(adata)
        print(
            f"  Loaded {path.name}: {adata.shape[0]} cells × {adata.shape[1]} features"
        )

    # Add batch information to each object
    print(f"\nAdding batch information to .obs['{batch_key}']...")

    # Extract batch IDs - go up enough levels to get experiment name
    # Typical path: .../ops0089_20251119/3-assembly/dino_features/anndata_objects/features_processed.h5ad
    batch_ids = []
    for path in adata_paths:
        path = Path(path)
        # Go up 4 levels: file -> anndata_objects -> dino_features -> 3-assembly -> experiment
        try:
            batch_id = path.parents[3].name  # Get experiment name
        except IndexError:
            # Fallback: use filename stem if path is not deep enough
            batch_id = path.stem
        batch_ids.append(batch_id)

    for i, (adata, batch_id) in enumerate(zip(adata_objects, batch_ids)):
        adata.obs[batch_key] = batch_id
        print(f"  Batch {i}: {batch_id} ({adata.shape[0]} cells)")

    # Concatenate
    print(f"\nConcatenating with join='{join}'...")
    adata_combined = ad.concat(
        adata_objects,
        join=join,
        merge="same",  # Merge .uns, .varm, .obsm with same keys
        index_unique=index_unique,
        label=batch_key,  # Store batch labels in .obs
        keys=batch_ids,
    )

    print(
        f"Combined shape: {adata_combined.shape[0]} cells × {adata_combined.shape[1]} features"
    )
    print(f"Batch counts:\n{adata_combined.obs[batch_key].value_counts()}")

    # Warn if features differ
    if join == "inner":
        n_features_per_batch = [adata.shape[1] for adata in adata_objects]
        if len(set(n_features_per_batch)) > 1:
            warnings.warn(
                f"Feature counts differ across batches: {n_features_per_batch}. "
                f"Using join='inner', kept only {adata_combined.shape[1]} common features."
            )

    return adata_combined


def recompute_embeddings(
    adata: ad.AnnData,
    n_pca_components: int = 128,
    n_umap_neighbors: int = 15,
    compute_pca: bool = True,
    compute_umap: bool = True,
    compute_phate: bool = True,
    use_existing_pca: bool = False,
) -> ad.AnnData:
    """
    Recompute PCA, UMAP, and PHATE on combined AnnData object.

    After concatenating multiple datasets, their individual PCA/UMAP/PHATE embeddings
    are in different coordinate systems. This function computes a shared
    embedding space across all datasets.

    Args:
        adata: AnnData object (typically from concatenate_anndata_objects)
        n_pca_components: Number of PCA components
                          Will be adjusted if n_samples is smaller
        n_umap_neighbors: Number of neighbors for UMAP and PHATE
                          Will be adjusted if n_samples is smaller
        compute_pca: Whether to compute PCA
                     Set False if you want to use raw features for UMAP/PHATE
        compute_umap: Whether to compute UMAP
        compute_phate: Whether to compute PHATE
        use_existing_pca: If True and "X_pca" exists, use it instead of recomputing
                          Useful when concatenating already-aggregated data

    Returns:
        AnnData object with updated .obsm["X_pca"], .obsm["X_umap"], and/or .obsm["X_phate"]

    Example:
        >>> adata_combined = concatenate_anndata_objects(paths)
        >>> adata_combined = recompute_embeddings(adata_combined)
        >>> sc.pl.umap(adata_combined, color=["batch", "label_str"])
        >>> sc.pl.embedding(adata_combined, basis='phate', color=["batch", "label_str"])

    Notes:
        - This modifies the input AnnData object in place AND returns it
        - Old PCA/UMAP/PHATE (if present) are overwritten
        - Use batch-aware methods (Harmony, Combat) for batch correction
    """
    n_samples = adata.shape[0]
    print(f"\nRecomputing embeddings for {n_samples} cells...")

    # Adjust PCA components to valid range
    # For arpack solver, must be strictly less than min(n_samples, n_features)
    max_pca_components = min(n_samples - 1, adata.shape[1] - 1)
    n_pca_components = min(n_pca_components, max_pca_components)

    if compute_pca and not use_existing_pca:
        print(f"Computing PCA with {n_pca_components} components...")
        adata = pca_embed(adata, n_components=n_pca_components, variance_plot=False)
        print(f"  PCA computed: {adata.obsm['X_pca'].shape}")

    if compute_umap:
        # Adjust neighbors to valid range
        n_neighbors = min(n_umap_neighbors, n_samples - 1)

        if use_existing_pca and "X_pca" in adata.obsm.keys():
            print(f"Using existing PCA with {adata.obsm['X_pca'].shape[1]} components")
            n_pcs = adata.obsm["X_pca"].shape[1]
        elif compute_pca or "X_pca" in adata.obsm.keys():
            n_pcs = n_pca_components
        else:
            # Use raw features
            n_pcs = 0
            print("Computing UMAP on raw features (no PCA)...")

        if n_neighbors >= 2:
            print(f"Computing neighbors (n_neighbors={n_neighbors}, n_pcs={n_pcs})...")
            sc.pp.neighbors(
                adata, n_pcs=n_pcs, n_neighbors=n_neighbors, metric="cosine"
            )

            print("Computing UMAP...")
            sc.tl.umap(adata, min_dist=0.1)
            print(f"  UMAP computed: {adata.obsm['X_umap'].shape}")

            # Compute PHATE if requested
            if compute_phate:
                print("Computing PHATE...")
                # Suppress warnings from PHATE and its dependencies (graphtools)
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", category=FutureWarning, module="phate"
                    )
                    warnings.filterwarnings(
                        "ignore", category=UserWarning, module="graphtools"
                    )
                    warnings.filterwarnings(
                        "ignore", category=RuntimeWarning, module="phate"
                    )
                    sce.tl.phate(
                        adata,
                        n_components=2,
                        k=n_neighbors,
                        n_pca=n_pcs,
                        knn_dist="cosine",
                        t="auto",
                    )
                print(f"  PHATE computed: {adata.obsm['X_phate'].shape}")
        else:
            warnings.warn(f"Too few samples ({n_samples}) for UMAP computation")

    return adata


def aggregate_to_level(
    adata: ad.AnnData,
    level: Literal["guide", "gene"],
    method: str = "mean",
    preserve_batch_info: bool = True,
    subsample_controls: bool = False,
    control_gene: str = "NTC",
    control_group_size: int = 4,
    random_seed: Optional[int] = None,
) -> ad.AnnData:
    """
    Aggregate cell-level data to guide or gene level.

    Args:
        adata: Cell-level AnnData object
        level: Aggregation level ("guide" or "gene")
        method: Aggregation method ("mean", "median")
        preserve_batch_info: Keep batch metadata in aggregated object
        subsample_controls: If True, split control gene guides into random groups
                           at gene level. Only applies when level="gene".
        control_gene: Gene name to subsample (default: "NTC")
        control_group_size: Number of guides per control group (default: 4)
        random_seed: Random seed for reproducible control grouping

    Returns:
        Aggregated AnnData object (no embeddings computed yet)

    Example:
        >>> adata_cell = ad.read_h5ad("features_processed.h5ad")
        >>> adata_guide = aggregate_to_level(adata_cell, "guide")
        >>> print(adata_guide.shape)  # (n_guides, n_features)

        >>> # Subsample NTC controls into groups of 4 guides
        >>> adata_gene = aggregate_to_level(
        ...     adata_cell,
        ...     level="gene",
        ...     subsample_controls=True,
        ...     control_group_size=4,
        ...     random_seed=42
        ... )
        >>> # NTC with 200 guides becomes NTC_1, NTC_2, ..., NTC_50
    """
    # Determine grouping column
    group_col = "sgRNA" if level == "guide" else "label_str"

    if group_col not in adata.obs.columns:
        raise ValueError(f"Column '{group_col}' not found in adata.obs")

    print(f"Aggregating to {level} level...")

    # Handle control gene subsampling if requested
    if subsample_controls and level == "gene":
        # Validate that sgRNA column exists (needed for grouping guides)
        if "sgRNA" not in adata.obs.columns:
            raise ValueError(
                "Control subsampling requires 'sgRNA' column in adata.obs. "
                "Cannot group guides without guide identifiers."
            )

        # Check if control gene exists
        if control_gene not in adata.obs["label_str"].values:
            raise ValueError(
                f"Control gene '{control_gene}' not found in data. "
                f"Available genes: {sorted(adata.obs['label_str'].unique())}"
            )

        # Get all unique guides for the control gene
        control_mask = adata.obs["label_str"] == control_gene
        control_guides = adata.obs.loc[control_mask, "sgRNA"].unique()
        n_control_guides = len(control_guides)

        print(
            f"  Subsampling {control_gene}: {n_control_guides} guides → groups of {control_group_size}"
        )

        # Shuffle guides
        rng = np.random.RandomState(random_seed)
        shuffled_guides = control_guides.copy()
        rng.shuffle(shuffled_guides)

        # Create guide-to-group mapping
        guide_to_group = {}
        group_num = 1
        for i in range(0, len(shuffled_guides), control_group_size):
            group_guides = shuffled_guides[i : i + control_group_size]
            group_label = f"{control_gene}_{group_num}"
            for guide in group_guides:
                guide_to_group[guide] = group_label
            group_num += 1

        n_groups = group_num - 1
        print(f"    Created {n_groups} groups")

        # Copy adata to avoid modifying original
        adata = adata.copy()

        # Remap control gene cells to their group labels
        def remap_label(row):
            if row["label_str"] == control_gene:
                return guide_to_group[row["sgRNA"]]
            return row["label_str"]

        adata.obs["label_str"] = adata.obs.apply(remap_label, axis=1)

    # Extract features for aggregation
    feature_cols = list(adata.var_names)

    # Check for duplicate var_names
    # unique_var_names = set(adata.var_names)
    # if len(unique_var_names) < len(adata.var_names):
    #     n_duplicates = len(adata.var_names) - len(unique_var_names)
    #     # Show some examples
    #     from collections import Counter
    #     var_counts = Counter(adata.var_names)
    #     duplicates = {k: v for k, v in var_counts.items() if v > 1}
    #     print(f"  DEBUG: Example duplicates (showing first 5): {list(duplicates.items())[:5]}")

    features_df = pd.DataFrame(adata.X, columns=feature_cols)
    features_df[group_col] = adata.obs[group_col].values

    # Preserve label_str when aggregating to guide level (needed for guide→gene aggregation later)
    preserve_label_str = False
    label_str_values = None
    if level == "guide" and "label_str" in adata.obs.columns:
        label_str_values = adata.obs["label_str"].values
        preserve_label_str = True

    # Add batch column if preserving batch info
    if preserve_batch_info and "batch" in adata.obs.columns:
        features_df["batch"] = adata.obs["batch"].values
        # Group by both level and batch
        group_cols = [group_col, "batch"]
    else:
        group_cols = group_col

    # Aggregate features
    # Use numpy array aggregation to avoid pandas column name issues with duplicates
    # Extract feature data into numpy array upfront for fast indexing

    # Get unique group values and their indices
    if isinstance(group_cols, list):
        # Multi-column grouping (with batch)
        group_df = features_df[group_cols]
        groups = group_df.groupby(group_cols, observed=False).groups
    else:
        # Single column grouping
        groups = features_df.groupby(group_cols, observed=False).groups

    # Pre-allocate result array
    n_groups = len(groups)
    n_features = len(feature_cols)
    X_agg = np.zeros((n_groups, n_features))
    group_keys = []

    # Extract feature data into numpy array ONCE (much faster than repeated pandas indexing)
    # Use adata.X directly to avoid pandas column name issues
    X_features = adata.X  # Already a numpy array from AnnData

    # Aggregate each group using numpy indexing
    for i, (group_key, indices) in enumerate(groups.items()):
        # Convert indices to numpy array for fast indexing
        indices_array = np.array(list(indices), dtype=int)

        # Extract feature values for this group using numpy indexing (very fast)
        group_data = X_features[indices_array, :]

        # Compute aggregation
        if method == "mean":
            X_agg[i] = np.mean(group_data, axis=0)
        elif method == "median":
            X_agg[i] = np.median(group_data, axis=0)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

        group_keys.append(group_key)

    # Create DataFrame from aggregated data with original group keys as index
    features_agg = pd.DataFrame(X_agg, columns=feature_cols, index=group_keys)

    # Set index name(s) to match grouping column(s)
    if isinstance(group_cols, list):
        features_agg.index.names = group_cols
    else:
        features_agg.index.name = group_cols

    # If we preserved label_str, also aggregate it (take first value per group)
    label_str_agg = None
    if preserve_label_str:
        # Create temporary df with grouping columns and label_str
        label_df = pd.DataFrame({group_col: adata.obs[group_col].values})
        if preserve_batch_info and "batch" in adata.obs.columns:
            label_df["batch"] = adata.obs["batch"].values
        label_df["label_str"] = label_str_values
        label_str_agg = label_df.groupby(group_cols, observed=False)[
            "label_str"
        ].first()

    # Handle MultiIndex from groupby
    if preserve_batch_info and "batch" in adata.obs.columns:
        # Reset index to convert MultiIndex to columns
        features_agg_reset = features_agg.reset_index()

        # Create simple index for AnnData
        simple_index = [
            f"{row[group_col]}_{row['batch']}"
            for _, row in features_agg_reset.iterrows()
        ]

        # Use X_agg directly (numpy array with correct shape) instead of pandas column selection
        # This avoids pandas expanding duplicate column names
        X_matrix = X_agg  # Already has shape (n_groups, n_features)

        # Create AnnData with simple index
        adata_agg = ad.AnnData(X=X_matrix)
        adata_agg.var_names = adata.var_names  # Use original var_names from input
        adata_agg.obs[group_col] = features_agg_reset[group_col].values
        adata_agg.obs["batch"] = features_agg_reset["batch"].values
        adata_agg.obs_names = simple_index

        # Add label_str if preserved
        if preserve_label_str and label_str_agg is not None:
            label_str_reset = label_str_agg.reset_index()
            adata_agg.obs["label_str"] = label_str_reset["label_str"].values
    else:
        # Simple case without batch info
        # Reset index to avoid index.name/column conflicts in h5ad format
        features_agg_reset = features_agg.reset_index()

        # Use X_agg directly (numpy array with correct shape) instead of pandas column selection
        # This avoids pandas expanding duplicate column names
        X_matrix = X_agg  # Already has shape (n_groups, n_features)

        # Create AnnData with simple integer index
        adata_agg = ad.AnnData(X=X_matrix)
        adata_agg.var_names = adata.var_names  # Use original var_names from input
        adata_agg.obs[group_col] = features_agg_reset[group_col].values

        # Add label_str if preserved
        if preserve_label_str and label_str_agg is not None:
            label_str_reset = label_str_agg.reset_index()
            adata_agg.obs["label_str"] = label_str_reset["label_str"].values

    print(
        f"  Aggregated to {adata_agg.shape[0]} {level}s × {adata_agg.shape[1]} features"
    )

    return adata_agg


def compute_embeddings(
    adata: ad.AnnData,
    n_pca_components: int = 128,
    n_neighbors: int = 15,
    compute_pca: bool = True,
    compute_umap: bool = True,
    compute_phate: bool = True,
    use_existing_pca: bool = False,
) -> ad.AnnData:
    """
    Compute PCA, UMAP, and PHATE embeddings on AnnData object.

    Unified function that handles all embedding computation with consistent
    parameters and error handling. This is an alias for recompute_embeddings()
    with clearer naming for use in aggregation pipelines.

    Args:
        adata: AnnData object
        n_pca_components: Number of PCA components (auto-adjusted to n_samples)
        n_neighbors: Number of neighbors for UMAP/PHATE (auto-adjusted)
        compute_pca: Whether to compute PCA
        compute_umap: Whether to compute UMAP
        compute_phate: Whether to compute PHATE
        use_existing_pca: Use existing X_pca if available

    Returns:
        AnnData object with embeddings in .obsm

    Example:
        >>> adata = aggregate_to_level(adata_cell, "gene")
        >>> adata = compute_embeddings(adata)
        >>> print(adata.obsm.keys())  # X_pca, X_umap, X_phate

    Note:
        This is an alias for recompute_embeddings() with clearer naming.
    """
    return recompute_embeddings(
        adata,
        n_pca_components=n_pca_components,
        n_umap_neighbors=n_neighbors,
        compute_pca=compute_pca,
        compute_umap=compute_umap,
        compute_phate=compute_phate,
        use_existing_pca=use_existing_pca,
    )


def create_embeddings_fit_on_aggregated_controls(
    adata: ad.AnnData,
    control_gene: str = "NTC",
    control_group_size: int = 4,
    random_seed: Optional[int] = None,
    n_pca_components: int = 128,
    n_umap_neighbors: int = 15,
    use_pca_for_umap: bool = True,
    compute_phate: bool = True,
) -> ad.AnnData:
    """
    Fit PCA/UMAP on data with aggregated controls, then transform subsampled controls.

    This reveals how control subgroups distribute in an embedding space defined
    by target genes and a single representative control observation.

    Workflow:
    1. Aggregate controls to single observation (fit data)
    2. Fit PCA and UMAP on fit data
    3. Aggregate controls to subsampled groups (transform data)
    4. Transform subsampled data onto fitted embeddings

    Args:
        adata: AnnData at guide level (with separate guide observations)
        control_gene: Control gene name (default: "NTC")
        control_group_size: Number of guides per control group (default: 4)
        random_seed: Random seed for reproducible control grouping
        n_pca_components: Number of PCA components
        n_umap_neighbors: Number of neighbors for UMAP
        use_pca_for_umap: If True, compute UMAP on PCA space; if False, use raw features
        compute_phate: Whether to compute PHATE as well

    Returns:
        AnnData at gene level with subsampled controls, transformed onto fitted embeddings.
        Contains:
        - .obsm['X_pca']: All observations in fitted PCA space
        - .obsm['X_umap']: All observations in fitted UMAP space
        - .obsm['X_phate']: (optional) PHATE embedding
        - .uns['embedding_fit_metadata']: Info about fitting procedure

    Example:
        >>> # Start with guide-level data
        >>> adata_guide = ad.read_h5ad("guide_level.h5ad")
        >>>
        >>> # Create gene-level with fitted embeddings
        >>> adata_gene = create_embeddings_fit_on_aggregated_controls(
        ...     adata_guide,
        ...     control_gene="NTC",
        ...     control_group_size=4,
        ...     random_seed=42
        ... )
        >>>
        >>> # NTC subgroups are positioned in embedding space defined by
        >>> # target genes + single aggregated NTC
        >>> sc.pl.umap(adata_gene, color="label_str")
    """
    import umap
    from sklearn.decomposition import PCA

    print("\n" + "=" * 80)
    print("FITTED EMBEDDINGS: Fit on Aggregated Controls, Transform Subsampled")
    print("=" * 80)

    # Step 1: Create aggregated version (single NTC)
    print("\nStep 1: Aggregating controls to single observation (for fitting)...")
    adata_fit = aggregate_to_level(
        adata,
        level="gene",
        method="mean",
        preserve_batch_info=False,
        subsample_controls=False,  # All NTC guides → single NTC
    )
    print(f"  Fit data shape: {adata_fit.shape}")

    # Step 2: Fit PCA on aggregated data
    print(f"\nStep 2: Fitting PCA ({n_pca_components} components)...")
    pca = PCA(n_components=n_pca_components, random_state=random_seed)
    X_pca_fit = pca.fit_transform(adata_fit.X)
    print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    # Step 3: Fit UMAP on aggregated data
    print(f"\nStep 3: Fitting UMAP (n_neighbors={n_umap_neighbors})...")
    if use_pca_for_umap:
        umap_input = X_pca_fit
        print("  Using PCA features for UMAP")
    else:
        umap_input = adata_fit.X
        print("  Using raw features for UMAP")

    umap_model = umap.UMAP(
        n_neighbors=min(n_umap_neighbors, adata_fit.shape[0] - 1),
        n_components=2,
        metric="cosine",
        random_state=random_seed,
    )
    X_umap_fit = umap_model.fit_transform(umap_input)
    print(f"  UMAP fitted on {adata_fit.shape[0]} observations")

    # Step 4: Create subsampled version (NTC_1, NTC_2, ..., NTC_N)
    print("\nStep 4: Aggregating with control subsampling (for transformation)...")
    adata_transform = aggregate_to_level(
        adata,
        level="gene",
        method="mean",
        preserve_batch_info=False,
        subsample_controls=True,
        control_gene=control_gene,
        control_group_size=control_group_size,
        random_seed=random_seed,
    )
    print(f"  Transform data shape: {adata_transform.shape}")

    # Step 5: Transform subsampled data onto fitted PCA
    print("\nStep 5: Transforming subsampled data onto fitted PCA...")
    X_pca_transform = pca.transform(adata_transform.X)
    adata_transform.obsm["X_pca"] = X_pca_transform
    print(f"  PCA transformed: {X_pca_transform.shape}")

    # Step 6: Transform subsampled data onto fitted UMAP
    print("\nStep 6: Transforming subsampled data onto fitted UMAP...")
    if use_pca_for_umap:
        umap_input_transform = X_pca_transform
    else:
        umap_input_transform = adata_transform.X

    X_umap_transform = umap_model.transform(umap_input_transform)
    adata_transform.obsm["X_umap"] = X_umap_transform
    print(f"  UMAP transformed: {X_umap_transform.shape}")

    # Step 7: Optional PHATE
    if compute_phate:
        print("\nStep 7: Computing PHATE on transformed data...")
        import scanpy.external as sce

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="phate")
            warnings.filterwarnings("ignore", category=UserWarning, module="graphtools")
            warnings.filterwarnings("ignore", category=RuntimeWarning, module="phate")

            # PHATE needs neighbors - compute on PCA space
            import scanpy as sc

            sc.pp.neighbors(
                adata_transform,
                n_neighbors=min(n_umap_neighbors, adata_transform.shape[0] - 1),
                n_pcs=n_pca_components if use_pca_for_umap else 0,
                use_rep="X_pca" if use_pca_for_umap else None,
                metric="cosine",
            )
            sce.tl.phate(
                adata_transform,
                n_components=2,
                k=min(n_umap_neighbors, adata_transform.shape[0] - 1),
                n_pca=n_pca_components if use_pca_for_umap else 0,
                knn_dist="cosine",
                t="auto",
            )
        print(f"  PHATE computed: {adata_transform.obsm['X_phate'].shape}")

    # Store metadata
    control_mask = adata_transform.obs["label_str"].str.startswith(f"{control_gene}_")
    n_control_groups = control_mask.sum()

    adata_transform.uns["embedding_fit_metadata"] = {
        "method": "fit_on_aggregated_controls",
        "control_gene": control_gene,
        "control_group_size": control_group_size,
        "n_control_groups": n_control_groups,
        "n_fit_observations": adata_fit.shape[0],
        "n_transform_observations": adata_transform.shape[0],
        "n_pca_components": n_pca_components,
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "use_pca_for_umap": use_pca_for_umap,
        "random_seed": random_seed,
    }

    print("\n" + "=" * 80)
    print("FITTED EMBEDDINGS COMPLETE")
    print("=" * 80)
    print(f"Fit on: {adata_fit.shape[0]} observations (controls aggregated)")
    print(
        f"Transformed: {adata_transform.shape[0]} observations ({n_control_groups} control groups)"
    )
    print("=" * 80 + "\n")

    return adata_transform


def create_aggregated_embeddings(
    adata: ad.AnnData,
    level: Literal["guide", "gene"],
    n_pca_components: int = 128,
    n_neighbors: int = 15,
    aggregation_method: str = "mean",
    preserve_batch_info: bool = True,
    subsample_controls: bool = False,
    control_gene: str = "NTC",
    control_group_size: int = 4,
    random_seed: Optional[int] = None,
) -> ad.AnnData:
    """
    Complete pipeline: aggregate + compute embeddings.

    Convenience function that combines aggregate_to_level() and
    compute_embeddings() in one call.

    Args:
        adata: Cell-level AnnData object
        level: Aggregation level ("guide" or "gene")
        n_pca_components: PCA components for embeddings
        n_neighbors: Neighbors for UMAP/PHATE
        aggregation_method: How to aggregate ("mean" or "median")
        preserve_batch_info: Keep batch metadata
        subsample_controls: Split control gene guides into random groups (gene level only)
        control_gene: Gene name to subsample (default: "NTC")
        control_group_size: Number of guides per control group (default: 4)
        random_seed: Random seed for reproducible control grouping

    Returns:
        Aggregated AnnData with PCA, UMAP, PHATE computed

    Example:
        >>> adata_cell = ad.read_h5ad("features_processed.h5ad")
        >>> adata_gene = create_aggregated_embeddings(adata_cell, "gene")
        >>> sc.pl.umap(adata_gene, color="label_str")

        >>> # With control subsampling
        >>> adata_gene = create_aggregated_embeddings(
        ...     adata_cell, "gene",
        ...     subsample_controls=True,
        ...     control_group_size=4,
        ...     random_seed=42
        ... )
    """
    # Step 1: Aggregate
    adata_agg = aggregate_to_level(
        adata,
        level=level,
        method=aggregation_method,
        preserve_batch_info=preserve_batch_info,
        subsample_controls=subsample_controls,
        control_gene=control_gene,
        control_group_size=control_group_size,
        random_seed=random_seed,
    )

    # Step 2: Compute embeddings
    adata_agg = compute_embeddings(
        adata_agg,
        n_pca_components=n_pca_components,
        n_neighbors=n_neighbors,
        compute_pca=True,
        compute_umap=True,
        compute_phate=True,
    )

    return adata_agg


def load_multiple_experiments(
    base_dir: Union[str, Path],
    experiments: List[str],
    feature_type: str = "features_processed",
    require_all: bool = True,
) -> List[Path]:
    """
    Helper function to load AnnData paths from multiple experiments.

    Args:
        base_dir: Base directory containing experiment folders
        experiments: List of experiment names (e.g., ["ops0089_20251119", "ops0084_20250101"])
        feature_type: Which h5ad file to load
                      Options: "features_processed", "guide_bulked_umap", "gene_bulked_umap"
        require_all: If True, raise error if any file missing
                     If False, skip missing files with warning

    Returns:
        List of paths to .h5ad files

    Example:
        >>> base_dir = "/hpc/projects/intracellular_dashboard/ops"
        >>> experiments = ["ops0089_20251119", "ops0084_20250101"]
        >>> paths = load_multiple_experiments(base_dir, experiments)
        >>> adata_combined = concatenate_anndata_objects(paths)
    """
    base_dir = Path(base_dir)
    paths = []

    for exp in experiments:
        exp_path = (
            base_dir
            / exp
            / "3-assembly"
            / "dino_features"
            / "anndata_objects"
            / f"{feature_type}.h5ad"
        )

        if exp_path.exists():
            paths.append(exp_path)
        else:
            msg = f"File not found: {exp_path}"
            if require_all:
                raise FileNotFoundError(msg)
            else:
                warnings.warn(msg)

    print(f"Found {len(paths)}/{len(experiments)} experiment files")
    return paths


def compare_batch_distributions(
    adata: ad.AnnData,
    batch_key: str = "batch",
    label_key: str = "label_str",
) -> pd.DataFrame:
    """
    Compare gene/guide distributions across batches.

    Useful for checking if batches are comparable before combining.

    Args:
        adata: AnnData object with batch information
        batch_key: Column in .obs with batch labels
        label_key: Column in .obs with gene/guide labels

    Returns:
        DataFrame with counts per batch × label

    Example:
        >>> adata_combined = concatenate_anndata_objects(paths)
        >>> dist = compare_batch_distributions(adata_combined)
        >>> print(dist)
        >>> # Check for imbalanced genes
        >>> imbalanced = dist.std(axis=1) / dist.mean(axis=1)
        >>> print(imbalanced[imbalanced > 0.5])
    """
    if batch_key not in adata.obs.columns:
        raise ValueError(f"Batch key '{batch_key}' not found in .obs")
    if label_key not in adata.obs.columns:
        raise ValueError(f"Label key '{label_key}' not found in .obs")

    # Cross-tabulation
    dist = pd.crosstab(
        adata.obs[label_key], adata.obs[batch_key], margins=True, margins_name="Total"
    )

    return dist


def split_by_batch(
    adata: ad.AnnData,
    batch_key: str = "batch",
) -> dict:
    """
    Split combined AnnData back into individual batch objects.

    Useful for batch-specific analysis after combining.

    Args:
        adata: Combined AnnData object
        batch_key: Column in .obs with batch labels

    Returns:
        Dictionary mapping batch_id -> AnnData object

    Example:
        >>> adata_combined = concatenate_anndata_objects(paths)
        >>> adata_combined = recompute_embeddings(adata_combined)
        >>> batches = split_by_batch(adata_combined)
        >>> for batch_id, adata_batch in batches.items():
        ...     print(f"{batch_id}: {adata_batch.shape}")
    """
    if batch_key not in adata.obs.columns:
        raise ValueError(f"Batch key '{batch_key}' not found in .obs")

    batches = {}
    for batch_id in adata.obs[batch_key].unique():
        mask = adata.obs[batch_key] == batch_id
        batches[batch_id] = adata[mask].copy()

    print(f"Split into {len(batches)} batches:")
    for batch_id, batch_adata in batches.items():
        print(f"  {batch_id}: {batch_adata.shape[0]} cells")

    return batches


def concatenate_features_by_channel(
    experiment: Optional[str] = None,
    channels: Optional[List[str]] = None,
    experiments_channels: Optional[List[tuple]] = None,
    feature_type: str = "dinov3",
    aggregation_level: Literal["guide", "gene"] = "gene",
    base_dir: Union[str, Path] = "/hpc/projects/intracellular_dashboard/ops",
    recompute_embeddings: bool = True,
    n_pca_components: int = 128,
    n_umap_neighbors: int = 15,
) -> ad.AnnData:
    """
    Concatenate features from multiple channels or experiments.

    Supports two strategies:

    Strategy 1 (Multi-Channel): Combines features from different imaging channels
    within the same experiment. Specify experiment + channels parameters.

    Strategy 2 (Multi-Organelle): Combines features from the same channel across
    different experiments, each targeting different organelles. Specify
    experiments_channels parameter.

    Features are concatenated horizontally (columns), while observations
    (genes/guides) are matched and aligned. Uses FeatureMetadata to create
    biologically-meaningful feature names.

    Args:
        experiment: Experiment name for Strategy 1 (e.g., "ops0089" or "ops0089_20251119")
        channels: List of channels for Strategy 1 (e.g., ["Phase2D", "GFP", "mCherry"])
        experiments_channels: List of (experiment, channel) tuples for Strategy 2
                             (e.g., [("ops0089", "GFP"), ("ops0108", "GFP")])
        feature_type: Feature extraction method
                      Options: "dinov3", "cellprofiler"
        aggregation_level: Which data level to combine
                          Options: "guide" (guide_bulked_umap), "gene" (gene_bulked_umap)
                          Note: Cell-level concatenation not supported (no stable IDs)
        base_dir: Base directory containing experiment folders
        recompute_embeddings: Whether to compute PCA/UMAP on combined features
        n_pca_components: Number of PCA components for combined features
        n_umap_neighbors: Number of neighbors for UMAP

    Returns:
        Combined AnnData object with:
        - .X: Horizontally concatenated features from all sources
        - .var_names: Biologically-informed feature names (e.g., "ops0089_EEA1_dinov3_0")
        - .obs: Metadata from first source (all sources have same observations)
        - .obsm: New PCA/UMAP computed on combined features (if recompute_embeddings=True)
        - .uns["combined_metadata"]: Provenance information including strategy used

    Example (Strategy 1 - Multi-Channel):
        >>> # Combine Phase2D, GFP, mCherry from same experiment
        >>> adata_multi = concatenate_features_by_channel(
        ...     experiment="ops0089",
        ...     channels=["Phase2D", "GFP", "mCherry"],
        ...     feature_type="dinov3",
        ...     aggregation_level="gene"
        ... )
        >>> print(adata_multi.shape)  # (n_genes, 3072) = 3 channels × 1024 features

    Example (Strategy 2 - Multi-Organelle):
        >>> # Combine GFP from multiple experiments with different markers
        >>> adata_multi = concatenate_features_by_channel(
        ...     experiments_channels=[
        ...         ("ops0089", "GFP"),  # EEA1 marker
        ...         ("ops0108", "GFP"),  # TOMM70A marker
        ...     ],
        ...     feature_type="dinov3",
        ...     aggregation_level="gene"
        ... )
        >>> print(adata_multi.shape)  # (n_genes, 2048) = 2 experiments × 1024 features

    Notes:
        - Only genes/guides present in ALL sources are kept
        - Channel-specific file naming required: {level}_{channel}.h5ad
        - Uses FeatureMetadata to create meaningful variable names with biological context
        - Original PCA/UMAP from individual sources are discarded

    Raises:
        ValueError: If aggregation_level is "cell" (not supported)
        ValueError: If parameters don't match one of the two strategies
        FileNotFoundError: If any source file is missing
    """
    if aggregation_level == "cell":
        raise ValueError(
            "Cell-level feature concatenation not supported. "
            "Cells don't have stable identifiers across different channel extractions. "
            "Use aggregation_level='guide' or 'gene' instead."
        )

    # Validate strategy parameters
    if experiments_channels is not None:
        if experiment is not None or channels is not None:
            raise ValueError(
                "Parameters are mutually exclusive. Use either:\n"
                "  - (experiment + channels) for Strategy 1: Multi-Channel\n"
                "  - experiments_channels for Strategy 2: Multi-Organelle\n"
                "Do not provide both."
            )
        strategy = "multi_organelle"
    elif experiment is not None and channels is not None:
        strategy = "multi_channel"
        # Convert to unified format
        experiments_channels = [(experiment, ch) for ch in channels]
    else:
        raise ValueError(
            "Must provide either:\n"
            "  - experiment + channels (Strategy 1: Multi-Channel)\n"
            "  - experiments_channels (Strategy 2: Multi-Organelle)"
        )

    # Store reference to recompute_embeddings function to avoid shadowing by parameter
    _recompute_fn = globals()["recompute_embeddings"]

    # Initialize metadata manager
    meta = FeatureMetadata()

    # Determine file prefix based on aggregation level
    if aggregation_level == "guide":
        file_prefix = "guide_bulked_umap"
    elif aggregation_level == "gene":
        file_prefix = "gene_bulked_umap"
    else:
        raise ValueError(f"Invalid aggregation_level: {aggregation_level}")

    # Build paths
    base_dir = Path(base_dir)

    print("=" * 80)
    if strategy == "multi_channel":
        print(f"MULTI-CHANNEL FEATURE CONCATENATION (Strategy 1)")
    else:
        print(f"MULTI-ORGANELLE FEATURE CONCATENATION (Strategy 2)")
    print("=" * 80)
    print(f"Sources: {len(experiments_channels)}")
    print(f"Feature type: {feature_type}")
    print(f"Aggregation level: {aggregation_level}")
    print("=" * 80)

    # Load AnnData objects for each source
    print(f"\nLoading {len(experiments_channels)} source(s)...")
    adata_by_source = {}

    for exp, channel in experiments_channels:
        exp_short = exp.split("_")[0]  # Handle ops0089_20251119 or ops0089

        # Find experiment directory (may have date suffix)
        exp_dirs = list(base_dir.glob(f"{exp_short}*"))
        if not exp_dirs:
            raise FileNotFoundError(
                f"Experiment directory not found: {base_dir}/{exp_short}*"
            )
        exp_dir = exp_dirs[0]  # Take first match

        if feature_type == "cellprofiler":
            anndata_dir = exp_dir / "3-assembly" / "cell-profiler" / "anndata_objects"
            # For CellProfiler, files are named by reporter, not channel
            # Check if 'channel' is already a reporter or needs mapping
            from ops_model.data.feature_metadata import FeatureMetadata

            meta_temp = FeatureMetadata()
            test_reporter = meta_temp.get_short_label(exp_short, channel)

            # If lookup returns "unknown", then 'channel' is already a reporter name
            # (from biological signal grouping in comprehensive mode)
            if test_reporter == "unknown" or test_reporter.startswith("unlabeled"):
                reporter = channel  # Already a reporter name
            else:
                reporter = test_reporter  # Mapped from channel

            file_path = anndata_dir / f"{file_prefix}_{reporter}.h5ad"
        else:
            anndata_dir = (
                exp_dir / "3-assembly" / f"{feature_type}_features" / "anndata_objects"
            )
            file_path = anndata_dir / f"{file_prefix}_{channel}.h5ad"

        if not file_path.exists():
            if feature_type == "cellprofiler":
                raise FileNotFoundError(
                    f"Source file not found: {file_path}\n"
                    f"Expected naming: {file_prefix}_{reporter}.h5ad (CellProfiler files use reporter names)"
                )
            else:
                raise FileNotFoundError(
                    f"Source file not found: {file_path}\n"
                    f"Expected naming: {file_prefix}_{channel}.h5ad"
                )

        adata = ad.read_h5ad(file_path)
        source_key = f"{exp_short}_{channel}"
        adata_by_source[source_key] = adata

        # Get biological information
        bio_signal = meta.get_biological_signal(exp_short, channel)
        print(f"  {exp_short}/{channel}: {adata.shape} - {bio_signal}")

    # Find common genes/guides across all sources
    print(f"\nFinding common {aggregation_level}s across sources...")
    label_key = "sgRNA" if aggregation_level == "guide" else "label_str"

    gene_sets = [set(adata.obs[label_key]) for adata in adata_by_source.values()]
    common_genes = set.intersection(*gene_sets)
    print(f"  Common {aggregation_level}s: {len(common_genes)}")

    if len(common_genes) == 0:
        raise ValueError(f"No common {aggregation_level}s found across all sources!")

    # Filter each dataset to common genes and sort for alignment
    print(f"\nAligning {aggregation_level}s across sources...")
    for source_key in adata_by_source.keys():
        adata = adata_by_source[source_key]
        mask = adata.obs[label_key].isin(common_genes)
        adata_filtered = adata[mask].copy()

        # Sort by label_key for consistent ordering
        sort_idx = adata_filtered.obs[label_key].argsort()
        adata_by_source[source_key] = adata_filtered[sort_idx].copy()

        print(f"  {source_key}: {adata_by_source[source_key].shape}")

    # Verify alignment
    print(f"\nVerifying alignment...")
    source_keys = list(adata_by_source.keys())
    reference_labels = adata_by_source[source_keys[0]].obs[label_key].values
    for source_key in source_keys[1:]:
        other_labels = adata_by_source[source_key].obs[label_key].values
        if not np.array_equal(reference_labels, other_labels):
            raise ValueError(
                f"Label mismatch between {source_keys[0]} and {source_key}"
            )
    print(f"  ✓ All sources aligned ({len(reference_labels)} {aggregation_level}s)")

    # Concatenate features horizontally
    print(f"\nConcatenating features...")
    X_list = [adata_by_source[source_key].X for source_key in source_keys]
    X_combined = np.concatenate(X_list, axis=1)
    print(f"  Combined shape: {X_combined.shape}")

    # Create biologically-informed variable names using FeatureMetadata
    print(f"\nCreating feature names with biological context...")
    var_names = []
    feature_slices = {}  # Track which features came from which source

    start_idx = 0
    for exp, channel in experiments_channels:
        exp_short = exp.split("_")[0]
        source_key = f"{exp_short}_{channel}"
        n_features = adata_by_source[source_key].shape[1]
        short_label = meta.get_short_label(exp_short, channel)

        # Create feature names
        source_var_names = [
            meta.create_feature_name(exp_short, channel, feature_type, i)
            for i in range(n_features)
        ]
        var_names.extend(source_var_names)

        # Track slice (use list for HDF5 compatibility)
        feature_slices[source_key] = [start_idx, start_idx + n_features]
        start_idx += n_features

        print(
            f"  {source_key} ({short_label}): features {feature_slices[source_key][0]}-{feature_slices[source_key][1]-1}"
        )

    # Create combined AnnData
    print(f"\nCreating combined AnnData object...")
    first_source_key = list(adata_by_source.keys())[0]
    adata_combined = ad.AnnData(
        X=X_combined,
        obs=adata_by_source[
            first_source_key
        ].obs.copy(),  # Use metadata from first source
    )
    adata_combined.var_names = var_names

    # Add metadata (convert tuples to lists for HDF5 compatibility)
    adata_combined.uns["combined_metadata"] = {
        "strategy": strategy,
        "experiments_channels": [[exp, ch] for exp, ch in experiments_channels],
        "feature_type": feature_type,
        "aggregation_level": aggregation_level,
        "n_sources": len(experiments_channels),
        "feature_slices": feature_slices,
        "channel_biology": {
            f"{exp.split('_')[0]}_{ch}": meta.get_channel_info(exp.split("_")[0], ch)
            for exp, ch in experiments_channels
        },
    }

    # Add individual source metadata
    for exp, channel in experiments_channels:
        exp_short = exp.split("_")[0]
        meta.add_to_anndata(adata_combined, exp_short, channel, feature_type)

    print(f"  ✓ Combined AnnData: {adata_combined.shape}")
    print(f"  Variable names (first 3): {adata_combined.var_names[:3].tolist()}")

    # Recompute embeddings on combined features
    if recompute_embeddings:
        print(
            f"\nRecomputing embeddings on combined {X_combined.shape[1]}-dimensional space..."
        )
        adata_combined = _recompute_fn(
            adata_combined,
            n_pca_components=n_pca_components,
            n_umap_neighbors=n_umap_neighbors,
            compute_pca=True,
            compute_umap=True,
            compute_phate=True,
        )

    print("\n" + "=" * 80)
    if strategy == "multi_channel":
        print("MULTI-CHANNEL CONCATENATION COMPLETE")
    else:
        print("MULTI-ORGANELLE CONCATENATION COMPLETE")
    print("=" * 80)
    print(f"Final shape: {adata_combined.shape}")
    print(f"Sources combined: {len(experiments_channels)}")
    print(
        f"Features per source: {[adata.shape[1] for adata in adata_by_source.values()]}"
    )
    print(f"Total features: {adata_combined.shape[1]}")
    print("=" * 80)

    return adata_combined


# ============================================================================
# Strategy 3: Comprehensive Multi-Experiment Combination
# ============================================================================


def _group_by_biological_signal(
    experiments_channels: List[Tuple[str, str]],
    meta: FeatureMetadata,
    verbose: bool = True,
    feature_type: str = "dinov3",
) -> Dict[str, List[Tuple[str, str]]]:
    """
    Group experiment/channel pairs by their biological signal.

    Uses FeatureMetadata to look up the biological meaning of each channel
    and groups those with the same meaning together.

    For CellProfiler, the "channel" parameter IS the biological signal/reporter
    (e.g., "SEC61B", "5xUPRE") so no metadata lookup is needed.

    Parameters
    ----------
    experiments_channels : List[Tuple[str, str]]
        List of (experiment, channel) pairs
    meta : FeatureMetadata
        Metadata manager with ops_channel_maps.yaml loaded
    verbose : bool
        Print grouping information
    feature_type : str
        Feature type ("dinov3", "cellprofiler", etc.)

    Returns
    -------
    Dict[str, List[Tuple[str, str]]]
        Mapping from biological signal to list of (experiment, channel) pairs

        Example: {
            "Phase": [("ops0089", "Phase2D"), ("ops0108", "Phase2D")],
            "early endosome, EEA1": [("ops0089", "GFP")],
            "mitochondria, TOMM70A": [("ops0108", "GFP")]
        }
    """
    biological_signals = {}

    for exp, channel in experiments_channels:
        # Get short name for metadata lookup
        exp_short = exp.split("_")[0]

        # For CellProfiler, channel IS the biological signal (reporter name)
        if feature_type == "cellprofiler":
            bio_signal = channel
        else:
            # For DinoV3/other, look up biological signal from metadata
            bio_signal = meta.get_biological_signal(exp_short, channel)

            if bio_signal is None:
                raise ValueError(
                    f"No metadata found for {exp_short}/{channel}. "
                    f"Please add to ops_channel_maps.yaml"
                )

        # Group by biological signal
        if bio_signal not in biological_signals:
            biological_signals[bio_signal] = []
        biological_signals[bio_signal].append((exp, channel))

    if verbose:
        print("\nBiological signal grouping:")
        for bio_signal, pairs in biological_signals.items():
            agg_type = "VERTICAL" if len(pairs) > 1 else "HORIZONTAL"
            print(f"  [{agg_type}] {bio_signal}:")
            for exp, ch in pairs:
                print(f"    - {exp}/{ch}")

    return biological_signals


def _combine_duplicate_observations(
    adata_concat: ad.AnnData,
    level: Literal["guide", "gene"],
    cell_counts: Dict[str, int],
    verbose: bool = True,
) -> ad.AnnData:
    """
    Combine duplicate observations from different experiments.

    When the same gene/guide appears in multiple experiments, we combine
    their features by taking the unweighted mean (each experiment contributes
    equally, regardless of cell count).

    Parameters
    ----------
    adata_concat : ad.AnnData
        Concatenated data with duplicate observations
        Example: RPL18_ops0089, RPL18_ops0108 (same gene, different experiments)
    level : Literal["guide", "gene"]
        Aggregation level
    cell_counts : Dict[str, int]
        Number of cells from each experiment
    verbose : bool
        Print progress

    Returns
    -------
    ad.AnnData
        Pooled data with one observation per unique gene/guide
        Contains metadata about which experiments contributed
    """
    label_key = "sgRNA" if level == "guide" else "label_str"

    if verbose:
        print(f"    Combining duplicate {level}s across experiments...")

    # Create DataFrame for aggregation
    features_df = pd.DataFrame(adata_concat.X, columns=adata_concat.var_names)
    features_df[label_key] = adata_concat.obs[label_key].values
    features_df["experiment"] = adata_concat.obs["experiment"].values

    # Check if label_str exists when processing guide level (needed for guide→gene aggregation)
    preserve_label_str = False
    label_str_agg = None
    if level == "guide" and "label_str" in adata_concat.obs.columns:
        features_df["label_str"] = adata_concat.obs["label_str"].values
        preserve_label_str = True

    # Unweighted mean: each experiment contributes equally
    # This prevents one large experiment from dominating
    feature_cols = adata_concat.var_names.tolist()
    features_agg = features_df.groupby(label_key)[feature_cols].mean()

    # If preserving label_str, aggregate it separately (take first value per guide)
    if preserve_label_str:
        label_str_agg = features_df.groupby(label_key)["label_str"].first()

    # Track which experiments contributed to each observation
    experiments_per_obs = (
        features_df.groupby(label_key)["experiment"]
        .apply(lambda x: list(x.unique()))
        .to_dict()
    )

    # Count how many experiments contributed
    n_experiments_per_obs = (
        features_df.groupby(label_key)["experiment"].nunique().to_dict()
    )

    # Create final AnnData
    # Reset index to avoid index.name/column conflicts in h5ad format
    features_agg_reset = features_agg.reset_index()

    adata_pooled = ad.AnnData(features_agg_reset[feature_cols].values)
    adata_pooled.var_names = feature_cols
    adata_pooled.obs[label_key] = features_agg_reset[label_key].values

    # Add label_str if it was preserved
    if preserve_label_str and label_str_agg is not None:
        label_str_reset = label_str_agg.reset_index()
        adata_pooled.obs["label_str"] = label_str_reset["label_str"].values

    # Add experiment count per observation
    adata_pooled.obs["n_experiments"] = [
        n_experiments_per_obs[obs_id] for obs_id in adata_pooled.obs[label_key]
    ]

    # Store vertical aggregation metadata
    adata_pooled.uns["vertical_metadata"] = {
        "experiments": list(cell_counts.keys()),
        "cell_counts_per_experiment": cell_counts,
        "total_cells": sum(cell_counts.values()),
        "n_observations_per_experiment": {
            exp: (adata_concat.obs["experiment"] == exp).sum()
            for exp in cell_counts.keys()
        },
        "aggregation_method": "unweighted_mean",
        "experiments_per_observation": experiments_per_obs,
    }

    if verbose:
        print(f"      Combined to {len(adata_pooled)} unique {level}s")
        print(f"      Total cells pooled: {sum(cell_counts.values())}")

    return adata_pooled


def _process_vertical_group(
    exp_channel_pairs: List[Tuple[str, str]],
    feature_type: str,
    base_dir: Path,
    target_level: Literal["guide", "gene"],
    join: str,
    verbose: bool = True,
    subsample_controls: bool = False,
    control_gene: str = "NTC",
    control_group_size: int = 4,
    random_seed: Optional[int] = None,
    normalize_on_pooling: bool = True,
    normalize_on_controls: bool = False,
) -> ad.AnnData:
    """
    Memory-efficient vertical aggregation for same biological signal.

    Strategy:
    1. Load cells from each experiment
    2. Z-score normalize cells (per-experiment)
    3. Aggregate to target level immediately (frees cell memory)
    4. Concatenate aggregated data (much smaller)
    5. Combine duplicate observations (e.g., same gene from multiple experiments)

    This avoids keeping all cells in memory at once.

    Parameters
    ----------
    exp_channel_pairs : List[Tuple[str, str]]
        List of (experiment, channel) pairs with same biological signal
    feature_type : str
        Feature type to load
    base_dir : Path
        Base OPS directory
    target_level : Literal["guide", "gene"]
        Target aggregation level
    join : str
        How to handle observations: "inner" or "outer"
    verbose : bool
        Print progress
    subsample_controls : bool
        Split control gene guides into random groups (gene level only)
    control_gene : str
        Gene name to subsample (default: "NTC")
    control_group_size : int
        Number of guides per control group (default: 4)
    random_seed : Optional[int]
        Random seed for reproducible control grouping
    normalize_on_pooling : bool
        If True, z-score normalize cells from each experiment before aggregation (default: True)
    normalize_on_controls : bool
        If True, compute normalization statistics from control cells only (default: False)

    Returns
    -------
    ad.AnnData
        Aggregated AnnData at target level with pooled observations
        Contains metadata about experiments and cell counts
    """
    if verbose:
        print(
            f"\n  Processing {len(exp_channel_pairs)} source(s) with vertical aggregation..."
        )

    aggregated_list = []
    cell_counts = {}

    for exp, channel in exp_channel_pairs:
        exp_short = exp.split("_")[0]

        # Find experiment directory
        exp_dirs = list(base_dir.glob(f"{exp_short}*"))
        if not exp_dirs:
            raise FileNotFoundError(
                f"Experiment directory not found: {base_dir}/{exp_short}*"
            )
        exp_dir = exp_dirs[0]

        # Load cell-level file
        if feature_type == "cellprofiler":
            anndata_dir = exp_dir / "3-assembly" / "cell-profiler" / "anndata_objects"
            # For CellProfiler, files are named by reporter
            # In comprehensive mode, 'channel' parameter is already the reporter (biological signal)
            # Check if this is a reporter name or needs mapping from channel
            from ops_model.data.feature_metadata import FeatureMetadata

            meta = FeatureMetadata()

            # Try to get reporter from channel mapping
            # If it returns 'unknown', then 'channel' is likely already a reporter name
            test_reporter = meta.get_short_label(exp_short, channel)
            if test_reporter == "unknown" or test_reporter.startswith("unlabeled"):
                # 'channel' is already a reporter name (biological signal)
                reporter = channel
            else:
                # 'channel' is an actual channel, map to reporter
                reporter = test_reporter

            cell_file = anndata_dir / f"features_processed_{reporter}.h5ad"
        else:
            anndata_dir = (
                exp_dir / "3-assembly" / f"{feature_type}_features" / "anndata_objects"
            )
            cell_file = anndata_dir / f"features_processed_{channel}.h5ad"

        if not cell_file.exists():
            raise FileNotFoundError(f"Cell-level file not found: {cell_file}")

        if verbose:
            print(f"    Loading {exp}/{channel}...")

        adata_cells = ad.read_h5ad(cell_file)

        # Track cell counts BEFORE aggregation
        cell_counts[exp] = len(adata_cells)

        if verbose:
            print(f"      {len(adata_cells)} cells loaded")

        # Normalize cells from this experiment
        if normalize_on_pooling:
            if verbose:
                print(f"      Normalizing cells from {exp}...")
            adata_cells = normalize_adata_zscore(
                adata_cells,
                normalize_on_controls=normalize_on_controls,
                control_gene=control_gene,
            )

        # Aggregate immediately to target level (frees cell-level memory)
        adata_agg = aggregate_to_level(
            adata_cells,
            level=target_level,
            method="mean",
            preserve_batch_info=False,
            subsample_controls=subsample_controls,
            control_gene=control_gene,
            control_group_size=control_group_size,
            random_seed=random_seed,
        )

        # Add experiment label
        adata_agg.obs["experiment"] = exp

        # Free memory
        del adata_cells

        aggregated_list.append(adata_agg)

        if verbose:
            print(f"      Aggregated to {len(adata_agg)} {target_level}s")

    # Concatenate aggregated data (much smaller than cells!)
    if verbose:
        print(f"    Concatenating {len(aggregated_list)} aggregated datasets...")

    adata_concat = ad.concat(aggregated_list, join=join, label="source_idx")

    if verbose:
        print(f"      Concatenated: {adata_concat.shape}")

    # Combine duplicate observations
    adata_pooled = _combine_duplicate_observations(
        adata_concat, level=target_level, cell_counts=cell_counts, verbose=verbose
    )

    return adata_pooled


def _process_horizontal_group(
    exp: str,
    channel: str,
    feature_type: str,
    base_dir: Path,
    target_level: Literal["guide", "gene"],
    verbose: bool = True,
    subsample_controls: bool = False,
    control_gene: str = "NTC",
    control_group_size: int = 4,
    random_seed: Optional[int] = None,
    normalize_on_pooling: bool = True,
    normalize_on_controls: bool = False,
) -> ad.AnnData:
    """
    Process single-source group (different biology).

    Since there's only one experiment/channel pair with this biological signal,
    no vertical pooling is needed. Just load cells, normalize, and aggregate.

    Parameters
    ----------
    exp : str
        Experiment name
    channel : str
        Channel name
    feature_type : str
        Feature type
    base_dir : Path
        Base OPS directory
    target_level : Literal["guide", "gene"]
        Target aggregation level
    verbose : bool
        Print progress
    subsample_controls : bool
        Split control gene guides into random groups (gene level only)
    control_gene : str
        Gene name to subsample (default: "NTC")
    control_group_size : int
        Number of guides per control group (default: 4)
    random_seed : Optional[int]
        Random seed for reproducible control grouping
    normalize_on_pooling : bool
        If True, z-score normalize cells before aggregation (default: True)
    normalize_on_controls : bool
        If True, compute normalization statistics from control cells only (default: False)

    Returns
    -------
    ad.AnnData
        Aggregated data at target level for this single source
    """
    if verbose:
        print(f"\n  Processing {exp}/{channel} (horizontal, single source)...")

    exp_short = exp.split("_")[0]

    # Find experiment directory
    exp_dirs = list(base_dir.glob(f"{exp_short}*"))
    if not exp_dirs:
        raise FileNotFoundError(
            f"Experiment directory not found: {base_dir}/{exp_short}*"
        )
    exp_dir = exp_dirs[0]

    # Load cell-level file
    if feature_type == "cellprofiler":
        anndata_dir = exp_dir / "3-assembly" / "cell-profiler" / "anndata_objects"
        # For CellProfiler, files are named by reporter
        # In comprehensive mode, 'channel' parameter is already the reporter (biological signal)
        # Check if this is a reporter name or needs mapping from channel
        from ops_model.data.feature_metadata import FeatureMetadata

        meta = FeatureMetadata()

        # Try to get reporter from channel mapping
        # If it returns 'unknown', then 'channel' is likely already a reporter name
        test_reporter = meta.get_short_label(exp_short, channel)
        if test_reporter == "unknown" or test_reporter.startswith("unlabeled"):
            # 'channel' is already a reporter name (biological signal)
            reporter = channel
        else:
            # 'channel' is an actual channel, map to reporter
            reporter = test_reporter

        cell_file = anndata_dir / f"features_processed_{reporter}.h5ad"
    else:
        anndata_dir = (
            exp_dir / "3-assembly" / f"{feature_type}_features" / "anndata_objects"
        )
        cell_file = anndata_dir / f"features_processed_{channel}.h5ad"

    if not cell_file.exists():
        raise FileNotFoundError(f"Cell-level file not found: {cell_file}")

    adata_cells = ad.read_h5ad(cell_file)

    if verbose:
        print(f"    Loaded {len(adata_cells)} cells")

    # Track cell count
    n_cells = len(adata_cells)

    # Normalize cells
    if normalize_on_pooling:
        if verbose:
            print(f"    Normalizing cells from {exp}...")
        adata_cells = normalize_adata_zscore(
            adata_cells,
            normalize_on_controls=normalize_on_controls,
            control_gene=control_gene,
        )

    # Aggregate to target level
    adata_agg = aggregate_to_level(
        adata_cells,
        level=target_level,
        method="mean",
        preserve_batch_info=False,
        subsample_controls=subsample_controls,
        control_gene=control_gene,
        control_group_size=control_group_size,
        random_seed=random_seed,
    )

    # Free memory
    del adata_cells

    # Store metadata
    adata_agg.uns["horizontal_metadata"] = {
        "experiment": exp,
        "channel": channel,
        "n_cells": n_cells,
        "n_observations": len(adata_agg),
    }

    if verbose:
        print(f"    Aggregated to {len(adata_agg)} {target_level}s")

    return adata_agg


def _align_biological_groups(
    group_adatas: Dict[str, ad.AnnData],
    target_level: Literal["guide", "gene"],
    join: str = "inner",
    verbose: bool = True,
) -> List[str]:
    """
    Find common observations across all biological groups.

    Parameters
    ----------
    group_adatas : Dict[str, ad.AnnData]
        Mapping from biological signal to processed AnnData
    target_level : Literal["guide", "gene"]
        Aggregation level
    join : str
        "inner" for intersection (default), "outer" for union
    verbose : bool
        Print alignment info

    Returns
    -------
    List[str]
        Sorted list of common observation IDs (genes or guides)
    """
    label_key = "sgRNA" if target_level == "guide" else "label_str"

    if verbose:
        print(f"\nAligning {target_level}s across biological groups (join={join})...")

    # Collect observation sets
    obs_sets = {}
    for bio_signal, adata in group_adatas.items():
        obs_set = set(adata.obs[label_key].unique())
        obs_sets[bio_signal] = obs_set
        if verbose:
            print(f"  {bio_signal}: {len(obs_set)} {target_level}s")

    # Compute intersection or union
    if join == "inner":
        common_obs = set.intersection(*obs_sets.values())
        join_desc = "intersection"
    else:  # outer
        common_obs = set.union(*obs_sets.values())
        join_desc = "union"

    if not common_obs:
        raise ValueError(
            f"No common {target_level}s found across biological groups (join={join}). "
            "This may indicate different gene sets in each experiment. "
            "Try join='outer' to include all observations."
        )

    # Sort for consistent ordering
    common_obs = sorted(common_obs)

    if verbose:
        print(f"\nAlignment result ({join_desc}): {len(common_obs)} {target_level}s")

    return common_obs


def _concatenate_horizontal(
    group_adatas: Dict[str, ad.AnnData],
    common_obs: List[str],
    biological_signals: Dict[str, List[Tuple[str, str]]],
    meta: FeatureMetadata,
    feature_type: str,
    target_level: Literal["guide", "gene"],
    verbose: bool = True,
) -> ad.AnnData:
    """
    Horizontally concatenate features from aligned biological groups.

    Parameters
    ----------
    group_adatas : Dict[str, ad.AnnData]
        Mapping from biological signal to processed AnnData
    common_obs : List[str]
        Common observation IDs to include
    biological_signals : Dict[str, List[Tuple[str, str]]]
        Original grouping information
    meta : FeatureMetadata
        Metadata manager
    feature_type : str
        Feature type
    target_level : Literal["guide", "gene"]
        Aggregation level
    verbose : bool
        Print progress

    Returns
    -------
    ad.AnnData
        Combined AnnData with horizontally concatenated features
    """
    label_key = "sgRNA" if target_level == "guide" else "label_str"

    if verbose:
        print(f"\nHorizontally concatenating features...")

    # Filter and sort each group
    filtered_groups = {}
    for bio_signal, adata in group_adatas.items():
        # Filter to common observations
        mask = adata.obs[label_key].isin(common_obs)
        adata_filtered = adata[mask].copy()

        # Sort by observation key for alignment
        sort_idx = adata_filtered.obs[label_key].argsort()
        adata_filtered = adata_filtered[sort_idx]

        filtered_groups[bio_signal] = adata_filtered

        if verbose:
            print(f"  {bio_signal}: {adata_filtered.shape}")

    # Verify alignment
    if verbose:
        print(f"\nVerifying alignment...")

    bio_signals_list = sorted(filtered_groups.keys())
    reference_labels = filtered_groups[bio_signals_list[0]].obs[label_key].values

    for bio_signal in bio_signals_list[1:]:
        other_labels = filtered_groups[bio_signal].obs[label_key].values
        if not np.array_equal(reference_labels, other_labels):
            raise ValueError(
                f"Label mismatch between {bio_signals_list[0]} and {bio_signal}"
            )

    if verbose:
        print(f"  ✓ All groups aligned ({len(reference_labels)} {target_level}s)")

    # Concatenate features horizontally
    if verbose:
        print(f"\nConcatenating features...")

    X_list = []
    var_names = []
    feature_slices = {}

    current_idx = 0
    for bio_signal in bio_signals_list:
        adata = filtered_groups[bio_signal]
        X_list.append(adata.X)

        # Get short label for variable naming
        exp_channel_pairs = biological_signals[bio_signal]
        exp_short = exp_channel_pairs[0][0].split("_")[0]
        channel = exp_channel_pairs[0][1]
        short_label = meta.get_short_label(exp_short, channel)

        # Get feature count (needed for tracking slices)
        n_features = adata.shape[1]

        # Create variable names based on feature type
        if feature_type == "cellprofiler":
            # Preserve original descriptive CellProfiler feature names
            # Names already contain reporter and measurement info
            # (e.g., 'single_object_SEC61B_cell_Area', 'cell_AreaShape_Area')
            # Duplicates (cell-level and coloc features) will get automatic suffixes from AnnData
            group_var_names = adata.var_names.tolist()
            var_names.extend(group_var_names)
        else:
            # DinoV3: generate indexed names (original behavior)
            # Creates: "Phase_0", "Phase_1", ..., "EEA1_0", ...
            group_var_names = [f"{short_label}_{i}" for i in range(n_features)]
            var_names.extend(group_var_names)

        # Track feature slices
        feature_slices[bio_signal] = {
            "start": current_idx,
            "end": current_idx + n_features,
            "n_features": n_features,
            "short_label": short_label,
        }
        current_idx += n_features

        if verbose:
            print(
                f"  {bio_signal} ({short_label}): features [{feature_slices[bio_signal]['start']}:{feature_slices[bio_signal]['end']}]"
            )

    # Create combined X matrix
    X_combined = np.hstack(X_list)

    if verbose:
        print(f"\n  Combined feature matrix: {X_combined.shape}")

    # Use first group's observations (all aligned)
    obs_df = filtered_groups[bio_signals_list[0]].obs.copy()

    # Create combined AnnData
    adata_combined = ad.AnnData(X=X_combined, obs=obs_df)
    adata_combined.var_names = var_names

    # Store feature slices for later reference
    adata_combined.uns["feature_slices"] = feature_slices
    adata_combined.uns["biological_signals"] = list(bio_signals_list)

    return adata_combined


def _create_comprehensive_metadata(
    biological_signals: Dict[str, List[Tuple[str, str]]],
    group_adatas: Dict[str, ad.AnnData],
    meta: FeatureMetadata,
    feature_type: str,
    feature_slices: Dict[str, Dict[str, Any]],
    target_level: str,
) -> Dict[str, Any]:
    """
    Create comprehensive metadata structure.

    Parameters
    ----------
    biological_signals : Dict[str, List[Tuple[str, str]]]
        Original grouping by biological signal
    group_adatas : Dict[str, ad.AnnData]
        Processed AnnData for each group
    meta : FeatureMetadata
        Metadata manager
    feature_type : str
        Feature type
    feature_slices : Dict[str, Dict[str, Any]]
        Feature index ranges for each biological signal
    target_level : str
        Aggregation level

    Returns
    -------
    Dict[str, Any]
        Comprehensive metadata structure
    """
    biological_groups = {}

    for bio_signal, exp_channel_pairs in biological_signals.items():
        adata = group_adatas[bio_signal]

        # Determine aggregation type
        agg_type = "vertical" if len(exp_channel_pairs) > 1 else "horizontal"

        # Extract cell counts and experiment info
        if agg_type == "vertical" and "vertical_metadata" in adata.uns:
            cell_counts = adata.uns["vertical_metadata"]["cell_counts_per_experiment"]
            n_cells_total = adata.uns["vertical_metadata"]["total_cells"]
            experiments = adata.uns["vertical_metadata"]["experiments"]
        elif "horizontal_metadata" in adata.uns:
            exp = adata.uns["horizontal_metadata"]["experiment"]
            n_cells = adata.uns["horizontal_metadata"]["n_cells"]
            cell_counts = {exp: n_cells}
            n_cells_total = n_cells
            experiments = [exp]
        else:
            # Fallback
            experiments = [exp for exp, _ in exp_channel_pairs]
            cell_counts = {}
            n_cells_total = 0

        # Get short label
        exp_short = exp_channel_pairs[0][0].split("_")[0]
        channel = exp_channel_pairs[0][1]
        short_label = meta.get_short_label(exp_short, channel)

        biological_groups[bio_signal] = {
            "biological_signal": bio_signal,
            "short_label": short_label,
            "aggregation_type": agg_type,
            "experiments": experiments,
            "channels": [ch for _, ch in exp_channel_pairs],
            "n_cells_per_experiment": cell_counts,
            "n_cells_total": n_cells_total,
            "n_features": feature_slices[bio_signal]["n_features"],
            "feature_range": [
                feature_slices[bio_signal]["start"],
                feature_slices[bio_signal]["end"],
            ],
        }

    metadata = {
        "strategy": "comprehensive",
        "feature_type": feature_type,
        "aggregation_level": target_level,
        "n_biological_signals": len(biological_signals),
        "biological_groups": biological_groups,
        "feature_slices": feature_slices,
    }

    return metadata


def _run_leiden_clustering(
    adata: ad.AnnData,
    resolutions: List[float],
    random_state: Optional[int] = None,
    verbose: bool = True,
) -> ad.AnnData:
    """
    Run Leiden clustering at multiple resolutions.

    Requires that sc.pp.neighbors() has already been called.

    Args:
        adata: AnnData object with neighbors graph
        resolutions: List of resolution parameters
        random_state: Random seed for reproducibility
        verbose: Print progress

    Returns:
        AnnData with leiden_{resolution} columns in .obs
    """
    if "neighbors" not in adata.uns:
        raise ValueError(
            "Neighbors graph not found. Run sc.pp.neighbors() first or set "
            "recompute_embeddings=True to compute embeddings before clustering."
        )

    if verbose:
        print(f"\nRunning Leiden clustering at {len(resolutions)} resolution(s)...")

    for res in resolutions:
        key = f"leiden_{res}"
        if verbose:
            print(f"  Resolution {res}...")
        sc.tl.leiden(adata, resolution=res, key_added=key, random_state=random_state)
        if verbose:
            n_clusters = adata.obs[key].nunique()
            print(f"    → {n_clusters} clusters")

    return adata


def concatenate_experiments_comprehensive(
    experiments_channels: List[Tuple[str, str]],
    feature_type: str = "dinov3",
    base_dir: Union[str, Path] = "/hpc/projects/intracellular_dashboard/ops",
    recompute_embeddings: bool = True,
    n_pca_components: int = 128,
    n_umap_neighbors: int = 15,
    join: str = "inner",
    verbose: bool = True,
    subsample_controls: bool = False,
    control_gene: str = "NTC",
    control_group_size: int = 4,
    random_seed: Optional[int] = None,
    fit_on_aggregated_controls: bool = False,
    use_pca_for_umap: bool = True,
    leiden_resolutions: Optional[List[float]] = None,
    normalize_on_pooling: bool = True,
    normalize_on_controls: bool = False,
) -> Tuple[ad.AnnData, ad.AnnData]:
    """
    Comprehensively combine experiments using biology-driven aggregation.

    This function automatically groups experiment/channel pairs by their biological
    signal (using FeatureMetadata), then:
    - Vertically aggregates (pools cells) for same biological signals
    - Horizontally concatenates (separate features) for different biological signals

    Memory-efficient: Aggregates cells to guide/gene level immediately per experiment,
    then combines aggregated data (avoids storing all cells in memory).

    Parameters
    ----------
    experiments_channels : List[Tuple[str, str]]
        List of (experiment, channel) pairs to combine.
        Example: [("ops0089", "Phase2D"), ("ops0089", "GFP"), ("ops0108", "Phase2D")]

    feature_type : str, default="dinov3"
        Type of features: "dinov3", "cellprofiler", etc.

    base_dir : Union[str, Path]
        Base directory for OPS data

    recompute_embeddings : bool, default=True
        Whether to compute PCA/UMAP on combined feature space

    n_pca_components : int, default=128
        Number of PCA components

    n_umap_neighbors : int, default=15
        Number of neighbors for UMAP

    join : str, default="inner"
        How to handle observations: "inner" (intersection) or "outer" (union)

    verbose : bool, default=True
        Print progress information

    subsample_controls : bool, default=False
        Split control gene guides into random groups at gene level

    control_gene : str, default="NTC"
        Gene name to subsample

    control_group_size : int, default=4
        Number of guides per control group

    random_seed : Optional[int], default=None
        Random seed for reproducible control grouping

    fit_on_aggregated_controls : bool, default=False
        If True, fit embeddings on data with aggregated controls, then transform
        subsampled controls. This reveals how control subgroups distribute in an
        embedding space defined by target genes + single control observation.

    use_pca_for_umap : bool, default=True
        Whether to use PCA features for UMAP (only relevant if fit_on_aggregated_controls=True)

    leiden_resolutions : Optional[List[float]], default=None
        List of resolution parameters for Leiden clustering. If provided, clustering
        will be performed at each resolution on both guide and gene levels, with results
        stored as "leiden_{resolution}" columns in .obs. Requires that embeddings are
        computed (recompute_embeddings=True). Example: [0.5, 1.0, 1.5]

    normalize_on_pooling : bool, default=True
        If True, z-score normalize cells from each experiment before aggregation.
        This ensures features from different experiments are on the same scale.
        If False, use raw features (assumes input files are already normalized or
        you want to preserve original units).

    normalize_on_controls : bool, default=False
        If True, compute normalization statistics (mean/std) from control cells only,
        then apply to all cells. If False, compute from all cells.
        Only used when normalize_on_pooling=True.

    Returns
    -------
    Tuple[ad.AnnData, ad.AnnData]
        (adata_guide, adata_gene) - Combined data at both aggregation levels

    Raises
    ------
    ValueError
        If experiments_channels is empty
        If tuples are malformed
        If no common observations found across groups
        If metadata not found for any experiment/channel
    FileNotFoundError
        If cell-level data files not found

    Example
    -------
    >>> experiments_channels = [
    ...     ("ops0089_20251119", "Phase2D"),  # Phase morphology
    ...     ("ops0089_20251119", "GFP"),      # early endosome, EEA1
    ...     ("ops0108_20251201", "Phase2D"),  # Phase morphology (same biology)
    ...     ("ops0108_20251201", "GFP"),      # mitochondria, TOMM70A
    ... ]
    >>> adata_guide, adata_gene = concatenate_experiments_comprehensive(
    ...     experiments_channels,
    ...     feature_type="dinov3"
    ... )
    >>> # Result: Phase features pooled, EEA1 and TOMM70A separate
    >>> print(adata_gene.shape)  # (n_genes, 3072) = 1024*3 features
    """

    # Validate inputs
    if not experiments_channels:
        raise ValueError("experiments_channels cannot be empty")

    if not all(
        isinstance(item, tuple) and len(item) == 2 for item in experiments_channels
    ):
        raise ValueError(
            "experiments_channels must be list of (experiment, channel) tuples. "
            f"Got: {experiments_channels}"
        )

    base_dir = Path(base_dir)

    if verbose:
        print("=" * 80)
        print("COMPREHENSIVE MULTI-EXPERIMENT COMBINATION")
        print("=" * 80)
        print(f"Sources: {len(experiments_channels)}")
        print(f"Feature type: {feature_type}")
        print(f"Join method: {join}")
        print("=" * 80)

    # Initialize metadata manager
    meta = FeatureMetadata()

    # Group by biological signal
    # This will raise ValueError if any metadata is missing
    biological_signals = _group_by_biological_signal(
        experiments_channels, meta, verbose, feature_type
    )

    if verbose:
        print(f"\nIdentified {len(biological_signals)} biological signal group(s)")

    # Edge case 1: All same biology → pure vertical aggregation
    if len(biological_signals) == 1:
        bio_signal = list(biological_signals.keys())[0]
        pairs = biological_signals[bio_signal]

        if verbose:
            print("\n⚠ All channels represent same biological signal")
            print(f"  Signal: {bio_signal}")
            print(f"  Sources: {len(pairs)}")
            print("  → Falling back to vertical concatenation")

        # Use vertical aggregation at guide level only
        adata_guide = _process_vertical_group(
            pairs,
            feature_type,
            base_dir,
            "guide",
            join,
            verbose,
            False,
            control_gene,
            control_group_size,
            random_seed,  # No subsampling at guide level
            normalize_on_pooling,
            normalize_on_controls,
        )

        # Recompute embeddings on guide level
        if recompute_embeddings:
            if verbose:
                print("\nRecomputing guide-level embeddings...")
            _recompute_fn = globals()["recompute_embeddings"]
            adata_guide = _recompute_fn(
                adata_guide,
                n_pca_components=n_pca_components,
                n_umap_neighbors=n_umap_neighbors,
            )

        # Create gene level from guide level
        if verbose:
            print("\nCreating gene-level aggregation from guide-level...")

        if fit_on_aggregated_controls and subsample_controls:
            # Use fitted embeddings approach
            adata_gene = create_embeddings_fit_on_aggregated_controls(
                adata_guide,
                control_gene=control_gene,
                control_group_size=control_group_size,
                random_seed=random_seed,
                n_pca_components=n_pca_components,
                n_umap_neighbors=n_umap_neighbors,
                use_pca_for_umap=use_pca_for_umap,
                compute_phate=True,
            )
        else:
            # Standard aggregation
            adata_gene = aggregate_to_level(
                adata_guide,
                level="gene",
                method="mean",
                preserve_batch_info=False,
                subsample_controls=subsample_controls,
                control_gene=control_gene,
                control_group_size=control_group_size,
                random_seed=random_seed,
            )
            if recompute_embeddings:
                adata_gene = _recompute_fn(
                    adata_gene,
                    n_pca_components=n_pca_components,
                    n_umap_neighbors=n_umap_neighbors,
                )

        if verbose:
            print("\n" + "=" * 80)
            print("VERTICAL COMBINATION COMPLETE (fallback)")
            print("=" * 80)
            print(f"Guide-level: {adata_guide.shape}")
            print(f"Gene-level: {adata_gene.shape}")
            print("=" * 80)

        return adata_guide, adata_gene

    # Edge case 2: All different biology → pure horizontal concatenation
    all_single_source = all(len(pairs) == 1 for pairs in biological_signals.values())
    if all_single_source:
        if verbose:
            print("\n⚠ All channels represent different biological signals")
            print(f"  Signals: {len(biological_signals)}")
            print("  → Falling back to horizontal concatenation")

        # Process each as horizontal group (guide level only)
        group_adatas_guide = {}

        for bio_signal, exp_channel_pairs in biological_signals.items():
            exp, channel = exp_channel_pairs[0]
            if verbose:
                print(f"\n  Processing {exp}/{channel}...")

            adata_guide = _process_horizontal_group(
                exp,
                channel,
                feature_type,
                base_dir,
                "guide",
                verbose,
                False,
                control_gene,
                control_group_size,
                random_seed,  # No subsampling at guide level
                normalize_on_pooling,
                normalize_on_controls,
            )

            group_adatas_guide[bio_signal] = adata_guide

        # Align and concatenate at guide level
        common_guides = _align_biological_groups(
            group_adatas_guide, "guide", join, verbose
        )

        adata_guide = _concatenate_horizontal(
            group_adatas_guide,
            common_guides,
            biological_signals,
            meta,
            feature_type,
            "guide",
            verbose,
        )

        # Recompute embeddings on guide level
        if recompute_embeddings:
            if verbose:
                print("\nRecomputing guide-level embeddings...")
            _recompute_fn = globals()["recompute_embeddings"]
            adata_guide = _recompute_fn(
                adata_guide,
                n_pca_components=n_pca_components,
                n_umap_neighbors=n_umap_neighbors,
            )

        # Create gene level from guide level
        if verbose:
            print("\nCreating gene-level aggregation from guide-level...")

        if fit_on_aggregated_controls and subsample_controls:
            # Use fitted embeddings approach
            adata_gene = create_embeddings_fit_on_aggregated_controls(
                adata_guide,
                control_gene=control_gene,
                control_group_size=control_group_size,
                random_seed=random_seed,
                n_pca_components=n_pca_components,
                n_umap_neighbors=n_umap_neighbors,
                use_pca_for_umap=use_pca_for_umap,
                compute_phate=True,
            )
        else:
            # Standard aggregation
            adata_gene = aggregate_to_level(
                adata_guide,
                level="gene",
                method="mean",
                preserve_batch_info=False,
                subsample_controls=subsample_controls,
                control_gene=control_gene,
                control_group_size=control_group_size,
                random_seed=random_seed,
            )
            if recompute_embeddings:
                adata_gene = _recompute_fn(
                    adata_gene,
                    n_pca_components=n_pca_components,
                    n_umap_neighbors=n_umap_neighbors,
                )

        if verbose:
            print("\n" + "=" * 80)
            print("HORIZONTAL COMBINATION COMPLETE (fallback)")
            print("=" * 80)
            print(f"Guide-level: {adata_guide.shape}")
            print(f"Gene-level: {adata_gene.shape}")
            print("=" * 80)

        return adata_guide, adata_gene

    if verbose:
        print("\nProcessing biological groups...")

    # Process at guide level only
    group_adatas_guide = {}

    for bio_signal, exp_channel_pairs in biological_signals.items():
        if len(exp_channel_pairs) > 1:
            # Vertical aggregation (same biology)
            if verbose:
                print(f"\n[VERTICAL] {bio_signal}")

            adata_guide = _process_vertical_group(
                exp_channel_pairs,
                feature_type,
                base_dir,
                "guide",
                join,
                verbose,
                False,
                control_gene,
                control_group_size,
                random_seed,  # No subsampling at guide level
                normalize_on_pooling,
                normalize_on_controls,
            )
        else:
            # Horizontal (single source)
            exp, channel = exp_channel_pairs[0]
            if verbose:
                print(f"\n[HORIZONTAL] {bio_signal}")

            adata_guide = _process_horizontal_group(
                exp,
                channel,
                feature_type,
                base_dir,
                "guide",
                verbose,
                False,
                control_gene,
                control_group_size,
                random_seed,  # No subsampling at guide level
                normalize_on_pooling,
                normalize_on_controls,
            )

        group_adatas_guide[bio_signal] = adata_guide

    # ========================================================================
    # Phase 3: Align observations and concatenate horizontally
    # ========================================================================

    # Align at guide level
    common_guides = _align_biological_groups(group_adatas_guide, "guide", join, verbose)

    # Concatenate horizontally at guide level
    adata_guide = _concatenate_horizontal(
        group_adatas_guide,
        common_guides,
        biological_signals,
        meta,
        feature_type,
        "guide",
        verbose,
    )

    # ========================================================================
    # Phase 4: Add comprehensive metadata
    # ========================================================================

    if verbose:
        print("\nAdding comprehensive metadata to guide-level...")

    # Add comprehensive metadata to guide level
    adata_guide.uns["comprehensive_metadata"] = _create_comprehensive_metadata(
        biological_signals=biological_signals,
        group_adatas=group_adatas_guide,
        meta=meta,
        feature_type=feature_type,
        feature_slices=adata_guide.uns["feature_slices"],
        target_level="guide",
    )

    # ========================================================================
    # Phase 5: Recompute embeddings on guide level
    # ========================================================================

    if recompute_embeddings:
        if verbose:
            print("\nRecomputing guide-level embeddings on combined feature space...")

        # Store reference to avoid shadowing
        _recompute_fn = globals()["recompute_embeddings"]

        adata_guide = _recompute_fn(
            adata_guide,
            n_pca_components=n_pca_components,
            n_umap_neighbors=n_umap_neighbors,
        )

    # ========================================================================
    # Phase 6: Create gene level from guide level
    # ========================================================================

    if verbose:
        print("\nCreating gene-level aggregation from guide-level...")

    if fit_on_aggregated_controls and subsample_controls:
        # Use fitted embeddings approach
        adata_gene = create_embeddings_fit_on_aggregated_controls(
            adata_guide,
            control_gene=control_gene,
            control_group_size=control_group_size,
            random_seed=random_seed,
            n_pca_components=n_pca_components,
            n_umap_neighbors=n_umap_neighbors,
            use_pca_for_umap=use_pca_for_umap,
            compute_phate=True,
        )
    else:
        # Standard aggregation
        adata_gene = aggregate_to_level(
            adata_guide,
            level="gene",
            method="mean",
            preserve_batch_info=False,
            subsample_controls=subsample_controls,
            control_gene=control_gene,
            control_group_size=control_group_size,
            random_seed=random_seed,
        )
        if recompute_embeddings:
            if verbose:
                print("\nRecomputing gene-level embeddings...")
            adata_gene = _recompute_fn(
                adata_gene,
                n_pca_components=n_pca_components,
                n_umap_neighbors=n_umap_neighbors,
            )

    # Add metadata to gene level (using guide-level group info)
    if verbose:
        print("\nAdding comprehensive metadata to gene-level...")

    # Copy feature_slices from guide level (same features)
    adata_gene.uns["feature_slices"] = adata_guide.uns["feature_slices"].copy()

    adata_gene.uns["comprehensive_metadata"] = _create_comprehensive_metadata(
        biological_signals=biological_signals,
        group_adatas=group_adatas_guide,  # Use guide-level groups for cell counts
        meta=meta,
        feature_type=feature_type,
        feature_slices=adata_gene.uns["feature_slices"],
        target_level="gene",
    )

    # ========================================================================
    # Leiden Clustering
    # ========================================================================

    if leiden_resolutions is not None and len(leiden_resolutions) > 0:
        if verbose:
            print("\n" + "=" * 80)
            print("LEIDEN CLUSTERING")
            print("=" * 80)

        # Cluster guide level
        if verbose:
            print("\nGuide-level clustering:")
        adata_guide = _run_leiden_clustering(
            adata_guide,
            resolutions=leiden_resolutions,
            random_state=random_seed,
            verbose=verbose,
        )

        # Cluster gene level
        if verbose:
            print("\nGene-level clustering:")
        adata_gene = _run_leiden_clustering(
            adata_gene,
            resolutions=leiden_resolutions,
            random_state=random_seed,
            verbose=verbose,
        )

        if verbose:
            print("=" * 80)

    # ========================================================================
    # Summary
    # ========================================================================

    if verbose:
        print("\n" + "=" * 80)
        print("COMPREHENSIVE COMBINATION COMPLETE")
        print("=" * 80)
        print(f"Guide-level: {adata_guide.shape}")
        print(f"Gene-level: {adata_gene.shape}")
        print(f"Biological signals: {len(biological_signals)}")
        for bio_signal, info in adata_gene.uns["comprehensive_metadata"][
            "biological_groups"
        ].items():
            agg_type = info["aggregation_type"]
            n_cells = info["n_cells_total"]
            print(f"  [{agg_type.upper():10s}] {bio_signal}: {n_cells} cells")
        print("=" * 80)

    return adata_guide, adata_gene


# Example usage script
def main():
    """Example usage of AnnData combination utilities."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Combine multiple AnnData objects from different experiments"
    )
    parser.add_argument(
        "--input_paths",
        nargs="+",
        required=True,
        help="Paths to .h5ad files to combine",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output path for combined .h5ad file",
    )
    parser.add_argument(
        "--batch_key", type=str, default="batch", help="Column name for batch tracking"
    )
    parser.add_argument(
        "--join",
        type=str,
        choices=["inner", "outer"],
        default="inner",
        help="How to handle different feature sets",
    )
    parser.add_argument(
        "--recompute_embeddings",
        action="store_true",
        help="Recompute PCA/UMAP on combined data",
    )
    parser.add_argument(
        "--n_pca", type=int, default=128, help="Number of PCA components"
    )

    args = parser.parse_args()

    # Combine
    adata_combined = concatenate_anndata_objects(
        args.input_paths, batch_key=args.batch_key, join=args.join
    )

    # Optional: recompute embeddings
    if args.recompute_embeddings:
        adata_combined = recompute_embeddings(
            adata_combined, n_pca_components=args.n_pca
        )

    # Save
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata_combined.write_h5ad(output_path)
    print(f"\nSaved combined AnnData to {output_path}")

    # Summary
    print("\nBatch summary:")
    print(adata_combined.obs[args.batch_key].value_counts())


if __name__ == "__main__":
    main()
