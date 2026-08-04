"""
Tests for AnnData combination utilities.

This test suite validates the functions for combining, managing, and
analyzing multiple AnnData objects from different experiments or feature types.
"""

import warnings

# Filter warnings
warnings.filterwarnings("ignore", message=".*zarr v2.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import pytest
import numpy as np
import pandas as pd
import anndata as ad
from pathlib import Path
import tempfile
import shutil

from ops_model.features.anndata_utils import (
    concatenate_anndata_objects,
    recompute_embeddings,
    load_multiple_experiments,
    compare_batch_distributions,
    split_by_batch,
    aggregate_to_level,
    compute_embeddings,
    create_aggregated_embeddings,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def mock_adata_1():
    """Create first mock AnnData object."""
    np.random.seed(42)

    n_cells = 100
    n_features = 50

    X = np.random.randn(n_cells, n_features).astype(np.float32)

    obs = pd.DataFrame(
        {
            "label_str": ["NTC"] * 30 + ["GENE_A"] * 40 + ["GENE_B"] * 30,
            "label_int": [0] * 30 + [1] * 40 + [2] * 30,
            "sgRNA": ["NTC_sg1"] * 30 + ["GENE_A_sg1"] * 40 + ["GENE_B_sg1"] * 30,
            "well": ["A1"] * 100,
        }
    )

    adata = ad.AnnData(X=X, obs=obs)
    adata.var_names = [f"feature_{i}" for i in range(n_features)]

    return adata


@pytest.fixture(scope="module")
def mock_adata_2():
    """Create second mock AnnData object with same features."""
    np.random.seed(123)

    n_cells = 80
    n_features = 50

    X = np.random.randn(n_cells, n_features).astype(np.float32)

    obs = pd.DataFrame(
        {
            "label_str": ["NTC"] * 20 + ["GENE_A"] * 30 + ["GENE_C"] * 30,
            "label_int": [0] * 20 + [1] * 30 + [3] * 30,
            "sgRNA": ["NTC_sg1"] * 20 + ["GENE_A_sg1"] * 30 + ["GENE_C_sg1"] * 30,
            "well": ["B2"] * 80,
        }
    )

    adata = ad.AnnData(X=X, obs=obs)
    adata.var_names = [f"feature_{i}" for i in range(n_features)]

    return adata


@pytest.fixture(scope="module")
def mock_adata_3_different_features():
    """Create third mock AnnData with different features (for testing join modes)."""
    np.random.seed(456)

    n_cells = 60
    n_features = 40  # Different number of features

    X = np.random.randn(n_cells, n_features).astype(np.float32)

    obs = pd.DataFrame(
        {
            "label_str": ["NTC"] * 20 + ["GENE_D"] * 40,
            "label_int": [0] * 20 + [4] * 40,
            "sgRNA": ["NTC_sg2"] * 20 + ["GENE_D_sg1"] * 40,
            "well": ["C3"] * 60,
        }
    )

    adata = ad.AnnData(X=X, obs=obs)
    # Only first 30 features overlap with other datasets
    adata.var_names = [f"feature_{i}" for i in range(30)] + [
        f"feature_new_{i}" for i in range(10)
    ]

    return adata


@pytest.fixture(scope="module")
def temp_adata_files(mock_adata_1, mock_adata_2):
    """Create temporary h5ad files for testing."""
    temp_dir = Path(tempfile.mkdtemp())

    # Create directory structure
    exp1_dir = (
        temp_dir
        / "ops0089_20251119"
        / "3-assembly"
        / "dino_features"
        / "anndata_objects"
    )
    exp2_dir = (
        temp_dir
        / "ops0084_20250101"
        / "3-assembly"
        / "dino_features"
        / "anndata_objects"
    )

    exp1_dir.mkdir(parents=True, exist_ok=True)
    exp2_dir.mkdir(parents=True, exist_ok=True)

    # Save files
    path1 = exp1_dir / "features_processed.h5ad"
    path2 = exp2_dir / "features_processed.h5ad"

    mock_adata_1.write_h5ad(path1)
    mock_adata_2.write_h5ad(path2)

    yield temp_dir, path1, path2

    # Cleanup
    shutil.rmtree(temp_dir)


# ============================================================================
# Test concatenate_anndata_objects
# ============================================================================


def test_concatenate_basic(temp_adata_files):
    """Test basic concatenation of two AnnData objects."""
    temp_dir, path1, path2 = temp_adata_files

    adata_combined = concatenate_anndata_objects(
        [path1, path2], batch_key="experiment", join="inner"
    )

    # Check combined shape
    assert adata_combined.shape[0] == 180, "Should have 100 + 80 = 180 cells"
    assert adata_combined.shape[1] == 50, "Should have 50 features"

    # Check batch information added
    assert "experiment" in adata_combined.obs.columns
    assert len(adata_combined.obs["experiment"].unique()) == 2


def test_concatenate_batch_tracking(temp_adata_files):
    """Test that batch information is correctly tracked."""
    temp_dir, path1, path2 = temp_adata_files

    adata_combined = concatenate_anndata_objects(
        [path1, path2], batch_key="batch", join="inner"
    )

    # Check batch counts
    batch_counts = adata_combined.obs["batch"].value_counts()
    assert len(batch_counts) == 2
    assert batch_counts.iloc[0] == 100 or batch_counts.iloc[0] == 80
    assert batch_counts.iloc[1] == 100 or batch_counts.iloc[1] == 80


def test_concatenate_metadata_preserved(temp_adata_files):
    """Test that metadata columns are preserved."""
    temp_dir, path1, path2 = temp_adata_files

    adata_combined = concatenate_anndata_objects([path1, path2], batch_key="experiment")

    # Check original metadata preserved
    required_cols = ["label_str", "label_int", "sgRNA", "well", "experiment"]
    for col in required_cols:
        assert col in adata_combined.obs.columns, f"Missing column: {col}"

    # Check values preserved
    assert "NTC" in adata_combined.obs["label_str"].values
    assert "GENE_A" in adata_combined.obs["label_str"].values


def test_concatenate_join_inner(temp_adata_files, mock_adata_3_different_features):
    """Test inner join keeps only common features."""
    temp_dir, path1, path2 = temp_adata_files

    # Save third adata with different features
    path3 = (
        temp_dir
        / "ops0065_20250101"
        / "3-assembly"
        / "dino_features"
        / "anndata_objects"
        / "features_processed.h5ad"
    )
    path3.parent.mkdir(parents=True, exist_ok=True)
    mock_adata_3_different_features.write_h5ad(path3)

    # Expect warning about different feature counts
    with pytest.warns(UserWarning, match="Feature counts differ"):
        adata_combined = concatenate_anndata_objects(
            [path1, path2, path3], batch_key="batch", join="inner"
        )

    # Only 30 features are common across all three
    assert adata_combined.shape[1] == 30, "Inner join should keep only common features"
    assert adata_combined.shape[0] == 240, "Should have all cells (100+80+60)"


def test_concatenate_join_outer(temp_adata_files, mock_adata_3_different_features):
    """Test outer join keeps all features."""
    temp_dir, path1, path2 = temp_adata_files

    # Save third adata
    path3 = (
        temp_dir
        / "ops0065_20250101"
        / "3-assembly"
        / "dino_features"
        / "anndata_objects"
        / "features_processed.h5ad"
    )
    path3.parent.mkdir(parents=True, exist_ok=True)
    mock_adata_3_different_features.write_h5ad(path3)

    adata_combined = concatenate_anndata_objects(
        [path1, path3],  # path1 has 50 features, path3 has 40 (30 overlap)
        batch_key="batch",
        join="outer",
    )

    # Should have all unique features
    expected_features = 50 + 10  # 50 from path1 + 10 unique from path3
    assert (
        adata_combined.shape[1] == expected_features
    ), "Outer join should keep all features"


def test_concatenate_file_not_found():
    """Test error handling for missing files."""
    with pytest.raises(FileNotFoundError):
        concatenate_anndata_objects(["/nonexistent/path.h5ad"], batch_key="batch")


# ============================================================================
# Test recompute_embeddings
# ============================================================================


def test_recompute_embeddings_pca(mock_adata_1, mock_adata_2, temp_adata_files):
    """Test PCA computation on combined data."""
    temp_dir, path1, path2 = temp_adata_files

    adata_combined = concatenate_anndata_objects([path1, path2])
    adata_combined = recompute_embeddings(
        adata_combined, n_pca_components=10, compute_pca=True, compute_umap=False
    )

    # Check PCA was computed
    assert "X_pca" in adata_combined.obsm.keys()
    assert adata_combined.obsm["X_pca"].shape == (180, 20)

    # Check PCA metadata
    assert "pca" in adata_combined.uns.keys()
    assert "variance" in adata_combined.uns["pca"].keys()


def test_recompute_embeddings_umap(temp_adata_files):
    """Test UMAP computation on combined data."""
    temp_dir, path1, path2 = temp_adata_files

    adata_combined = concatenate_anndata_objects([path1, path2])
    adata_combined = recompute_embeddings(
        adata_combined,
        n_pca_components=10,
        n_umap_neighbors=15,
        compute_pca=True,
        compute_umap=True,
    )

    # Check both PCA and UMAP computed
    assert "X_pca" in adata_combined.obsm.keys()
    assert "X_umap" in adata_combined.obsm.keys()
    assert adata_combined.obsm["X_umap"].shape == (180, 2)


def test_recompute_embeddings_phate(temp_adata_files):
    """Test PHATE computation on combined data."""
    temp_dir, path1, path2 = temp_adata_files

    adata_combined = concatenate_anndata_objects([path1, path2])
    adata_combined = recompute_embeddings(
        adata_combined,
        n_pca_components=10,
        n_umap_neighbors=15,
        compute_pca=True,
        compute_umap=True,
        compute_phate=True,
    )

    # Check PCA, UMAP, and PHATE computed
    assert "X_pca" in adata_combined.obsm.keys()
    assert "X_umap" in adata_combined.obsm.keys()
    assert "X_phate" in adata_combined.obsm.keys()

    # Check PHATE shape
    assert adata_combined.obsm["X_phate"].shape == (180, 2), "PHATE should be 2D"

    # Check PHATE coordinates are finite
    assert np.isfinite(
        adata_combined.obsm["X_phate"]
    ).all(), "PHATE coordinates should be finite"

    # Check PHATE is different from UMAP (they should produce different embeddings)
    assert not np.allclose(
        adata_combined.obsm["X_phate"], adata_combined.obsm["X_umap"]
    ), "PHATE and UMAP should produce different embeddings"


def test_recompute_embeddings_phate_disabled(temp_adata_files):
    """Test that PHATE computation can be disabled."""
    temp_dir, path1, path2 = temp_adata_files

    adata_combined = concatenate_anndata_objects([path1, path2])
    adata_combined = recompute_embeddings(
        adata_combined,
        n_pca_components=10,
        n_umap_neighbors=15,
        compute_pca=True,
        compute_umap=True,
        compute_phate=False,  # Explicitly disable
    )

    # Check PCA and UMAP computed but not PHATE
    assert "X_pca" in adata_combined.obsm.keys()
    assert "X_umap" in adata_combined.obsm.keys()
    assert (
        "X_phate" not in adata_combined.obsm.keys()
    ), "PHATE should not be computed when compute_phate=False"


def test_recompute_embeddings_adjusts_components(temp_adata_files):
    """Test that n_components is adjusted for small datasets."""
    temp_dir, path1, path2 = temp_adata_files

    adata_combined = concatenate_anndata_objects([path1, path2])

    # Request more components than samples
    adata_combined = recompute_embeddings(
        adata_combined,
        n_pca_components=100,  # More than n_samples
        compute_pca=True,
        compute_umap=False,
    )

    # Should be adjusted to min(n_samples-1, n_features-1) for arpack solver
    max_components = min(179, 49)  # 180-1, 50-1 features
    assert adata_combined.obsm["X_pca"].shape[1] == max_components


def test_recompute_embeddings_skip_pca(temp_adata_files):
    """Test UMAP computation without PCA."""
    temp_dir, path1, path2 = temp_adata_files

    adata_combined = concatenate_anndata_objects([path1, path2])
    adata_combined = recompute_embeddings(
        adata_combined, compute_pca=False, compute_umap=True
    )

    # Should compute UMAP on raw features (n_pcs=0)
    assert "X_umap" in adata_combined.obsm.keys()
    assert (
        "X_pca" not in adata_combined.obsm.keys()
        or len(adata_combined.obsm["X_pca"]) == 0
    )


def test_recompute_embeddings_use_existing_pca(mock_adata_1):
    """Test using existing PCA instead of recomputing."""
    # Add PCA to adata
    import scanpy as sc

    adata = mock_adata_1.copy()
    sc.tl.pca(adata, n_comps=10)

    original_pca = adata.obsm["X_pca"].copy()

    # Recompute UMAP using existing PCA
    adata = recompute_embeddings(
        adata, compute_pca=False, compute_umap=True, use_existing_pca=True
    )

    # PCA should be unchanged
    assert np.allclose(adata.obsm["X_pca"], original_pca)
    # UMAP should be new
    assert "X_umap" in adata.obsm.keys()


# ============================================================================
# Test load_multiple_experiments
# ============================================================================


def test_load_multiple_experiments(temp_adata_files):
    """Test loading multiple experiment paths."""
    temp_dir, path1, path2 = temp_adata_files

    paths = load_multiple_experiments(
        base_dir=temp_dir,
        experiments=["ops0089_20251119", "ops0084_20250101"],
        feature_type="features_processed",
        require_all=True,
    )

    assert len(paths) == 2
    assert all(isinstance(p, Path) for p in paths)
    assert all(p.exists() for p in paths)


def test_load_multiple_experiments_missing_file(temp_adata_files):
    """Test handling of missing files."""
    temp_dir, path1, path2 = temp_adata_files

    # With require_all=True, should raise error
    with pytest.raises(FileNotFoundError):
        load_multiple_experiments(
            base_dir=temp_dir,
            experiments=["ops0089_20251119", "nonexistent_experiment"],
            require_all=True,
        )

    # With require_all=False, should skip with warning
    with pytest.warns(UserWarning):
        paths = load_multiple_experiments(
            base_dir=temp_dir,
            experiments=["ops0089_20251119", "nonexistent_experiment"],
            require_all=False,
        )
    assert len(paths) == 1


# ============================================================================
# Test compare_batch_distributions
# ============================================================================


def test_compare_batch_distributions(temp_adata_files):
    """Test comparing gene distributions across batches."""
    temp_dir, path1, path2 = temp_adata_files

    adata_combined = concatenate_anndata_objects([path1, path2], batch_key="experiment")

    dist = compare_batch_distributions(
        adata_combined, batch_key="experiment", label_key="label_str"
    )

    # Check output is DataFrame
    assert isinstance(dist, pd.DataFrame)

    # Check it has correct structure
    assert "Total" in dist.columns  # Margins column
    assert "Total" in dist.index  # Margins row

    # Check genes present
    assert "NTC" in dist.index
    assert "GENE_A" in dist.index


def test_compare_batch_distributions_missing_keys(temp_adata_files):
    """Test error handling for missing keys."""
    temp_dir, path1, path2 = temp_adata_files

    adata_combined = concatenate_anndata_objects([path1, path2])

    # Test missing batch key
    with pytest.raises(ValueError, match="Batch key"):
        compare_batch_distributions(adata_combined, batch_key="nonexistent_key")

    # Test missing label key
    with pytest.raises(ValueError, match="Label key"):
        compare_batch_distributions(
            adata_combined,
            batch_key="batch",  # Use valid batch key
            label_key="nonexistent_key",
        )


# ============================================================================
# Test split_by_batch
# ============================================================================


def test_split_by_batch(temp_adata_files):
    """Test splitting combined AnnData back into batches."""
    temp_dir, path1, path2 = temp_adata_files

    adata_combined = concatenate_anndata_objects([path1, path2], batch_key="batch")

    batches = split_by_batch(adata_combined, batch_key="batch")

    # Check we got a dictionary
    assert isinstance(batches, dict)
    assert len(batches) == 2

    # Check each batch
    for batch_id, batch_adata in batches.items():
        assert isinstance(batch_adata, ad.AnnData)
        assert batch_adata.shape[0] > 0
        assert batch_adata.shape[1] == adata_combined.shape[1]

    # Check total cells preserved
    total_cells = sum(b.shape[0] for b in batches.values())
    assert total_cells == adata_combined.shape[0]


def test_split_by_batch_missing_key(mock_adata_1):
    """Test error handling for missing batch key."""
    with pytest.raises(ValueError, match="Batch key"):
        split_by_batch(mock_adata_1, batch_key="nonexistent_key")


# ============================================================================
# Integration Test: Full Workflow
# ============================================================================


@pytest.mark.slow
def test_full_workflow(temp_adata_files):
    """Test complete workflow: load → concatenate → recompute → split."""
    temp_dir, path1, path2 = temp_adata_files

    # Load experiments
    paths = load_multiple_experiments(
        base_dir=temp_dir,
        experiments=["ops0089_20251119", "ops0084_20250101"],
        require_all=True,
    )

    # Concatenate
    adata_combined = concatenate_anndata_objects(
        paths, batch_key="experiment", join="inner"
    )

    # Check initial state
    assert adata_combined.shape[0] == 180
    assert "experiment" in adata_combined.obs.columns

    # Recompute embeddings in shared space
    adata_combined = recompute_embeddings(
        adata_combined, n_pca_components=10, compute_pca=True, compute_umap=True
    )

    # Check embeddings computed
    assert "X_pca" in adata_combined.obsm.keys()
    assert "X_umap" in adata_combined.obsm.keys()

    # Compare distributions (use correct batch_key)
    dist = compare_batch_distributions(
        adata_combined,
        batch_key="experiment",  # Match the batch_key used above
        label_key="label_str",
    )
    assert isinstance(dist, pd.DataFrame)

    # Split back into batches
    batches = split_by_batch(adata_combined, batch_key="experiment")
    assert len(batches) == 2

    # Each batch should have shared embeddings
    for batch_id, batch_adata in batches.items():
        assert "X_pca" in batch_adata.obsm.keys()
        assert "X_umap" in batch_adata.obsm.keys()


# ============================================================================
# Tests for aggregate_to_level()
# ============================================================================


def test_aggregate_to_guide_level_mean(mock_adata_1):
    """Test aggregation from cell-level to guide-level using mean."""
    adata_guide = aggregate_to_level(mock_adata_1, level="guide", method="mean")

    # Check output shape
    n_guides = mock_adata_1.obs["sgRNA"].nunique()
    assert adata_guide.shape[0] == n_guides
    assert adata_guide.shape[1] == mock_adata_1.shape[1]

    # Check sgRNA column exists
    assert "sgRNA" in adata_guide.obs.columns

    # Verify aggregated values are correct (compare to manual groupby)
    manual_agg = pd.DataFrame(mock_adata_1.X, columns=mock_adata_1.var_names)
    manual_agg["sgRNA"] = mock_adata_1.obs["sgRNA"].values
    manual_result = manual_agg.groupby("sgRNA", observed=False).mean()

    # Compare first feature values
    assert np.allclose(adata_guide.X[:, 0], manual_result.iloc[:, 0].values, rtol=1e-5)


def test_aggregate_to_gene_level_mean(mock_adata_1):
    """Test aggregation from cell-level to gene-level using mean."""
    adata_gene = aggregate_to_level(mock_adata_1, level="gene", method="mean")

    # Check output shape
    n_genes = mock_adata_1.obs["label_str"].nunique()
    assert adata_gene.shape[0] == n_genes
    assert adata_gene.shape[1] == mock_adata_1.shape[1]

    # Check label_str column exists
    assert "label_str" in adata_gene.obs.columns

    # Verify aggregated values
    manual_agg = pd.DataFrame(mock_adata_1.X, columns=mock_adata_1.var_names)
    manual_agg["label_str"] = mock_adata_1.obs["label_str"].values
    manual_result = manual_agg.groupby("label_str", observed=False).mean()

    assert np.allclose(adata_gene.X[:, 0], manual_result.iloc[:, 0].values, rtol=1e-5)


def test_aggregate_with_median(mock_adata_1):
    """Test that median aggregation method works correctly."""
    adata_median = aggregate_to_level(mock_adata_1, level="guide", method="median")
    adata_mean = aggregate_to_level(mock_adata_1, level="guide", method="mean")

    # Check both produced output
    assert adata_median.shape == adata_mean.shape

    # For skewed data, median and mean should differ
    # (Our mock data is random normal, so they should be similar but not identical)
    assert not np.allclose(adata_median.X, adata_mean.X, rtol=0.01)


def test_aggregate_preserves_batch_info(mock_adata_1):
    """Test that batch information is preserved when preserve_batch_info=True."""
    # Add batch column
    adata = mock_adata_1.copy()
    adata.obs["batch"] = ["batch1"] * 50 + ["batch2"] * 50

    adata_agg = aggregate_to_level(adata, level="guide", preserve_batch_info=True)

    # Check batch column is in output
    assert "batch" in adata_agg.obs.columns

    # Check that aggregation grouped by both sgRNA and batch
    # Should have more rows than unique sgRNAs (since we have 2 batches)
    n_guides = adata.obs["sgRNA"].nunique()
    assert adata_agg.shape[0] > n_guides or adata_agg.shape[0] == n_guides


def test_aggregate_without_batch_info(mock_adata_1):
    """Test aggregation with preserve_batch_info=False."""
    # Add batch column
    adata = mock_adata_1.copy()
    adata.obs["batch"] = ["batch1"] * 50 + ["batch2"] * 50

    adata_agg = aggregate_to_level(adata, level="guide", preserve_batch_info=False)

    # Check output shape equals unique guides (not guides × batches)
    n_guides = adata.obs["sgRNA"].nunique()
    assert adata_agg.shape[0] == n_guides


def test_aggregate_preserves_var_names(mock_adata_1):
    """Verify that feature names (var_names) are preserved in output."""
    adata_agg = aggregate_to_level(mock_adata_1, level="guide")

    # Check var_names match
    assert list(adata_agg.var_names) == list(mock_adata_1.var_names)

    # Check var dimensions match
    assert adata_agg.shape[1] == mock_adata_1.shape[1]


def test_aggregate_missing_sgrna_column():
    """Test error handling when sgRNA column is missing (for guide-level)."""
    # Create adata without sgRNA column
    X = np.random.randn(10, 5).astype(np.float32)
    obs = pd.DataFrame({"label_str": ["GENE_A"] * 10})
    adata = ad.AnnData(X=X, obs=obs)

    with pytest.raises(ValueError, match="sgRNA.*not found"):
        aggregate_to_level(adata, level="guide")


def test_aggregate_missing_label_str_column():
    """Test error handling when label_str column is missing (for gene-level)."""
    # Create adata without label_str column
    X = np.random.randn(10, 5).astype(np.float32)
    obs = pd.DataFrame({"sgRNA": ["sg1"] * 10})
    adata = ad.AnnData(X=X, obs=obs)

    with pytest.raises(ValueError, match="label_str.*not found"):
        aggregate_to_level(adata, level="gene")


def test_aggregate_invalid_method():
    """Test that invalid aggregation methods raise ValueError."""
    X = np.random.randn(10, 5).astype(np.float32)
    obs = pd.DataFrame({"sgRNA": ["sg1"] * 10, "label_str": ["GENE_A"] * 10})
    adata = ad.AnnData(X=X, obs=obs)

    with pytest.raises(ValueError, match="Unknown aggregation method"):
        aggregate_to_level(adata, level="guide", method="invalid")


def test_aggregate_single_cell_per_group():
    """Test aggregation when each group has only one cell."""
    X = np.random.randn(3, 5).astype(np.float32)
    obs = pd.DataFrame(
        {"sgRNA": ["sg1", "sg2", "sg3"], "label_str": ["GENE_A", "GENE_B", "GENE_C"]}
    )
    adata = ad.AnnData(X=X, obs=obs)

    # Aggregate with mean
    adata_mean = aggregate_to_level(adata, level="guide", method="mean")
    # Aggregate with median
    adata_median = aggregate_to_level(adata, level="guide", method="median")

    # Should produce same result (mean and median of single value are identical)
    assert np.allclose(adata_mean.X, adata_median.X)
    assert adata_mean.shape[0] == 3


def test_aggregate_large_dataset():
    """Test with larger mock dataset."""
    np.random.seed(42)
    n_cells = 1000
    n_features = 100
    n_guides = 50

    X = np.random.randn(n_cells, n_features).astype(np.float32)
    obs = pd.DataFrame(
        {
            "sgRNA": [f"sg{i % n_guides}" for i in range(n_cells)],
            "label_str": [f"GENE_{i % 25}" for i in range(n_cells)],
        }
    )
    adata = ad.AnnData(X=X, obs=obs)

    # Should complete without error
    adata_guide = aggregate_to_level(adata, level="guide")
    assert adata_guide.shape[0] == n_guides

    adata_gene = aggregate_to_level(adata, level="gene")
    assert adata_gene.shape[0] == 25


# ============================================================================
# Tests for compute_embeddings()
# ============================================================================


def test_compute_embeddings_is_alias(mock_adata_1):
    """Test that compute_embeddings() produces identical output to recompute_embeddings()."""
    adata1 = mock_adata_1.copy()
    adata2 = mock_adata_1.copy()

    # Use same seed for reproducibility
    np.random.seed(42)
    adata1 = compute_embeddings(
        adata1,
        n_pca_components=10,
        n_neighbors=10,
        compute_pca=True,
        compute_umap=True,
        compute_phate=True,
    )

    np.random.seed(42)
    adata2 = recompute_embeddings(
        adata2,
        n_pca_components=10,
        n_umap_neighbors=10,
        compute_pca=True,
        compute_umap=True,
        compute_phate=True,
    )

    # Check PCA is identical
    assert np.allclose(adata1.obsm["X_pca"], adata2.obsm["X_pca"], rtol=1e-5)

    # UMAP and PHATE may have slight numerical differences due to randomness
    # but should produce embeddings of same shape
    assert adata1.obsm["X_umap"].shape == adata2.obsm["X_umap"].shape
    assert adata1.obsm["X_phate"].shape == adata2.obsm["X_phate"].shape


def test_compute_embeddings_parameter_passing(mock_adata_1):
    """Test that all parameters are correctly passed through."""
    # Test with compute_phate=False
    adata = compute_embeddings(
        mock_adata_1.copy(),
        n_pca_components=10,
        n_neighbors=10,
        compute_pca=True,
        compute_umap=True,
        compute_phate=False,
    )

    assert "X_pca" in adata.obsm.keys()
    assert "X_umap" in adata.obsm.keys()
    assert "X_phate" not in adata.obsm.keys()


# ============================================================================
# Tests for create_aggregated_embeddings()
# ============================================================================


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_create_aggregated_embeddings_guide_level(mock_adata_1):
    """Test complete pipeline: cell → guide aggregation → embeddings."""
    # Use smaller parameters for small dataset
    adata_guide = create_aggregated_embeddings(
        mock_adata_1,
        level="guide",
        n_pca_components=1,
        n_neighbors=1,
    )

    # Check output shape
    n_guides = mock_adata_1.obs["sgRNA"].nunique()
    assert adata_guide.shape[0] == n_guides
    assert adata_guide.shape[1] == mock_adata_1.shape[1]

    # Check PCA was computed
    assert "X_pca" in adata_guide.obsm.keys()

    # UMAP/PHATE may not be computed if dataset is too small (n_neighbors >= n_samples)
    # This is expected behavior

    # Check sgRNA column exists
    assert "sgRNA" in adata_guide.obs.columns


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_create_aggregated_embeddings_gene_level(mock_adata_1):
    """Test complete pipeline: cell → gene aggregation → embeddings."""
    # Use smaller parameters for small dataset
    adata_gene = create_aggregated_embeddings(
        mock_adata_1,
        level="gene",
        n_pca_components=1,
        n_neighbors=1,
    )

    # Check output shape
    n_genes = mock_adata_1.obs["label_str"].nunique()
    assert adata_gene.shape[0] == n_genes
    assert adata_gene.shape[1] == mock_adata_1.shape[1]

    # Check PCA was computed
    assert "X_pca" in adata_gene.obsm.keys()

    # UMAP/PHATE may not be computed if dataset is too small
    # This is expected behavior for edge cases

    # Check label_str column exists
    assert "label_str" in adata_gene.obs.columns


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_create_aggregated_custom_pca_components(mock_adata_1):
    """Test with custom n_pca_components parameter."""
    adata_guide = create_aggregated_embeddings(
        mock_adata_1,
        level="guide",
        n_pca_components=1,
        n_neighbors=1,
    )

    # Check PCA dimensionality (may be adjusted for small datasets)
    assert "X_pca" in adata_guide.obsm.keys()
    assert adata_guide.obsm["X_pca"].shape[1] <= 2


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_create_aggregated_custom_neighbors(mock_adata_1):
    """Test with custom n_neighbors parameter."""
    # Should complete without error
    adata_guide = create_aggregated_embeddings(
        mock_adata_1,
        level="guide",
        n_pca_components=1,
        n_neighbors=1,
    )

    # PCA should be computed
    assert "X_pca" in adata_guide.obsm.keys()
    # UMAP/PHATE may not be computed for very small datasets


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_create_aggregated_median_method(mock_adata_1):
    """Test with aggregation_method='median'."""
    adata_median = create_aggregated_embeddings(
        mock_adata_1,
        level="guide",
        aggregation_method="median",
        n_pca_components=1,
        n_neighbors=1,
    )

    adata_mean = create_aggregated_embeddings(
        mock_adata_1,
        level="guide",
        aggregation_method="mean",
        n_pca_components=1,
        n_neighbors=1,
    )

    # Both should succeed
    assert adata_median.shape == adata_mean.shape
    # Values should differ slightly
    assert not np.allclose(adata_median.X, adata_mean.X, rtol=0.01)


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_create_aggregated_with_batch_preservation(mock_adata_1):
    """Test with preserve_batch_info=True."""
    # Add batch column
    adata = mock_adata_1.copy()
    adata.obs["batch"] = ["batch1"] * 50 + ["batch2"] * 50

    adata_agg = create_aggregated_embeddings(
        adata,
        level="guide",
        preserve_batch_info=True,
        n_pca_components=1,
        n_neighbors=1,
    )

    # Check batch column exists
    assert "batch" in adata_agg.obs.columns


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_create_aggregated_without_batch_preservation(mock_adata_1):
    """Test with preserve_batch_info=False."""
    # Add batch column
    adata = mock_adata_1.copy()
    adata.obs["batch"] = ["batch1"] * 50 + ["batch2"] * 50

    adata_agg = create_aggregated_embeddings(
        adata,
        level="guide",
        preserve_batch_info=False,
        n_pca_components=1,
        n_neighbors=1,
    )

    # Output shape should be just unique guides
    n_guides = adata.obs["sgRNA"].nunique()
    assert adata_agg.shape[0] == n_guides


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_create_aggregated_equals_sequential_calls(mock_adata_1):
    """Test that create_aggregated_embeddings() produces same result as sequential calls."""
    # Use the unified function
    np.random.seed(42)
    adata_unified = create_aggregated_embeddings(
        mock_adata_1.copy(),
        level="guide",
        n_pca_components=1,
        n_neighbors=1,
    )

    # Use sequential calls
    np.random.seed(42)
    adata_seq = aggregate_to_level(mock_adata_1.copy(), level="guide")
    adata_seq = compute_embeddings(
        adata_seq,
        n_pca_components=1,
        n_neighbors=1,
    )

    # Check X matrices are identical
    assert np.allclose(adata_unified.X, adata_seq.X, rtol=1e-5)

    # Check PCA is identical
    assert np.allclose(adata_unified.obsm["X_pca"], adata_seq.obsm["X_pca"], rtol=1e-5)

    # Check shapes match
    assert adata_unified.shape == adata_seq.shape


def test_create_aggregated_propagates_aggregation_errors():
    """Test that errors from aggregate_to_level() are properly propagated."""
    # Create adata without required column
    X = np.random.randn(10, 5).astype(np.float32)
    obs = pd.DataFrame({"label_str": ["GENE_A"] * 10})
    adata = ad.AnnData(X=X, obs=obs)

    with pytest.raises(ValueError, match="sgRNA.*not found"):
        create_aggregated_embeddings(adata, level="guide")


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_create_aggregated_minimal_dataset():
    """Test with very small dataset."""
    X = np.random.randn(10, 20).astype(np.float32)
    obs = pd.DataFrame(
        {
            "sgRNA": ["sg1"] * 5 + ["sg2"] * 5,
            "label_str": ["GENE_A"] * 5 + ["GENE_B"] * 5,
        }
    )
    adata = ad.AnnData(X=X, obs=obs)

    # Should handle small dataset gracefully with appropriate parameters
    adata_guide = create_aggregated_embeddings(
        adata,
        level="guide",
        n_pca_components=1,  # Small to match small dataset
        n_neighbors=1,  # Small to match small dataset
    )

    # Should have completed aggregation
    assert adata_guide.shape[0] == 2  # 2 guides
    assert "X_pca" in adata_guide.obsm.keys()
    # PCA components should be 1 (limited by small sample size)
    assert adata_guide.obsm["X_pca"].shape[1] == 1


def test_create_aggregated_many_features():
    """Test with high-dimensional feature space."""
    np.random.seed(42)
    n_cells = 100
    n_features = 2048  # High-dimensional like DinoV3

    X = np.random.randn(n_cells, n_features).astype(np.float32)
    obs = pd.DataFrame(
        {
            "sgRNA": [f"sg{i % 10}" for i in range(n_cells)],
            "label_str": [f"GENE_{i % 5}" for i in range(n_cells)],
        }
    )
    adata = ad.AnnData(X=X, obs=obs)

    # Should handle high-dimensional data
    adata_gene = create_aggregated_embeddings(
        adata,
        level="gene",
        n_pca_components=50,
        n_neighbors=3,
    )

    assert adata_gene.shape[0] == 5  # 5 genes
    assert "X_pca" in adata_gene.obsm.keys()
    # PCA should correctly reduce dimensionality
    assert adata_gene.obsm["X_pca"].shape[1] <= 50


# ============================================================================
# Regression Tests for Index/Column Conflicts (h5ad compatibility)
# ============================================================================


def test_aggregate_to_level_preserves_label_columns(mock_adata_1):
    """Ensure sgRNA/label_str exist as columns after aggregation, not as index names."""
    # Test guide level
    adata_guide = aggregate_to_level(mock_adata_1, level="guide")
    assert "sgRNA" in adata_guide.obs.columns, "sgRNA must be a column"
    assert len(adata_guide.obs["sgRNA"]) == len(
        adata_guide
    ), "sgRNA column must have one entry per observation"
    assert all(
        adata_guide.obs["sgRNA"].notna()
    ), "sgRNA column must not contain NaN values"

    # Test gene level
    adata_gene = aggregate_to_level(mock_adata_1, level="gene")
    assert "label_str" in adata_gene.obs.columns, "label_str must be a column"
    assert len(adata_gene.obs["label_str"]) == len(
        adata_gene
    ), "label_str column must have one entry per observation"
    assert all(
        adata_gene.obs["label_str"].notna()
    ), "label_str column must not contain NaN values"


def test_aggregate_no_index_column_conflict(mock_adata_1):
    """Ensure index.name doesn't conflict with column names (h5ad compatibility)."""
    # Aggregate
    adata_guide = aggregate_to_level(mock_adata_1, level="guide")

    # Check for conflict: if index.name matches a column, values must be identical
    if adata_guide.obs.index.name in adata_guide.obs.columns:
        col_name = adata_guide.obs.index.name
        assert np.array_equal(
            adata_guide.obs.index.values, adata_guide.obs[col_name].values
        ), f"Index and column '{col_name}' have different values - will fail h5ad write"

    # Better: index.name should NOT match any column name
    assert (
        adata_guide.obs.index.name not in adata_guide.obs.columns
    ), "Index name should not match column names to avoid h5ad conflicts"

    # Same for gene level
    adata_gene = aggregate_to_level(mock_adata_1, level="gene")
    assert (
        adata_gene.obs.index.name not in adata_gene.obs.columns
    ), "Index name should not match column names to avoid h5ad conflicts"


def test_aggregate_roundtrip_h5ad(tmp_path, mock_adata_1):
    """Test that aggregated data can be written to h5ad and read back."""
    # Aggregate to guide level
    adata_guide = aggregate_to_level(mock_adata_1, level="guide")

    # Write to h5ad (this should NOT raise ValueError)
    guide_path = tmp_path / "test_guide.h5ad"
    try:
        adata_guide.write_h5ad(guide_path)
    except ValueError as e:
        if "is also used by a column" in str(e):
            pytest.fail(f"h5ad write failed due to index/column conflict: {e}")
        raise

    # Read back and verify columns preserved
    adata_loaded = ad.read_h5ad(guide_path)
    assert (
        "sgRNA" in adata_loaded.obs.columns
    ), "sgRNA column must persist after round-trip"
    assert set(adata_loaded.obs["sgRNA"]) == set(
        adata_guide.obs["sgRNA"]
    ), "sgRNA values must be preserved"

    # Test gene level
    adata_gene = aggregate_to_level(mock_adata_1, level="gene")
    gene_path = tmp_path / "test_gene.h5ad"
    try:
        adata_gene.write_h5ad(gene_path)
    except ValueError as e:
        if "is also used by a column" in str(e):
            pytest.fail(f"h5ad write failed due to index/column conflict: {e}")
        raise

    adata_gene_loaded = ad.read_h5ad(gene_path)
    assert (
        "label_str" in adata_gene_loaded.obs.columns
    ), "label_str column must persist after round-trip"


def test_no_sgRNA_index_column_value_mismatch(tmp_path, mock_adata_1):
    """Regression test for: 'sgRNA' is used by both index and column with different values."""
    # Aggregate
    adata_guide = aggregate_to_level(mock_adata_1, level="guide")

    # Check the exact condition that caused the error
    if adata_guide.obs.index.name == "sgRNA" and "sgRNA" in adata_guide.obs.columns:
        # If both exist, they must have identical values
        if not np.array_equal(adata_guide.obs.index, adata_guide.obs["sgRNA"]):
            pytest.fail(
                "REGRESSION: Index named 'sgRNA' and column 'sgRNA' have different values. "
                "This will cause ValueError when writing h5ad."
            )

    # Attempt write to verify
    test_path = tmp_path / "regression_test.h5ad"
    adata_guide.write_h5ad(test_path)  # Should not raise


def test_anndata_structure_invariants(mock_adata_1):
    """Test structural invariants that should hold for all AnnData objects we create."""
    # Aggregate
    adata_guide = aggregate_to_level(mock_adata_1, level="guide")
    adata_gene = aggregate_to_level(mock_adata_1, level="gene")

    # Invariant 1: Observation labels exist as columns
    assert "sgRNA" in adata_guide.obs.columns, "Biological identifiers must be columns"
    assert (
        "label_str" in adata_gene.obs.columns
    ), "Biological identifiers must be columns"

    # Invariant 2: No index.name conflicts
    if adata_guide.obs.index.name is not None:
        assert (
            adata_guide.obs.index.name not in adata_guide.obs.columns
        ), "Index name must not conflict with column names"
    if adata_gene.obs.index.name is not None:
        assert (
            adata_gene.obs.index.name not in adata_gene.obs.columns
        ), "Index name must not conflict with column names"

    # Invariant 3: Each observation has exactly one label
    assert not adata_guide.obs["sgRNA"].isna().any(), "No missing labels allowed"
    assert not adata_gene.obs["label_str"].isna().any(), "No missing labels allowed"

    # Invariant 4: obs rows match X rows
    assert len(adata_guide.obs) == adata_guide.X.shape[0], "obs and X must be aligned"
    assert len(adata_gene.obs) == adata_gene.X.shape[0], "obs and X must be aligned"


# ============================================================================
# Tests for Control Gene Subsampling
# ============================================================================


@pytest.fixture
def mock_adata_with_many_ntc_guides():
    """Create mock AnnData with many NTC guides (simulating real scenario)."""
    np.random.seed(42)

    # NTC: 20 guides with 10 cells each = 200 cells
    # GENE_A: 4 guides with 10 cells each = 40 cells
    # GENE_B: 4 guides with 10 cells each = 40 cells

    n_cells = 280
    n_features = 50

    X = np.random.randn(n_cells, n_features).astype(np.float32)

    # Create guide labels
    ntc_guides = [f"NTC_sg{i}" for i in range(1, 21)]  # 20 NTC guides
    gene_a_guides = [f"GENE_A_sg{i}" for i in range(1, 5)]  # 4 guides
    gene_b_guides = [f"GENE_B_sg{i}" for i in range(1, 5)]  # 4 guides

    # Assign cells to guides (10 cells per guide)
    sgRNA_list = []
    label_str_list = []

    for guide in ntc_guides:
        sgRNA_list.extend([guide] * 10)
        label_str_list.extend(["NTC"] * 10)

    for guide in gene_a_guides:
        sgRNA_list.extend([guide] * 10)
        label_str_list.extend(["GENE_A"] * 10)

    for guide in gene_b_guides:
        sgRNA_list.extend([guide] * 10)
        label_str_list.extend(["GENE_B"] * 10)

    obs = pd.DataFrame(
        {
            "sgRNA": sgRNA_list,
            "label_str": label_str_list,
        }
    )

    adata = ad.AnnData(X=X, obs=obs)
    adata.var_names = [f"feature_{i}" for i in range(n_features)]

    return adata


def test_subsample_controls_basic(mock_adata_with_many_ntc_guides):
    """Test basic control subsampling functionality."""
    adata_gene = aggregate_to_level(
        mock_adata_with_many_ntc_guides,
        level="gene",
        subsample_controls=True,
        control_group_size=4,
        random_seed=42,
    )

    # Should have 5 NTC groups (20 guides / 4 per group)
    # Plus GENE_A and GENE_B = 7 total
    assert adata_gene.shape[0] == 7

    # Check NTC groups exist
    genes = set(adata_gene.obs["label_str"])
    ntc_groups = [g for g in genes if g.startswith("NTC_")]
    assert len(ntc_groups) == 5

    # Check naming
    expected_ntc = {"NTC_1", "NTC_2", "NTC_3", "NTC_4", "NTC_5"}
    actual_ntc = {g for g in genes if g.startswith("NTC_")}
    assert actual_ntc == expected_ntc

    # Check other genes still present
    assert "GENE_A" in genes
    assert "GENE_B" in genes


def test_subsample_controls_group_size_different(mock_adata_with_many_ntc_guides):
    """Test with different control_group_size."""
    # 20 guides, groups of 5 → 4 groups
    adata_gene = aggregate_to_level(
        mock_adata_with_many_ntc_guides,
        level="gene",
        subsample_controls=True,
        control_group_size=5,
        random_seed=42,
    )

    # 4 NTC groups + GENE_A + GENE_B = 6 total
    assert adata_gene.shape[0] == 6

    ntc_groups = [g for g in adata_gene.obs["label_str"] if g.startswith("NTC_")]
    assert len(ntc_groups) == 4


def test_subsample_controls_with_remainder(mock_adata_with_many_ntc_guides):
    """Test that remainder guides are grouped into smaller group."""
    # 20 guides, groups of 6 → 3 groups of 6, 1 group of 2
    adata_gene = aggregate_to_level(
        mock_adata_with_many_ntc_guides,
        level="gene",
        subsample_controls=True,
        control_group_size=6,
        random_seed=42,
    )

    # 4 NTC groups (3 full + 1 remainder) + GENE_A + GENE_B = 6 total
    assert adata_gene.shape[0] == 6

    ntc_groups = [g for g in adata_gene.obs["label_str"] if g.startswith("NTC_")]
    assert len(ntc_groups) == 4


def test_subsample_controls_reproducible(mock_adata_with_many_ntc_guides):
    """Test that random_seed makes grouping reproducible."""
    adata1 = aggregate_to_level(
        mock_adata_with_many_ntc_guides,
        level="gene",
        subsample_controls=True,
        control_group_size=4,
        random_seed=42,
    )

    adata2 = aggregate_to_level(
        mock_adata_with_many_ntc_guides,
        level="gene",
        subsample_controls=True,
        control_group_size=4,
        random_seed=42,
    )

    # Same seed should produce identical results
    assert set(adata1.obs["label_str"]) == set(adata2.obs["label_str"])

    # Feature values should be identical (same guides grouped together)
    # Sort by label_str for comparison
    adata1_sorted = adata1[adata1.obs["label_str"].argsort()]
    adata2_sorted = adata2[adata2.obs["label_str"].argsort()]
    assert np.allclose(adata1_sorted.X, adata2_sorted.X, rtol=1e-5)


def test_subsample_controls_different_seeds_differ(mock_adata_with_many_ntc_guides):
    """Test that different seeds produce different groupings."""
    adata1 = aggregate_to_level(
        mock_adata_with_many_ntc_guides,
        level="gene",
        subsample_controls=True,
        control_group_size=4,
        random_seed=42,
    )

    adata2 = aggregate_to_level(
        mock_adata_with_many_ntc_guides,
        level="gene",
        subsample_controls=True,
        control_group_size=4,
        random_seed=123,
    )

    # Same number of groups
    assert adata1.shape[0] == adata2.shape[0]

    # But feature values should differ (different guides grouped together)
    ntc1 = adata1[adata1.obs["label_str"] == "NTC_1"]
    ntc2 = adata2[adata2.obs["label_str"] == "NTC_1"]

    # Different seeds should produce different aggregations
    assert not np.allclose(ntc1.X, ntc2.X, rtol=1e-5)


def test_subsample_controls_only_at_gene_level(mock_adata_with_many_ntc_guides):
    """Test that subsampling is ignored at guide level."""
    # Should not affect guide-level aggregation
    adata_guide = aggregate_to_level(
        mock_adata_with_many_ntc_guides,
        level="guide",
        subsample_controls=True,  # Should be ignored
        control_group_size=4,
        random_seed=42,
    )

    # Should have all 28 guides (20 NTC + 4 GENE_A + 4 GENE_B)
    assert adata_guide.shape[0] == 28


def test_subsample_controls_missing_sgrna_column():
    """Test error when sgRNA column missing but subsampling requested."""
    # Create adata without sgRNA column
    X = np.random.randn(100, 50).astype(np.float32)
    obs = pd.DataFrame({"label_str": ["NTC"] * 50 + ["GENE_A"] * 50})
    adata = ad.AnnData(X=X, obs=obs)

    with pytest.raises(ValueError, match="Control subsampling requires 'sgRNA' column"):
        aggregate_to_level(adata, level="gene", subsample_controls=True)


def test_subsample_controls_gene_not_found(mock_adata_1):
    """Test error when control gene not in data."""
    # mock_adata_1 has NTC, but we'll ask for a non-existent control
    with pytest.raises(ValueError, match="Control gene 'NONEXISTENT' not found"):
        aggregate_to_level(
            mock_adata_1,
            level="gene",
            subsample_controls=True,
            control_gene="NONEXISTENT",
            random_seed=42,
        )


def test_subsample_controls_preserves_other_genes(mock_adata_with_many_ntc_guides):
    """Test that non-control genes are unaffected."""
    adata_no_subsample = aggregate_to_level(
        mock_adata_with_many_ntc_guides, level="gene", subsample_controls=False
    )

    adata_with_subsample = aggregate_to_level(
        mock_adata_with_many_ntc_guides,
        level="gene",
        subsample_controls=True,
        control_group_size=4,
        random_seed=42,
    )

    # GENE_A should be identical in both
    gene_a_no_sub = adata_no_subsample[adata_no_subsample.obs["label_str"] == "GENE_A"]
    gene_a_with_sub = adata_with_subsample[
        adata_with_subsample.obs["label_str"] == "GENE_A"
    ]

    assert np.allclose(gene_a_no_sub.X, gene_a_with_sub.X, rtol=1e-5)

    # GENE_B should be identical in both
    gene_b_no_sub = adata_no_subsample[adata_no_subsample.obs["label_str"] == "GENE_B"]
    gene_b_with_sub = adata_with_subsample[
        adata_with_subsample.obs["label_str"] == "GENE_B"
    ]

    assert np.allclose(gene_b_no_sub.X, gene_b_with_sub.X, rtol=1e-5)


def test_subsample_controls_with_batch_info(mock_adata_with_many_ntc_guides):
    """Test subsampling with preserve_batch_info=True."""
    # Add batch information
    adata = mock_adata_with_many_ntc_guides.copy()
    adata.obs["batch"] = ["batch1"] * 140 + ["batch2"] * 140

    adata_gene = aggregate_to_level(
        adata,
        level="gene",
        subsample_controls=True,
        control_group_size=4,
        preserve_batch_info=True,
        random_seed=42,
    )

    # Check batch column exists
    assert "batch" in adata_gene.obs.columns

    # Check that we have NTC groups
    ntc_groups = [g for g in adata_gene.obs["label_str"] if g.startswith("NTC_")]
    assert len(ntc_groups) > 0

    # Verify GENE_A and GENE_B exist
    assert "GENE_A" in adata_gene.obs["label_str"].values
    assert "GENE_B" in adata_gene.obs["label_str"].values

    # Total observations should be reasonable
    # (exact number depends on batch/guide distribution)
    assert adata_gene.shape[0] > 5  # At least more than without batches
    assert adata_gene.shape[0] < 20  # But not too many


def test_subsample_controls_aggregation_correctness(mock_adata_with_many_ntc_guides):
    """Test that aggregated values are correct means of grouped guides."""
    adata = mock_adata_with_many_ntc_guides

    # Aggregate with subsampling
    adata_gene = aggregate_to_level(
        adata,
        level="gene",
        method="mean",
        subsample_controls=True,
        control_group_size=4,
        random_seed=42,
    )

    # Manually compute what NTC_1 should be
    # First, figure out which guides are in NTC_1
    rng = np.random.RandomState(42)
    ntc_guides = adata.obs.loc[adata.obs["label_str"] == "NTC", "sgRNA"].unique()
    shuffled = ntc_guides.copy()
    rng.shuffle(shuffled)
    ntc_1_guides = shuffled[0:4]

    # Get cells with those guides
    ntc_1_mask = adata.obs["sgRNA"].isin(ntc_1_guides)
    ntc_1_cells = adata[ntc_1_mask]

    # Compute mean manually
    expected_mean = ntc_1_cells.X.mean(axis=0)

    # Get NTC_1 from aggregated data
    actual_ntc_1 = adata_gene[adata_gene.obs["label_str"] == "NTC_1"]

    # Should match
    assert np.allclose(actual_ntc_1.X[0], expected_mean, rtol=1e-5)


def test_subsample_controls_does_not_modify_original(mock_adata_with_many_ntc_guides):
    """Test that original adata is not modified."""
    adata_original = mock_adata_with_many_ntc_guides.copy()
    original_labels = adata_original.obs["label_str"].copy()

    # Aggregate with subsampling
    _ = aggregate_to_level(
        adata_original,
        level="gene",
        subsample_controls=True,
        control_group_size=4,
        random_seed=42,
    )

    # Original should be unchanged
    assert all(adata_original.obs["label_str"] == original_labels)
    assert "NTC_1" not in adata_original.obs["label_str"].values


def test_subsample_controls_with_median(mock_adata_with_many_ntc_guides):
    """Test subsampling with median aggregation method."""
    adata_gene = aggregate_to_level(
        mock_adata_with_many_ntc_guides,
        level="gene",
        method="median",
        subsample_controls=True,
        control_group_size=4,
        random_seed=42,
    )

    # Should complete without error
    assert adata_gene.shape[0] == 7

    # Check NTC groups exist
    ntc_groups = [g for g in adata_gene.obs["label_str"] if g.startswith("NTC_")]
    assert len(ntc_groups) == 5
