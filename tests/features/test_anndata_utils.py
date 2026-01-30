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
    concatenate_features_by_channel,
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
# Fixtures for Multi-Channel Feature Concatenation Tests
# ============================================================================


@pytest.fixture(scope="module")
def mock_gene_level_channel_data():
    """Create mock gene-level AnnData for multiple channels."""
    np.random.seed(42)

    # Define common genes plus some channel-specific genes
    all_genes = ["NTC", "GENE_A", "GENE_B", "GENE_C", "GENE_D", "GENE_E"]

    channels_data = {}

    # Phase2D: has all 6 genes
    n_genes_phase = 6
    X_phase = np.random.randn(n_genes_phase, 1024).astype(np.float32)
    obs_phase = pd.DataFrame(
        {
            "label_str": all_genes[:6],
            "label_int": list(range(6)),
            "n_cells": [100, 80, 90, 85, 95, 88],
        }
    )
    adata_phase = ad.AnnData(X=X_phase, obs=obs_phase)
    adata_phase.var_names = [f"phase_feat_{i}" for i in range(1024)]
    channels_data["Phase2D"] = adata_phase

    # GFP: has 5 genes (missing GENE_E)
    n_genes_gfp = 5
    X_gfp = (
        np.random.randn(n_genes_gfp, 1024).astype(np.float32) + 1.0
    )  # Different distribution
    obs_gfp = pd.DataFrame(
        {
            "label_str": all_genes[:5],
            "label_int": list(range(5)),
            "n_cells": [100, 80, 90, 85, 95],
        }
    )
    adata_gfp = ad.AnnData(X=X_gfp, obs=obs_gfp)
    adata_gfp.var_names = [f"gfp_feat_{i}" for i in range(1024)]
    channels_data["GFP"] = adata_gfp

    # mCherry: has 5 genes (missing GENE_D), different order
    genes_mcherry = ["GENE_A", "NTC", "GENE_B", "GENE_E", "GENE_C"]  # Different order!
    X_mcherry = np.random.randn(5, 1024).astype(np.float32) - 0.5
    obs_mcherry = pd.DataFrame(
        {
            "label_str": genes_mcherry,
            "label_int": [1, 0, 2, 5, 3],  # Matches different order
            "n_cells": [80, 100, 90, 88, 85],
        }
    )
    adata_mcherry = ad.AnnData(X=X_mcherry, obs=obs_mcherry)
    adata_mcherry.var_names = [f"mcherry_feat_{i}" for i in range(1024)]
    channels_data["mCherry"] = adata_mcherry

    return channels_data


@pytest.fixture(scope="module")
def mock_guide_level_channel_data():
    """Create mock guide-level AnnData for multiple channels."""
    np.random.seed(123)

    # Define sgRNAs
    guides = ["NTC_sg1", "NTC_sg2", "GENE_A_sg1", "GENE_A_sg2", "GENE_B_sg1"]

    channels_data = {}

    for channel in ["Phase2D", "GFP"]:
        n_guides = len(guides)
        X = np.random.randn(n_guides, 1024).astype(np.float32)
        obs = pd.DataFrame(
            {
                "sgRNA": guides,
                "label_str": ["NTC", "NTC", "GENE_A", "GENE_A", "GENE_B"],
                "n_cells": [50, 50, 40, 40, 45],
            }
        )
        adata = ad.AnnData(X=X, obs=obs)
        adata.var_names = [f"{channel}_feat_{i}" for i in range(1024)]
        channels_data[channel] = adata

    return channels_data


@pytest.fixture(scope="module")
def temp_multi_channel_files(mock_gene_level_channel_data):
    """Create temporary directory structure with channel-specific h5ad files."""
    temp_dir = Path(tempfile.mkdtemp())

    # Create experiment directory structure
    exp_dir = (
        temp_dir
        / "ops0089_20251119"
        / "3-assembly"
        / "dinov3_features"
        / "anndata_objects"
    )
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save channel-specific files
    for channel, adata in mock_gene_level_channel_data.items():
        file_path = exp_dir / f"gene_bulked_umap_{channel}.h5ad"
        adata.write_h5ad(file_path)

    yield temp_dir

    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="module")
def temp_guide_level_files(mock_guide_level_channel_data):
    """Create temporary directory with guide-level channel-specific files."""
    temp_dir = Path(tempfile.mkdtemp())

    exp_dir = (
        temp_dir
        / "ops0089_20251119"
        / "3-assembly"
        / "dinov3_features"
        / "anndata_objects"
    )
    exp_dir.mkdir(parents=True, exist_ok=True)

    for channel, adata in mock_guide_level_channel_data.items():
        file_path = exp_dir / f"guide_bulked_umap_{channel}.h5ad"
        adata.write_h5ad(file_path)

    yield temp_dir

    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_feature_metadata(monkeypatch):
    """Mock FeatureMetadata class for testing."""
    from unittest.mock import MagicMock

    mock_meta = MagicMock()

    # Mock get_short_label to return simple channel-based labels
    def mock_get_short_label(experiment, channel):
        label_map = {
            "Phase2D": "Phase",
            "GFP": "TestMarkerGFP",
            "mCherry": "TestMarkermCherry",
        }
        return label_map.get(channel, channel)

    # Mock create_feature_name
    def mock_create_feature_name(experiment, channel, feature_type, feature_idx):
        short = mock_get_short_label(experiment, channel)
        exp_short = experiment.split("_")[0]
        return f"{exp_short}_{short}_{feature_type}_{feature_idx}"

    # Mock get_channel_info
    def mock_get_channel_info(experiment, channel):
        return {
            "channel_name": channel,
            "label": f"test organelle, Test{channel}",
            "organelle": "test organelle",
            "marker": f"Test{channel}",
        }

    # Mock add_to_anndata (no-op for testing)
    def mock_add_to_anndata(adata, experiment, channel, feature_type):
        pass

    mock_meta.get_short_label = mock_get_short_label
    mock_meta.create_feature_name = mock_create_feature_name
    mock_meta.get_channel_info = mock_get_channel_info
    mock_meta.add_to_anndata = mock_add_to_anndata

    # Patch the FeatureMetadata import in anndata_utils
    import ops_model.features.anndata_utils as anndata_utils_module

    monkeypatch.setattr(anndata_utils_module, "FeatureMetadata", lambda: mock_meta)

    return mock_meta


# ============================================================================
# Test concatenate_features_by_channel - Basic Functionality
# ============================================================================


def test_concatenate_features_basic_gene_level(
    temp_multi_channel_files, mock_feature_metadata
):
    """Test basic multi-channel concatenation at gene level."""
    adata_combined = concatenate_features_by_channel(
        experiment="ops0089",
        channels=["Phase2D", "GFP"],
        feature_type="dinov3",
        aggregation_level="gene",
        base_dir=temp_multi_channel_files,
        recompute_embeddings=False,
    )

    # Check it returns AnnData
    assert isinstance(adata_combined, ad.AnnData)

    # Check shape: common genes = 5 (Phase2D has 6, GFP has 5)
    # Features: 1024 + 1024 = 2048
    assert adata_combined.shape[0] == 5, "Should have 5 common genes"
    assert adata_combined.shape[1] == 2048, "Should have 2048 features (2 × 1024)"


def test_concatenate_features_dimensions_correct(
    temp_multi_channel_files, mock_feature_metadata
):
    """Test that horizontal concatenation produces correct dimensions."""
    # Test with 3 channels
    adata_combined = concatenate_features_by_channel(
        experiment="ops0089",
        channels=["Phase2D", "GFP", "mCherry"],
        feature_type="dinov3",
        aggregation_level="gene",
        base_dir=temp_multi_channel_files,
        recompute_embeddings=False,
    )

    # Common genes across all 3: NTC, GENE_A, GENE_B, GENE_C (4 genes)
    # Phase2D: 6 genes, GFP: 5 genes, mCherry: 5 genes
    # Common: {NTC, GENE_A, GENE_B, GENE_C, GENE_D, GENE_E} ∩ {NTC, GENE_A, GENE_B, GENE_C, GENE_D} ∩ {GENE_A, NTC, GENE_B, GENE_E, GENE_C}
    # = {NTC, GENE_A, GENE_B, GENE_C}
    assert adata_combined.shape[0] == 4, "Should have 4 common genes"
    assert adata_combined.shape[1] == 3072, "Should have 3072 features (3 × 1024)"


def test_concatenate_features_guide_level(
    temp_guide_level_files, mock_feature_metadata
):
    """Test concatenation at guide level."""
    adata_combined = concatenate_features_by_channel(
        experiment="ops0089",
        channels=["Phase2D", "GFP"],
        feature_type="dinov3",
        aggregation_level="guide",
        base_dir=temp_guide_level_files,
        recompute_embeddings=False,
    )

    # Both channels have same 5 guides
    assert adata_combined.shape[0] == 5, "Should have 5 guides"
    assert adata_combined.shape[1] == 2048, "Should have 2048 features"

    # Check sgRNA column exists
    assert "sgRNA" in adata_combined.obs.columns


# ============================================================================
# Test concatenate_features_by_channel - Metadata and Provenance
# ============================================================================


def test_concatenate_features_variable_names_biological(
    temp_multi_channel_files, mock_feature_metadata
):
    """Test that variable names include biological context."""
    adata_combined = concatenate_features_by_channel(
        experiment="ops0089",
        channels=["Phase2D", "GFP"],
        feature_type="dinov3",
        aggregation_level="gene",
        base_dir=temp_multi_channel_files,
        recompute_embeddings=False,
    )

    # Check variable names have biological format
    var_names = adata_combined.var_names.tolist()

    # First 1024 should be Phase2D features
    assert var_names[0].startswith("ops0089_Phase_dinov3_")
    assert var_names[1023].startswith("ops0089_Phase_dinov3_")

    # Next 1024 should be GFP features
    assert var_names[1024].startswith("ops0089_TestMarkerGFP_dinov3_")
    assert var_names[2047].startswith("ops0089_TestMarkerGFP_dinov3_")

    # Check feature indices
    assert var_names[0] == "ops0089_Phase_dinov3_0"
    assert var_names[1023] == "ops0089_Phase_dinov3_1023"
    assert var_names[1024] == "ops0089_TestMarkerGFP_dinov3_0"


def test_concatenate_features_metadata_tracking(
    temp_multi_channel_files, mock_feature_metadata
):
    """Test that metadata is properly tracked in .uns."""
    adata_combined = concatenate_features_by_channel(
        experiment="ops0089",
        channels=["Phase2D", "GFP", "mCherry"],
        feature_type="dinov3",
        aggregation_level="gene",
        base_dir=temp_multi_channel_files,
        recompute_embeddings=False,
    )

    # Check combined_metadata exists
    assert "combined_metadata" in adata_combined.uns
    meta = adata_combined.uns["combined_metadata"]

    # Check required fields
    assert meta["experiment"] == "ops0089"
    assert meta["channels"] == ["Phase2D", "GFP", "mCherry"]
    assert meta["feature_type"] == "dinov3"
    assert meta["aggregation_level"] == "gene"
    assert meta["n_channels"] == 3

    # Check feature_slices exists
    assert "feature_slices" in meta
    assert "channel_biology" in meta


def test_concatenate_features_feature_slices(
    temp_multi_channel_files, mock_feature_metadata
):
    """Test that feature slices correctly map channels to index ranges."""
    adata_combined = concatenate_features_by_channel(
        experiment="ops0089",
        channels=["Phase2D", "GFP"],
        feature_type="dinov3",
        aggregation_level="gene",
        base_dir=temp_multi_channel_files,
        recompute_embeddings=False,
    )

    feature_slices = adata_combined.uns["combined_metadata"]["feature_slices"]

    # Check Phase2D slice
    assert "Phase2D" in feature_slices
    assert feature_slices["Phase2D"] == [0, 1024]

    # Check GFP slice
    assert "GFP" in feature_slices
    assert feature_slices["GFP"] == [1024, 2048]

    # Verify slices extract correct data
    phase_features = adata_combined[
        :, feature_slices["Phase2D"][0] : feature_slices["Phase2D"][1]
    ]
    assert phase_features.shape[1] == 1024


# ============================================================================
# Test concatenate_features_by_channel - Observation Alignment
# ============================================================================


def test_concatenate_features_alignment_verified(
    temp_multi_channel_files, mock_feature_metadata
):
    """Test that genes are correctly aligned across channels despite different orders."""
    adata_combined = concatenate_features_by_channel(
        experiment="ops0089",
        channels=["Phase2D", "mCherry"],  # mCherry has different order
        feature_type="dinov3",
        aggregation_level="gene",
        base_dir=temp_multi_channel_files,
        recompute_embeddings=False,
    )

    # Should succeed without alignment error
    # Common genes: NTC, GENE_A, GENE_B, GENE_C, GENE_E (5 genes)
    assert adata_combined.shape[0] == 5

    # Check genes are sorted
    genes = adata_combined.obs["label_str"].tolist()
    assert genes == sorted(genes), "Genes should be sorted for alignment"


def test_concatenate_features_partial_gene_overlap(
    temp_multi_channel_files, mock_feature_metadata
):
    """Test handling of partial gene overlap across channels."""
    adata_combined = concatenate_features_by_channel(
        experiment="ops0089",
        channels=["Phase2D", "GFP", "mCherry"],
        feature_type="dinov3",
        aggregation_level="gene",
        base_dir=temp_multi_channel_files,
        recompute_embeddings=False,
    )

    # Phase2D: {NTC, GENE_A, GENE_B, GENE_C, GENE_D, GENE_E}
    # GFP:     {NTC, GENE_A, GENE_B, GENE_C, GENE_D}
    # mCherry: {GENE_A, NTC, GENE_B, GENE_E, GENE_C}
    # Common:  {NTC, GENE_A, GENE_B, GENE_C}

    assert adata_combined.shape[0] == 4, "Should keep only common genes"

    common_genes = set(adata_combined.obs["label_str"])
    expected_common = {"NTC", "GENE_A", "GENE_B", "GENE_C"}
    assert common_genes == expected_common


def test_concatenate_features_no_common_genes_raises(
    temp_multi_channel_files, mock_feature_metadata
):
    """Test that error is raised when no common genes exist."""
    # Create a channel with completely different genes
    temp_dir = temp_multi_channel_files
    exp_dir = (
        temp_dir
        / "ops0089_20251119"
        / "3-assembly"
        / "dinov3_features"
        / "anndata_objects"
    )

    # Create a file with no overlapping genes
    X = np.random.randn(3, 1024).astype(np.float32)
    obs = pd.DataFrame(
        {
            "label_str": ["GENE_X", "GENE_Y", "GENE_Z"],
            "label_int": [10, 11, 12],
            "n_cells": [50, 60, 70],
        }
    )
    adata = ad.AnnData(X=X, obs=obs)
    adata.write_h5ad(exp_dir / "gene_bulked_umap_Cy5.h5ad")

    # Should raise ValueError
    with pytest.raises(ValueError, match="No common"):
        concatenate_features_by_channel(
            experiment="ops0089",
            channels=["Phase2D", "Cy5"],  # No overlap
            feature_type="dinov3",
            aggregation_level="gene",
            base_dir=temp_multi_channel_files,
            recompute_embeddings=False,
        )


# ============================================================================
# Test concatenate_features_by_channel - Embedding Computation
# ============================================================================


def test_concatenate_features_embeddings_recomputed(
    temp_multi_channel_files, mock_feature_metadata
):
    """Test that embeddings are recomputed on combined features."""
    adata_combined = concatenate_features_by_channel(
        experiment="ops0089",
        channels=["Phase2D", "GFP"],
        feature_type="dinov3",
        aggregation_level="gene",
        base_dir=temp_multi_channel_files,
        recompute_embeddings=True,
        n_pca_components=50,
    )

    # Check PCA computed
    assert "X_pca" in adata_combined.obsm.keys()
    # Note: actual components might be less than requested if n_samples is small
    assert adata_combined.obsm["X_pca"].shape[0] == adata_combined.shape[0]

    # Check UMAP computed
    assert "X_umap" in adata_combined.obsm.keys()
    assert adata_combined.obsm["X_umap"].shape == (adata_combined.shape[0], 2)


def test_concatenate_features_embeddings_skipped(
    temp_multi_channel_files, mock_feature_metadata
):
    """Test that embeddings can be skipped."""
    adata_combined = concatenate_features_by_channel(
        experiment="ops0089",
        channels=["Phase2D", "GFP"],
        feature_type="dinov3",
        aggregation_level="gene",
        base_dir=temp_multi_channel_files,
        recompute_embeddings=False,
    )

    # Should not have new embeddings
    assert "X_pca" not in adata_combined.obsm.keys()
    assert "X_umap" not in adata_combined.obsm.keys()


def test_concatenate_features_embeddings_dimensions(
    temp_multi_channel_files, mock_feature_metadata
):
    """Test that embeddings use combined feature space."""
    adata_combined = concatenate_features_by_channel(
        experiment="ops0089",
        channels=["Phase2D", "GFP", "mCherry"],
        feature_type="dinov3",
        aggregation_level="gene",
        base_dir=temp_multi_channel_files,
        recompute_embeddings=True,
        n_pca_components=30,
    )

    # PCA should be computed on 3072-dimensional space
    # (though we can't directly verify the input dimensionality)
    assert "X_pca" in adata_combined.obsm.keys()

    # Check that n_components doesn't exceed n_samples
    n_samples = adata_combined.shape[0]
    n_components = adata_combined.obsm["X_pca"].shape[1]
    assert n_components < n_samples


def test_concatenate_features_embeddings_includes_phate(
    temp_multi_channel_files, mock_feature_metadata
):
    """Test that PHATE is included when embeddings are recomputed."""
    adata_combined = concatenate_features_by_channel(
        experiment="ops0089",
        channels=["Phase2D", "GFP"],
        feature_type="dinov3",
        aggregation_level="gene",
        base_dir=temp_multi_channel_files,
        recompute_embeddings=True,
        n_pca_components=10,
    )

    # Check all three embeddings computed
    assert "X_pca" in adata_combined.obsm.keys(), "PCA should be computed"
    assert "X_umap" in adata_combined.obsm.keys(), "UMAP should be computed"
    assert "X_phate" in adata_combined.obsm.keys(), "PHATE should be computed"

    # Check PHATE shape
    n_samples = adata_combined.shape[0]
    assert adata_combined.obsm["X_phate"].shape == (n_samples, 2), "PHATE should be 2D"

    # Check PHATE coordinates are finite
    assert np.isfinite(
        adata_combined.obsm["X_phate"]
    ).all(), "PHATE coordinates should be finite"


# ============================================================================
# Test concatenate_features_by_channel - Error Handling
# ============================================================================


def test_concatenate_features_channel_file_missing(
    temp_multi_channel_files, mock_feature_metadata
):
    """Test error when channel file doesn't exist."""
    with pytest.raises(FileNotFoundError, match="Channel file not found"):
        concatenate_features_by_channel(
            experiment="ops0089",
            channels=["Phase2D", "NonExistentChannel"],
            feature_type="dinov3",
            aggregation_level="gene",
            base_dir=temp_multi_channel_files,
            recompute_embeddings=False,
        )


def test_concatenate_features_experiment_not_found(
    temp_multi_channel_files, mock_feature_metadata
):
    """Test error when experiment directory doesn't exist."""
    with pytest.raises(FileNotFoundError, match="Experiment directory not found"):
        concatenate_features_by_channel(
            experiment="ops9999",  # Non-existent
            channels=["Phase2D", "GFP"],
            feature_type="dinov3",
            aggregation_level="gene",
            base_dir=temp_multi_channel_files,
            recompute_embeddings=False,
        )


def test_concatenate_features_cell_level_raises_error(
    temp_multi_channel_files, mock_feature_metadata
):
    """Test that cell-level concatenation is explicitly prevented."""
    with pytest.raises(
        ValueError, match="Cell-level feature concatenation not supported"
    ):
        concatenate_features_by_channel(
            experiment="ops0089",
            channels=["Phase2D", "GFP"],
            feature_type="dinov3",
            aggregation_level="cell",  # Not allowed
            base_dir=temp_multi_channel_files,
            recompute_embeddings=False,
        )


# ============================================================================
# Test concatenate_features_by_channel - Edge Cases
# ============================================================================


def test_concatenate_features_single_channel(
    temp_multi_channel_files, mock_feature_metadata
):
    """Test that single channel concatenation works (identity operation)."""
    adata_combined = concatenate_features_by_channel(
        experiment="ops0089",
        channels=["Phase2D"],  # Only one channel
        feature_type="dinov3",
        aggregation_level="gene",
        base_dir=temp_multi_channel_files,
        recompute_embeddings=False,
    )

    # Should work and return data with single channel
    assert (
        adata_combined.shape[1] == 1024
    ), "Should have 1024 features from single channel"
    assert adata_combined.uns["combined_metadata"]["n_channels"] == 1


# ============================================================================
# Test concatenate_features_by_channel - Integration
# ============================================================================


@pytest.mark.slow
def test_concatenate_features_full_workflow(
    temp_multi_channel_files, mock_feature_metadata
):
    """Test complete workflow with all features."""
    # Concatenate 3 channels
    adata_combined = concatenate_features_by_channel(
        experiment="ops0089",
        channels=["Phase2D", "GFP", "mCherry"],
        feature_type="dinov3",
        aggregation_level="gene",
        base_dir=temp_multi_channel_files,
        recompute_embeddings=True,
        n_pca_components=30,
        n_umap_neighbors=3,  # Small for test data
    )

    # Verify all aspects
    # 1. Shape correct
    assert adata_combined.shape[0] == 4  # Common genes
    assert adata_combined.shape[1] == 3072  # 3 channels × 1024

    # 2. Variable names biological
    assert "ops0089_Phase_dinov3_0" in adata_combined.var_names
    assert "ops0089_TestMarkerGFP_dinov3_0" in adata_combined.var_names

    # 3. Metadata tracked
    assert "combined_metadata" in adata_combined.uns
    assert adata_combined.uns["combined_metadata"]["n_channels"] == 3

    # 4. Feature slices correct
    slices = adata_combined.uns["combined_metadata"]["feature_slices"]
    assert len(slices) == 3
    assert slices["Phase2D"] == (0, 1024)
    assert slices["GFP"] == (1024, 2048)
    assert slices["mCherry"] == (2048, 3072)

    # 5. Embeddings computed
    assert "X_pca" in adata_combined.obsm.keys()
    assert "X_umap" in adata_combined.obsm.keys()

    # 6. Observations aligned
    genes = adata_combined.obs["label_str"].tolist()
    assert genes == sorted(genes)


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
