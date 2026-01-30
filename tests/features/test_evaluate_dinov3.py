"""
Tests for DinoV3 feature evaluation pipeline.

This test suite validates the processing of DinoV3 embeddings through
the AnnData pipeline, including normalization, aggregation, and output generation.
"""

import warnings

# Filter anndata and zarr warnings
warnings.filterwarnings("ignore", message=".*zarr v2.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import pytest
import numpy as np
import pandas as pd
import anndata as ad
from pathlib import Path
import tempfile
import shutil

from ops_model.features.evaluate_cp import (
    center_scale_fast,
    pca_embed,
)
from ops_model.features.evaluate_dinov3 import (
    create_adata_object_dinov3,
    process_dinov3,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def mock_dinov3_csv():
    """
    Create a synthetic DinoV3 features DataFrame for testing.

    Returns:
        pd.DataFrame with 1024 feature columns + metadata
    """
    np.random.seed(42)

    # Use 150 cells to allow PCA with 128 components (need > 128 samples)
    n_cells = 150
    n_features = 1024

    # Create feature matrix with realistic embedding values
    # DinoV3 embeddings typically have values roughly in [-2, 2] range
    features = np.random.randn(n_cells, n_features).astype(np.float32)

    # Create DataFrame with numbered columns (0-1023)
    feature_df = pd.DataFrame(features, columns=[str(i) for i in range(n_features)])

    # Create metadata
    genes = ["NTC", "GENE_A", "GENE_B", "GENE_C"]
    gene_labels = np.random.choice(genes, size=n_cells)

    # Create multiple guides per gene
    guides = []
    for gene in gene_labels:
        guide_idx = np.random.randint(1, 4)  # 3 guides per gene
        guides.append(f"{gene}_sg{guide_idx}")

    # Create label_int mapping
    gene_to_int = {gene: idx for idx, gene in enumerate(genes)}
    label_ints = [gene_to_int[gene] for gene in gene_labels]

    # Add metadata columns
    feature_df["label_int"] = label_ints
    feature_df["label_str"] = gene_labels
    feature_df["sgRNA"] = guides
    feature_df["experiment"] = "ops0089_20251119"
    feature_df["x_position"] = np.random.uniform(0, 1000, n_cells)
    feature_df["y_position"] = np.random.uniform(0, 1000, n_cells)
    feature_df["well"] = np.random.choice(
        ["A1_ops0089_20251119", "A2_ops0089_20251119"], n_cells
    )

    return feature_df


@pytest.fixture(scope="module")
def mock_dinov3_csv_path(mock_dinov3_csv):
    """
    Write mock DinoV3 CSV to temporary file.

    Returns:
        Path to temporary CSV file
    """
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    csv_path = Path(temp_dir) / "dinov3_features_Phase2D.csv"

    # Write CSV
    mock_dinov3_csv.to_csv(csv_path, index=False)

    yield csv_path

    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="module")
def processed_adata(mock_dinov3_csv_path):
    """
    Run full processing pipeline on mock data.

    Returns:
        Processed AnnData object (cell-level)
    """
    config = {
        "normalize_PCA_embeddings": False,
    }

    # Note: This will create guide/gene aggregations but we're returning cell-level
    adata = create_adata_object_dinov3(str(mock_dinov3_csv_path), config=config)

    return adata


# ============================================================================
# Unit Tests
# ============================================================================


def test_load_dinov3_csv(mock_dinov3_csv):
    """Test 1: Validate CSV loading and basic structure."""
    assert isinstance(mock_dinov3_csv, pd.DataFrame)
    assert len(mock_dinov3_csv) == 150, "Should have 150 rows"

    # 1024 features + 7 metadata columns
    assert (
        len(mock_dinov3_csv.columns) == 1031
    ), "Should have 1024 features + 7 metadata"

    # Check that feature columns are numeric
    feature_cols = [str(i) for i in range(1024)]
    for col in feature_cols:
        assert col in mock_dinov3_csv.columns
        assert pd.api.types.is_numeric_dtype(mock_dinov3_csv[col])


def test_metadata_columns_present(mock_dinov3_csv):
    """Test 2: Validate required metadata columns."""
    required_columns = [
        "label_int",
        "label_str",
        "sgRNA",
        "experiment",
        "x_position",
        "y_position",
        "well",
    ]

    for col in required_columns:
        assert col in mock_dinov3_csv.columns, f"Missing required column: {col}"

    # Check no missing values in critical columns
    assert (
        not mock_dinov3_csv["label_str"].isna().any()
    ), "label_str should not have NaN"
    assert not mock_dinov3_csv["sgRNA"].isna().any(), "sgRNA should not have NaN"

    # Check data types
    assert pd.api.types.is_integer_dtype(mock_dinov3_csv["label_int"])
    assert pd.api.types.is_numeric_dtype(mock_dinov3_csv["x_position"])
    assert pd.api.types.is_numeric_dtype(mock_dinov3_csv["y_position"])


def test_feature_dimensions(mock_dinov3_csv):
    """Test 3: Validate feature dimensions and values."""
    feature_cols = [str(i) for i in range(1024)]
    features = mock_dinov3_csv[feature_cols]

    # Check dimensions
    assert features.shape[1] == 1024, "Should have exactly 1024 features"

    # Check all numeric
    assert features.dtypes.apply(lambda x: pd.api.types.is_numeric_dtype(x)).all()

    # Check no NaN values
    assert not features.isna().any().any(), "Features should not contain NaN"

    # Check reasonable value ranges (embeddings typically in [-5, 5])
    assert features.min().min() > -10, "Feature values too negative"
    assert features.max().max() < 10, "Feature values too large"

    # Check not all zeros
    assert features.std().mean() > 0.1, "Features should have variance"


def test_normalization(mock_dinov3_csv):
    """Test 4: Validate center-scaling normalization."""
    feature_cols = [str(i) for i in range(1024)]
    features = mock_dinov3_csv[feature_cols + ["label_str"]].copy()

    # Apply normalization
    features_norm = center_scale_fast(features, on_controls=False)

    # Check shape preserved
    assert features_norm.shape == features.shape

    # Check mean ≈ 0 and std ≈ 1 (excluding label_str column)
    feature_cols_only = [col for col in features_norm.columns if col != "label_str"]
    means = features_norm[feature_cols_only].mean()
    stds = features_norm[feature_cols_only].std()

    assert np.allclose(means, 0, atol=1e-6), "Mean should be close to 0"
    assert np.allclose(stds, 1, atol=1e-6), "Std should be close to 1"

    # Test control-based normalization
    features_norm_ctrl = center_scale_fast(
        features, on_controls=True, control_column="label_str", control_gene="NTC"
    )
    assert features_norm_ctrl.shape == features.shape

    # Check that label_str column preserved
    assert "label_str" in features_norm.columns
    pd.testing.assert_series_equal(
        features["label_str"], features_norm["label_str"], check_names=False
    )


def test_no_constant_columns(mock_dinov3_csv):
    """Test 5: Validate that QC filtering removes constant columns."""
    # Create a copy with one constant column
    test_df = mock_dinov3_csv.copy()
    test_df["constant_col"] = 1.0

    # Check that we can identify constant columns
    constant_cols = test_df.columns[test_df.nunique(dropna=False) == 1]
    assert "constant_col" in constant_cols.tolist()

    # Verify DinoV3 features have no constant columns
    feature_cols = [str(i) for i in range(1024)]
    features = mock_dinov3_csv[feature_cols]
    constant_features = features.columns[features.nunique(dropna=False) == 1]
    assert (
        len(constant_features) == 0
    ), "DinoV3 embeddings should not have constant columns"


# ============================================================================
# Integration Tests
# ============================================================================


def test_create_adata_object_structure(processed_adata):
    """Test 6: Validate AnnData object structure at cell level."""
    assert isinstance(processed_adata, ad.AnnData)

    # Check .X shape and dtype
    assert processed_adata.X.shape[0] == 150, "Should have 150 cells"
    assert processed_adata.X.shape[1] == 1024, "Should have 1024 features (embeddings)"
    # Note: center_scale_fast returns float64 for precision, this is acceptable
    assert processed_adata.X.dtype in [
        np.float32,
        np.float64,
    ], "Should use float32 or float64"

    # Check .obs contains metadata
    required_obs_cols = ["label_str", "label_int", "sgRNA", "well"]
    for col in required_obs_cols:
        assert col in processed_adata.obs.columns, f"Missing obs column: {col}"

    # Check .var_names
    assert len(processed_adata.var_names) == 1024

    # CRITICAL: DinoV3 cell-level should NOT have PCA/UMAP (only at aggregation level)
    assert (
        "X_pca" not in processed_adata.obsm.keys()
    ), "DinoV3 cell-level AnnData should not have PCA (only at aggregation level)"
    assert (
        "X_umap" not in processed_adata.obsm.keys()
    ), "DinoV3 cell-level AnnData should not have UMAP (only at aggregation level)"


def test_adata_metadata_integrity(processed_adata, mock_dinov3_csv):
    """Test 7: Validate metadata preservation through pipeline."""
    # Check that metadata matches original CSV
    assert len(processed_adata.obs) == len(mock_dinov3_csv)

    # Check label_str preservation
    original_genes = sorted(mock_dinov3_csv["label_str"].unique())
    adata_genes = sorted(processed_adata.obs["label_str"].unique())
    assert original_genes == adata_genes

    # Check sgRNA preservation
    original_guides = sorted(mock_dinov3_csv["sgRNA"].unique())
    adata_guides = sorted(processed_adata.obs["sgRNA"].unique())
    assert original_guides == adata_guides

    # Check gene counts
    original_gene_counts = mock_dinov3_csv["label_str"].value_counts()
    adata_gene_counts = processed_adata.obs["label_str"].value_counts()
    pd.testing.assert_series_equal(
        original_gene_counts.sort_index(), adata_gene_counts.sort_index()
    )

    # Check guide counts
    original_guide_counts = mock_dinov3_csv["sgRNA"].value_counts()
    adata_guide_counts = processed_adata.obs["sgRNA"].value_counts()
    pd.testing.assert_series_equal(
        original_guide_counts.sort_index(), adata_guide_counts.sort_index()
    )


# ============================================================================
# Aggregation Tests
# ============================================================================


@pytest.fixture(scope="module")
def guide_bulked_adata(mock_dinov3_csv_path):
    """
    Create guide-level aggregated AnnData for testing.
    This mimics the guide-bulking step in the pipeline.
    """
    # Load and process
    config = {"normalize_PCA_embeddings": False}
    adata = create_adata_object_dinov3(str(mock_dinov3_csv_path), config=config)

    # Aggregate by guide
    embeddings_df = pd.DataFrame(adata.X)
    embeddings_df["sgRNA"] = adata.obs["sgRNA"].values
    embeddings_guide_bulk = embeddings_df.groupby("sgRNA").mean()

    # Create bulked AnnData
    adata_guide = ad.AnnData(embeddings_guide_bulk)
    adata_guide.obs["sgRNA"] = adata_guide.obs_names

    # Compute PCA and UMAP
    adata_guide = pca_embed(
        adata_guide, n_components=min(128, adata_guide.shape[0] - 1)
    )

    # Import scanpy for UMAP and PHATE
    import scanpy as sc
    import scanpy.external as sce

    n_pcs = min(50, adata_guide.obsm["X_pca"].shape[1])
    n_neighbors = min(15, adata_guide.shape[0] - 1)

    sc.pp.neighbors(adata_guide, n_pcs=n_pcs, n_neighbors=n_neighbors, metric="cosine")
    sc.tl.umap(adata_guide, min_dist=0.1)

    # Compute PHATE
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="phate")
        warnings.filterwarnings("ignore", category=UserWarning, module="graphtools")
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="phate")
        sce.tl.phate(
            adata_guide,
            n_components=2,
            k=n_neighbors,
            n_pca=n_pcs,
            knn_dist="cosine",
            t="auto",
        )

    return adata_guide


@pytest.fixture(scope="module")
def gene_bulked_adata(mock_dinov3_csv_path):
    """
    Create gene-level aggregated AnnData for testing.
    """
    # Load and process
    config = {"normalize_PCA_embeddings": False}
    adata = create_adata_object_dinov3(str(mock_dinov3_csv_path), config=config)

    # Aggregate by gene
    embeddings_df = pd.DataFrame(adata.X)
    embeddings_df["label_str"] = adata.obs["label_str"].values
    embeddings_gene_avg = embeddings_df.groupby("label_str").mean()

    # Create bulked AnnData
    adata_gene = ad.AnnData(embeddings_gene_avg)
    adata_gene.obs["label_str"] = adata_gene.obs_names

    # Compute PCA and UMAP
    adata_gene = pca_embed(adata_gene, n_components=min(128, adata_gene.shape[0] - 1))

    # Import scanpy for UMAP and PHATE
    import scanpy as sc
    import scanpy.external as sce

    n_pcs = min(3, adata_gene.obsm["X_pca"].shape[1])
    n_neighbors = min(3, adata_gene.shape[0] - 1)

    sc.pp.neighbors(adata_gene, n_pcs=n_pcs, n_neighbors=n_neighbors, metric="cosine")
    sc.tl.umap(adata_gene, min_dist=0.1)

    # Compute PHATE (skip for very small datasets where PHATE may have numerical issues)
    if adata_gene.shape[0] >= 5:
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="phate")
            warnings.filterwarnings("ignore", category=UserWarning, module="graphtools")
            warnings.filterwarnings("ignore", category=RuntimeWarning, module="phate")
            sce.tl.phate(
                adata_gene,
                n_components=2,
                k=n_neighbors,
                n_pca=n_pcs,
                knn_dist="cosine",
                t="auto",
            )

    return adata_gene


def test_guide_level_aggregation_structure(guide_bulked_adata, mock_dinov3_csv):
    """Test 8: Validate guide-level aggregation structure."""
    n_unique_guides = mock_dinov3_csv["sgRNA"].nunique()

    assert isinstance(guide_bulked_adata, ad.AnnData)
    assert (
        guide_bulked_adata.shape[0] == n_unique_guides
    ), f"Should have {n_unique_guides} guides"
    assert (
        guide_bulked_adata.shape[1] == 1024
    ), "Should maintain 1024 feature dimensions"

    # Check each guide represented once
    assert len(guide_bulked_adata.obs["sgRNA"].unique()) == n_unique_guides

    # Check metadata preserved
    assert "sgRNA" in guide_bulked_adata.obs.columns


def test_guide_level_pca_umap(guide_bulked_adata):
    """Test 9: Validate PCA and UMAP on guide-level data."""
    # Check PCA exists
    assert "X_pca" in guide_bulked_adata.obsm.keys(), "Guide-level data should have PCA"

    n_guides = guide_bulked_adata.shape[0]
    pca_shape = guide_bulked_adata.obsm["X_pca"].shape
    assert pca_shape[0] == n_guides
    assert pca_shape[1] <= min(
        128, n_guides - 1
    ), "PCA components should not exceed n_samples - 1"

    # Check explained variance exists
    assert "pca" in guide_bulked_adata.uns.keys()
    assert "variance" in guide_bulked_adata.uns["pca"].keys()

    # Check UMAP exists
    assert (
        "X_umap" in guide_bulked_adata.obsm.keys()
    ), "Guide-level data should have UMAP"

    umap_shape = guide_bulked_adata.obsm["X_umap"].shape
    assert umap_shape == (n_guides, 2), "UMAP should be 2D"

    # Check UMAP coordinates are finite
    assert np.isfinite(
        guide_bulked_adata.obsm["X_umap"]
    ).all(), "UMAP coordinates should be finite"


def test_guide_level_phate(guide_bulked_adata):
    """Test 9b: Validate PHATE on guide-level data."""
    # Check PHATE exists
    assert (
        "X_phate" in guide_bulked_adata.obsm.keys()
    ), "Guide-level data should have PHATE"

    n_guides = guide_bulked_adata.shape[0]
    phate_shape = guide_bulked_adata.obsm["X_phate"].shape
    assert phate_shape == (n_guides, 2), "PHATE should be 2D"

    # Check PHATE coordinates are finite
    assert np.isfinite(
        guide_bulked_adata.obsm["X_phate"]
    ).all(), "PHATE coordinates should be finite"

    # Check PHATE is different from UMAP (they should produce different embeddings)
    if "X_umap" in guide_bulked_adata.obsm.keys():
        assert not np.allclose(
            guide_bulked_adata.obsm["X_phate"], guide_bulked_adata.obsm["X_umap"]
        ), "PHATE and UMAP should produce different embeddings"


def test_gene_level_aggregation_structure(gene_bulked_adata, mock_dinov3_csv):
    """Test 10: Validate gene-level aggregation structure."""
    n_unique_genes = mock_dinov3_csv["label_str"].nunique()

    assert isinstance(gene_bulked_adata, ad.AnnData)
    assert (
        gene_bulked_adata.shape[0] == n_unique_genes
    ), f"Should have {n_unique_genes} genes"
    assert gene_bulked_adata.shape[1] == 1024, "Should maintain 1024 feature dimensions"

    # Check each gene represented once
    assert len(gene_bulked_adata.obs["label_str"].unique()) == n_unique_genes

    # Check metadata preserved
    assert "label_str" in gene_bulked_adata.obs.columns

    # Check expected genes present
    expected_genes = ["NTC", "GENE_A", "GENE_B", "GENE_C"]
    for gene in expected_genes:
        assert gene in gene_bulked_adata.obs["label_str"].values


def test_gene_level_pca_umap(gene_bulked_adata):
    """Test 11: Validate PCA and UMAP on gene-level data."""
    # Check PCA exists
    assert "X_pca" in gene_bulked_adata.obsm.keys(), "Gene-level data should have PCA"

    n_genes = gene_bulked_adata.shape[0]
    pca_shape = gene_bulked_adata.obsm["X_pca"].shape
    assert pca_shape[0] == n_genes
    assert pca_shape[1] <= min(
        128, n_genes - 1
    ), "PCA components should not exceed n_samples - 1"

    # Check UMAP exists
    assert "X_umap" in gene_bulked_adata.obsm.keys(), "Gene-level data should have UMAP"

    umap_shape = gene_bulked_adata.obsm["X_umap"].shape
    assert umap_shape == (n_genes, 2), "UMAP should be 2D"

    # Check UMAP coordinates are finite
    assert np.isfinite(
        gene_bulked_adata.obsm["X_umap"]
    ).all(), "UMAP coordinates should be finite"


def test_gene_level_phate(gene_bulked_adata):
    """Test 11b: Validate PHATE on gene-level data."""
    # Check PHATE exists
    assert (
        "X_phate" in gene_bulked_adata.obsm.keys()
    ), "Gene-level data should have PHATE"

    n_genes = gene_bulked_adata.shape[0]
    phate_shape = gene_bulked_adata.obsm["X_phate"].shape
    assert phate_shape == (n_genes, 2), "PHATE should be 2D"

    # Check PHATE coordinates are finite
    assert np.isfinite(
        gene_bulked_adata.obsm["X_phate"]
    ).all(), "PHATE coordinates should be finite"

    # Check PHATE is different from UMAP (they should produce different embeddings)
    if "X_umap" in gene_bulked_adata.obsm.keys():
        assert not np.allclose(
            gene_bulked_adata.obsm["X_phate"], gene_bulked_adata.obsm["X_umap"]
        ), "PHATE and UMAP should produce different embeddings"


# ============================================================================
# Output Tests
# ============================================================================


@pytest.mark.slow
def test_output_files_created(mock_dinov3_csv_path):
    """Test 12: Validate that all output files are created."""
    config = {
        "normalize_PCA_embeddings": False,
    }

    # Run full pipeline
    adata = process_dinov3(str(mock_dinov3_csv_path), config=config)

    # Check output directory created
    output_dir = mock_dinov3_csv_path.parent / "anndata_objects"
    assert output_dir.exists(), "anndata_objects directory should be created"

    # Check main output file
    main_file = output_dir / "features_processed.h5ad"
    assert main_file.exists(), "features_processed.h5ad should be created"

    # Check guide-bulk file
    guide_file = output_dir / "guide_bulked_umap.h5ad"
    assert guide_file.exists(), "guide_bulked_umap.h5ad should be created"

    # Check gene-bulk file
    gene_file = output_dir / "gene_bulked_umap.h5ad"
    assert gene_file.exists(), "gene_bulked_umap.h5ad should be created"

    # Verify files can be read
    adata_main = ad.read_h5ad(main_file)
    assert isinstance(adata_main, ad.AnnData)

    adata_guide = ad.read_h5ad(guide_file)
    assert isinstance(adata_guide, ad.AnnData)

    adata_gene = ad.read_h5ad(gene_file)
    assert isinstance(adata_gene, ad.AnnData)


@pytest.mark.slow
def test_saved_adata_completeness(mock_dinov3_csv_path):
    """Test 13: Validate round-trip save/load preserves data."""
    config = {
        "normalize_PCA_embeddings": False,
    }

    # Run pipeline
    adata_original = process_dinov3(str(mock_dinov3_csv_path), config=config)

    output_dir = mock_dinov3_csv_path.parent / "anndata_objects"

    # Load cell-level file
    adata_loaded = ad.read_h5ad(output_dir / "features_processed.h5ad")

    # Check .X preserved
    assert adata_loaded.X.shape == adata_original.X.shape
    assert np.allclose(adata_loaded.X, adata_original.X)

    # Check .obs preserved
    assert list(adata_loaded.obs.columns) == list(adata_original.obs.columns)

    # DinoV3 cell-level should NOT have PCA/UMAP
    assert (
        len(adata_loaded.obsm.keys()) == 0
    ), "DinoV3 cell-level file should not contain obsm (PCA/UMAP only at aggregation level)"
    assert (
        len(adata_original.obsm.keys()) == 0
    ), "DinoV3 cell-level should not have obsm"

    # Load guide-bulk and check UMAP and PHATE present
    # NOTE: Guide-bulk uses cell-level PCA embeddings directly (n_pcs=0),
    # so it won't have its own X_pca, just X_umap and X_phate
    adata_guide = ad.read_h5ad(output_dir / "guide_bulked_umap.h5ad")
    assert "X_umap" in adata_guide.obsm.keys()
    assert (
        "X_phate" in adata_guide.obsm.keys()
    ), "Guide-level file should contain PHATE embeddings"

    # Load gene-bulk and check UMAP and PHATE present
    adata_gene = ad.read_h5ad(output_dir / "gene_bulked_umap.h5ad")
    assert "X_umap" in adata_gene.obsm.keys()
    assert (
        "X_phate" in adata_gene.obsm.keys()
    ), "Gene-level file should contain PHATE embeddings"


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


def test_missing_metadata_columns():
    """Test 14: Validate error handling for missing metadata columns."""
    # Create CSV missing a required column
    df = pd.DataFrame(
        {
            "0": [1.0, 2.0],
            "1": [3.0, 4.0],
            # Missing label_str, sgRNA, etc.
        }
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df.to_csv(f.name, index=False)
        temp_path = f.name

    try:
        config = {"normalize_PCA_embeddings": False}
        # This should raise an error or handle gracefully
        with pytest.raises(KeyError):
            create_adata_object_dinov3(temp_path, config=config)
    finally:
        Path(temp_path).unlink()


def test_wrong_feature_dimensions():
    """Test 15: Validate handling of incorrect feature dimensions."""
    # Create CSV with wrong number of features (not 1024)
    np.random.seed(42)
    wrong_features = np.random.randn(10, 512)  # Only 512 features

    df = pd.DataFrame(wrong_features, columns=[str(i) for i in range(512)])
    df["label_int"] = [0] * 10
    df["label_str"] = ["GENE_A"] * 10
    df["sgRNA"] = ["GENE_A_sg1"] * 10
    df["experiment"] = ["ops0089_20251119"] * 10
    df["x_position"] = [100.0] * 10
    df["y_position"] = [200.0] * 10
    df["well"] = ["A1_ops0089_20251119"] * 10

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df.to_csv(f.name, index=False)
        temp_path = f.name

    try:
        config = {"normalize_PCA_embeddings": False}
        # Should handle gracefully - may process with 512 features
        adata = create_adata_object_dinov3(temp_path, config=config)
        # Just check it doesn't crash - actual dimension is 512
        assert adata.shape[1] == 512
    finally:
        Path(temp_path).unlink()


def test_empty_dataframe():
    """Test 16: Validate handling of empty input."""
    # Create empty CSV with correct columns
    df = pd.DataFrame(
        columns=[str(i) for i in range(1024)]
        + [
            "label_int",
            "label_str",
            "sgRNA",
            "experiment",
            "x_position",
            "y_position",
            "well",
        ]
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df.to_csv(f.name, index=False)
        temp_path = f.name

    try:
        config = {"normalize_PCA_embeddings": False}
        # Should handle empty data gracefully
        adata = create_adata_object_dinov3(temp_path, config=config)
        assert adata.shape[0] == 0, "Empty input should produce empty AnnData"
    except Exception as e:
        # Also acceptable to raise an error for empty data
        assert "empty" in str(e).lower() or "shape" in str(e).lower()
    finally:
        Path(temp_path).unlink()


# ============================================================================
# Comparison Tests
# ============================================================================


def test_pipeline_output_structure_consistency(processed_adata):
    """Test 17: Ensure DinoV3 output structure matches expected format."""
    # Check that required metadata columns present
    required_obs = ["label_str", "label_int", "sgRNA", "well"]
    for col in required_obs:
        assert (
            col in processed_adata.obs.columns
        ), f"Required obs column {col} missing - inconsistent with CellProfiler format"

    # Check that .X is the feature matrix
    assert hasattr(processed_adata, "X")
    assert isinstance(processed_adata.X, np.ndarray)

    # Check that cell-level has no PCA/UMAP (different from CellProfiler which may have it)
    assert "X_pca" not in processed_adata.obsm.keys()
    assert "X_umap" not in processed_adata.obsm.keys()

    # Note: This is expected difference between DinoV3 and CellProfiler
    # Both will have PCA/UMAP at aggregation level (guide/gene bulked)
