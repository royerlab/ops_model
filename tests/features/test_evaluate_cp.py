"""
Tests for CellProfiler feature evaluation pipeline (evaluate_cp.py).

Covers create_adata_object() and the NONFEATURE_COLUMNS constant.
FeatureMetadata is mocked so tests run without database access.
"""

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from ops_model.features.evaluate_cp import NONFEATURE_COLUMNS, create_adata_object

# Minimal config that satisfies all branches in create_adata_object
_CONFIG = {"cell_type": "A549", "processing": {}}


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def mock_cp_csv():
    """
    Synthetic CellProfiler feature DataFrame.

    Uses Cells_AreaShape_* column names (not single_object_*) so that
    channel detection yields an empty set and FeatureMetadata methods
    beyond the constructor are never called.
    """
    np.random.seed(7)
    n_cells = 60
    genes = ["NTC", "GENE_A", "GENE_B"]
    gene_labels = np.random.choice(genes, size=n_cells)
    gene_to_int = {g: i for i, g in enumerate(genes)}

    df = pd.DataFrame(
        {
            "Cells_AreaShape_Area": np.random.uniform(100, 500, n_cells),
            "Cells_AreaShape_Eccentricity": np.random.uniform(0, 1, n_cells),
            "Cells_Intensity_MeanIntensity": np.random.uniform(0, 1, n_cells),
            "label_str": gene_labels,
            "label_int": [gene_to_int[g] for g in gene_labels],
            "sgRNA": [f"{g}_sg{np.random.randint(1, 4)}" for g in gene_labels],
            "experiment": "ops0138_20260305",
            "x_position": np.random.uniform(0, 1000, n_cells),
            "y_position": np.random.uniform(0, 1000, n_cells),
            "well": np.random.choice(
                ["A1_ops0138_20260305", "A2_ops0138_20260305"], n_cells
            ),
        }
    )
    return df


@pytest.fixture(scope="module")
def mock_cp_csv_path(mock_cp_csv, tmp_path_factory):
    """Write mock CSV to a temporary file."""
    temp_dir = tmp_path_factory.mktemp("cp_test")
    path = temp_dir / "cp_features_Phase2D.csv"
    mock_cp_csv.to_csv(path, index=False)
    yield path


@pytest.fixture(scope="module")
def processed_cp_adata(mock_cp_csv_path):
    """Run create_adata_object on mock data with FeatureMetadata mocked."""
    with patch("ops_utils.data.feature_metadata.FeatureMetadata") as mock_fm:
        mock_instance = MagicMock()
        # No channel renaming — return column name unchanged
        mock_instance.replace_channel_in_feature_name.side_effect = (
            lambda col, exp: col
        )
        mock_fm.return_value = mock_instance
        adata = create_adata_object(str(mock_cp_csv_path), config=_CONFIG)
    return adata


# ============================================================================
# Unit tests
# ============================================================================


def test_nonfeature_columns_definition():
    """NONFEATURE_COLUMNS contains exactly the expected metadata keys."""
    expected = {"label_str", "label_int", "sgRNA", "well", "experiment", "x_position", "y_position"}
    assert set(NONFEATURE_COLUMNS) == expected


# ============================================================================
# Integration tests
# ============================================================================


def test_create_adata_object_cp_structure(processed_cp_adata, mock_cp_csv):
    """AnnData has correct shape and all required .obs columns."""
    assert isinstance(processed_cp_adata, ad.AnnData)
    assert processed_cp_adata.shape[0] == len(mock_cp_csv)

    feature_cols = [c for c in mock_cp_csv.columns if c not in NONFEATURE_COLUMNS]
    assert processed_cp_adata.shape[1] == len(feature_cols)

    for col in ["perturbation", "sgRNA", "well", "experiment", "x_position", "y_position"]:
        assert col in processed_cp_adata.obs.columns, f"Missing obs column: {col}"


def test_create_adata_object_cp_metadata_integrity(processed_cp_adata, mock_cp_csv):
    """Gene labels and sgRNA values survive the pipeline unchanged."""
    assert sorted(processed_cp_adata.obs["perturbation"].unique()) == sorted(
        mock_cp_csv["label_str"].unique()
    )
    assert sorted(processed_cp_adata.obs["sgRNA"].unique()) == sorted(
        mock_cp_csv["sgRNA"].unique()
    )


def test_nonfeature_columns_excluded_from_X(processed_cp_adata):
    """No metadata column leaks into the feature matrix."""
    for col in NONFEATURE_COLUMNS:
        assert col not in processed_cp_adata.var_names, (
            f"Metadata column '{col}' should not appear in .var_names"
        )


def test_create_adata_object_cp_drops_constant_columns(mock_cp_csv, tmp_path):
    """Constant-value feature columns are removed by QC."""
    df = mock_cp_csv.copy()
    df["Cells_AreaShape_Constant"] = 42.0

    path = tmp_path / "cp_features_Phase2D.csv"
    df.to_csv(path, index=False)

    with patch("ops_utils.data.feature_metadata.FeatureMetadata") as mock_fm:
        mock_instance = MagicMock()
        mock_instance.replace_channel_in_feature_name.side_effect = (
            lambda col, exp: col
        )
        mock_fm.return_value = mock_instance
        adata = create_adata_object(str(path), config=_CONFIG)

    assert "Cells_AreaShape_Constant" not in adata.var_names
