"""Tests for configurable guide_col in AnndataValidator.

Covers the PR2 refactor that lets the per-construct identifier column be
named anything (e.g. "minibinder_perturbation") instead of hardcoded "sgRNA".
"""

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from ops_model.post_process.anndata_processing.anndata_validator import (
    AnndataSpec,
    AnndataValidator,
    DEFAULT_GUIDE_COL,
)


def _make_cell_obs(guide_col_name: str, guide_values: list[str]) -> pd.DataFrame:
    n = len(guide_values)
    return pd.DataFrame(
        {
            guide_col_name: guide_values,
            "perturbation": [f"GENE_{i}" for i in range(n)],
            "reporter": ["GFP"] * n,
            "well": ["A1"] * n,
            "experiment": ["ops0089"] * n,
            "x_position": [float(i) for i in range(n)],
            "y_position": [float(i) for i in range(n)],
        }
    )


def _make_cell_adata(
    guide_col_name: str,
    guide_values: list[str],
    set_uns_guide_col: bool,
) -> ad.AnnData:
    obs = _make_cell_obs(guide_col_name, guide_values)
    adata = ad.AnnData(X=np.random.rand(len(obs), 4).astype(np.float32), obs=obs)
    adata.uns["cell_type"] = "A549"
    adata.uns["embedding_type"] = "dinov3"
    if set_uns_guide_col:
        adata.uns["guide_col"] = guide_col_name
    return adata


def test_default_guide_col_is_sgrna():
    assert DEFAULT_GUIDE_COL == "sgRNA"


def test_anndataspec_defaults_to_sgrna():
    spec = AnndataSpec()
    cell_fields = spec.get_schema("cell")["required_fields"]
    names = [f.name for f in cell_fields]
    assert "sgRNA" in names


def test_anndataspec_uses_custom_guide_col():
    spec = AnndataSpec(guide_col="minibinder_perturbation")
    cell_fields = spec.get_schema("cell")["required_fields"]
    names = [f.name for f in cell_fields]
    assert "minibinder_perturbation" in names
    assert "sgRNA" not in names

    guide_fields = spec.get_schema("guide")["required_fields"]
    g_names = [f.name for f in guide_fields]
    assert "minibinder_perturbation" in g_names
    assert "sgRNA" not in g_names


def test_legacy_crispr_anndata_validates_with_default():
    """AnnData without uns['guide_col'] falls back to default 'sgRNA'."""
    adata = _make_cell_adata("sgRNA", ["g1", "g2", "g3"], set_uns_guide_col=False)
    report = AnndataValidator(strict=False).validate(adata, level="cell")
    assert report.is_valid, [(e.field, e.message) for e in report.errors]


def test_minibinder_anndata_validates_when_uns_guide_col_set():
    """AnnData with obs['minibinder_perturbation'] + uns['guide_col'] passes."""
    adata = _make_cell_adata(
        "minibinder_perturbation",
        ["2_1921", "2_1010", "no-peptide"],
        set_uns_guide_col=True,
    )
    report = AnndataValidator(strict=False).validate(adata, level="cell")
    assert report.is_valid, [(e.field, e.message) for e in report.errors]


def test_minibinder_data_without_uns_key_fails_on_default_column():
    """If uns['guide_col'] is absent, validator falls back to 'sgRNA' and fails."""
    adata = _make_cell_adata(
        "minibinder_perturbation",
        ["2_1921", "2_1010", "no-peptide"],
        set_uns_guide_col=False,
    )
    report = AnndataValidator(strict=False).validate(adata, level="cell")
    assert not report.is_valid
    assert any(e.field == "sgRNA" for e in report.errors), report.errors


def test_validator_can_be_initialized_with_explicit_guide_col():
    """Explicit guide_col on the validator overrides the default fallback."""
    adata = _make_cell_adata(
        "minibinder_perturbation",
        ["2_1921", "2_1010", "no-peptide"],
        set_uns_guide_col=False,
    )
    v = AnndataValidator(strict=False, guide_col="minibinder_perturbation")
    report = v.validate(adata, level="cell")
    assert report.is_valid, [(e.field, e.message) for e in report.errors]


def test_guide_level_uses_dynamic_column():
    """Guide-level validation should require the configured column."""
    n = 6
    obs = pd.DataFrame(
        {
            "minibinder_perturbation": [f"p{i}" for i in range(n)],
            "perturbation": ["G"] * n,
            "reporter": ["GFP"] * n,
            "n_cells": [10] * n,
        }
    )
    adata = ad.AnnData(X=np.random.rand(n, 4).astype(np.float32), obs=obs)
    adata.uns["cell_type"] = "A549"
    adata.uns["embedding_type"] = "dinov3"
    adata.uns["guide_col"] = "minibinder_perturbation"
    adata.uns["aggregation_method"] = "mean"

    report = AnndataValidator(strict=False).validate(adata, level="guide")
    assert report.is_valid, [(e.field, e.message) for e in report.errors]


def test_infer_schema_level_uses_guide_col():
    """infer_schema_level reads guide_col from uns for the guide-level heuristic."""
    n = 10
    obs = pd.DataFrame(
        {
            "minibinder_perturbation": [f"p{i}" for i in range(n)],
            "perturbation": ["G"] * n,
            "reporter": ["GFP"] * n,
            "n_cells": [10] * n,
        }
    )
    adata = ad.AnnData(X=np.random.rand(n, 4).astype(np.float32), obs=obs)
    adata.uns["guide_col"] = "minibinder_perturbation"

    level = AnndataValidator(strict=False).infer_schema_level(adata)
    assert level == "guide"
