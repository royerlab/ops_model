"""Tests for ops_model.eval.evaluate_guide."""

from __future__ import annotations

from unittest.mock import patch

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import warnings

from ops_model.eval.evaluate_guide import evaluate_guide_level
from tests.eval.conftest import (
    make_guide_adata,
    make_activity_map,
)

EXPECTED_KEYS = {
    "pct_perturbations_active",
    "mean_map_active",
    "pct_pos_controls_active",
    "mean_map_pos_controls",
    "pct_perturbations_distinct",
    "mean_map_distinct",
    "mean_cosine_sim_within_gene",
    "silhouette_within_gene",
}

PERTURBATIONS = ["A", "A", "B", "B"]

# Embeddings: perfect separation — A guides identical, B guides identical, A⊥B
_EMBEDDINGS_PERFECT = np.array(
    [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
)
_EMBEDDINGS_RANDOM = np.random.default_rng(0).random((4, 3)).astype(np.float32)

POS_CONTROLS = {"complex1": {"genes": ["A", "B"]}}


def _mock_all(mock_activity, mock_distinct, mock_load, all_active=True, map_value=1.0):
    activity_df = make_activity_map(["A", "B"], all_active=all_active, map_value=map_value)
    mock_activity.return_value = (activity_df, float(all_active))
    distinct_df = make_activity_map(["A", "B"], all_active=all_active, map_value=map_value)
    mock_distinct.return_value = (distinct_df, float(all_active))
    mock_load.return_value = POS_CONTROLS


# ---------------------------------------------------------------------------
# Coverage tests
# ---------------------------------------------------------------------------

def test_raises_on_missing_sgRNA():
    adata = ad.AnnData(
        X=np.eye(3),
        obs=pd.DataFrame({"perturbation": ["A", "B", "C"]}, index=["A", "B", "C"]),
    )
    adata.uns["aggregation_method"] = "mean"
    adata.uns["cell_type"] = "HeLa"
    adata.uns["embedding_type"] = "test"
    with pytest.raises(Exception):
        evaluate_guide_level(adata)


def test_raises_on_missing_perturbation():
    adata = ad.AnnData(
        X=np.eye(3),
        obs=pd.DataFrame({"sgRNA": ["sg_0", "sg_1", "sg_2"]}, index=["sg_0", "sg_1", "sg_2"]),
    )
    adata.uns["aggregation_method"] = "mean"
    with pytest.raises(Exception):
        evaluate_guide_level(adata)


@patch("ops_model.eval.evaluate_guide._load_pos_controls")
@patch("ops_model.eval.evaluate_guide.phenotypic_distinctivness")
@patch("ops_model.eval.evaluate_guide.phenotypic_activity_assesment")
def test_returns_expected_keys(mock_activity, mock_distinct, mock_load):
    _mock_all(mock_activity, mock_distinct, mock_load)
    adata = make_guide_adata(PERTURBATIONS, _EMBEDDINGS_PERFECT)
    result, _ = evaluate_guide_level(adata)
    assert set(result.keys()) == EXPECTED_KEYS


@patch("ops_model.eval.evaluate_guide._load_pos_controls")
@patch("ops_model.eval.evaluate_guide.phenotypic_distinctivness")
@patch("ops_model.eval.evaluate_guide.phenotypic_activity_assesment")
def test_returns_activity_map(mock_activity, mock_distinct, mock_load):
    _mock_all(mock_activity, mock_distinct, mock_load)
    adata = make_guide_adata(PERTURBATIONS, _EMBEDDINGS_PERFECT)
    _, activity_map = evaluate_guide_level(adata)
    assert isinstance(activity_map, pd.DataFrame)
    assert "perturbation" in activity_map.columns
    assert "below_corrected_p" in activity_map.columns


@patch("ops_model.eval.evaluate_guide._load_pos_controls")
@patch("ops_model.eval.evaluate_guide.phenotypic_distinctivness")
@patch("ops_model.eval.evaluate_guide.phenotypic_activity_assesment")
def test_runs_without_error(mock_activity, mock_distinct, mock_load):
    _mock_all(mock_activity, mock_distinct, mock_load)
    adata = make_guide_adata(PERTURBATIONS, _EMBEDDINGS_PERFECT)
    evaluate_guide_level(adata)  # should not raise


# ---------------------------------------------------------------------------
# Metric correctness tests
# ---------------------------------------------------------------------------

@patch("ops_model.eval.evaluate_guide._load_pos_controls")
@patch("ops_model.eval.evaluate_guide.phenotypic_distinctivness")
@patch("ops_model.eval.evaluate_guide.phenotypic_activity_assesment")
def test_perfect_separation_map_metrics(mock_activity, mock_distinct, mock_load):
    """Mocked perfect separation → all mAP scalars are 1.0, all pct are 1.0."""
    _mock_all(mock_activity, mock_distinct, mock_load, all_active=True, map_value=1.0)
    adata = make_guide_adata(PERTURBATIONS, _EMBEDDINGS_PERFECT)
    result, _ = evaluate_guide_level(adata)
    assert np.isclose(result["pct_perturbations_active"], 1.0)
    assert np.isclose(result["mean_map_active"], 1.0)
    assert np.isclose(result["pct_pos_controls_active"], 1.0)
    assert np.isclose(result["mean_map_pos_controls"], 1.0)
    assert np.isclose(result["pct_perturbations_distinct"], 1.0)
    assert np.isclose(result["mean_map_distinct"], 1.0)


@patch("ops_model.eval.evaluate_guide._load_pos_controls")
@patch("ops_model.eval.evaluate_guide.phenotypic_distinctivness")
@patch("ops_model.eval.evaluate_guide.phenotypic_activity_assesment")
def test_no_separation_runs_without_error(mock_activity, mock_distinct, mock_load):
    _mock_all(mock_activity, mock_distinct, mock_load, all_active=False, map_value=0.5)
    adata = make_guide_adata(PERTURBATIONS, _EMBEDDINGS_RANDOM)
    evaluate_guide_level(adata)  # should not raise (returns tuple, but we don't need it)


@patch("ops_model.eval.evaluate_guide._load_pos_controls")
@patch("ops_model.eval.evaluate_guide.phenotypic_distinctivness")
@patch("ops_model.eval.evaluate_guide.phenotypic_activity_assesment")
def test_identical_guides_cosine_sim_one(mock_activity, mock_distinct, mock_load):
    """Identical guides within each perturbation → cosine sim = 1.0."""
    _mock_all(mock_activity, mock_distinct, mock_load)
    embeddings = np.array(
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    )
    adata = make_guide_adata(PERTURBATIONS, embeddings)
    result, _ = evaluate_guide_level(adata)
    assert np.isclose(result["mean_cosine_sim_within_gene"], 1.0)


@patch("ops_model.eval.evaluate_guide._load_pos_controls")
@patch("ops_model.eval.evaluate_guide.phenotypic_distinctivness")
@patch("ops_model.eval.evaluate_guide.phenotypic_activity_assesment")
def test_orthogonal_guides_cosine_sim_zero(mock_activity, mock_distinct, mock_load):
    """Orthogonal guides within each perturbation → cosine sim = 0.0."""
    _mock_all(mock_activity, mock_distinct, mock_load)
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    adata = make_guide_adata(PERTURBATIONS, embeddings)
    result, _ = evaluate_guide_level(adata)
    assert np.isclose(result["mean_cosine_sim_within_gene"], 0.0)


@patch("ops_model.eval.evaluate_guide._load_pos_controls")
@patch("ops_model.eval.evaluate_guide.phenotypic_distinctivness")
@patch("ops_model.eval.evaluate_guide.phenotypic_activity_assesment")
def test_perfect_separation_silhouette_one(mock_activity, mock_distinct, mock_load):
    """Perfect cluster separation → silhouette = 1.0."""
    _mock_all(mock_activity, mock_distinct, mock_load)
    embeddings = np.array(
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    )
    adata = make_guide_adata(PERTURBATIONS, embeddings)
    result, _ = evaluate_guide_level(adata)
    assert np.isclose(result["silhouette_within_gene"], 1.0)
