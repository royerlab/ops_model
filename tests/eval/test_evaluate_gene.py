"""Tests for ops_model.eval.evaluate_gene."""

from __future__ import annotations

from unittest.mock import patch

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import warnings

from ops_model.eval.evaluate_gene import evaluate_gene_level
from tests.eval.conftest import (
    make_gene_adata,
    make_activity_map,
    make_consistency_map,
    make_clusters,
)

EXPECTED_KEYS = {
    "pct_complexes_significant_manual",
    "mean_map_complexes_manual",
    "pct_complexes_significant_corum",
    "mean_map_complexes_corum",
    "mean_cosine_sim_within_complex",
    "silhouette_within_complex",
}

PERTURBATIONS = ["GENE_A1", "GENE_A2", "GENE_B1", "GENE_B2"]
CLUSTERS = make_clusters(PERTURBATIONS)

_EMBEDDINGS_PERFECT = np.array(
    [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
)
_EMBEDDINGS_RANDOM = np.random.default_rng(1).random((4, 3)).astype(np.float32)


ACTIVITY_MAP = make_activity_map(PERTURBATIONS, all_active=True)


def _mock_all(
    mock_manual,
    mock_corum,
    mock_load,
    all_significant=True,
    map_value=1.0,
):
    consistency_df = make_consistency_map(2, all_significant=all_significant, map_value=map_value)
    mock_manual.return_value = (consistency_df, float(all_significant))
    mock_corum.return_value = (consistency_df, float(all_significant))
    mock_load.return_value = CLUSTERS


# ---------------------------------------------------------------------------
# Coverage tests
# ---------------------------------------------------------------------------

def test_raises_on_missing_n_cells():
    adata = ad.AnnData(
        X=np.eye(3),
        obs=pd.DataFrame(
            {
                "perturbation": ["A", "B", "C"],
                "guides": [["sg_0"]] * 3,
                "n_experiments": [1] * 3,
            },
            index=["A", "B", "C"],
        ),
    )
    adata.uns["aggregation_method"] = "mean"
    adata.uns["cell_type"] = "HeLa"
    adata.uns["embedding_type"] = "test"
    with pytest.raises(Exception):
        evaluate_gene_level(adata)


def test_raises_on_missing_perturbation():
    adata = ad.AnnData(
        X=np.eye(3),
        obs=pd.DataFrame(
            {"n_cells": [10, 10, 10], "guides": [["sg_0"]] * 3, "n_experiments": [1] * 3},
            index=["A", "B", "C"],
        ),
    )
    adata.uns["aggregation_method"] = "mean"
    with pytest.raises(Exception):
        evaluate_gene_level(adata)


@patch("ops_model.eval.evaluate_gene._load_gene_clusters")
@patch("ops_model.eval.evaluate_gene.phenotypic_consistency_corum")
@patch("ops_model.eval.evaluate_gene.phenotypic_consistency_manual_annotation")
def test_returns_expected_keys(mock_manual, mock_corum, mock_load):
    _mock_all(mock_manual, mock_corum, mock_load)
    adata = make_gene_adata(PERTURBATIONS, _EMBEDDINGS_PERFECT)
    result = evaluate_gene_level(adata, activity_map=ACTIVITY_MAP)
    assert set(result.keys()) == EXPECTED_KEYS


@patch("ops_model.eval.evaluate_gene._load_gene_clusters")
@patch("ops_model.eval.evaluate_gene.phenotypic_consistency_corum")
@patch("ops_model.eval.evaluate_gene.phenotypic_consistency_manual_annotation")
def test_runs_without_error(mock_manual, mock_corum, mock_load):
    _mock_all(mock_manual, mock_corum, mock_load)
    adata = make_gene_adata(PERTURBATIONS, _EMBEDDINGS_PERFECT)
    evaluate_gene_level(adata, activity_map=ACTIVITY_MAP)  # should not raise


@patch("ops_model.eval.evaluate_gene._load_gene_clusters")
@patch("ops_model.eval.evaluate_gene.phenotypic_consistency_corum")
@patch("ops_model.eval.evaluate_gene.phenotypic_consistency_manual_annotation")
def test_no_activity_map_warns_and_runs(mock_manual, mock_corum, mock_load):
    """No activity_map → UserWarning issued, function still runs."""
    _mock_all(mock_manual, mock_corum, mock_load)
    adata = make_gene_adata(PERTURBATIONS, _EMBEDDINGS_PERFECT)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        evaluate_gene_level(adata)  # no activity_map
    assert any(issubclass(w.category, UserWarning) for w in caught)


# ---------------------------------------------------------------------------
# Metric correctness tests
# ---------------------------------------------------------------------------

@patch("ops_model.eval.evaluate_gene._load_gene_clusters")
@patch("ops_model.eval.evaluate_gene.phenotypic_consistency_corum")
@patch("ops_model.eval.evaluate_gene.phenotypic_consistency_manual_annotation")
def test_perfect_complex_structure_map_metrics(mock_manual, mock_corum, mock_load):
    """Mocked perfect structure → all mAP scalars are 1.0, all pct are 1.0."""
    _mock_all(mock_manual, mock_corum, mock_load, all_significant=True, map_value=1.0)
    adata = make_gene_adata(PERTURBATIONS, _EMBEDDINGS_PERFECT)
    result = evaluate_gene_level(adata, activity_map=ACTIVITY_MAP)
    assert np.isclose(result["pct_complexes_significant_manual"], 1.0)
    assert np.isclose(result["mean_map_complexes_manual"], 1.0)
    assert np.isclose(result["pct_complexes_significant_corum"], 1.0)
    assert np.isclose(result["mean_map_complexes_corum"], 1.0)


@patch("ops_model.eval.evaluate_gene._load_gene_clusters")
@patch("ops_model.eval.evaluate_gene.phenotypic_consistency_corum")
@patch("ops_model.eval.evaluate_gene.phenotypic_consistency_manual_annotation")
def test_no_structure_runs_without_error(mock_manual, mock_corum, mock_load):
    _mock_all(mock_manual, mock_corum, mock_load, all_significant=False, map_value=0.5)
    adata = make_gene_adata(PERTURBATIONS, _EMBEDDINGS_RANDOM)
    evaluate_gene_level(adata, activity_map=ACTIVITY_MAP)  # should not raise


@patch("ops_model.eval.evaluate_gene._load_gene_clusters")
@patch("ops_model.eval.evaluate_gene.phenotypic_consistency_corum")
@patch("ops_model.eval.evaluate_gene.phenotypic_consistency_manual_annotation")
def test_identical_complex_members_cosine_sim_one(mock_manual, mock_corum, mock_load):
    """Identical embeddings within each complex → cosine sim = 1.0."""
    _mock_all(mock_manual, mock_corum, mock_load)
    adata = make_gene_adata(PERTURBATIONS, _EMBEDDINGS_PERFECT)
    result = evaluate_gene_level(adata, activity_map=ACTIVITY_MAP)
    assert np.isclose(result["mean_cosine_sim_within_complex"], 1.0)


@patch("ops_model.eval.evaluate_gene._load_gene_clusters")
@patch("ops_model.eval.evaluate_gene.phenotypic_consistency_corum")
@patch("ops_model.eval.evaluate_gene.phenotypic_consistency_manual_annotation")
def test_orthogonal_complex_members_cosine_sim_zero(mock_manual, mock_corum, mock_load):
    """Orthogonal embeddings within each complex → cosine sim = 0.0."""
    _mock_all(mock_manual, mock_corum, mock_load)
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    adata = make_gene_adata(PERTURBATIONS, embeddings)
    result = evaluate_gene_level(adata, activity_map=ACTIVITY_MAP)
    assert np.isclose(result["mean_cosine_sim_within_complex"], 0.0)


@patch("ops_model.eval.evaluate_gene._load_gene_clusters")
@patch("ops_model.eval.evaluate_gene.phenotypic_consistency_corum")
@patch("ops_model.eval.evaluate_gene.phenotypic_consistency_manual_annotation")
def test_genes_absent_from_complexes_do_not_affect_cosine(mock_manual, mock_corum, mock_load):
    """Genes not in any complex should be ignored in the cosine similarity score."""
    _mock_all(mock_manual, mock_corum, mock_load)
    perturbations = PERTURBATIONS + ["EXTRA_GENE"]
    embeddings = np.vstack([_EMBEDDINGS_PERFECT, [[0.5, 0.5, 0.0]]])
    adata = make_gene_adata(perturbations, embeddings)
    result = evaluate_gene_level(adata, activity_map=ACTIVITY_MAP)
    assert np.isclose(result["mean_cosine_sim_within_complex"], 1.0)


@patch("ops_model.eval.evaluate_gene._load_gene_clusters")
@patch("ops_model.eval.evaluate_gene.phenotypic_consistency_corum")
@patch("ops_model.eval.evaluate_gene.phenotypic_consistency_manual_annotation")
def test_perfect_separation_silhouette_one(mock_manual, mock_corum, mock_load):
    """Perfect complex structure → silhouette = 1.0."""
    _mock_all(mock_manual, mock_corum, mock_load)
    adata = make_gene_adata(PERTURBATIONS, _EMBEDDINGS_PERFECT)
    result = evaluate_gene_level(adata, activity_map=ACTIVITY_MAP)
    assert np.isclose(result["silhouette_within_complex"], 1.0)
