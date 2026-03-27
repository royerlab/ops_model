"""Tests for ops_model.eval.metrics."""

from __future__ import annotations

import numpy as np
import pytest

from ops_model.eval.metrics import mean_cosine_sim_within_groups
from tests.eval.conftest import make_guide_adata


# ---------------------------------------------------------------------------
# Coverage tests
# ---------------------------------------------------------------------------

def test_returns_float():
    adata = make_guide_adata(["A", "A"], np.array([[1.0, 0.0], [1.0, 0.0]]))
    result = mean_cosine_sim_within_groups(adata, [[0, 1]])
    assert isinstance(result, float)


def test_result_in_valid_range():
    adata = make_guide_adata(["A", "A"], np.array([[1.0, 0.0], [0.5, 0.5]]))
    result = mean_cosine_sim_within_groups(adata, [[0, 1]])
    assert -1.0 <= result <= 1.0


def test_skips_groups_smaller_than_two():
    # Both groups have 1 member — all skipped → NaN
    adata = make_guide_adata(["A", "B"], np.eye(2))
    result = mean_cosine_sim_within_groups(adata, [[0], [1]])
    assert np.isnan(result)


def test_empty_groups_list_returns_nan():
    adata = make_guide_adata(["A"], np.array([[1.0, 0.0]]))
    result = mean_cosine_sim_within_groups(adata, [])
    assert np.isnan(result)


def test_single_group():
    adata = make_guide_adata(["A", "A"], np.array([[1.0, 0.0], [1.0, 0.0]]))
    result = mean_cosine_sim_within_groups(adata, [[0, 1]])
    assert np.isclose(result, 1.0)


def test_multiple_groups_unequal_size():
    embeddings = np.array(
        [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
    )
    adata = make_guide_adata(["A", "A", "A", "B", "B"], embeddings)
    result = mean_cosine_sim_within_groups(adata, [[0, 1, 2], [3, 4]])
    assert np.isclose(result, 1.0)


# ---------------------------------------------------------------------------
# Metric correctness tests (synthetic embeddings)
# ---------------------------------------------------------------------------

def test_identical_vectors_returns_one():
    adata = make_guide_adata(["A", "A"], np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    result = mean_cosine_sim_within_groups(adata, [[0, 1]])
    assert np.isclose(result, 1.0)


def test_orthogonal_vectors_returns_zero():
    adata = make_guide_adata(["A", "A"], np.array([[1.0, 0.0], [0.0, 1.0]]))
    result = mean_cosine_sim_within_groups(adata, [[0, 1]])
    assert np.isclose(result, 0.0)


def test_anti_parallel_vectors_returns_minus_one():
    adata = make_guide_adata(["A", "A"], np.array([[1.0, 0.0], [-1.0, 0.0]]))
    result = mean_cosine_sim_within_groups(adata, [[0, 1]])
    assert np.isclose(result, -1.0)


def test_known_analytic_value():
    # angle between [1,0] and [1,1]/sqrt(2) is 45° → cos = sqrt(2)/2
    embeddings = np.array([[1.0, 0.0], [1.0, 1.0]])
    adata = make_guide_adata(["A", "A"], embeddings)
    result = mean_cosine_sim_within_groups(adata, [[0, 1]])
    assert np.isclose(result, np.sqrt(2) / 2, atol=1e-5)
