"""Shared fixtures for eval tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import anndata as ad


# ---------------------------------------------------------------------------
# Factory functions (used directly in tests that need custom data)
# ---------------------------------------------------------------------------

def make_guide_adata(perturbations: list[str], embeddings: np.ndarray) -> ad.AnnData:
    """Build a minimal valid guide-level AnnData.

    Parameters
    ----------
    perturbations : list of str
        Perturbation label for each row.
    embeddings : ndarray of shape (n_guides, n_features)
    """
    n = len(perturbations)
    assert embeddings.shape[0] == n
    obs = pd.DataFrame(
        {
            "perturbation": perturbations,
            "sgRNA": [f"sg_{i}" for i in range(n)],
            "n_cells": [10] * n,
        }
    )
    obs.index = obs["sgRNA"].values
    obs.index.name = None
    adata = ad.AnnData(X=embeddings.astype(np.float32), obs=obs)
    adata.uns["aggregation_method"] = "mean"
    adata.uns["cell_type"] = "HeLa"
    adata.uns["embedding_type"] = "test"
    return adata


def make_gene_adata(perturbations: list[str], embeddings: np.ndarray) -> ad.AnnData:
    """Build a minimal valid gene-level AnnData.

    Parameters
    ----------
    perturbations : list of str
        Perturbation label for each row (one per gene).
    embeddings : ndarray of shape (n_genes, n_features)
    """
    n = len(perturbations)
    assert embeddings.shape[0] == n
    obs = pd.DataFrame(
        {
            "perturbation": perturbations,
            "n_cells": [20] * n,
            "guides": [["sg_0", "sg_1"]] * n,
            "n_experiments": [1] * n,
        }
    )
    obs.index = perturbations
    adata = ad.AnnData(X=embeddings.astype(np.float32), obs=obs)
    adata.uns["aggregation_method"] = "mean"
    adata.uns["cell_type"] = "HeLa"
    adata.uns["embedding_type"] = "test"
    return adata


def make_clusters(perturbations: list[str]) -> dict:
    """Build a clusters dict grouping consecutive pairs of perturbations.

    Parameters
    ----------
    perturbations : list of str
        Genes to group. Must have even length.
    """
    assert len(perturbations) % 2 == 0, "perturbations must have even length"
    clusters = {}
    for i in range(0, len(perturbations), 2):
        cluster_name = f"complex_{i // 2}"
        clusters[cluster_name] = {"genes": [perturbations[i], perturbations[i + 1]]}
    return clusters


def make_activity_map(
    perturbations: list[str],
    all_active: bool = True,
    map_value: float = 1.0,
) -> pd.DataFrame:
    """Build a minimal activity_map DataFrame as returned by phenotypic_activity_assesment."""
    return pd.DataFrame(
        {
            "perturbation": perturbations,
            "below_corrected_p": [all_active] * len(perturbations),
            "mean_average_precision": [map_value] * len(perturbations),
            "corrected_p_value": [0.01 if all_active else 0.5] * len(perturbations),
        }
    )


def make_consistency_map(
    n_complexes: int,
    all_significant: bool = True,
    map_value: float = 1.0,
) -> pd.DataFrame:
    """Build a minimal consistency_map DataFrame as returned by phenotypic_consistency_*."""
    return pd.DataFrame(
        {
            "complex_id": [f"c{i}" for i in range(n_complexes)],
            "below_corrected_p": [all_significant] * n_complexes,
            "mean_average_precision": [map_value] * n_complexes,
            "corrected_p_value": [0.01 if all_significant else 0.5] * n_complexes,
        }
    )


# ---------------------------------------------------------------------------
# pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def guide_adata_identical():
    """2 perturbations × 2 guides each; identical embeddings within perturbation."""
    embeddings = np.array(
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    )
    return make_guide_adata(["A", "A", "B", "B"], embeddings)


@pytest.fixture
def guide_adata_orthogonal():
    """2 perturbations × 2 guides each; orthogonal embeddings within perturbation."""
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    return make_guide_adata(["A", "A", "B", "B"], embeddings)


@pytest.fixture
def gene_adata_identical():
    """4 genes in 2 complexes; identical embeddings within complex."""
    perturbations = ["GENE_A1", "GENE_A2", "GENE_B1", "GENE_B2"]
    embeddings = np.array(
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    )
    return make_gene_adata(perturbations, embeddings)


@pytest.fixture
def gene_adata_orthogonal():
    """4 genes in 2 complexes; orthogonal embeddings within complex."""
    perturbations = ["GENE_A1", "GENE_A2", "GENE_B1", "GENE_B2"]
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    return make_gene_adata(perturbations, embeddings)


@pytest.fixture
def gene_clusters_fixture():
    """Matching clusters dict for gene_adata_* fixtures."""
    return make_clusters(["GENE_A1", "GENE_A2", "GENE_B1", "GENE_B2"])
