"""Tests for configurable guide_col in anndata_utils aggregation paths.

Covers PR3 of the refactor — replacing hardcoded "sgRNA" with the
adata.uns["guide_col"] lookup so minibinder-style experiments aggregate
correctly without aliasing peptide identifiers onto an "sgRNA" column.
"""

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from ops_model.features.anndata_utils import (
    DEFAULT_GUIDE_COL,
    _guide_col,
    aggregate_to_level,
    hconcat_by_perturbation,
)


def _make_cell_adata(guide_col_name: str, set_uns_guide_col: bool) -> ad.AnnData:
    rng = np.random.default_rng(0)
    n_cells_per_guide = 3
    guides_per_gene = 2
    genes = ["CDC5L", "POLR2C", "NEG_CTRL"]

    rows = []
    for gene in genes:
        for g in range(guides_per_gene):
            if gene == "NEG_CTRL":
                guide_label = "no-peptide"
            else:
                guide_label = f"{gene}_g{g}"
            for c in range(n_cells_per_guide):
                rows.append(
                    {
                        guide_col_name: guide_label,
                        "perturbation": gene,
                        "label_str": gene,
                        "reporter": "GFP",
                        "well": "A1",
                        "experiment": "ops0154",
                        "x_position": float(c),
                        "y_position": float(c),
                    }
                )

    obs = pd.DataFrame(rows)
    X = rng.standard_normal(size=(len(rows), 5)).astype(np.float32)
    adata = ad.AnnData(X=X, obs=obs)
    adata.uns["cell_type"] = "A549"
    adata.uns["embedding_type"] = "dinov3"
    if set_uns_guide_col:
        adata.uns["guide_col"] = guide_col_name
    return adata


def test_guide_col_helper_default():
    adata = ad.AnnData(X=np.zeros((1, 1), dtype=np.float32))
    assert _guide_col(adata) == DEFAULT_GUIDE_COL


def test_guide_col_helper_reads_uns():
    adata = ad.AnnData(X=np.zeros((1, 1), dtype=np.float32))
    adata.uns["guide_col"] = "minibinder_perturbation"
    assert _guide_col(adata) == "minibinder_perturbation"


def test_aggregate_cell_to_guide_minibinder():
    """Cell→guide aggregation uses the minibinder column, not sgRNA."""
    adata = _make_cell_adata("minibinder_perturbation", set_uns_guide_col=True)
    adata_guide = aggregate_to_level(adata, level="guide")
    assert "minibinder_perturbation" in adata_guide.obs.columns
    assert "sgRNA" not in adata_guide.obs.columns
    assert adata_guide.uns["guide_col"] == "minibinder_perturbation"
    assert sorted(adata_guide.obs["minibinder_perturbation"].unique()) == [
        "CDC5L_g0",
        "CDC5L_g1",
        "POLR2C_g0",
        "POLR2C_g1",
        "no-peptide",
    ]


def test_aggregate_cell_to_gene_minibinder_no_nan_bug():
    """The original NaN bug: gene-level aggregation must not choke on the
    neg_ctrl 'no-peptide' group when guide_col is properly configured."""
    adata = _make_cell_adata("minibinder_perturbation", set_uns_guide_col=True)
    adata_gene = aggregate_to_level(adata, level="gene")
    assert "guides" in adata_gene.obs.columns
    assert adata_gene.uns["guide_col"] == "minibinder_perturbation"
    # The negative-control row contains the literal "no-peptide" string,
    # not a NaN, so pipe-joining works.
    neg_row = adata_gene.obs[adata_gene.obs["perturbation"] == "NEG_CTRL"].iloc[0]
    assert neg_row["guides"] == "no-peptide"


def test_aggregate_cell_to_gene_crispr_default():
    """Legacy CRISPR path (no guide_col in uns) still aggregates via 'sgRNA'."""
    adata = _make_cell_adata("sgRNA", set_uns_guide_col=False)
    adata_gene = aggregate_to_level(adata, level="gene")
    assert "guides" in adata_gene.obs.columns
    # No-peptide-like value is now keyed under 'sgRNA' so we just check the
    # neg_ctrl row got aggregated successfully.
    assert (adata_gene.obs["perturbation"] == "NEG_CTRL").any()


def test_aggregate_guide_to_gene_minibinder():
    """Guide → gene aggregation also threads guide_col through."""
    adata_cell = _make_cell_adata("minibinder_perturbation", set_uns_guide_col=True)
    adata_guide = aggregate_to_level(adata_cell, level="guide")
    adata_gene = aggregate_to_level(adata_guide, level="gene")
    assert adata_gene.uns["guide_col"] == "minibinder_perturbation"
    assert "guides" in adata_gene.obs.columns


def test_hconcat_by_perturbation_uses_guide_col():
    """Horizontal-concat join key follows adata.uns['guide_col']."""
    a = _make_cell_adata("minibinder_perturbation", set_uns_guide_col=True)
    a_guide = aggregate_to_level(a, level="guide")
    # Build a second block with a different feature set but same guides
    b_guide = a_guide.copy()
    b_guide.X = b_guide.X + 1.0  # change features
    b_guide.var_names = [f"alt_{v}" for v in b_guide.var_names]
    merged = hconcat_by_perturbation([a_guide, b_guide], level="guide")
    # 5 unique guides aligned, 10 stacked features
    assert merged.shape == (5, 10)
    assert "minibinder_perturbation" in merged.obs.columns
