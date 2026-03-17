"""Tests for ops_model.eval.run_eval CLI."""

from __future__ import annotations

import csv
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from ops_model.eval.run_eval import main, _default_output_path


# ---------------------------------------------------------------------------
# Coverage tests
# ---------------------------------------------------------------------------

def test_default_output_path_in_same_directory(tmp_path):
    embedding = str(tmp_path / "embeddings.h5ad")
    output = _default_output_path(embedding)
    assert Path(output).parent == tmp_path
    assert output.endswith("_eval.csv")


def test_raises_when_no_embedding_provided(monkeypatch):
    monkeypatch.setattr("sys.argv", ["run_eval"])
    with pytest.raises(SystemExit):
        main()


def test_guide_only_produces_guide_columns(tmp_path, monkeypatch):
    output_csv = str(tmp_path / "out.csv")
    guide_h5ad = str(tmp_path / "guide.h5ad")
    guide_results = {
        "pct_perturbations_active": 0.8,
        "mean_map_active": 0.7,
        "pct_pos_controls_active": 1.0,
        "mean_map_pos_controls": 0.9,
        "pct_perturbations_distinct": 0.6,
        "mean_map_distinct": 0.65,
        "mean_cosine_sim_within_gene": 0.9,
        "silhouette_within_gene": 0.85,
    }

    monkeypatch.setattr("sys.argv", ["run_eval", "--guide_embedding", guide_h5ad, "--output", output_csv])

    with patch("ops_model.eval.run_eval.ad.read_h5ad", return_value=MagicMock()):
        with patch("ops_model.eval.run_eval.evaluate_guide_level", return_value=(guide_results, MagicMock())):
            main()

    with open(output_csv) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 1
    assert "guide_embedding_path" in rows[0]
    assert "gene_embedding_path" not in rows[0]
    assert "pct_perturbations_active" in rows[0]
    assert "pct_complexes_significant_manual" not in rows[0]


def test_gene_only_produces_gene_columns(tmp_path, monkeypatch):
    output_csv = str(tmp_path / "out.csv")
    gene_h5ad = str(tmp_path / "gene.h5ad")
    gene_results = {
        "pct_complexes_significant_manual": 0.9,
        "mean_map_complexes_manual": 0.85,
        "pct_complexes_significant_corum": 0.7,
        "mean_map_complexes_corum": 0.75,
        "mean_cosine_sim_within_complex": 0.8,
        "silhouette_within_complex": 0.7,
    }

    monkeypatch.setattr("sys.argv", ["run_eval", "--gene_embedding", gene_h5ad, "--output", output_csv])

    with patch("ops_model.eval.run_eval.ad.read_h5ad", return_value=MagicMock()):
        with patch("ops_model.eval.run_eval.evaluate_gene_level", return_value=gene_results):
            main()

    with open(output_csv) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 1
    assert "gene_embedding_path" in rows[0]
    assert "guide_embedding_path" not in rows[0]
    assert "pct_complexes_significant_manual" in rows[0]
    assert "pct_perturbations_active" not in rows[0]


def test_both_embeddings_merged_into_one_row(tmp_path, monkeypatch):
    output_csv = str(tmp_path / "out.csv")
    guide_h5ad = str(tmp_path / "guide.h5ad")
    gene_h5ad = str(tmp_path / "gene.h5ad")
    guide_results = {"pct_perturbations_active": 0.8, "mean_cosine_sim_within_gene": 0.9,
                     "silhouette_within_gene": 0.85, "mean_map_active": 0.7,
                     "pct_pos_controls_active": 1.0, "mean_map_pos_controls": 0.9,
                     "pct_perturbations_distinct": 0.6, "mean_map_distinct": 0.65}
    gene_results = {"pct_complexes_significant_manual": 0.9, "mean_cosine_sim_within_complex": 0.8,
                    "silhouette_within_complex": 0.7, "mean_map_complexes_manual": 0.85,
                    "pct_complexes_significant_corum": 0.7, "mean_map_complexes_corum": 0.75}

    monkeypatch.setattr(
        "sys.argv",
        ["run_eval", "--guide_embedding", guide_h5ad, "--gene_embedding", gene_h5ad, "--output", output_csv],
    )

    with patch("ops_model.eval.run_eval.ad.read_h5ad", return_value=MagicMock()):
        with patch("ops_model.eval.run_eval.evaluate_guide_level", return_value=(guide_results, MagicMock())):
            with patch("ops_model.eval.run_eval.evaluate_gene_level", return_value=gene_results):
                main()

    with open(output_csv) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 1
    assert "guide_embedding_path" in rows[0]
    assert "gene_embedding_path" in rows[0]
    assert "pct_perturbations_active" in rows[0]
    assert "pct_complexes_significant_manual" in rows[0]


def test_default_output_path_used_when_not_specified(tmp_path, monkeypatch):
    guide_h5ad = str(tmp_path / "guide.h5ad")
    guide_results = {"pct_perturbations_active": 0.8, "mean_cosine_sim_within_gene": 0.9,
                     "silhouette_within_gene": 0.85, "mean_map_active": 0.7,
                     "pct_pos_controls_active": 1.0, "mean_map_pos_controls": 0.9,
                     "pct_perturbations_distinct": 0.6, "mean_map_distinct": 0.65}

    monkeypatch.setattr("sys.argv", ["run_eval", "--guide_embedding", guide_h5ad])

    with patch("ops_model.eval.run_eval.ad.read_h5ad", return_value=MagicMock()):
        with patch("ops_model.eval.run_eval.evaluate_guide_level", return_value=(guide_results, MagicMock())):
            main()

    output_files = list(tmp_path.glob("*_eval.csv"))
    assert len(output_files) == 1
