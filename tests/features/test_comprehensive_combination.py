"""
Comprehensive tests for Strategy 3: Comprehensive Multi-Experiment Combination.

Tests cover:
- Biological signal grouping
- Vertical aggregation (same biology)
- Horizontal concatenation (different biology)
- Observation alignment
- Metadata creation
- Fallback cases
- Full integration
"""

import pytest
import numpy as np
import pandas as pd
import anndata as ad
from pathlib import Path

from ops_model.data.feature_metadata import FeatureMetadata
from ops_model.features.anndata_utils import (
    _group_by_biological_signal,
    _combine_duplicate_observations,
    _process_vertical_group,
    _process_horizontal_group,
    _align_biological_groups,
    _concatenate_horizontal,
    _create_comprehensive_metadata,
    concatenate_experiments_comprehensive,
)


# ============================================================================
# Phase 1: Biological Signal Grouping Tests
# ============================================================================


class TestBiologicalGrouping:
    """Test biological signal grouping logic."""

    def test_group_by_biological_signal_mixed(self):
        """Test grouping with mixed same/different biology."""
        meta = FeatureMetadata()

        experiments_channels = [
            ("ops0089", "Phase2D"),
            ("ops0089", "GFP"),
            ("ops0108", "Phase2D"),
            ("ops0108", "GFP"),
        ]

        groups = _group_by_biological_signal(experiments_channels, meta, verbose=False)

        # Should have 3 groups: Phase (2 sources), EEA1 (1), TOMM70A (1)
        assert len(groups) == 3
        assert "Phase" in groups
        assert len(groups["Phase"]) == 2

    def test_group_all_same_biology(self):
        """Test when all channels represent same biology."""
        meta = FeatureMetadata()

        experiments_channels = [
            ("ops0089", "Phase2D"),
            ("ops0108", "Phase2D"),
        ]

        groups = _group_by_biological_signal(experiments_channels, meta, verbose=False)

        # Should have only one group
        assert len(groups) == 1
        assert "Phase" in groups
        assert len(groups["Phase"]) == 2

    def test_group_all_different_biology(self):
        """Test when all channels are different."""
        meta = FeatureMetadata()

        experiments_channels = [
            ("ops0089", "GFP"),
            ("ops0108", "GFP"),
            ("ops0084", "GFP"),
        ]

        groups = _group_by_biological_signal(experiments_channels, meta, verbose=False)

        # Should have three separate groups (different markers)
        assert len(groups) == 3
        assert all(len(pairs) == 1 for pairs in groups.values())

    def test_channel_normalization(self):
        """Test that Phase2D maps to BF correctly."""
        meta = FeatureMetadata()

        # Using Phase2D should map to BF in YAML
        experiments_channels = [("ops0089", "Phase2D")]

        groups = _group_by_biological_signal(experiments_channels, meta, verbose=False)

        # Should find Phase group
        assert "Phase" in groups


# ============================================================================
# Phase 2: Vertical Aggregation Tests
# ============================================================================


class TestVerticalAggregation:
    """Test vertical aggregation logic."""

    def test_combine_duplicate_observations(self):
        """Test combining same gene from different experiments."""
        # Create mock data
        X = np.array(
            [
                [1.0, 2.0, 3.0],  # RPL18 from ops0089
                [3.0, 4.0, 5.0],  # RPL18 from ops0108
                [5.0, 6.0, 7.0],  # TP53 from ops0089
            ]
        )

        obs = pd.DataFrame(
            {
                "label_str": ["RPL18", "RPL18", "TP53"],
                "experiment": ["ops0089", "ops0108", "ops0089"],
            }
        )

        adata = ad.AnnData(X=X, obs=obs)
        adata.var_names = ["feat_0", "feat_1", "feat_2"]

        cell_counts = {"ops0089": 1000, "ops0108": 1200}

        adata_pooled = _combine_duplicate_observations(
            adata, level="gene", cell_counts=cell_counts, verbose=False
        )

        # Check shape
        assert adata_pooled.shape == (2, 3)

        # Check RPL18 is averaged (unweighted)
        rpl18_idx = np.where(adata_pooled.obs["label_str"] == "RPL18")[0][0]
        expected_rpl18 = np.array([2.0, 3.0, 4.0])
        assert np.allclose(adata_pooled.X[rpl18_idx], expected_rpl18)

        # Check metadata
        assert "vertical_metadata" in adata_pooled.uns
        assert adata_pooled.uns["vertical_metadata"]["total_cells"] == 2200
        assert (
            adata_pooled.uns["vertical_metadata"]["aggregation_method"]
            == "unweighted_mean"
        )

    def test_combine_no_duplicates(self):
        """Test combining when no duplicate observations."""
        X = np.array(
            [
                [1.0, 2.0],  # Gene A from ops0089
                [3.0, 4.0],  # Gene B from ops0108
            ]
        )

        obs = pd.DataFrame(
            {"label_str": ["GENE_A", "GENE_B"], "experiment": ["ops0089", "ops0108"]}
        )

        adata = ad.AnnData(X=X, obs=obs)
        adata.var_names = ["feat_0", "feat_1"]

        cell_counts = {"ops0089": 100, "ops0108": 200}

        adata_pooled = _combine_duplicate_observations(
            adata, level="gene", cell_counts=cell_counts, verbose=False
        )

        # Should have both genes (no combining needed)
        assert adata_pooled.shape == (2, 2)


# ============================================================================
# Phase 3: Horizontal Concatenation Tests
# ============================================================================


class TestHorizontalConcatenation:
    """Test horizontal concatenation and alignment."""

    def test_align_inner_join(self):
        """Test inner join alignment."""
        obs1 = pd.DataFrame({"label_str": ["A", "B", "C"]})
        obs2 = pd.DataFrame({"label_str": ["B", "C", "D"]})

        adata1 = ad.AnnData(X=np.zeros((3, 10)), obs=obs1)
        adata2 = ad.AnnData(X=np.zeros((3, 10)), obs=obs2)

        group_adatas = {"group1": adata1, "group2": adata2}

        common = _align_biological_groups(
            group_adatas, "gene", join="inner", verbose=False
        )

        # Should have B and C (intersection)
        assert set(common) == {"B", "C"}

    def test_align_outer_join(self):
        """Test outer join alignment."""
        obs1 = pd.DataFrame({"label_str": ["A", "B"]})
        obs2 = pd.DataFrame({"label_str": ["C", "D"]})

        adata1 = ad.AnnData(X=np.zeros((2, 10)), obs=obs1)
        adata2 = ad.AnnData(X=np.zeros((2, 10)), obs=obs2)

        group_adatas = {"group1": adata1, "group2": adata2}

        common = _align_biological_groups(
            group_adatas, "gene", join="outer", verbose=False
        )

        # Should have all (union)
        assert set(common) == {"A", "B", "C", "D"}

    def test_align_no_common_raises(self):
        """Test that no common observations raises error."""
        obs1 = pd.DataFrame({"label_str": ["A", "B"]})
        obs2 = pd.DataFrame({"label_str": ["C", "D"]})

        adata1 = ad.AnnData(X=np.zeros((2, 10)), obs=obs1)
        adata2 = ad.AnnData(X=np.zeros((2, 10)), obs=obs2)

        group_adatas = {"group1": adata1, "group2": adata2}

        with pytest.raises(ValueError, match="No common"):
            _align_biological_groups(group_adatas, "gene", join="inner", verbose=False)

    def test_concatenate_horizontal_features(self):
        """Test horizontal feature concatenation."""
        genes = ["RPL18", "TP53", "GAPDH"]

        # Create three groups with different feature values
        obs1 = pd.DataFrame({"label_str": genes})
        adata1 = ad.AnnData(X=np.ones((3, 5)), obs=obs1)
        adata1.var_names = [f"feat_{i}" for i in range(5)]

        obs2 = pd.DataFrame({"label_str": genes})
        adata2 = ad.AnnData(X=np.ones((3, 3)) * 2, obs=obs2)
        adata2.var_names = [f"feat_{i}" for i in range(3)]

        group_adatas = {
            "Phase": adata1,
            "early endosome, EEA1": adata2,
        }

        biological_signals = {
            "Phase": [("ops0089", "Phase2D")],
            "early endosome, EEA1": [("ops0089", "GFP")],
        }

        meta = FeatureMetadata()

        adata_combined = _concatenate_horizontal(
            group_adatas=group_adatas,
            common_obs=genes,
            biological_signals=biological_signals,
            meta=meta,
            feature_type="dinov3",
            target_level="gene",
            verbose=False,
        )

        # Check shape
        assert adata_combined.shape == (3, 8)  # 3 genes × 8 features

        # Check feature values are correct
        assert np.allclose(adata_combined.X[0, 0:5], 1.0)  # Phase features
        assert np.allclose(adata_combined.X[0, 5:8], 2.0)  # EEA1 features

        # Check metadata
        assert "feature_slices" in adata_combined.uns
        assert "Phase" in adata_combined.uns["feature_slices"]


# ============================================================================
# Phase 4: Metadata Tests
# ============================================================================


class TestMetadata:
    """Test comprehensive metadata creation."""

    def test_create_comprehensive_metadata(self):
        """Test metadata structure creation."""
        # Create mock group AnnData objects
        obs_guide = pd.DataFrame({"sgRNA": ["guide1", "guide2"]})
        adata_phase = ad.AnnData(X=np.ones((2, 1024)), obs=obs_guide)
        adata_phase.uns["vertical_metadata"] = {
            "experiments": ["ops0089", "ops0108"],
            "cell_counts_per_experiment": {"ops0089": 1000, "ops0108": 1200},
            "total_cells": 2200,
        }

        adata_eea1 = ad.AnnData(X=np.ones((2, 1024)), obs=obs_guide)
        adata_eea1.uns["horizontal_metadata"] = {
            "experiment": "ops0089",
            "channel": "GFP",
            "n_cells": 1000,
        }

        group_adatas = {"Phase": adata_phase, "early endosome, EEA1": adata_eea1}

        biological_signals = {
            "Phase": [("ops0089", "Phase2D"), ("ops0108", "Phase2D")],
            "early endosome, EEA1": [("ops0089", "GFP")],
        }

        feature_slices = {
            "Phase": {"start": 0, "end": 1024, "n_features": 1024},
            "early endosome, EEA1": {"start": 1024, "end": 2048, "n_features": 1024},
        }

        meta = FeatureMetadata()

        metadata = _create_comprehensive_metadata(
            biological_signals=biological_signals,
            group_adatas=group_adatas,
            meta=meta,
            feature_type="dinov3",
            feature_slices=feature_slices,
            target_level="guide",
        )

        # Check structure
        assert metadata["strategy"] == "comprehensive"
        assert metadata["feature_type"] == "dinov3"
        assert metadata["n_biological_signals"] == 2

        # Check Phase group (vertical)
        phase_info = metadata["biological_groups"]["Phase"]
        assert phase_info["aggregation_type"] == "vertical"
        assert phase_info["n_cells_total"] == 2200
        assert len(phase_info["experiments"]) == 2

        # Check EEA1 group (horizontal)
        eea1_info = metadata["biological_groups"]["early endosome, EEA1"]
        assert eea1_info["aggregation_type"] == "horizontal"
        assert eea1_info["n_cells_total"] == 1000
        assert len(eea1_info["experiments"]) == 1


# ============================================================================
# Phase 5: Edge Cases and Validation Tests
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_experiments_channels_raises(self):
        """Test that empty input raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            concatenate_experiments_comprehensive(
                experiments_channels=[], feature_type="dinov3"
            )

    def test_malformed_tuples_raises(self):
        """Test that malformed tuples raise error."""
        with pytest.raises(ValueError, match="must be list of"):
            concatenate_experiments_comprehensive(
                experiments_channels=[("ops0089",)],  # Missing channel
                feature_type="dinov3",
            )

    def test_missing_metadata_raises(self):
        """Test that missing metadata raises error."""
        with pytest.raises(ValueError, match="No metadata found"):
            experiments_channels = [("ops9999", "GFP")]
            meta = FeatureMetadata()
            _group_by_biological_signal(experiments_channels, meta, verbose=False)


# ============================================================================
# Integration Tests (require real data)
# ============================================================================


@pytest.mark.integration
@pytest.mark.skipif(
    not Path("/hpc/projects/intracellular_dashboard/ops").exists(),
    reason="Requires access to HPC data",
)
class TestIntegration:
    """Integration tests with real data."""

    def test_comprehensive_with_real_data(self):
        """Test full workflow with real OPS data."""
        experiments_channels = [
            ("ops0089_20251119", "Phase2D"),
            ("ops0089_20251119", "GFP"),
        ]

        adata_guide, adata_gene = concatenate_experiments_comprehensive(
            experiments_channels, feature_type="dinov3", verbose=False
        )

        # Validate basic structure
        assert adata_guide.shape[0] > 0
        assert adata_gene.shape[0] > 0
        assert "comprehensive_metadata" in adata_gene.uns

        # Check embeddings
        assert "X_pca" in adata_gene.obsm
        assert "X_umap" in adata_gene.obsm

    def test_fallback_vertical_with_real_data(self):
        """Test vertical fallback with real data."""
        experiments_channels = [
            ("ops0089_20251119", "Phase2D"),
            ("ops0108_20251201", "Phase2D"),
        ]

        adata_guide, adata_gene = concatenate_experiments_comprehensive(
            experiments_channels, feature_type="dinov3", verbose=False
        )

        # Should have single feature set (vertical fallback)
        assert adata_gene.shape[1] == 1024

    def test_fallback_horizontal_with_real_data(self):
        """Test horizontal fallback with real data."""
        experiments_channels = [
            ("ops0089_20251119", "GFP"),
            ("ops0108_20251201", "GFP"),
        ]

        adata_guide, adata_gene = concatenate_experiments_comprehensive(
            experiments_channels, feature_type="dinov3", verbose=False
        )

        # Should have two feature sets (horizontal fallback)
        assert adata_gene.shape[1] == 2048  # 2 × 1024


# ============================================================================
# Parametrized Tests
# ============================================================================


@pytest.mark.parametrize("join_type", ["inner", "outer"])
def test_alignment_with_different_joins(join_type):
    """Test alignment works with both join types."""
    obs1 = pd.DataFrame({"label_str": ["A", "B", "C"]})
    obs2 = pd.DataFrame({"label_str": ["B", "C", "D"]})

    adata1 = ad.AnnData(X=np.zeros((3, 10)), obs=obs1)
    adata2 = ad.AnnData(X=np.zeros((3, 10)), obs=obs2)

    group_adatas = {"group1": adata1, "group2": adata2}

    common = _align_biological_groups(
        group_adatas, "gene", join=join_type, verbose=False
    )

    if join_type == "inner":
        assert set(common) == {"B", "C"}
    else:  # outer
        assert set(common) == {"A", "B", "C", "D"}


@pytest.mark.parametrize("level", ["guide", "gene"])
def test_combine_duplicates_at_different_levels(level):
    """Test combining duplicates works at both guide and gene levels."""
    if level == "guide":
        label_key = "sgRNA"
    else:
        label_key = "label_str"

    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    obs = pd.DataFrame(
        {label_key: ["item1", "item1"], "experiment": ["ops0089", "ops0108"]}
    )

    adata = ad.AnnData(X=X, obs=obs)
    adata.var_names = ["feat_0", "feat_1"]
    cell_counts = {"ops0089": 100, "ops0108": 100}

    adata_pooled = _combine_duplicate_observations(
        adata, level=level, cell_counts=cell_counts, verbose=False
    )

    assert adata_pooled.shape == (1, 2)
    assert np.allclose(adata_pooled.X[0], [2.0, 3.0])


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_feature_metadata():
    """Fixture providing FeatureMetadata instance."""
    return FeatureMetadata()


@pytest.fixture
def mock_gene_data():
    """Fixture providing mock gene-level data."""
    genes = ["RPL18", "TP53", "GAPDH"]
    obs = pd.DataFrame({"label_str": genes})
    X = np.random.rand(3, 10)
    return ad.AnnData(X=X, obs=obs)
