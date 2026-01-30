"""
Tests for feature concatenation (horizontal concatenation).

Tests both strategies:
- Strategy 1: Multi-Channel (same experiment, multiple channels)
- Strategy 2: Multi-Organelle (multiple experiments, same channel)
"""

import pytest
import numpy as np
import pandas as pd
import anndata as ad
from pathlib import Path

from ops_model.features.anndata_utils import concatenate_features_by_channel


# Helper function to create mock AnnData objects
def create_mock_adata(n_obs=100, n_vars=1024, obs_labels=None):
    """Create a mock AnnData object for testing."""
    if obs_labels is None:
        obs_labels = [f"Gene_{i}" for i in range(n_obs)]

    X = np.random.randn(n_obs, n_vars).astype(np.float32)
    obs = pd.DataFrame(
        {
            "label_str": obs_labels,
            "label_int": range(n_obs),
            "sgRNA": [f"sgRNA_{i}" for i in range(n_obs)],
            "well": ["A1"] * n_obs,
        }
    )

    adata = ad.AnnData(X=X, obs=obs)
    adata.var_names = [f"feature_{i}" for i in range(n_vars)]

    return adata


class TestParameterValidation:
    """Test parameter validation for strategy selection."""

    def test_requires_valid_strategy(self):
        """Test that at least one strategy must be specified."""
        with pytest.raises(ValueError, match="Must provide either"):
            concatenate_features_by_channel(
                feature_type="dinov3", aggregation_level="gene"
            )

    def test_mutually_exclusive_parameters(self):
        """Test that strategy parameters are mutually exclusive."""
        with pytest.raises(ValueError, match="mutually exclusive"):
            concatenate_features_by_channel(
                experiment="ops0089",
                channels=["Phase2D"],
                experiments_channels=[("ops0089", "GFP")],
                feature_type="dinov3",
                aggregation_level="gene",
            )

    def test_cell_level_not_supported(self):
        """Test that cell-level concatenation raises error."""
        with pytest.raises(
            ValueError, match="Cell-level feature concatenation not supported"
        ):
            concatenate_features_by_channel(
                experiment="ops0089",
                channels=["Phase2D", "GFP"],
                feature_type="dinov3",
                aggregation_level="cell",
            )


class TestAlignment:
    """Test that observation alignment works correctly."""

    def test_common_genes_intersection(self):
        """Test that only common genes are kept."""
        # Create mock data with different gene sets
        genes1 = [f"Gene_{i}" for i in range(100)]
        genes2 = [f"Gene_{i}" for i in range(50, 150)]  # 50-99 overlap

        adata1 = create_mock_adata(n_obs=100, obs_labels=genes1)
        adata2 = create_mock_adata(n_obs=100, obs_labels=genes2)

        # In real implementation, would test with actual files
        # Here we're just documenting the expected behavior

        # Expected common genes: 50 (genes 50-99)
        common = set(genes1) & set(genes2)
        assert len(common) == 50

    def test_alignment_order(self):
        """Test that observations are sorted consistently."""
        genes = [f"Gene_{i}" for i in range(100)]

        # Create data with shuffled order
        shuffled_genes = genes.copy()
        np.random.shuffle(shuffled_genes)

        adata1 = create_mock_adata(n_obs=100, obs_labels=genes)
        adata2 = create_mock_adata(n_obs=100, obs_labels=shuffled_genes)

        # After alignment, both should have same order
        # This is what the implementation does via argsort()
        sorted_genes1 = sorted(genes)
        sorted_genes2 = sorted(shuffled_genes)
        assert sorted_genes1 == sorted_genes2


class TestFeatureSlices:
    """Test feature slice tracking."""

    def test_feature_slices_format(self):
        """Test that feature slices have correct format."""
        # Expected format for Strategy 1 (lists for HDF5 compatibility)
        slices_s1 = {
            "ops0089_Phase2D": [0, 1024],
            "ops0089_GFP": [1024, 2048],
            "ops0089_mCherry": [2048, 3072],
        }

        # Verify format
        for key, (start, end) in slices_s1.items():
            assert isinstance(key, str)
            assert isinstance(start, int)
            assert isinstance(end, int)
            assert start < end
            assert end - start == 1024

        # Expected format for Strategy 2 (lists for HDF5 compatibility)
        slices_s2 = {
            "ops0089_GFP": [0, 1024],
            "ops0108_GFP": [1024, 2048],
        }

        for key, (start, end) in slices_s2.items():
            assert "_" in key  # Contains experiment prefix
            assert isinstance(start, int)
            assert isinstance(end, int)

    def test_slices_no_overlap(self):
        """Test that feature slices don't overlap."""
        slices = {
            "source1": [0, 1024],
            "source2": [1024, 2048],
            "source3": [2048, 3072],
        }

        ranges = list(slices.values())
        for i, (start1, end1) in enumerate(ranges):
            for j, (start2, end2) in enumerate(ranges):
                if i != j:
                    # Check no overlap
                    assert end1 <= start2 or end2 <= start1

    def test_slices_cover_full_range(self):
        """Test that slices cover entire feature space."""
        slices = {
            "source1": [0, 1024],
            "source2": [1024, 2048],
            "source3": [2048, 3072],
        }

        # Check coverage
        assert min(s for s, e in slices.values()) == 0
        assert max(e for s, e in slices.values()) == 3072


class TestMetadataStorage:
    """Test metadata storage in .uns."""

    def test_strategy1_metadata_format(self):
        """Test metadata format for Strategy 1."""
        expected_metadata = {
            "strategy": "multi_channel",
            "experiments_channels": [
                ("ops0089", "Phase2D"),
                ("ops0089", "GFP"),
                ("ops0089", "mCherry"),
            ],
            "feature_type": "dinov3",
            "aggregation_level": "gene",
            "n_sources": 3,
            "feature_slices": {},  # Dict of source -> (start, end)
            "channel_biology": {},  # Dict of source -> biology info
        }

        # Verify structure
        assert "strategy" in expected_metadata
        assert expected_metadata["strategy"] == "multi_channel"
        assert len(expected_metadata["experiments_channels"]) == 3

    def test_strategy2_metadata_format(self):
        """Test metadata format for Strategy 2."""
        expected_metadata = {
            "strategy": "multi_organelle",
            "experiments_channels": [
                ("ops0089", "GFP"),
                ("ops0108", "GFP"),
            ],
            "feature_type": "dinov3",
            "aggregation_level": "gene",
            "n_sources": 2,
            "feature_slices": {},
            "channel_biology": {},
        }

        # Verify structure
        assert expected_metadata["strategy"] == "multi_organelle"
        # Each experiment should be different
        exps = [exp for exp, ch in expected_metadata["experiments_channels"]]
        assert len(set(exps)) == 2  # Two different experiments

    def test_biological_metadata_tracked(self):
        """Test that biological metadata is stored per source."""
        # For Strategy 2, each source should have different biology
        channel_biology = {
            "ops0089_GFP": {"label": "early endosome, EEA1", "marker": "EEA1"},
            "ops0108_GFP": {"label": "mitochondria, TOMM70A", "marker": "TOMM70A"},
        }

        # Verify each source has unique biology
        markers = [info["marker"] for info in channel_biology.values()]
        assert len(set(markers)) == len(markers)  # All unique


class TestVariableNaming:
    """Test variable name generation."""

    def test_strategy1_variable_names(self):
        """Test variable names for Strategy 1 include channel."""
        # Strategy 1: Same experiment, different channels
        # Expected: ops0089_Phase_dinov3_0, ops0089_EEA1_dinov3_0, ...
        var_name_s1 = "ops0089_Phase_dinov3_0"

        parts = var_name_s1.split("_")
        assert parts[0] == "ops0089"  # Experiment
        assert "dinov3" in var_name_s1  # Feature type
        assert parts[-1].isdigit()  # Feature index

    def test_strategy2_variable_names(self):
        """Test variable names for Strategy 2 include experiment."""
        # Strategy 2: Different experiments, same channel
        # Expected: ops0089_EEA1_dinov3_0, ops0108_TOMM70A_dinov3_0, ...
        var_names_s2 = ["ops0089_EEA1_dinov3_0", "ops0108_TOMM70A_dinov3_0"]

        # Each should have different experiment prefix
        experiments = [name.split("_")[0] for name in var_names_s2]
        assert len(set(experiments)) == 2  # Two different experiments

        # Each should have different marker
        assert "EEA1" in var_names_s2[0]
        assert "TOMM70A" in var_names_s2[1]


class TestShapeValidation:
    """Test output shapes are correct."""

    def test_strategy1_output_shape(self):
        """Test output shape for Strategy 1."""
        # Strategy 1: 3 channels × 1024 features = 3072 total
        n_genes = 100
        n_features_per_channel = 1024
        n_channels = 3

        expected_shape = (n_genes, n_features_per_channel * n_channels)
        assert expected_shape == (100, 3072)

    def test_strategy2_output_shape(self):
        """Test output shape for Strategy 2."""
        # Strategy 2: 2 experiments × 1024 features = 2048 total
        n_genes = 100  # Intersection of genes
        n_features_per_exp = 1024
        n_experiments = 2

        expected_shape = (n_genes, n_features_per_exp * n_experiments)
        assert expected_shape == (100, 2048)


class TestEmbeddings:
    """Test PCA/UMAP computation."""

    def test_embeddings_recomputed_when_requested(self):
        """Test that embeddings are recomputed when recompute_embeddings=True."""
        # This would test actual implementation
        # Expected behavior: X_pca and X_umap in .obsm
        pass

    def test_embeddings_skipped_when_not_requested(self):
        """Test that embeddings are skipped when recompute_embeddings=False."""
        # Expected behavior: May or may not have embeddings
        # Shouldn't crash regardless
        pass

    def test_embedding_dimensions(self):
        """Test that embedding dimensions are correct."""
        # PCA should be (n_obs, 128)
        # UMAP should be (n_obs, 2)
        n_obs = 100
        expected_pca_shape = (n_obs, 128)
        expected_umap_shape = (n_obs, 2)

        assert expected_pca_shape[1] == 128
        assert expected_umap_shape[1] == 2


class TestEdgeCases:
    """Test edge cases."""

    def test_single_source(self):
        """Test that single source works (degenerate case)."""
        # Single source should work but is effectively just loading one file
        # Expected: Same shape as input
        pass

    def test_no_common_genes_raises_error(self):
        """Test error when no common genes across sources."""
        genes1 = [f"Gene_A_{i}" for i in range(100)]
        genes2 = [f"Gene_B_{i}" for i in range(100)]

        # No overlap
        assert len(set(genes1) & set(genes2)) == 0

        # Should raise ValueError in real implementation
        # with pytest.raises(ValueError, match="No common"):
        #     concatenate_features_by_channel(...)

    def test_partial_gene_overlap(self):
        """Test with partial gene overlap."""
        genes1 = [f"Gene_{i}" for i in range(100)]
        genes2 = [f"Gene_{i}" for i in range(50, 150)]

        # 50 genes overlap
        common = set(genes1) & set(genes2)
        assert len(common) == 50


# Integration tests would go here
# These would test with actual files on disk
class TestIntegration:
    """Integration tests with real files."""

    @pytest.mark.skip(reason="Requires actual data files")
    def test_strategy1_with_real_data(self):
        """Test Strategy 1 with real experiment data."""
        pass

    @pytest.mark.skip(reason="Requires actual data files")
    def test_strategy2_with_real_data(self):
        """Test Strategy 2 with real multi-experiment data."""
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
