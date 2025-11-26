import warnings

# Filter anndata zarr deprecation warning BEFORE importing anndata
warnings.filterwarnings("ignore", message=".*zarr v2.*", category=DeprecationWarning)

import numpy as np
import anndata as ad
import pytest

from ops_model.data.embeddings.embeddding_metrics import (
    alignment_and_uniformity,
    mean_similarity,
)


@pytest.fixture(scope="module")
def constant_adata():
    n_cells = 5000
    n_features = 50

    # All features are constant (zeros)
    X = np.repeat(np.arange(n_features).reshape(1, -1), n_cells, axis=0).astype(float)

    # Create observations metadata
    obs = {
        "label_str": ["gene_A"] * (n_cells // 2) + ["gene_B"] * (n_cells // 2),
        "label_int": [0] * (n_cells // 2) + [1] * (n_cells // 2),
    }

    # Create AnnData object
    adata = ad.AnnData(X=X, obs=obs)
    return adata


def test_mean_similarity(constant_adata):
    mean_sim, std_sim = mean_similarity(constant_adata, n_samples=1000, batch_size=100)
    assert np.isclose(
        mean_sim, 1.0
    ), f"Expected mean similarity close to 1.0, got {mean_sim}"
    assert np.isclose(
        std_sim, 0.0
    ), f"Expected std similarity close to 0.0, got {std_sim}"


def test_alignment_uniformity(constant_adata):
    return
