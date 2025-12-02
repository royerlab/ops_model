from typing import Optional

import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt

from ops_model.data.paths import OpsPaths


def plot_pca(
    adata: ad.AnnData,
    save_path: str,
    description: Optional[str] = None,
):
    sc.pl.pca_variance_ratio(adata, n_pcs=100, log=False, save=False, show=False)
    if description:
        plt.title(description)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    return


def pca_90_percent_variance(adata: ad.AnnData) -> int:
    cumulative_variance = adata.uns["pca"]["variance_ratio"].cumsum()
    if cumulative_variance[-1] < 0.9:
        return (
            cumulative_variance[99],
            f"First 100 PCs explain {cumulative_variance[-1]:.2f} variance",
        )
    else:
        num_components = (cumulative_variance < 0.9).sum() + 1
        return num_components, f"First {num_components} PCs explain 90% variance"


def generate_pca_section(
    experiment: str,
):
    path = OpsPaths(experiment).cell_profiler_out
    save_dir = path.parent / "anndata_objects"
    assert save_dir.exists(), f"Anndata objects directory does not exist: {save_dir}"
    checkpoint_path = save_dir / "features_processed.h5ad"
    adata = ad.read_h5ad(checkpoint_path)
    assert "X_pca" in adata.obsm, "PCA embeddings not found in AnnData object."
    plots_dir = OpsPaths(experiment).embedding_plot_dir
    plots_dir.mkdir(parents=True, exist_ok=True)

    cum_var, descr = pca_90_percent_variance(adata)

    plot_pca(
        adata=adata,
        save_path=plots_dir / "pca_variance_ratio.png",
        description=descr,
    )

    return


if __name__ == "__main__":
    pass
