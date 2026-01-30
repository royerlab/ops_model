from typing import Optional
from pathlib import Path

import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt

from ops_model.data.paths import OpsPaths


def plot_pca(
    adata: ad.AnnData,
    save_path: Optional[str] = None,
    description: Optional[str] = None,
    report_dir: Optional[str] = None,
    filename: str = "pca_variance_ratio.png",
):
    """
    Plot PCA variance ratio.

    Args:
        adata: AnnData object with PCA results
        save_path: Legacy parameter - direct path to save file
        description: Plot title/description
        report_dir: Path to report directory (preferred over save_path)
        filename: Filename to use when saving to report_dir
    """
    sc.pl.pca_variance_ratio(adata, n_pcs=100, log=False, save=False, show=False)
    if description:
        plt.title(description)

    # Determine save path
    if report_dir is not None:
        save_path = Path(report_dir) / "plots" / filename

    if save_path is not None:
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


if __name__ == "__main__":
    pass
