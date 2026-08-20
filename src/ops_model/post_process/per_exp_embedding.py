"""Per-experiment embedding post-processing.

Runs the pca_optimization aggregation machinery for a SINGLE experiment, one
marker (reporter) at a time, reusing ``aggregate_channels`` so all of its rich
outputs are produced (UMAP / PHATE overlays + interactive HTMLs, mAP
consistency / distinctiveness bars, sweep plots, coord CSVs, gene/guide h5ads).
On top of that it adds a gene x gene correlation heatmap (PNG / SVG / interactive
HTML) and records every post-processing decision in ``decisions.yaml``.

There is no cross-experiment correction and no second-pass PCA — those only
matter when combining multiple experiments / markers; here each marker of the
single experiment is embedded on its own.

All outputs land in the existing CellDINO dir:
``<experiment>/3-assembly/cell_dino_features_v2/embeddings/<marker>/``.
"""

from __future__ import annotations

import glob
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import yaml


# --- Post-processing decisions ------------------------------------------------
# Captured in embeddings/<marker>/decisions.yaml instead of being encoded in
# directory names (the multi-exp pipeline uses a nested path-per-decision layout).
@dataclass
class EmbeddingDecisions:
    distance: str = "cosine"            # cosine | euclidean (mAP / consistency scoring)
    pca_variance: float = 0.80          # fraction of variance kept by the correlation-heatmap PCA
    norm_method: str = "ntc"            # ntc | global
    zscore_per_experiment: bool = True
    agg_method: str = "mean"            # cells->guides / guides->genes reduction: mean | median
    umap_type: str = "max"
    random_seed: int = 42
    # populated per marker at runtime
    marker: str | None = None
    n_genes: int | None = None
    n_pcs: int | None = None
    source_gene_h5ad: str | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EmbeddingDecisions":
        data = yaml.safe_load(Path(path).read_text()) or {}
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


def _pca_to_variance(X: np.ndarray, variance: float, seed: int) -> tuple[np.ndarray, int]:
    """PCA keeping enough components to explain ``variance`` fraction.

    sklearn accepts a float n_components as a variance target directly.
    """
    from sklearn.decomposition import PCA

    pca = PCA(n_components=variance, svd_solver="full", random_state=seed)
    X_pca = pca.fit_transform(X)
    return X_pca, int(pca.n_components_)


def _correlation_heatmap(X_ops: np.ndarray, labels: list[str], out_stem: Path, marker: str):
    """gene x gene correlation of mean-centered embeddings -> PNG + SVG + HTML.

    This is the "image half" of the paper joint heatmap:
        X_ops_c  = X_ops - X_ops.mean(axis=0, keepdims=True)
        corr_ops = np.corrcoef(X_ops_c)
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["pdf.fonttype"] = 42  # editable text in vector output

    X_ops_c = X_ops - X_ops.mean(axis=0, keepdims=True)
    corr_ops = np.corrcoef(X_ops_c)

    # static PNG + SVG
    n = corr_ops.shape[0]
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(corr_ops, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_title(f"{marker}: gene x gene embedding correlation (n={n})")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    fig.tight_layout()
    fig.savefig(f"{out_stem}.png", dpi=200)
    fig.savefig(f"{out_stem}.svg")
    plt.close(fig)

    # interactive HTML (hover shows the gene pair + r)
    import plotly.graph_objects as go

    fig = go.Figure(
        go.Heatmap(
            z=corr_ops, x=labels, y=labels, zmin=-1, zmax=1, colorscale="RdBu_r",
            colorbar=dict(title="Pearson r"),
        )
    )
    fig.update_layout(
        title=f"{marker}: gene x gene embedding correlation (n={n})",
        width=900, height=900, xaxis_showticklabels=False, yaxis_showticklabels=False,
    )
    fig.write_html(f"{out_stem}.html", include_plotlyjs="cdn")

    return corr_ops


def run_marker(
    gene_h5ad: Path,
    guide_h5ad: Path,
    out_dir: Path,
    decisions: EmbeddingDecisions,
) -> dict:
    """Post-process a single marker: rich aggregate_channels outputs + corr heatmap.

    ``gene_h5ad`` / ``guide_h5ad`` are the CellDINO combine's gene_bulked /
    guide_bulked h5ads for this marker.
    """
    import anndata as ad

    from ops_model.post_process.combination.pca_optimization.phase2 import (
        aggregate_channels,
    )

    marker = decisions.marker
    out_dir.mkdir(parents=True, exist_ok=True)

    # aggregate_channels reads <out_dir>/per_channel/<prefix>_{guide,gene}.h5ad.
    # CellDINO writes {guide,gene}_bulked_<marker>.h5ad, so bridge the filename with
    # a hardlink (same inode: no data copy, no symlink / broken-link risk); fall
    # back to a copy only across filesystems.
    import os
    import shutil

    per_channel = out_dir / "per_channel"
    per_channel.mkdir(parents=True, exist_ok=True)
    for src, name in ((guide_h5ad, f"{marker}_guide.h5ad"), (gene_h5ad, f"{marker}_gene.h5ad")):
        dst = per_channel / name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        try:
            os.link(Path(src).resolve(), dst)
        except OSError:
            shutil.copy2(Path(src).resolve(), dst)

    # Reuse the full aggregation + plotting machinery (UMAP/PHATE/mAP/sweep/HTML).
    agg_result = aggregate_channels(
        output_dir=str(out_dir),
        norm_method=decisions.norm_method,
        per_unit_subdir="per_channel",
        distance=decisions.distance,
        random_seed=decisions.random_seed,
        agg_method=decisions.agg_method,
        umap_type=decisions.umap_type,
    )

    # Correlation heatmap on the gene-level embedding (mean-centered), PCA-reduced
    # to the configured variance fraction.
    gene = ad.read_h5ad(gene_h5ad)
    X = np.asarray(gene.X)
    if decisions.zscore_per_experiment:
        X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-8)
    X_ops, n_pcs = _pca_to_variance(X, decisions.pca_variance, decisions.random_seed)
    pert_col = "perturbation" if "perturbation" in gene.obs.columns else gene.obs.columns[0]
    labels = gene.obs[pert_col].astype(str).tolist()

    decisions.n_genes = int(X_ops.shape[0])
    decisions.n_pcs = n_pcs
    decisions.source_gene_h5ad = str(gene_h5ad)

    _correlation_heatmap(X_ops, labels, out_dir / "corr_heatmap", marker)

    (out_dir / "decisions.yaml").write_text(
        yaml.safe_dump(asdict(decisions), sort_keys=False)
    )
    return {"marker": marker, "aggregate": agg_result, "n_genes": decisions.n_genes, "n_pcs": n_pcs}


def run_per_exp_embeddings(
    experiment: str,
    feature_dir: str | Path | None = None,
    embeddings_dir: str | Path | None = None,
    decisions_yaml: str | Path | None = None,
    **decision_overrides,
) -> list[dict]:
    """Post-process every marker of one experiment's CellDINO embeddings.

    Discovers gene_bulked_<marker>.h5ad in the CellDINO anndata_objects dir and
    runs :func:`run_marker` for each, writing to <embeddings_dir>/<marker>/.
    """
    from ops_utils.data.experiment import OpsDataset

    ds = OpsDataset(experiment)
    celldino_dir = ds.results / "cell_dino_features_v2"
    if feature_dir is None:
        feature_dir = celldino_dir / "anndata_objects"
    feature_dir = Path(feature_dir)
    if embeddings_dir is None:
        # Keep outputs inside the existing CellDINO dir (no separate tree).
        embeddings_dir = celldino_dir / "embeddings"
    embeddings_dir = Path(embeddings_dir)

    base = EmbeddingDecisions.from_yaml(decisions_yaml) if decisions_yaml else EmbeddingDecisions()
    for k, v in decision_overrides.items():
        if hasattr(base, k) and v is not None:
            setattr(base, k, v)

    gene_files = sorted(glob.glob(str(feature_dir / "gene_bulked_*.h5ad")))
    if not gene_files:
        raise FileNotFoundError(
            f"No gene_bulked_*.h5ad in {feature_dir}. Run celldino_inference "
            f"(extraction + combine) before embedding post-processing."
        )

    results = []
    for gf in gene_files:
        marker = Path(gf).stem.replace("gene_bulked_", "")
        guide = feature_dir / f"guide_bulked_{marker}.h5ad"
        if not guide.exists():
            raise FileNotFoundError(f"Missing guide_bulked for {marker}: {guide}")
        dec = EmbeddingDecisions(**{**asdict(base), "marker": marker})
        results.append(run_marker(Path(gf), guide, embeddings_dir / marker, dec))
    return results
