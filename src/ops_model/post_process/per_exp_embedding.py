"""Per-experiment embedding post-processing.

Runs the pca_optimization aggregation machinery for one experiment (or a pool of
experiments), one marker (reporter) at a time, reusing ``aggregate_channels`` so all of its rich
outputs are produced (UMAP / PHATE overlays + interactive HTMLs, mAP
consistency / distinctiveness bars, sweep plots, coord CSVs, gene/guide h5ads).
On top of that it adds a gene x gene correlation heatmap (PNG / SVG / interactive
HTML + a downloadable CSV of the correlation values) and records every
post-processing decision in ``decisions.yaml``.

There is no second-pass PCA — that only matters when combining many markers.

A run can also pool **several experiments** and/or restrict to a **subset of
wells**. Because the CellDINO ``{guide,gene}_bulked`` h5ads carry no well
information, any such run re-bulks from the cell-level
``features_processed_<marker>.h5ad``: the selected wells of every experiment are
pooled, cell features are z-scored per experiment when more than one experiment
is in the pool (the same normalization the multi-exp pooled-signal path applies
before PCA), and the guide/gene levels are re-aggregated. The experiments, the
wells kept and the wells dropped are recorded in ``decisions.yaml`` alongside the
post-processing decisions.

Outputs land in the CellDINO dir of the experiment
(``<experiment>/3-assembly/cell_dino_features_v2/embeddings/<marker>/``);
multi-experiment runs go to
``/hpc/projects/icd.fast.ops/analysis/embeddings/<experiments>/<marker>/``.
"""

from __future__ import annotations

import glob
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import yaml

# Multi-experiment runs don't belong to any single experiment's output tree.
COMBINED_EMBEDDINGS_ROOT = Path("/hpc/projects/icd.fast.ops/analysis/embeddings")


# --- Post-processing decisions ------------------------------------------------
# Captured in embeddings/<marker>/decisions.yaml instead of being encoded in
# directory names (the multi-exp pipeline uses a nested path-per-decision layout).
@dataclass
class EmbeddingDecisions:
    distance: str = "cosine"            # cosine | euclidean (mAP / consistency scoring)
    pca_variance: float = 0.80          # fraction of variance kept by the correlation-heatmap PCA
    zscore_heatmap_features: bool = True  # z-score gene-level features before that PCA
    norm_method: str = "ntc"            # ntc | global
    # Standardize cell features within each experiment before pooling; only has an
    # effect when >1 experiment is pooled (a single experiment is left as-is).
    zscore_per_experiment: bool = True
    agg_method: str = "mean"            # cells->guides / guides->genes reduction: mean | median
    umap_type: str = "max"
    # Leiden clustering resolutions for the overlays + GO enrichment. The full
    # multi-exp default is ~13 resolutions; per-exp we only need a few (GO
    # enrichment is the long pole and scales with resolution count).
    leiden_resolutions: list = field(default_factory=lambda: [4.0, 10.0, 30.0])
    random_seed: int = 42
    # populated per marker at runtime
    marker: str | None = None
    n_genes: int | None = None
    n_pcs: int | None = None
    source_gene_h5ad: str | None = None
    # provenance: what went into this embedding
    experiments: list | None = None
    wells_included: dict | None = None      # {experiment: [well, ...]}
    wells_excluded: dict | None = None      # {experiment: [well, ...]} (present but dropped)
    n_cells: int | None = None
    source_cell_h5ads: list | None = None

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


def _split_terms(raw):
    """Annotation cells hold Python-list strings ("['A', 'B']"), plain text, or empty.

    Parsing them as ';'-delimited text yielded terms like "[]" and one long list as a
    single term, which made every enrichment meaningless.
    """
    import ast as _ast

    if raw is None or not isinstance(raw, str) or not raw.strip():
        return set()
    t = raw.strip()
    if t.startswith("[") and t.endswith("]"):
        try:
            vals = _ast.literal_eval(t)
            return {str(v).strip() for v in vals if str(v).strip()}
        except Exception:
            t = t.strip("[]")
    return {x.strip().strip("'\"") for x in t.replace("|", ";").split(";") if x.strip().strip("'\"")}


def cross_pass_retention_report(
    X_a: np.ndarray, X_b: np.ndarray, labels: list, out_stem,
    names: tuple = ("live", "fixed"), low_quantile: float = 0.10,
    panel_csv: str = "/hpc/projects/icd.fast.ops/configs/annotated_gene_panel_July2025.csv",
    n_label: int = 25, active_genes=None, strength=None,
):
    """Which genes lose their phenotype between two passes, and what biology that is.

    A heatmap shows THAT structure degrades; this answers WHAT degrades. Per gene we
    score "retention" = correlation between its row of the pass-A gene x gene matrix
    and the same row in pass B. A gene whose relational context survives scores high;
    one whose neighbours change scores low. Then the low-retention tail is tested for
    enrichment against the annotated gene panel (CORUM complexes, GO / REACTOME
    pathways, curated category) so the answer is named biology rather than a picture.

    Outputs: ranked retention CSV, enrichment CSV, and a two-panel figure
    (retention distribution with the tail marked + labelled worst genes; enriched
    terms among them).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from scipy.stats import fisher_exact

    Path(out_stem).parent.mkdir(parents=True, exist_ok=True)

    def _corr(X):
        return np.corrcoef(X - X.mean(axis=0, keepdims=True))

    Ca, Cb = _corr(X_a), _corr(X_b)
    n = len(labels)
    off = ~np.eye(n, dtype=bool)
    ret = np.array([np.corrcoef(Ca[i][off[i]], Cb[i][off[i]])[0, 1] for i in range(n)])

    df = pd.DataFrame({"gene": [str(l) for l in labels], "retention": ret})
    if active_genes is not None:
        # Only genes with a real phenotype in pass A can meaningfully "lose" it.
        # Without this the low-retention tail is dominated by negative controls,
        # whose correlation profile is noise and so never reproduces.
        keep = {str(g) for g in active_genes}
        df["phenotype_in_a"] = df["gene"].isin(keep)
    else:
        df["phenotype_in_a"] = True
    df = df.sort_values("retention")
    scored = df[df["phenotype_in_a"]]
    cut = float(np.nanquantile(scored["retention"], low_quantile))
    df["low_retention"] = df["phenotype_in_a"] & (df["retention"] <= cut)
    df.to_csv(f"{out_stem}_retention.csv", index=False)

    # ---- enrichment of the low-retention tail against the annotated panel ----
    term_cols = {
        "CORUM complex": "In_corum_complexes",
        "same-complex partner": "In_same_complex_with",
        "REACTOME pathway": "In_REACT_pathways",
        "GO pathway": "In_go_pathways",
        "category": "Gene_Category",
    }
    enr = []
    try:
        panel = pd.read_csv(panel_csv, low_memory=False)
        panel = panel[panel["Gene.name"].notna()]
        low = set(df.loc[df.low_retention, "gene"])
        rest = set(df.loc[df["phenotype_in_a"] & ~df["low_retention"], "gene"])
        for kind, col in term_cols.items():
            if col not in panel.columns:
                continue
            gene_terms = {}
            for g, raw in zip(panel["Gene.name"], panel[col]):
                for t in _split_terms(raw):
                    gene_terms.setdefault(t, set()).add(str(g))
            for term, genes in gene_terms.items():
                a = len(genes & low)
                if a < 3:                      # too small to be interesting
                    continue
                b = len(low) - a
                c = len(genes & rest)
                d = len(rest) - c
                if min(a + c, b + d) == 0:
                    continue
                odds, p = fisher_exact([[a, b], [c, d]], alternative="greater")
                enr.append({"kind": kind, "term": term, "n_low": a,
                            "n_other": c, "odds_ratio": float(odds), "p": float(p)})
    except Exception as exc:
        print(f"  [enrichment skipped] {exc}")

    edf = pd.DataFrame(enr)
    if len(edf):
        edf = edf.sort_values("p")
        m = len(edf)
        edf["q_bh"] = (edf["p"] * m / np.arange(1, m + 1)).cummin().clip(upper=1.0)
        edf.to_csv(f"{out_stem}_enrichment.csv", index=False)

    # ---- figure ----
    fig, axes = plt.subplots(1, 2, figsize=(20, 8),
                             gridspec_kw={"width_ratios": [1.0, 1.25]})
    ax = axes[0]
    ax.hist(ret[~np.isnan(ret)], bins=60, color="#4C72B0", alpha=0.85, edgecolor="black", lw=0.4)
    ax.axvline(cut, color="#C44E52", lw=2.5, ls="--",
               label=f"low-retention cut (q={low_quantile:g}) = {cut:.3f}")
    ax.set_xlabel(f"phenotype retention  ({names[0]} vs {names[1]} correlation profile)", fontsize=15)
    ax.set_ylabel("genes", fontsize=15)
    ax.set_title(f"Per-gene retention (n={n})\nmedian={np.nanmedian(ret):.3f}", fontsize=17)
    ax.legend(fontsize=12); ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=13)

    # Gene-level view: a strong live phenotype that is NOT retained is the
    # interesting case. Annotation-space enrichment is kept as a CSV but is
    # underpowered at this n, so the plot names individual genes instead.
    ax = axes[1]
    if strength is not None:
        sdf = df.merge(pd.DataFrame({"gene": [str(k) for k in strength],
                                     "strength": list(strength.values())}), on="gene", how="inner")
        sdf = sdf.dropna(subset=["strength", "retention"])
        ax.scatter(sdf["strength"], sdf["retention"], s=18, c="#9aa4b1",
                   alpha=0.55, linewidths=0, label="gene")
        hi = sdf["strength"] >= sdf["strength"].quantile(0.75)
        lo = sdf["retention"] <= cut
        pick = sdf[hi & lo].sort_values("retention")
        ax.scatter(pick["strength"], pick["retention"], s=52, c="#C44E52",
                   edgecolors="black", linewidths=0.6, zorder=4,
                   label=f"strong phenotype, poor retention (n={len(pick)})")
        ax.axhline(cut, color="#C44E52", ls="--", lw=1.8, alpha=0.8)
        ax.axvline(float(sdf["strength"].quantile(0.75)), color="grey", ls=":", lw=1.5)
        for _, r in pick.head(n_label).iterrows():
            ax.annotate(r["gene"], (r["strength"], r["retention"]), fontsize=11,
                        xytext=(4, 3), textcoords="offset points")
        ax.set_xlabel(f"phenotype strength in {names[0]}  (activity mAP)", fontsize=15)
        ax.set_ylabel("phenotype retention", fontsize=15)
        ax.set_title("Genes whose real phenotype did not survive", fontsize=17)
        ax.legend(fontsize=11, loc="lower right")
        ax.tick_params(labelsize=13)
        pick.to_csv(f"{out_stem}_strong_but_lost.csv", index=False)
    else:
        worst = df[df["phenotype_in_a"]].head(n_label).iloc[::-1]
        y = np.arange(len(worst))
        ax.barh(y, worst["retention"], color="#C44E52", alpha=0.85, edgecolor="black", lw=0.5)
        ax.set_yticks(y); ax.set_yticklabels(worst["gene"], fontsize=11)
        ax.set_xlabel("phenotype retention", fontsize=15)
        ax.set_title("Lowest-retention genes", fontsize=17)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"{names[0]} vs {names[1]}: phenotype retention and its biology", fontsize=20)
    fig.tight_layout()
    fig.savefig(f"{out_stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(f"{out_stem}.svg", bbox_inches="tight")
    plt.close(fig)
    return df, edf


def cross_pass_correlation_heatmap(
    X_a: np.ndarray, X_b: np.ndarray, labels: list, out_stem,
    names: tuple = ("live", "fixed"), n_clusters: int = 12, **kwargs
):
    """Compare gene x gene correlation structure between two imaging passes.

    Three panels sharing ONE gene ordering (hierarchical clustering of pass A), so the
    same block appears in the same place in all three:

        A            gene x gene correlation in pass A
        B            the same genes in pass B
        A - B        where the passes agree (~0) and disagree (signed)

    Ordering matters more than it sounds: unordered, both matrices look like noise
    because related genes are scattered. Clustering on A and reusing that order is
    what makes a block that survives in B visible as agreement, and one that
    dissolves visible as disagreement.

    Correlation is computed in the raw shared feature space -- both passes are
    embedded by the same model, so the axes correspond. (PCA fitted per pass gives
    arbitrarily rotated axes and makes the comparison meaningless.)

    Returns (corr_a, corr_b, order, cluster_ids) in the clustered order.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
    from scipy.spatial.distance import squareform

    plt.rcParams["pdf.fonttype"] = 42
    Path(out_stem).parent.mkdir(parents=True, exist_ok=True)

    def _corr(X):
        return np.corrcoef(X - X.mean(axis=0, keepdims=True))

    Ca, Cb = _corr(X_a), _corr(X_b)

    # Cluster on pass A; 1-r as distance, averaged to enforce exact symmetry.
    d = 1.0 - Ca
    np.fill_diagonal(d, 0.0)
    Z = linkage(squareform((d + d.T) / 2.0, checks=False), method="average")
    order = leaves_list(Z)
    clusters = fcluster(Z, t=n_clusters, criterion="maxclust")

    Ca_o, Cb_o = Ca[np.ix_(order, order)], Cb[np.ix_(order, order)]
    diff = Ca_o - Cb_o
    lab_o = [str(labels[i]) for i in order]

    pd.DataFrame(Ca_o, index=lab_o, columns=lab_o).to_csv(f"{out_stem}_{names[0]}.csv")
    pd.DataFrame(Cb_o, index=lab_o, columns=lab_o).to_csv(f"{out_stem}_{names[1]}.csv")
    pd.DataFrame(diff, index=lab_o, columns=lab_o).to_csv(f"{out_stem}_difference.csv")
    pd.DataFrame({"gene": lab_o, "cluster": clusters[order]}).to_csv(
        f"{out_stem}_clusters.csv", index=False)

    # Symmetric limits from the data so faint structure is visible rather than
    # swamped by a hard -1..1 scale.
    lim = float(np.percentile(np.abs(np.concatenate([Ca_o.ravel(), Cb_o.ravel()])), 99))
    dlim = float(np.percentile(np.abs(diff.ravel()), 99))
    n = len(lab_o)

    fig, axes = plt.subplots(1, 3, figsize=(26, 9))
    # optional mAP strips above each panel, in the same clustered order, so a block of
    # structure can be read against how much signal those genes actually carry
    strips = kwargs.get("map_scores") or {}
    for ax, (M, title, v) in zip(axes, (
        (Ca_o, f"{names[0]}", lim), (Cb_o, f"{names[1]}", lim),
        (diff, f"{names[0]} - {names[1]}", dlim),
    )):
        im = ax.imshow(M, cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest")
        ax.set_title(f"{title}\n(n={n} genes, clustered on {names[0]})", fontsize=18)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    # Cluster boundaries, so blocks are readable as blocks
    bounds = np.where(np.diff(clusters[order]) != 0)[0] + 0.5
    for ax in axes:
        for b in bounds:
            ax.axhline(b, color="k", lw=0.6, alpha=0.5)
            ax.axvline(b, color="k", lw=0.6, alpha=0.5)
    if strips:
        for ax in axes:
            div = ax.inset_axes([0, 1.015, 1, 0.045])
            vals = np.array([[strips.get(g, np.nan) for g in lab_o]], dtype=float)
            div.imshow(vals, aspect="auto", cmap="viridis")
            div.set_xticks([]); div.set_yticks([])
            div.set_ylabel("mAP", rotation=0, ha="right", va="center", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{out_stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(f"{out_stem}.svg", bbox_inches="tight")
    plt.close(fig)
    return Ca_o, Cb_o, order, clusters[order]


def _correlation_heatmap(X_ops: np.ndarray, labels: list[str], out_stem: Path, marker: str,
                         X_other: "np.ndarray | None" = None,
                         axis_labels: "tuple[str, str] | None" = None,
                         zlim: "float | None" = None):
    """gene x gene correlation of mean-centered embeddings -> PNG + SVG + HTML.

    This is the "image half" of the paper joint heatmap:
        X_ops_c  = X_ops - X_ops.mean(axis=0, keepdims=True)
        corr_ops = np.corrcoef(X_ops_c)

    With ``X_other`` the result is instead the CROSS-correlation between two embeddings
    of the same genes (rows = X_ops, columns = X_other) -- e.g. a live pass against a
    joined reimage pass. That matrix is not symmetric, and its diagonal is the
    same-gene agreement between the two. ``axis_labels`` names the two axes.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["pdf.fonttype"] = 42  # editable text in vector output

    Path(out_stem).parent.mkdir(parents=True, exist_ok=True)

    X_ops_c = X_ops - X_ops.mean(axis=0, keepdims=True)
    if X_other is None:
        corr_ops = np.corrcoef(X_ops_c)
    else:
        # Cross-correlation: stack both, correlate, keep the off-diagonal block so
        # rows stay X_ops and columns X_other.
        other_c = X_other - X_other.mean(axis=0, keepdims=True)
        n_a = X_ops_c.shape[0]
        corr_ops = np.corrcoef(np.vstack([X_ops_c, other_c]))[:n_a, n_a:]

    # Downloadable values: gene x gene correlation matrix as CSV (labelled).
    import pandas as pd

    pd.DataFrame(corr_ops, index=labels, columns=labels).to_csv(f"{out_stem}.csv")

    # static PNG + SVG
    n = corr_ops.shape[0]
    lim = float(zlim) if zlim else max(
        float(np.nanpercentile(np.abs(corr_ops[~np.eye(*corr_ops.shape, dtype=bool)]), 99)),
        1e-3)
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(corr_ops, cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
    ax.set_title(f"{marker}: gene x gene embedding correlation (n={n})")
    if axis_labels:
        ax.set_ylabel(axis_labels[0]); ax.set_xlabel(axis_labels[1])
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    fig.tight_layout()
    fig.savefig(f"{out_stem}.png", dpi=200)
    fig.savefig(f"{out_stem}.svg")
    plt.close(fig)

    # interactive HTML (hover shows the gene pair + r)
    import plotly.graph_objects as go

    ylab, xlab = axis_labels if axis_labels else ("gene", "gene")
    # A 1001x1001 matrix embeds ~1e6 floats and yields a ~13MB page. Block-average to
    # a viewable resolution and round: the structure is what is being read, not
    # individual cells, and the full matrix is already written as CSV alongside.
    z_html, x_html, y_html = corr_ops, labels, labels
    _MAXPX = 420
    if corr_ops.shape[0] > _MAXPX:
        k = int(np.ceil(corr_ops.shape[0] / _MAXPX))
        m = (corr_ops.shape[0] // k) * k
        z_html = corr_ops[:m, :m].reshape(m // k, k, m // k, k).mean(axis=(1, 3))
        x_html = [labels[i] for i in range(0, m, k)]
        y_html = x_html
    z_html = np.round(z_html, 3)
    fig = go.Figure(
        go.Heatmap(
            z=z_html, x=x_html, y=y_html, zmin=-lim, zmax=lim, colorscale="RdBu_r",
            colorbar=dict(title="Pearson r"),
            hovertemplate=(f"{ylab}: %{{y}}<br>{xlab}: %{{x}}"
                           "<br>r: %{z:.3f}<extra></extra>"),
        )
    )
    fig.update_layout(
        title=dict(text=(f"{marker}: gene x gene embedding correlation "
                         f"(n={n}, scale +-{lim:.2f}"
                         + (f", binned to {z_html.shape[0]}px" if z_html.shape[0] < n else "")
                         + ")"), font=dict(size=19)),
        width=760, height=720, font=dict(size=15),
        margin=dict(l=90, r=20, t=80, b=80),
        xaxis=dict(title=dict(text=xlab, font=dict(size=17)), showticklabels=False),
        yaxis=dict(title=dict(text=ylab, font=dict(size=17)), showticklabels=False,
                   autorange="reversed"),
    )
    fig.write_html(f"{out_stem}.html", include_plotlyjs="cdn")

    return corr_ops


def _organize_plots(plots_dir: Path) -> None:
    """Group the flat aggregate_channels plot files into category subdirs so the
    output isn't a sprawling list of PNGs. Files are matched by name; anything
    unmatched and any existing subdir (leiden/, canonical_leiden/,
    marker_overlay/, ...) is left in place.
    """
    if not plots_dir.is_dir():
        return
    # (subdir, predicate) in priority order — first match wins.
    groups = [
        ("umap", lambda n: "umap" in n),
        ("phate", lambda n: "phate" in n),
        # EBI complex + binary overlays share one subdir; checked before
        # map_metrics (they contain "ebi").
        ("ebi_overlay", lambda n: "ebi_complex_overlay" in n or "ebi_binary_overlay" in n),
        ("map_metrics", lambda n: n.startswith("map_") or "violin" in n or "consistency" in n
                                   or "distinctiveness" in n or "activity" in n or "ebi" in n),
        ("sweep", lambda n: "sweep" in n),
        ("channel_qc", lambda n: "peak" in n or "per_channel" in n),
    ]
    for f in list(plots_dir.iterdir()):
        if not f.is_file():
            continue  # leave existing subdirs untouched
        n = f.name.lower()
        for sub, match in groups:
            if match(n):
                dest = plots_dir / sub
                dest.mkdir(exist_ok=True)
                f.rename(dest / f.name)
                break


# --- experiment / well selection ---------------------------------------------

def _well_key(value) -> str:
    """Canonical well key: ``A/1/0_ops0175_20260706`` | ``A/1/0`` | ``A/1`` | ``A1`` -> ``A1``."""
    token = str(value).split("_")[0]
    parts = [p for p in token.split("/") if p]
    return f"{parts[0]}{parts[1]}" if len(parts) >= 2 else token


def _well_spec(spec, experiment: str) -> set[str] | None:
    """Resolve a well spec for one experiment into a set of well keys.

    ``spec`` is either a flat list applied to every experiment, or a
    ``{experiment: [wells]}`` dict (keys may be the short id, e.g. ``ops0175``).
    """
    if not spec:
        return None
    if isinstance(spec, dict):
        for key, wells in spec.items():
            if experiment == key or experiment.startswith(f"{key}_"):
                return {_well_key(w) for w in wells}
        return None
    return {_well_key(w) for w in spec}


def _cell_h5ad(feature_dir: Path, marker: str) -> Path:
    return Path(feature_dir) / f"features_processed_{marker}.h5ad"


def _write_decisions(out_dir: Path, decisions: EmbeddingDecisions) -> Path:
    """Write decisions.yaml. Called at the start of a job (so the experiments /
    wells are on disk while it runs) and again at the end with the runtime fields."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "decisions.yaml"
    path.write_text(yaml.safe_dump(asdict(decisions), sort_keys=False))
    return path


def _well_census(cell_h5ad: Path) -> dict[str, int]:
    """``{well_key: n_cells}`` for a cell-level h5ad (obs only, X stays on disk)."""
    import anndata as ad

    obs = ad.read_h5ad(cell_h5ad, backed="r").obs
    counts = obs["well"].map(_well_key).astype(str).value_counts()
    return {str(k): int(v) for k, v in counts.items()}


def _select_wells(census: dict[str, int], include, exclude, experiment: str) -> list[str]:
    """Wells of ``experiment`` to keep, after applying the include/exclude specs."""
    keep = _well_spec(include, experiment)
    drop = _well_spec(exclude, experiment)
    present = set(census)
    if keep:
        missing = keep - present
        if missing:
            raise ValueError(
                f"{experiment}: requested wells {sorted(missing)} not in the CellDINO "
                f"features (present: {sorted(present)})"
            )
        present &= keep
    if drop:
        present -= drop
    if not present:
        raise ValueError(
            f"{experiment}: well selection left no cells "
            f"(include={keep and sorted(keep)}, exclude={drop and sorted(drop)})"
        )
    return sorted(present)


def _bulk_from_cells(
    cell_h5ads: dict,
    wells: dict,
    out_dir: Path,
    decisions: EmbeddingDecisions,
) -> tuple[Path, Path]:
    """Re-bulk one marker's guide/gene h5ads from the selected experiments/wells.

    Pools the kept wells of every experiment, per-experiment z-scores the cell
    features when more than one experiment is pooled, then aggregates to guide and
    gene level the same way the CellDINO combine does. Writes straight into
    ``<out_dir>/per_channel/`` (where ``aggregate_channels`` reads its blocks) and
    records the census on ``decisions``.
    """
    import anndata as ad

    from ops_model.features.anndata_utils import create_aggregated_embeddings

    experiments = list(cell_h5ads)
    blocks, included, excluded = [], {}, {}
    for exp in experiments:
        path = Path(cell_h5ads[exp])
        adata = ad.read_h5ad(path)
        keys = adata.obs["well"].map(_well_key).astype(str)
        keep = set(wells[exp])
        missing = keep - set(keys.unique())
        if missing:
            raise ValueError(f"{path}: wells {sorted(missing)} absent for this marker")
        mask = keys.isin(keep).values
        included[exp] = sorted(keep)
        excluded[exp] = sorted(set(keys.unique()) - keep)
        blocks.append(adata[mask].copy())
        print(
            f"[embed] {exp}: {int(mask.sum())}/{adata.n_obs} cells, wells "
            f"{included[exp]}" + (f", dropped {excluded[exp]}" if excluded[exp] else "")
        )
        del adata

    cells = (
        blocks[0]
        if len(blocks) == 1
        else ad.concat(blocks, join="inner", keys=experiments,
                       index_unique="_", uns_merge="first")
    )
    del blocks

    X = np.asarray(cells.X, dtype=np.float32)
    if len(experiments) > 1 and decisions.zscore_per_experiment:
        from sklearn.preprocessing import StandardScaler

        exp_ids = cells.obs["experiment"].astype(str).values
        for exp in np.unique(exp_ids):
            m = exp_ids == exp
            X[m] = StandardScaler().fit_transform(X[m])
        print(f"[embed] per-experiment z-score applied ({len(np.unique(exp_ids))} experiments)")
    cells.X = X

    per_channel = out_dir / "per_channel"
    per_channel.mkdir(parents=True, exist_ok=True)
    out = {}
    for level in ("guide", "gene"):
        agg = create_aggregated_embeddings(
            cells,
            level=level,
            aggregation_method=decisions.agg_method,
            # Pooling experiments needs one row per construct (aggregate_channels
            # matches blocks on the construct key); a single experiment keeps the
            # per-experiment grouping so the output matches the CellDINO combine.
            preserve_batch_info=len(experiments) == 1,
            random_seed=decisions.random_seed,
        )
        if len(experiments) > 1:
            # Pooled rows have no single source experiment.
            agg.obs["experiment"] = ",".join(experiments)
        agg.uns["experiment"] = ",".join(experiments)
        agg.uns["n_cells"] = int(cells.n_obs)
        agg.uns["n_features_raw"] = int(cells.n_vars)
        out[level] = per_channel / f"{decisions.marker}_{level}.h5ad"
        agg.write_h5ad(out[level])
        print(f"[embed] wrote {out[level]} ({agg.n_obs} x {agg.n_vars})")

    decisions.experiments = experiments
    decisions.wells_included = included
    decisions.wells_excluded = {e: w for e, w in excluded.items() if w}
    decisions.n_cells = int(cells.n_obs)
    decisions.source_cell_h5ads = [str(cell_h5ads[e]) for e in experiments]
    return out["gene"], out["guide"]


def run_marker_pooled(
    cell_h5ads: dict,
    wells: dict,
    out_dir: Path,
    decisions: EmbeddingDecisions,
) -> dict:
    """Re-bulk a marker from cell level (experiment/well selection), then post-process it."""
    decisions.experiments = list(cell_h5ads)
    decisions.wells_included = {e: list(wells[e]) for e in cell_h5ads}
    print(f"[embed] {_write_decisions(out_dir, decisions)}")
    gene_h5ad, guide_h5ad = _bulk_from_cells(cell_h5ads, wells, out_dir, decisions)
    return run_marker(gene_h5ad, guide_h5ad, out_dir, decisions)


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
    _write_decisions(out_dir, decisions)

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
        if dst.exists() and Path(src).resolve() == dst.resolve():
            continue  # re-bulked in place by _bulk_from_cells
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
        leiden_resolutions=tuple(decisions.leiden_resolutions),
    )

    # Group the flat aggregate_channels plots into category subdirs (de-sprawl).
    _organize_plots(out_dir / "plots")
    # canonical_leiden holds the GO-term-annotated cluster embeddings (top GO
    # term labelled at each cluster centroid). Move it under leiden/ as
    # go_annotated/ (clearer than the upstream "canonical" name).
    _plots = out_dir / "plots"
    _cl = _plots / "canonical_leiden"
    if _cl.is_dir():
        (_plots / "leiden").mkdir(exist_ok=True)
        _dest = _plots / "leiden" / "go_annotated"
        if _dest.exists():
            shutil.rmtree(_dest)
        shutil.move(str(_cl), str(_dest))

    # Correlation heatmap on the gene-level embedding (mean-centered), PCA-reduced
    # to the configured variance fraction.
    gene = ad.read_h5ad(gene_h5ad)
    X = np.asarray(gene.X)
    if decisions.zscore_heatmap_features:
        X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-8)
    X_ops, n_pcs = _pca_to_variance(X, decisions.pca_variance, decisions.random_seed)
    pert_col = "perturbation" if "perturbation" in gene.obs.columns else gene.obs.columns[0]
    labels = gene.obs[pert_col].astype(str).tolist()

    decisions.n_genes = int(X_ops.shape[0])
    decisions.n_pcs = n_pcs
    decisions.source_gene_h5ad = str(gene_h5ad)

    _correlation_heatmap(X_ops, labels, out_dir / "correlation_heatmap" / "corr_heatmap", marker)

    _write_decisions(out_dir, decisions)
    return {"marker": marker, "aggregate": agg_result, "n_genes": decisions.n_genes, "n_pcs": n_pcs}


def _run_name(experiments: list[str]) -> str:
    """Readable output-dir name for a pooled run, e.g. ``ops0175-ops0176``.

    Wells aren't in the name — ``decisions.yaml`` records which ones were used.
    """
    return "-".join(e.split("_")[0] for e in experiments)


def run_per_exp_embeddings(
    experiment: str | list[str],
    feature_dir: str | Path | None = None,
    embeddings_dir: str | Path | None = None,
    decisions_yaml: str | Path | None = None,
    slurm: bool = True,
    slurm_params: dict | None = None,
    experiments: list[str] | None = None,
    include_wells=None,
    exclude_wells=None,
    **decision_overrides,
) -> list[dict]:
    """Post-process every marker of one or more experiments' CellDINO embeddings.

    Fans one job per marker out to SLURM (each marker is independent), writing to
    <embeddings_dir>/<marker>/. Only discovery and job submission run locally; all
    compute (re-bulking + aggregate_channels + heatmap) runs on SLURM. Pass
    ``slurm=False`` to run in-process instead.

    Selection:
      - ``experiments``: pool several experiments into one embedding (cells are
        pooled and z-scored per experiment; markers = those present in all of them).
      - ``include_wells`` / ``exclude_wells``: a flat list of wells (``["A1", "A2"]``,
        also accepted as ``A/1`` / ``A/1/0``) applied to every experiment, or a
        ``{experiment: [wells]}`` dict.

    Any selection that actually drops cells — or any multi-experiment run —
    re-bulks guide/gene from ``features_processed_<marker>.h5ad``; otherwise the
    existing ``{guide,gene}_bulked`` h5ads are reused. Either way the experiments
    and per-experiment wells kept/dropped are written into ``decisions.yaml``.
    """
    from ops_utils.data.experiment import OpsDataset

    exps = [experiment] if isinstance(experiment, str) else list(experiment)
    if experiments:
        exps = list(experiments)
    if feature_dir is not None and len(exps) > 1:
        raise ValueError("feature_dir applies to a single experiment; got " + ", ".join(exps))

    celldino_dirs = {e: OpsDataset(e).results / "cell_dino_features_v2" for e in exps}
    feature_dirs = {
        e: Path(feature_dir) if feature_dir is not None else celldino_dirs[e] / "anndata_objects"
        for e in exps
    }

    base = EmbeddingDecisions.from_yaml(decisions_yaml) if decisions_yaml else EmbeddingDecisions()
    for k, v in decision_overrides.items():
        if hasattr(base, k) and v is not None:
            setattr(base, k, v)

    # Markers: from the bulked files for a plain single-experiment run (unchanged
    # behaviour, works even without the cell-level h5ad); from the cell-level
    # files — intersected across experiments — whenever a selection is in play.
    selecting = bool(include_wells or exclude_wells) or len(exps) > 1
    if selecting:
        per_exp_markers = {
            e: {Path(p).stem.replace("features_processed_", "")
                for p in glob.glob(str(feature_dirs[e] / "features_processed_*.h5ad"))}
            for e in exps
        }
        for e, found in per_exp_markers.items():
            if not found:
                raise FileNotFoundError(
                    f"No features_processed_*.h5ad in {feature_dirs[e]}; the cell-level "
                    f"features are required to select wells / pool experiments."
                )
        markers = sorted(set.intersection(*per_exp_markers.values()))
        if not markers:
            raise FileNotFoundError(
                "No marker is present in every experiment: "
                + "; ".join(f"{e}={sorted(m)}" for e, m in per_exp_markers.items())
            )
        skipped = sorted(set.union(*per_exp_markers.values()) - set(markers))
        if skipped:
            print(f"[embed] skipping markers missing from some experiment: {skipped}")
    else:
        markers = sorted(
            Path(p).stem.replace("gene_bulked_", "")
            for p in glob.glob(str(feature_dirs[exps[0]] / "gene_bulked_*.h5ad"))
        )
        if not markers:
            raise FileNotFoundError(
                f"No gene_bulked_*.h5ad in {feature_dirs[exps[0]]}. Run celldino_inference "
                f"(extraction + combine) before embedding post-processing."
            )

    # Resolve the well selection once, locally, off the first marker's cell obs:
    # it drives the output-dir name, the rebuild decision and the job kwargs, and
    # an unknown well fails here rather than an hour into a SLURM job.
    census, wells = {}, {}
    for e in exps:
        cell_path = _cell_h5ad(feature_dirs[e], markers[0])
        if not cell_path.exists():
            print(f"[embed] {e}: no {cell_path.name}; well provenance unavailable")
            continue
        census[e] = _well_census(cell_path)
        wells[e] = _select_wells(census[e], include_wells, exclude_wells, e)
    dropped = {e: sorted(set(census[e]) - set(wells[e])) for e in census}
    rebuild = len(exps) > 1 or any(dropped.values())

    if embeddings_dir is None:
        if len(exps) > 1:
            embeddings_dir = COMBINED_EMBEDDINGS_ROOT / _run_name(exps)
        else:
            # Keep outputs inside the existing CellDINO dir (no separate tree).
            embeddings_dir = celldino_dirs[exps[0]] / "embeddings"
    embeddings_dir = Path(embeddings_dir)
    print(
        f"[embed] {len(markers)} marker(s) {markers} from {exps} -> {embeddings_dir}\n"
        f"[embed] wells {wells or 'all'}"
        + (f", dropping {({e: d for e, d in dropped.items() if d})}" if any(dropped.values()) else "")
        + f" | {'re-bulking from cell level' if rebuild else 'reusing bulked h5ads'}"
    )

    jobs = []
    for marker in markers:
        dec = EmbeddingDecisions(**{
            **asdict(base),
            "marker": marker,
            "experiments": exps,
            "wells_included": {e: wells[e] for e in wells} or None,
            "wells_excluded": {e: d for e, d in dropped.items() if d} or None,
        })
        name = f"embed_{'_'.join(e.split('_')[0] for e in exps)}_{marker}"
        if rebuild:
            kwargs = {
                "cell_h5ads": {e: _cell_h5ad(feature_dirs[e], marker) for e in exps},
                "wells": {e: wells[e] for e in exps},
                "out_dir": embeddings_dir / marker,
                "decisions": dec,
            }
            func = run_marker_pooled
        else:
            gene = feature_dirs[exps[0]] / f"gene_bulked_{marker}.h5ad"
            guide = feature_dirs[exps[0]] / f"guide_bulked_{marker}.h5ad"
            for p in (gene, guide):
                if not p.exists():
                    raise FileNotFoundError(f"Missing bulked h5ad for {marker}: {p}")
            dec.n_cells = sum(census.get(exps[0], {}).get(w, 0) for w in wells.get(exps[0], [])) or None
            dec.source_cell_h5ads = (
                [str(_cell_h5ad(feature_dirs[exps[0]], marker))] if census else None
            )
            kwargs = {
                "gene_h5ad": gene,
                "guide_h5ad": guide,
                "out_dir": embeddings_dir / marker,
                "decisions": dec,
            }
            func = run_marker
        jobs.append({
            "name": name,
            "func": func,
            "kwargs": kwargs,
            "metadata": {"experiment": ",".join(exps), "marker": marker},
        })

    if not slurm:
        return [j["func"](**j["kwargs"]) for j in jobs]

    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs

    params = {
        "timeout_min": 720,
        "mem": "64G",
        "cpus_per_task": 8,
        "slurm_partition": "cpu",
    }
    if slurm_params:
        params.update(slurm_params)
    run_name = _run_name(exps) if len(exps) > 1 else exps[0]
    return submit_parallel_jobs(
        jobs_to_submit=jobs,
        experiment=f"{run_name}_embeddings",
        slurm_params=params,
        log_dir=f"slurm_embeddings_postprocess/{run_name}",
        manifest_prefix="embedding_postprocess",
        wait_for_completion=True,
        verbose=True,
    )


def main(argv=None):
    """CLI for ad-hoc runs (multi-experiment pools can't go through the per-experiment DAG)."""
    import argparse

    p = argparse.ArgumentParser(description="CellDINO embedding post-processing")
    p.add_argument("experiments", nargs="+", help="one or more experiments to embed together")
    p.add_argument("--wells", nargs="+", help="include only these wells (e.g. A1 A2)")
    p.add_argument("--exclude-wells", nargs="+", help="drop these wells")
    p.add_argument("--embeddings-dir", help="override the output dir")
    p.add_argument("--decisions-yaml", help="decisions.yaml to seed the run from")
    p.add_argument("--no-slurm", action="store_true", help="run in-process instead of on SLURM")
    a = p.parse_args(argv)
    return run_per_exp_embeddings(
        a.experiments,
        embeddings_dir=a.embeddings_dir,
        decisions_yaml=a.decisions_yaml,
        slurm=not a.no_slurm,
        include_wells=a.wells,
        exclude_wells=a.exclude_wells,
    )


if __name__ == "__main__":
    main()
