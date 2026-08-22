#!/usr/bin/env python3
"""Compare two imaging passes of the same plate in CellDINO embedding space.

A "pass" is one CellDINO marker directory under
``3-assembly/cell_dino_features_v2/embeddings/<marker>/`` -- e.g. ``Phase`` (live
acquisition) and ``Phase_fixed`` (a joined reimage pass, see convert/v3_reimage).

Reports, all paired on the same items so the comparison is like-for-like:

  1. reproducibility  per-gene distinctiveness and per-EBI-complex consistency mAP in
                      pass A vs pass B, EBI complexes named from the same yaml the
                      metric is built from. Points on the diagonal reproduce.
  2. retention        per-gene agreement of the gene x gene correlation profile
                      between passes, conditioned on the gene having a real phenotype
                      in pass A (see per_exp_embedding.cross_pass_retention_report).
  3. structure        clustered gene x gene correlation, both passes plus difference
                      (per_exp_embedding.cross_pass_correlation_heatmap).

Usage:
    python -m ops_model.post_process.cross_pass_compare -e ops0185_20260730
    python -m ops_model.post_process.cross_pass_compare -e ops0185 --a Phase --b Phase_fixed
    python -m ops_model.post_process.cross_pass_compare -e ops0185 --only reproducibility
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

VALUE = "mean_average_precision"
SIG = "below_corrected_p"
EBI_YAML = Path("/hpc/projects/icd.fast.ops/configs/gene_clusters/"
                "EBI_complexes_v1_old_gene_names.yaml")


# ── helpers ──────────────────────────────────────────────────────────────────

def wrap_label(text, width: int = 24) -> str:
    """Wrap a long complex name onto several lines (names are unusable truncated)."""
    out, line = [], ""
    for word in str(text).replace(",", ", ").split():
        if len(line) + len(word) + 1 > width:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(line)
    return "\n".join(out)


def place_labels(ax, xs, ys, texts, fontsize, pad_frac: float = 0.030,
                 max_pass: int = 2000, leader: bool = True):
    """Centre labels near their points and dodge until nothing overlaps.

    Uses ax.text, NOT ax.annotate with textcoords="offset points": on such an
    annotation set_position() sets the OFFSET IN POINTS, so writing data coordinates
    into it silently produces nonsense placement. A Text artist's position is plain
    data coordinates, which is what the solver below computes.

    The layout is solved in pixel space from a single canvas draw -- a label box does
    not change size when it moves, only position, so one measurement is exact and no
    re-render per iteration is needed.
    """
    import numpy as _np

    fig = ax.figure
    (ylo, yhi), (xlo, xhi) = ax.get_ylim(), ax.get_xlim()
    y_nudge = (yhi - ylo) * 0.022
    arts = [ax.text(x, y + y_nudge, t, fontsize=fontsize, ha="center", va="center",
                    linespacing=1.05, zorder=6)
            for x, y, t in zip(xs, ys, texts)]
    if not arts:
        return arts

    fig.canvas.draw()
    boxes = [a.get_window_extent() for a in arts]
    w = _np.array([b.width for b in boxes], dtype=float)
    h = _np.array([b.height for b in boxes], dtype=float)

    pos = _np.array([ax.transData.transform(a.get_position()) for a in arts], dtype=float)
    c0 = ax.transData.transform((xlo, ylo))
    c1 = ax.transData.transform((xhi, yhi))
    stepx = abs(c1[0] - c0[0]) * pad_frac
    stepy = abs(c1[1] - c0[1]) * pad_frac
    lo_x, hi_x = sorted([c0[0], c1[0]])
    lo_y, hi_y = sorted([c0[1], c1[1]])
    n = len(arts)

    for _ in range(max_pass):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                if abs(pos[i, 0] - pos[j, 0]) >= (w[i] + w[j]) / 2.0:
                    continue
                if abs(pos[i, 1] - pos[j, 1]) >= (h[i] + h[j]) / 2.0:
                    continue
                moved = True
                if abs(pos[i, 1] - pos[j, 1]) < stepy * 1.5:
                    hi, lo = (i, j) if pos[i, 1] >= pos[j, 1] else (j, i)
                    pos[hi, 1] += stepy
                    pos[lo, 1] -= stepy
                else:
                    rt, lf = (i, j) if pos[i, 0] >= pos[j, 0] else (j, i)
                    pos[rt, 0] += stepx
                    pos[lf, 0] -= stepx
        pos[:, 0] = _np.clip(pos[:, 0], lo_x + w / 2, hi_x - w / 2)
        pos[:, 1] = _np.clip(pos[:, 1], lo_y + h / 2, hi_y - h / 2)
        if not moved:
            break

    final = ax.transData.inverted().transform(pos)
    for a, (lx, ly) in zip(arts, final):
        a.set_position((float(lx), float(ly)))
    if leader:
        sx = (xhi - xlo) * pad_frac
        sy = (yhi - ylo) * pad_frac
        for (lx, ly), x, y in zip(final, xs, ys):
            if abs(lx - x) > sx * 0.6 or abs(ly - y) > sy * 1.2:
                ax.plot([x, lx], [y, ly], lw=0.7, color="0.55", zorder=2, alpha=0.85)
    return arts


def ebi_complex_names(yaml_path: Path = EBI_YAML) -> tuple[dict, dict]:
    """complex_num -> (name, n_genes) from the yaml the EBI mAP metric is built from."""
    import yaml

    names, sizes = {}, {}
    if not Path(yaml_path).exists():
        return names, sizes
    for key, entry in (yaml.safe_load(Path(yaml_path).read_text()) or {}).items():
        if isinstance(entry, dict) and "name" in entry:
            names[str(key)] = entry["name"]
            sizes[str(key)] = len(entry.get("genes") or [])
    return names, sizes


def _metrics_dir(exp_root: Path, marker: str) -> Path:
    return exp_root / "embeddings" / marker / "metrics"


def paired_metric(exp_root: Path, marker_a: str, marker_b: str,
                  metric: str, key: str) -> pd.DataFrame:
    a = pd.read_csv(_metrics_dir(exp_root, marker_a) / f"{metric}.csv")
    b = pd.read_csv(_metrics_dir(exp_root, marker_b) / f"{metric}.csv")
    cols = [key, VALUE] + ([SIG] if SIG in a.columns else [])
    return a[cols].merge(b[cols], on=key, suffixes=("_a", "_b"))


def _interactive_scatter(panels, out_html: Path, title: str, names: tuple):
    """One compact interactive file for both comparisons.

    Small canvas with proportionally larger markers and type, so it reads without
    scrolling. The non-significant cloud (~900 of ~1000 genes) is drawn with hover
    disabled -- carrying a hover string per point was most of the file size and none
    of the value.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=1, cols=len(panels), horizontal_spacing=0.09,
                        subplot_titles=[p["title"] for p in panels])
    for col, panel in enumerate(panels, start=1):
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", showlegend=False,
                                 hoverinfo="skip",
                                 line=dict(dash="dash", color="grey", width=1)),
                      row=1, col=col)
        ns = panel["df"][panel["df"]["class"] == "not significant"]
        if len(ns):
            fig.add_trace(go.Scatter(
                x=ns[f"{VALUE}_a"], y=ns[f"{VALUE}_b"], mode="markers",
                name=f"not significant ({len(ns)})", hoverinfo="skip",
                legendgroup="ns", showlegend=(col == 1),
                marker=dict(color="#c8ccd2", size=7, opacity=0.6)), row=1, col=col)
        for cls, colour, size in ((f"{names[0]} only", "#C44E52", 14),
                                  (f"{names[1]} only", "#DD8452", 14),
                                  ("significant in both", "#4C72B0", 17)):
            d = panel["df"][panel["df"]["class"] == cls]
            if not len(d):
                continue
            fig.add_trace(go.Scatter(
                x=d[f"{VALUE}_a"], y=d[f"{VALUE}_b"], mode="markers", name=f"{cls} ({len(d)})",
                legendgroup=cls, showlegend=(col == 1),
                marker=dict(color=colour, size=size, line=dict(width=0.7, color="black")),
                text=d["hover"], hovertemplate="%{text}<extra></extra>"), row=1, col=col)
        fig.update_xaxes(title_text=f"{names[0]} mAP", range=[-0.03, 1.05],
                         row=1, col=col, title_font_size=17, tickfont_size=15)
        fig.update_yaxes(title_text=f"{names[1]} mAP", range=[-0.03, 1.05],
                         row=1, col=col, title_font_size=17, tickfont_size=15)
    fig.update_annotations(font_size=18)          # subplot titles
    fig.update_layout(
        title=dict(text=title, font=dict(size=20)),
        width=1180, height=560, hovermode="closest",
        margin=dict(l=70, r=20, t=90, b=90), font=dict(size=15),
        legend=dict(orientation="h", y=-0.22, font=dict(size=14)))
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs="cdn",
                   config={"displaylogo": False,
                           "modeBarButtonsToRemove": ["select2d", "lasso2d"]})
    return out_html


def _classify(df, names):
    """Label each row by which pass called it significant (values are separate)."""
    sa = df.get(f"{SIG}_a", pd.Series(True, index=df.index)).fillna(False)
    sb = df.get(f"{SIG}_b", pd.Series(True, index=df.index)).fillna(False)
    out = pd.Series("not significant", index=df.index)
    out[sa & sb] = "significant in both"
    out[sa & ~sb] = f"{names[0]} only"
    out[~sa & sb] = f"{names[1]} only"
    return out


# ── report 1: reproducibility ────────────────────────────────────────────────

def reproducibility(exp_root: Path, marker_a: str, marker_b: str, out_dir: Path,
                    names: tuple = ("live", "fixed"), n_label_genes: int = 30,
                    n_label_complexes: int = 22, font_scale: float = 1.5):
    """Per-gene and per-complex mAP in both passes, on a shared diagonal.

    Deliberately NOT ranked-by-magnitude language: the plot shows where signal is and
    whether it reproduces, and a point being far from the origin is visible without
    asserting it is important.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    cnames, csizes = ebi_complex_names()
    plt.rcParams.update({"font.size": 14 * font_scale})
    fig, axes = plt.subplots(1, 2, figsize=(28, 12))

    def _panel(ax, df, key, label_col, labels, n_label, title, wrap=False,
               size_col=None, min_n_label=None):
        ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1.5, zorder=1)
        sig_a = df.get(f"{SIG}_a", pd.Series(True, index=df.index)).fillna(False)
        sig_b = df.get(f"{SIG}_b", pd.Series(True, index=df.index)).fillna(False)
        ns, only_a, both = df[~sig_a], df[sig_a & ~sig_b], df[sig_a & sig_b]
        ax.scatter(ns[f"{VALUE}_a"], ns[f"{VALUE}_b"], s=21, c="#c8ccd2",
                   alpha=0.6, lw=0, label=f"not significant in {names[0]}")
        ax.scatter(only_a[f"{VALUE}_a"], only_a[f"{VALUE}_b"], s=105, c="#C44E52",
                   edgecolors="k", lw=0.5, label=f"{names[0]} only")
        def _sz(d, base):
            # Marker area tracks complex size: most EBI complexes here have only 2-3
            # genes, where mAP is a couple of retrievals and 0/1 values are common.
            if size_col is None:
                return base
            return base * (0.45 + 0.55 * (d[size_col].fillna(2) / 6.0).clip(0.3, 2.5))
        ax.scatter(both[f"{VALUE}_a"], both[f"{VALUE}_b"], s=_sz(both, 165), c="#4C72B0",
                   edgecolors="k", lw=0.6, label="reproducible")
        ax.set_ylim(-0.04, 1.16)
        lab = df[sig_a | sig_b]
        if min_n_label is not None and size_col is not None:
            big_enough = lab[lab[size_col].fillna(0) >= min_n_label]
            lab = big_enough if len(big_enough) else lab
        lab = lab.head(n_label)
        texts = [f"{wrap_label(t)}\n(n={n})" if wrap else str(t)
                 for t, n in zip(lab[label_col], lab.get("n_genes", [0] * len(lab)))]
        place_labels(ax, lab[f"{VALUE}_a"].tolist(), lab[f"{VALUE}_b"].tolist(),
                     texts, fontsize=14 * font_scale * 0.67)
        ax.set_xlabel(f"{names[0]} mAP", fontsize=17 * font_scale)
        ax.set_ylabel(f"{names[1]} mAP", fontsize=17 * font_scale)
        ax.set_title(title, fontsize=18 * font_scale)
        ax.legend(fontsize=12 * font_scale, loc="center left",
                  bbox_to_anchor=(1.01, 0.5), frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        return ns, only_a, both

    genes = paired_metric(exp_root, marker_a, marker_b,
                          "phenotypic_distinctiveness", "perturbation")
    genes["rank_by"] = genes[[f"{VALUE}_a", f"{VALUE}_b"]].max(axis=1)
    genes = genes.sort_values("rank_by", ascending=False)
    _panel(axes[0], genes, "perturbation", "perturbation", names, n_label_genes,
           "geneKO distinctiveness")

    cx = paired_metric(exp_root, marker_a, marker_b,
                       "phenotypic_consistency_ebi", "complex_num")
    cx["complex"] = cx["complex_num"].astype(str).map(cnames).fillna(
        "complex " + cx["complex_num"].astype(str))
    cx["n_genes"] = cx["complex_num"].astype(str).map(csizes)
    cx["rank_by"] = cx[[f"{VALUE}_a", f"{VALUE}_b"]].max(axis=1)
    cx = cx.sort_values("rank_by", ascending=False)
    _panel(axes[1], cx, "complex_num", "complex", names, n_label_complexes,
           "EBI complex consistency", wrap=True, size_col="n_genes", min_n_label=4)

    # interactive twin
    genes_h = genes.copy()
    genes_h["class"] = _classify(genes_h, names)
    genes_h["hover"] = (genes_h["perturbation"].astype(str)
                        + "<br>" + names[0] + " mAP: " + genes_h[f"{VALUE}_a"].round(3).astype(str)
                        + "<br>" + names[1] + " mAP: " + genes_h[f"{VALUE}_b"].round(3).astype(str)
                        + "<br>" + genes_h["class"])
    cx_h = cx.copy()
    cx_h["class"] = _classify(cx_h, names)
    cx_h["hover"] = (cx_h["complex"].astype(str)
                     + "<br>genes: " + cx_h["n_genes"].astype(str)
                     + "<br>" + names[0] + " mAP: " + cx_h[f"{VALUE}_a"].round(3).astype(str)
                     + "<br>" + names[1] + " mAP: " + cx_h[f"{VALUE}_b"].round(3).astype(str)
                     + "<br>" + cx_h["class"])
    _interactive_scatter(
        [{"df": genes_h, "title": "geneKO distinctiveness"},
         {"df": cx_h, "title": "EBI complex consistency"}],
        out_dir / "reproducibility.html",
        f"{exp_root.parents[1].name}: {names[0]} vs {names[1]} phenotype reproducibility",
        names)

    fig.suptitle(f"{exp_root.parents[1].name}: {names[0]} vs {names[1]} "
                 f"phenotype reproducibility", fontsize=21 * font_scale)
    fig.tight_layout(rect=(0, 0, 0.97, 0.96))
    for ext in ("png", "svg"):
        fig.savefig(out_dir / f"reproducibility.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)

    genes.drop(columns=["rank_by"]).to_csv(out_dir / "genes_distinctiveness.csv", index=False)
    cx.drop(columns=["rank_by"]).to_csv(out_dir / "ebi_complexes_named.csv", index=False)
    return genes, cx


def _cluster_order(M):
    """Leaf order from average-linkage clustering of a similarity matrix."""
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform
    d = 1.0 - M
    np.fill_diagonal(d, 0.0)
    return leaves_list(linkage(squareform((d + d.T) / 2.0, checks=False), method="average"))


def _fold_panel(Ca, Cb, labels, names):
    """Pass A above the diagonal, pass B below, clustered on A."""
    order = _cluster_order(Ca)
    lab = [labels[i] for i in order]
    A, B = Ca[np.ix_(order, order)], Cb[np.ix_(order, order)]
    n = len(lab)
    iu = np.triu_indices(n, k=1)
    z = np.full((n, n), np.nan)
    z[iu] = A[iu]
    z[(iu[1], iu[0])] = B[iu]
    which = np.full((n, n), "", dtype=object)
    which[iu] = names[0]
    which[(iu[1], iu[0])] = names[1]
    lim = max(float(np.nanpercentile(np.abs(A[~np.eye(n, dtype=bool)]), 98)), 1e-3)
    return z, which, lab, lim, {names[0]: A, names[1]: B}


def _diff_panel(Ca, Cb, labels):
    """Difference matrix with its own clustering.

    Ordered by how similarly entities shed connections (correlation of their
    difference profiles), which groups them by mode of loss rather than by live
    structure -- that is the point of looking at the difference separately.
    """
    D = Ca - Cb
    prof = np.corrcoef(D - D.mean(axis=0, keepdims=True))
    prof = np.nan_to_num(prof, nan=0.0)
    order = _cluster_order(prof)
    lab = [labels[i] for i in order]
    Do = D[np.ix_(order, order)]
    n = len(lab)
    lim = max(float(np.nanpercentile(np.abs(Do[~np.eye(n, dtype=bool)]), 98)), 1e-3)
    return Do, lab, lim, {"difference": Do}


def _render_canvas(entries, out_html: Path, names: tuple, mode: str, title: str):
    """One canvas, one panel per entity (geneKO, complex) side by side.

    mode="fold": each panel carries both passes across its diagonal.
    mode="diff": each panel is pass A - pass B with its own clustering.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    ncol = len(entries)
    subtitles, panels = [], []
    for ent in entries:
        if mode == "fold":
            z, which, lab, lim, mats_out = _fold_panel(ent["Ca"], ent["Cb"],
                                                       ent["labels"], names)
            subtitles.append(f"{ent['name']}: {names[0]} upper right / {names[1]} lower left"
                             f"  ({len(lab)})")
            panels.append((z, which, lab, lim, ent, mats_out))
        else:
            z, lab, lim, mats_out = _diff_panel(ent["Ca"], ent["Cb"], ent["labels"])
            subtitles.append(f"{ent['name']}: {names[0]} - {names[1]}  ({len(lab)})")
            panels.append((z, None, lab, lim, ent, mats_out))

    # Every matrix is square, so a panel renders square only when its pixel width
    # equals the plot height -- equal column widths sized off the height. Forcing
    # it with scaleanchor instead leaves the axis domain wider than the drawn
    # heatmap, which is what parked the colourbar out in whitespace.
    L, R, TOP, BOT, HEIGHT = 185, 250, 105, 250, 1000
    panel_px = HEIGHT - TOP - BOT
    gap_px = 170
    plot_w = ncol * panel_px + (ncol - 1) * gap_px
    fig = make_subplots(rows=1, cols=ncol, horizontal_spacing=gap_px / plot_w,
                        subplot_titles=subtitles)
    for col, (z, which, lab, lim, ent, _m) in enumerate(panels, start=1):
        hover = ("%{customdata}<br>%{y}<br>vs %{x}<br>r: %{z:.3f}<extra></extra>"
                 if which is not None else
                 f"{names[0]} - {names[1]}<br>%{{y}}<br>vs %{{x}}"
                 "<br>\u0394: %{z:.3f}<extra></extra>")
        # Last panel's labels go on its right, otherwise long complex names sit on
        # top of the neighbouring heatmap; its colourbar then has to clear them.
        outward = "right" if col == ncol and ncol > 1 else "left"
        dom = fig.layout["xaxis" if col == 1 else f"xaxis{col}"].domain or (0.0, 1.0)
        cbar_x = (1.13 if col == ncol and ncol > 1 else float(dom[1]) + 0.008)
        fig.add_trace(go.Heatmap(
            z=np.round(z, 3), x=lab, y=lab, zmin=-lim, zmax=lim, colorscale="RdBu_r",
            customdata=which, showscale=True, hovertemplate=hover,
            colorbar=dict(title="\u0394r" if which is None else "r", len=0.8,
                          thickness=13, x=cbar_x, xanchor="left")),
            row=1, col=col)
        fig.update_xaxes(tickangle=90, tickfont_size=ent["tickfont"], row=1, col=col)
        fig.update_yaxes(tickfont_size=ent["tickfont"], autorange="reversed",
                         side=outward, row=1, col=col)
    fig.update_annotations(font_size=15)
    fig.update_layout(title=dict(text=title, font=dict(size=19)),
                      width=L + R + plot_w, height=HEIGHT,
                      margin=dict(l=L, r=R, t=TOP, b=BOT), font=dict(size=13))
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs="cdn", config={"displaylogo": False})

    stem = out_html.with_suffix("")
    for z, which, lab, lim, ent, mats_out in panels:
        slug = ent["name"].lower().replace(" ", "_")
        for who, M in mats_out.items():
            pd.DataFrame(M, index=lab, columns=lab).to_csv(
                f"{stem}_{slug}_{who}.csv")
    _render_canvas_png(panels, Path(f"{stem}.png"), names, mode, title)
    return out_html


def _render_canvas_png(panels, out_png: Path, names: tuple, mode: str, title: str):
    """Static twin of the interactive canvas.

    Drawn with matplotlib rather than exported from plotly, which would need
    kaleido; the layout rules (proportional widths, square cells) are the same.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Equal boxes, not width ratios: with aspect="equal" every square matrix fills
    # the same box, so geneKO and complex heatmaps come out the same physical size
    # (their cells differ instead, which is what the item counts should change).
    fig, axes = plt.subplots(
        1, len(panels), figsize=(9.0 * len(panels), 9.4),
        gridspec_kw={"wspace": 0.55})
    axes = np.atleast_1d(axes)
    for ax, (z, which, lab, lim, ent, _m) in zip(axes, panels):
        im = ax.imshow(z, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="equal",
                       interpolation="nearest")
        ax.set_xticks(range(len(lab)))
        ax.set_xticklabels(lab, rotation=90, fontsize=ent["tickfont"] - 1)
        ax.set_yticks(range(len(lab)))
        ax.set_yticklabels(lab, fontsize=ent["tickfont"] - 1)
        sub = (f"{ent['name']}: {names[0]} upper right / {names[1]} lower left"
               if mode == "fold" else f"{ent['name']}: {names[0]} - {names[1]}")
        ax.set_title(f"{sub}  ({len(lab)})", fontsize=12)
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
        cb.set_label("\u0394r" if mode != "fold" else "r", fontsize=11)
    fig.suptitle(title, fontsize=15)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_png


def _corr_from_rows(X):
    return np.corrcoef(X - X.mean(axis=0, keepdims=True))


def _pass_matrices(exp_root: Path, markers: tuple):
    out = {}
    import anndata as ad
    for marker in markers:
        g = ad.read_h5ad(exp_root / "anndata_objects" / f"gene_bulked_{marker}.h5ad")
        col = "perturbation" if "perturbation" in g.obs.columns else g.obs.columns[0]
        out[marker] = (np.asarray(g.X, dtype="float64"), g.obs[col].astype(str).tolist())
    return out


def geneko_entry(exp_root: Path, marker_a: str, marker_b: str, mats,
                 max_genes: int = 90):
    """geneKO correlation matrices, restricted to genes that carry signal.

    All 1001 genes is pixel soup with no room for labels. Note the selection is
    effectively pass-A-driven here: every gene significant in fixed is also
    significant in live, so the union is the live-significant set.
    """
    dist = pd.read_csv(_metrics_dir(exp_root, marker_a) / "phenotypic_distinctiveness.csv")
    dist_b = pd.read_csv(_metrics_dir(exp_root, marker_b) / "phenotypic_distinctiveness.csv")
    sig = set(dist.loc[dist[SIG] == True, "perturbation"].astype(str)) | \
          set(dist_b.loc[dist_b[SIG] == True, "perturbation"].astype(str))
    strength = dict(zip(dist["perturbation"].astype(str), dist[VALUE]))
    keep = set(mats[marker_b][1])
    genes = [g for g in mats[marker_a][1] if g in sig and g in keep]
    genes = sorted(genes, key=lambda g: -strength.get(g, 0.0))[:max_genes]
    if len(genes) < 4:
        return None

    def _sub(marker):
        X, labels = mats[marker]
        return _corr_from_rows(X[[labels.index(g) for g in genes]])

    return dict(name="geneKO", labels=genes, tickfont=7,
                Ca=_sub(marker_a), Cb=_sub(marker_b))


def complex_entry(exp_root: Path, marker_a: str, marker_b: str, mats,
                  min_genes: int = 2, max_complexes: int = 80):
    """EBI complex matrices from complex-mean gene profiles, labelled by name."""
    import yaml
    cnames, _ = ebi_complex_names()
    members = {}
    if EBI_YAML.exists():
        for k, e in (yaml.safe_load(EBI_YAML.read_text()) or {}).items():
            if isinstance(e, dict) and e.get("genes"):
                members[str(k)] = [str(g) for g in e["genes"]]

    cx = paired_metric(exp_root, marker_a, marker_b,
                       "phenotypic_consistency_ebi", "complex_num")
    sig = cx[(cx[f"{SIG}_a"] == True) | (cx[f"{SIG}_b"] == True)]
    sig = sig.assign(rank_by=sig[[f"{VALUE}_a", f"{VALUE}_b"]].max(axis=1)) \
             .sort_values("rank_by", ascending=False)

    idx_a = {g: i for i, g in enumerate(mats[marker_a][1])}
    idx_b = {g: i for i, g in enumerate(mats[marker_b][1])}
    rows_a, rows_b, labels = [], [], []
    for _, r in sig.iterrows():
        genes = [g for g in members.get(str(r["complex_num"]), [])
                 if g in idx_a and g in idx_b]
        if len(genes) < min_genes:
            continue
        rows_a.append(mats[marker_a][0][[idx_a[g] for g in genes]].mean(axis=0))
        rows_b.append(mats[marker_b][0][[idx_b[g] for g in genes]].mean(axis=0))
        nm = cnames.get(str(r["complex_num"]), f"complex {r['complex_num']}")
        labels.append(f"{nm[:44]} (n={len(genes)})")
        if len(labels) >= max_complexes:
            break
    if len(labels) < 4:
        return None
    return dict(name="EBI complex", labels=labels, tickfont=8,
                Ca=_corr_from_rows(np.vstack(rows_a)),
                Cb=_corr_from_rows(np.vstack(rows_b)))


def structure_heatmaps(exp_root: Path, marker_a: str, marker_b: str, out_dir: Path,
                       names: tuple = ("live", "fixed")):
    """Two canvases: passes folded across the diagonal, then their difference.

    geneKO and complex sit side by side on each canvas so the same comparison is
    read at both levels without opening two files.
    """
    mats = _pass_matrices(exp_root, (marker_a, marker_b))
    entries = [e for e in (geneko_entry(exp_root, marker_a, marker_b, mats),
                           complex_entry(exp_root, marker_a, marker_b, mats)) if e]
    if not entries:
        print("  [structure] no signal-bearing entities; skipping heatmaps")
        return []
    written = [
        _render_canvas(entries, out_dir / "structure_split_halves.html", names, "fold",
                       f"Correlation structure, {names[0]} vs {names[1]} "
                       "(clustered on " + names[0] + ")"),
        _render_canvas(entries, out_dir / "structure_difference.html", names, "diff",
                       f"Correlation loss on fixation, {names[0]} - {names[1]} "
                       "(clustered on the difference)"),
    ]
    for w in written:
        print(f"  {' + '.join(e['name'] for e in entries)} -> {w}", flush=True)
    return written


# ── driver ───────────────────────────────────────────────────────────────────

def _gene_matrix(exp_root: Path, marker: str):
    import anndata as ad

    h5 = exp_root / "anndata_objects" / f"gene_bulked_{marker}.h5ad"
    g = ad.read_h5ad(h5)
    col = "perturbation" if "perturbation" in g.obs.columns else g.obs.columns[0]
    return np.asarray(g.X, dtype="float64"), g.obs[col].astype(str).tolist()


def compare_passes(experiment: str, marker_a: str = "Phase", marker_b: str = "Phase_fixed",
                   feature_dir: str | None = None, out_dir: str | None = None,
                   names: tuple = ("live", "fixed"), only: str | None = None):
    """Run the cross-pass comparison for one experiment. Returns the output dir."""
    from ops_utils.data.experiment import OpsDataset
    from ops_model.post_process.per_exp_embedding import (
        cross_pass_correlation_heatmap, cross_pass_retention_report,
    )

    root = Path(feature_dir) if feature_dir else (
        Path(OpsDataset(experiment).results) / "cell_dino_features_v2")
    # Default to embeddings/<nameA>_vs_<nameB> using the DISPLAY names (live_vs_fixed),
    # not the marker ids: the marker-id path (Phase_vs_Phase_fixed) is a second
    # directory for the same comparison and just makes outputs hard to find.
    out = Path(out_dir) if out_dir else (
        root / "embeddings" / f"{names[0]}_vs_{names[1]}")
    out.mkdir(parents=True, exist_ok=True)
    print(f"[cross-pass] {experiment}: {marker_a} vs {marker_b}\n  -> {out}", flush=True)

    if only in (None, "reproducibility"):
        g, c = reproducibility(root, marker_a, marker_b, out / "reproducibility", names)
        for lbl, df, sa, sb in (("genes", g, f"{SIG}_a", f"{SIG}_b"),
                                ("EBI complexes", c, f"{SIG}_a", f"{SIG}_b")):
            n_a = int(df[sa].sum()); n_both = int((df[sa] & df[sb]).sum())
            print(f"  {lbl}: significant in {names[0]} = {n_a}, "
                  f"reproduced in {names[1]} = {n_both}", flush=True)

    if only in (None, "retention", "structure"):
        Xa, la = _gene_matrix(root, marker_a)
        Xb, lb = _gene_matrix(root, marker_b)
        shared = sorted(set(la) & set(lb))
        Xa = Xa[[la.index(x) for x in shared]]
        Xb = Xb[[lb.index(x) for x in shared]]
        print(f"  shared genes: {len(shared)}", flush=True)

        if only in (None, "retention"):
            dist = pd.read_csv(_metrics_dir(root, marker_a) / "phenotypic_distinctiveness.csv")
            active = dist.loc[dist[SIG] == True, "perturbation"].astype(str).tolist()
            strength = dict(zip(dist["perturbation"].astype(str), dist[VALUE]))
            cross_pass_retention_report(
                Xa, Xb, shared, out / "retention" / "retention",
                names=names, active_genes=active, strength=strength)

        if only in (None, "structure"):
            structure_heatmaps(root, marker_a, marker_b, out / "structure", names)
            dist = pd.read_csv(_metrics_dir(root, marker_a) / "phenotypic_distinctiveness.csv")
            maps = dict(zip(dist["perturbation"].astype(str), dist[VALUE]))
            # Cluster once on pass A and reuse that order for BOTH figures, otherwise
            # the within-pass panels and the cross-pass panel are not comparable
            # (the cross one was previously in alphabetical order).
            _, _, order, _ = cross_pass_correlation_heatmap(
                Xa, Xb, shared, out / "structure" / "within_pass_structure",
                names=names, map_scores=maps)
            ordered = [shared[i] for i in order]
            from ops_model.post_process.per_exp_embedding import _correlation_heatmap
            _correlation_heatmap(
                Xa[order], ordered, out / "structure" / "cross_pass_same_gene",
                f"{names[0]} rows vs {names[1]} columns (diagonal = same gene)",
                X_other=Xb[order],
                axis_labels=(f"{names[0]} ({marker_a})", f"{names[1]} ({marker_b})"))
    return out


# ── driver ────────────────────────────────────────────────────────────────────

def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(
        description="Compare two imaging passes (live vs fixed) at gene and "
                    "complex level: reproducibility, retention, structure.")
    ap.add_argument("-e", "--experiment", required=True)
    ap.add_argument("--a", default="Phase", help="marker id for pass A")
    ap.add_argument("--b", default="Phase_fixed", help="marker id for pass B")
    ap.add_argument("--names", default="live,fixed",
                    help="display names for pass A,B (drives the output dir name)")
    ap.add_argument("--feature-dir", default=None,
                    help="override <experiment>/cell_dino_features_v2")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--only", default=None,
                    choices=["reproducibility", "retention", "structure"],
                    help="run a single section instead of all three")
    args = ap.parse_args(argv)

    names = tuple(n.strip() for n in args.names.split(",")[:2])
    out = compare_passes(args.experiment, args.a, args.b, args.feature_dir,
                         args.out_dir, names, args.only)
    print(f"[cross-pass] done -> {out}", flush=True)
    return out


if __name__ == "__main__":
    main()
