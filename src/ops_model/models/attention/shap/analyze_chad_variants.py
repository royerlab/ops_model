"""Full breakdown of N SHAP variants — what differs, what overlaps, what dominates.

Reusable across any group of SHAP CSVs that share the same schema
(`gene`, `viz_channel`, `shap_rank`, `feature`, `auroc`, ...). Designed
for the CHAD 3-variant attention comparison but works for any N ≥ 2
variants: top-attention 3-way, all-cells 3-way, or a mixed 6-way
when the 6 SHAP runs are complete.

Compares along four axes:

  1. Classifier quality   — AUROC distribution per variant.
  2. Feature dominance    — which features hog the top-5 slots within
                              each variant; cumulative coverage curve.
  3. Cross-variant overlap — pairwise top-K Jaccard heatmap +
                              per-classifier Jaccard distribution.
  4. Category breakdown   — share of top-5 slots per feature category
                              (intensity / morphology / network / locality
                              / moments), side-by-side.

Plus a Venn-style "which features are unique vs shared" partition.

Outputs:
  • <output>.pdf          — multi-page report (one section per axis).
  • <output>.summary.csv  — per-variant numerical summary + category mix.
  • <output>.dominant.csv — top-20 most-frequent features per variant.

Variant CSV spec format (one --csv per variant):
    --csv NAME=/path/to/ko_shap_features.csv
    --csv NAME=/path/to/ntc_shap_features.csv@CONTRAST    (filters by `contrast` column)

Usage:
    # The 3 CHAD attention variants (default behavior):
    python analyze_chad_variants.py

    # 3 all-cells variants (after they're generated):
    python analyze_chad_variants.py \\
        --csv all-distinct=/hpc/.../all_cells_distinct_chad/ntc_shap_features.csv@distinct \\
        --csv all-ntc=/hpc/.../all_cells_ntc_chad/ntc_shap_features.csv@ntc \\
        --csv all-global=/hpc/.../all_cells_global_chad/ntc_shap_features.csv@global \\
        --output /hpc/mydata/gav.sturm/all_cells_3way_breakdown.pdf

    # Mixed 6-way:
    python analyze_chad_variants.py \\
        --csv attn-distinct=...     \\
        --csv attn-ntc=...          \\
        --csv attn-global=...       \\
        --csv all-distinct=...@distinct \\
        --csv all-ntc=...@ntc       \\
        --csv all-global=...@global \\
        --output /tmp/6way.pdf
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

ALEX_BASE = Path("/hpc/projects/icd.fast.ops/models/alex_lin_attention")
# Default CSV set: the 3 CHAD top-attention variants. Override with one
# or more --csv flags; if any --csv is passed, the defaults are skipped.
DEFAULT_VARIANTS = [
    ("distinct", ALEX_BASE / "attention_distinct_chad/ko_shap_features.csv", None),
    ("ntc",      ALEX_BASE / "attention_ntc_chad/ko_shap_features.csv",      None),
    ("global",   ALEX_BASE / "attention_global_chad/ko_shap_features.csv",   None),
]
# Color palette — used when a variant name doesn't have an explicit color.
# Pinned colors for common variant names (matches the broader codebase's
# scheme) come first; the palette below picks up any extras.
_NAMED_COLORS = {
    "distinct":      "#1F46A6", "ntc":         "#C81E1E", "global":   "#2A9D8F",
    "attn-distinct": "#1F46A6", "attn-ntc":    "#C81E1E", "attn-global": "#2A9D8F",
    "all-distinct":  "#F4A261", "all-ntc":     "#9D4EDD", "all-global":  "#666666",
}
_PALETTE = [
    "#1F46A6", "#C81E1E", "#2A9D8F", "#F4A261", "#9D4EDD",
    "#666666", "#E76F51", "#264653", "#E9C46A", "#6A040F",
]


def _color_for(name: str, idx: int) -> str:
    if name in _NAMED_COLORS:
        return _NAMED_COLORS[name]
    return _PALETTE[idx % len(_PALETTE)]


def _parse_csv_spec(spec: str) -> tuple[str, Path, str | None]:
    """Parse 'name=path' or 'name=path@contrast' into (name, Path, contrast)."""
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            f"--csv expects 'name=path' or 'name=path@contrast', got {spec!r}"
        )
    name, rhs = spec.split("=", 1)
    if "@" in rhs:
        path, contrast = rhs.rsplit("@", 1)
    else:
        path, contrast = rhs, None
    return name.strip(), Path(path).expanduser(), (contrast or None)


# ---------------------------------------------------------------------------
# Feature category inference
# ---------------------------------------------------------------------------
_CAT_PATTERNS = [
    ("intensity",    r"intensity"),
    ("moments",      r"hu_moment|moments_weighted_hu|central_moment|inertia_eigval|"
                     r"moments_normalized|humoment|centralmoment|normalizedmoment|"
                     r"spatialmoment|inertiatensoreigenvalues"),
    ("network",      r"branch|skeleton|tortuosity|connected_component|network_length|"
                     r"branching|num_endpoints|num_nodes|num_branches"),
    ("locality",     r"distance_from_cell_edge|distance_from_nucleus|"
                     r"normalized_radial|centroid"),
    ("size",         r"area|perimeter|axis_(major|minor)_length|"
                     r"equivalent_diameter|convexarea"),
    ("shape",        r"aspect_ratio|circularity|solidity|extent|orientation|"
                     r"eccentricity|compactness|euler_number"),
]
def _categorize(feat: str) -> str:
    f = feat.lower()
    for cat, pat in _CAT_PATTERNS:
        if re.search(pat, f):
            return cat
    return "other"


# ---------------------------------------------------------------------------
# Loaders + accessors
# ---------------------------------------------------------------------------
def _load(name: str, path: Path, contrast: str | None = None) -> pd.DataFrame:
    """Read a SHAP CSV and normalize it. When `contrast` is supplied AND
    the CSV has a `contrast` column (all-cells variant), filter to that
    contrast. Channel-key column is added for case-insensitive matching
    across the top-attention vs all-cells naming conventions."""
    df = pd.read_csv(path)
    if contrast and "contrast" in df.columns:
        before = len(df)
        df = df[df["contrast"].astype(str) == contrast].copy()
        print(f"  [{name}] filtered to contrast={contrast!r}: "
              f"{len(df):,}/{before:,} rows")
    df["_variant"] = name
    df["_cat"] = df["feature"].astype(str).map(_categorize)
    df["_chan_key"] = df["viz_channel"].astype(str).str.lower().str.strip()
    return df


def _top_k_sets(df: pd.DataFrame, k: int = 5) -> dict[tuple, set[str]]:
    top = (df.sort_values(["gene", "_chan_key", "shap_rank"])
             .groupby(["gene", "_chan_key"], observed=True).head(k))
    return {(g, c): set(grp["feature"])
            for (g, c), grp in top.groupby(["gene", "_chan_key"], observed=True)}


def _auroc_per_row(df: pd.DataFrame) -> dict[tuple, float]:
    return (df.groupby(["gene", "_chan_key"], observed=True)["auroc"]
              .first().to_dict())


# ---------------------------------------------------------------------------
# Plot panels
# ---------------------------------------------------------------------------
def _panel_auroc(ax, dfs: dict[str, pd.DataFrame], colors: dict[str, str]):
    variants = list(dfs)
    data, labels, bar_colors = [], [], []
    for v in variants:
        a = dfs[v].groupby(["gene", "_chan_key"], observed=True)["auroc"].first()
        data.append(a.to_numpy())
        labels.append(v)
        bar_colors.append(colors[v])
    parts = ax.violinplot(data, positions=range(1, len(data) + 1),
                           widths=0.7, showmeans=False, showextrema=False,
                           showmedians=True)
    for body, c in zip(parts["bodies"], bar_colors):
        body.set_facecolor(c); body.set_alpha(0.7); body.set_edgecolor("black")
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, label="chance")
    ax.axhline(1.0, color="red", linestyle=":", linewidth=0.8, label="saturated")
    ax.set_ylim(0.4, 1.05); ax.set_ylabel("AUROC")
    ax.set_title("Per-(complex × channel) AUROC distribution",
                  fontweight="bold")
    for i, a in enumerate(data, start=1):
        med, sat = float(np.median(a)), float((a >= 0.95).mean())
        ax.text(i, med + 0.02, f"med={med:.2f}\n≥0.95: {sat:.0%}",
                ha="center", fontsize=8)
    ax.legend(loc="lower right", fontsize=8)


def _panel_jaccard_heatmap(ax, dfs: dict[str, pd.DataFrame], common: set, sets):
    variants = list(dfs)
    n = len(variants)
    mat = np.zeros((n, n))
    for i, a in enumerate(variants):
        for j, b in enumerate(variants):
            if i == j:
                mat[i, j] = 1.0; continue
            js = []
            for key in common:
                sa, sb = sets[a].get(key, set()), sets[b].get(key, set())
                u = sa | sb
                if u: js.append(len(sa & sb) / len(u))
            mat[i, j] = float(np.mean(js)) if js else np.nan
    im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(variants); ax.set_yticklabels(variants)
    for i in range(n):
        for j in range(n):
            v = mat[i, j]
            ax.text(j, i, "—" if np.isnan(v) else f"{v:.2f}",
                    ha="center", va="center",
                    color="white" if v < 0.5 else "black", fontsize=10)
    ax.set_title(f"Top-5 Jaccard overlap (mean over {len(common):,} classifiers)",
                  fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _panel_jaccard_distribution(ax, dfs: dict[str, pd.DataFrame], common: set, sets):
    variants = list(dfs)
    pairs = [(a, b) for i, a in enumerate(variants)
             for b in variants[i + 1:]]
    data, labels = [], []
    for a, b in pairs:
        js = []
        for key in common:
            sa, sb = sets[a].get(key, set()), sets[b].get(key, set())
            u = sa | sb
            if u: js.append(len(sa & sb) / len(u))
        data.append(np.asarray(js))
        labels.append(f"{a}\nvs {b}")
    parts = ax.violinplot(data, positions=range(1, len(data) + 1), widths=0.7,
                           showmeans=False, showextrema=False, showmedians=True)
    for body in parts["bodies"]:
        body.set_facecolor("#888888"); body.set_alpha(0.7)
        body.set_edgecolor("black")
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(-0.05, 1.05); ax.set_ylabel("Jaccard")
    ax.set_title("Per-classifier Jaccard (top-5 features)",
                  fontweight="bold")
    for i, a in enumerate(data, start=1):
        med, mean = float(np.median(a)), float(a.mean())
        eq = int((a >= 0.999).sum())
        ax.text(i, max(med, mean) + 0.05,
                f"med={med:.2f}\nmean={mean:.2f}\n{eq} exact",
                ha="center", fontsize=8)


def _panel_top_features(ax, df: pd.DataFrame, variant: str,
                         colors: dict[str, str], top_k: int = 20):
    """Horizontal bar chart of the N most-frequent features in this
    variant's top-5 (across all complex × channel classifiers)."""
    top5 = df[df["shap_rank"] <= 5]
    counts = top5["feature"].value_counts().head(top_k)
    n_class = top5.groupby(["gene", "_chan_key"], observed=True).ngroups
    feats = list(counts.index)[::-1]
    vals = list(counts.values)[::-1]
    ax.barh(range(len(feats)), vals, color=colors[variant], alpha=0.85,
             edgecolor="black", linewidth=0.4)
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(feats, fontsize=7)
    ax.set_xlabel(f"# classifiers (of {n_class:,}) with feature in top-5")
    ax.set_title(f"{variant}", fontsize=10, fontweight="bold",
                  color=colors[variant])
    # Annotate the top bar with its share
    top_share = vals[-1] / n_class if n_class else 0
    ax.text(0.98, 0.02, f"top-feature share: {100*top_share:.1f}%",
             transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
             bbox=dict(facecolor="white", edgecolor="gray", alpha=0.85))


def _panel_coverage_curves(ax, dfs: dict[str, pd.DataFrame],
                             colors: dict[str, str]):
    """For each variant: how many distinct features do you need to cover
    X% of all top-5 slots? Flat curves = a few features dominate; steep
    curves = signal is spread thin across many features."""
    for v, df in dfs.items():
        top5 = df[df["shap_rank"] <= 5]
        counts = top5["feature"].value_counts().values
        cum = np.cumsum(counts) / counts.sum()
        ax.plot(np.arange(1, len(cum) + 1), cum, color=colors[v],
                 linewidth=2, label=f"{v} ({len(counts):,} unique)")
    ax.set_xlabel("# unique features (sorted by frequency)")
    ax.set_ylabel("Cumulative fraction of top-5 slots covered")
    ax.set_xscale("log")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
    ax.axhline(0.9, color="gray", linestyle=":", linewidth=0.8)
    ax.set_title("Feature dominance — cumulative coverage of top-5 slots",
                  fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.25)


def _panel_category_breakdown(ax, dfs: dict[str, pd.DataFrame]):
    """Stacked horizontal bar of feature-category share of top-5 slots
    per variant."""
    cats = ["intensity", "size", "shape", "locality", "network",
            "moments", "other"]
    cat_colors = {
        "intensity": "#E07A5F", "size": "#3D5A80",
        "shape":     "#81B29A", "locality": "#F2CC8F",
        "network":   "#9D4EDD", "moments": "#666666",
        "other":     "#CCCCCC",
    }
    variants = list(dfs)
    shares = np.zeros((len(variants), len(cats)))
    for vi, v in enumerate(variants):
        top5 = dfs[v][dfs[v]["shap_rank"] <= 5]
        counts = top5["_cat"].value_counts()
        total = counts.sum()
        for ci, cat in enumerate(cats):
            shares[vi, ci] = counts.get(cat, 0) / total if total else 0

    y_pos = np.arange(len(variants))
    left = np.zeros(len(variants))
    for ci, cat in enumerate(cats):
        ax.barh(y_pos, shares[:, ci], left=left, color=cat_colors[cat],
                 edgecolor="white", linewidth=0.5, label=cat)
        for vi in range(len(variants)):
            v = shares[vi, ci]
            if v > 0.04:  # Only annotate slices > 4%
                ax.text(left[vi] + v / 2, y_pos[vi], f"{v*100:.0f}%",
                         ha="center", va="center", fontsize=8,
                         color="white" if cat in ("moments", "size", "network") else "black")
        left += shares[:, ci]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(variants)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of top-5 slots")
    ax.set_title("Feature-category mix per variant",
                  fontweight="bold")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9,
               title="Category", title_fontsize=9)


def _panel_feature_overlap_venn(ax, dfs: dict[str, pd.DataFrame], top_n=100):
    """Set-partition of the top-N most-frequent features in each variant.

    For N=2 → reports "shared" + "unique to A" + "unique to B".
    For N=3 → reports the standard 3-way Venn regions (all-3, three
              pairwise-only, three uniques).
    For N≥4 → reports "shared across all", "shared across ≥k-of-N" for
              each k, plus uniques. Generic across any variant count.

    Output is a text summary on the axes (Venn diagrams don't scale
    well past 3 sets — readable text is the better display).
    """
    variants = list(dfs)
    n = len(variants)
    feat_sets = {}
    for v in variants:
        top5 = dfs[v][dfs[v]["shap_rank"] <= 5]
        feat_sets[v] = set(top5["feature"].value_counts().head(top_n).index)

    # Build feature → set-of-variants-it-appears-in.
    membership: dict[str, set[str]] = {}
    for v, fs in feat_sets.items():
        for f in fs:
            membership.setdefault(f, set()).add(v)

    # Group features by their membership-set size + identity.
    by_size: dict[int, dict[frozenset, list[str]]] = {}
    for f, mem in membership.items():
        by_size.setdefault(len(mem), {}).setdefault(frozenset(mem), []).append(f)

    ax.axis("off")
    lines = [f"Top-{top_n} most-frequent features per variant",
             f"Membership partition across {n} variants:", ""]
    # Largest groups first (features in ALL variants, then N-1, etc.)
    for size in sorted(by_size, reverse=True):
        for mem, feats in sorted(by_size[size].items(),
                                  key=lambda kv: (-len(kv[1]), sorted(kv[0]))):
            mem_str = " ∩ ".join(sorted(mem))
            if size == n:
                label = f"all {n}: {mem_str}"
            elif size == 1:
                only = next(iter(mem))
                label = f"unique to {only}"
            else:
                label = f"{size} of {n}: {mem_str}"
            sample = ", ".join(sorted(feats)[:3])
            more = f" ... (+{len(feats) - 3} more)" if len(feats) > 3 else ""
            lines.append(f"  • {len(feats):>3d}  {label}")
            lines.append(f"        e.g. {sample}{more}")
    ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes,
             va="top", ha="left", fontsize=9, family="monospace")
    ax.set_title(f"Feature membership across variants (top-{top_n} each)",
                  fontweight="bold", loc="left")


# ---------------------------------------------------------------------------
def _render_modality_pages(pdf, dfs: dict[str, pd.DataFrame],
                             colors: dict[str, str], top_k: int,
                             modality_label: str):
    """Emit the 4 analysis pages for a single (already-filtered) dfs.

    Pages: (1) AUROC + Jaccard overlap, (2) dominant features per variant,
    (3) coverage + category mix, (4) Venn-style membership partition.
    Title-cases every page with `modality_label` so the multi-modality
    report reads cleanly.
    """
    sets = {v: _top_k_sets(df, top_k) for v, df in dfs.items()}
    common = (set.intersection(*[set(s) for s in sets.values()])
              if sets else set())
    n_variants = len(dfs)

    # Page 1 — AUROC + Jaccard
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    _panel_auroc(axes[0], dfs, colors)
    _panel_jaccard_heatmap(axes[1], dfs, common, sets)
    _panel_jaccard_distribution(axes[2], dfs, common, sets)
    fig.suptitle(f"[{modality_label}] {n_variants}-variant SHAP — "
                 f"quantitative similarity",
                  fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    # Page 2 — Top-features bar per variant
    ncols = min(3, n_variants)
    nrows = int(np.ceil(n_variants / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(6 * ncols, 6 * nrows), squeeze=False)
    for i, v in enumerate(dfs):
        ax = axes[i // ncols, i % ncols]
        _panel_top_features(ax, dfs[v], v, colors, top_k=20)
    for j in range(n_variants, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")
    fig.suptitle(f"[{modality_label}] Top-20 features dominating "
                  f"top-5 slots, per variant",
                  fontsize=12, fontweight="bold", y=1.00)
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    # Page 3 — coverage + category
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.0))
    _panel_coverage_curves(axes[0], dfs, colors)
    _panel_category_breakdown(axes[1], dfs)
    fig.suptitle(f"[{modality_label}] Where the signal concentrates",
                  fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    # Page 4 — Venn partition
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    _panel_feature_overlap_venn(ax, dfs, top_n=100)
    ax.set_title(f"[{modality_label}] Feature membership "
                  f"across variants (top-100 each)",
                  fontweight="bold", loc="left")
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def _summary_rows(dfs: dict[str, pd.DataFrame], modality_label: str) -> list[dict]:
    rows = []
    for v, df in dfs.items():
        if df.empty:
            continue
        top5 = df[df["shap_rank"] <= 5]
        feat_counts = top5["feature"].value_counts()
        n_class = top5.groupby(["gene", "_chan_key"], observed=True).ngroups
        cats = top5["_cat"].value_counts(normalize=True).to_dict()
        auroc_per_class = df.groupby(["gene", "_chan_key"], observed=True)["auroc"].first()
        rows.append({
            "modality": modality_label,
            "variant": v,
            "n_rows": len(df),
            "n_complexes": df["gene"].nunique(),
            "n_channels": df["viz_channel"].nunique(),
            "n_classifiers": n_class,
            "auroc_median": float(auroc_per_class.median()),
            "auroc_frac_ge_0p95": float((auroc_per_class >= 0.95).mean()),
            "n_unique_top_features": int(len(feat_counts)),
            "top_feature_share_pct": round(100 * feat_counts.iloc[0] / n_class, 1)
                if len(feat_counts) else 0.0,
            "top10_features": "; ".join(f"{f}({c})" for f, c in feat_counts.head(10).items()),
            **{f"cat_{k}_pct": round(100 * v_, 1) for k, v_ in cats.items()},
        })
    return rows


def _dominant_rows(dfs: dict[str, pd.DataFrame], modality_label: str) -> list[dict]:
    rows = []
    for v, df in dfs.items():
        if df.empty:
            continue
        top5 = df[df["shap_rank"] <= 5]
        n_class = top5.groupby(["gene", "_chan_key"], observed=True).ngroups
        for rank, (feat, cnt) in enumerate(
            top5["feature"].value_counts().head(20).items(), start=1
        ):
            rows.append({
                "modality": modality_label,
                "variant": v, "rank": rank, "feature": feat,
                "n_classifiers_in_top5": int(cnt),
                "share_pct": round(100 * cnt / n_class, 2),
                "category": _categorize(feat),
            })
    return rows


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--csv", action="append", type=_parse_csv_spec, default=None,
        help="VARIANT_NAME=PATH[@CONTRAST]. Repeat for each variant. "
             "Defaults to the 3 CHAD top-attention variants when omitted.",
    )
    p.add_argument(
        "--modality", choices=("both", "phase", "fluor", "all"),
        default="both",
        help="Which modality to analyze. `both` (default) emits two "
             "sections per metric (phase + fluor); `phase`/`fluor` emit "
             "just that one; `all` pools phase+fluor into one section.",
    )
    p.add_argument("--top-k", type=int, default=5,
                    help="K for top-K feature set Jaccard. Default 5.")
    p.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parents[3] /
                "outputs" / "ko_shap_analysis" / "chad_3way_breakdown.pdf",
        help="Output PDF path. Default: <repo>/outputs/ko_shap_analysis/.",
    )
    args = p.parse_args()

    # Resolve variant spec list. CLI --csv overrides the default 3.
    variant_specs = args.csv if args.csv else DEFAULT_VARIANTS

    dfs: dict[str, pd.DataFrame] = {}
    colors: dict[str, str] = {}
    for idx, (name, path, contrast) in enumerate(variant_specs):
        if not Path(path).exists():
            print(f"  [skip] {name}: missing {path}")
            continue
        df = _load(name, Path(path), contrast)
        if df.empty:
            print(f"  [skip] {name}: 0 rows after filter")
            continue
        dfs[name] = df
        colors[name] = _color_for(name, idx)
        print(f"  loaded {name}: {len(df):,} rows, {df['gene'].nunique()} complexes, "
              f"{df['viz_channel'].nunique()} channels  (color {colors[name]})")

    if not dfs:
        raise SystemExit("No variants loaded — provide --csv NAME=PATH[@CONTRAST] flags.")

    # Pick which modality sections to emit.
    if args.modality == "all":
        sections = [("all", dfs)]
    elif args.modality == "phase":
        sections = [("phase", {v: df[df["modality"] == "phase"]
                                for v, df in dfs.items()})]
    elif args.modality == "fluor":
        sections = [("fluor", {v: df[df["modality"] == "fluor"]
                                for v, df in dfs.items()})]
    else:  # both
        sections = [
            ("phase", {v: df[df["modality"] == "phase"] for v, df in dfs.items()}),
            ("fluor", {v: df[df["modality"] == "fluor"] for v, df in dfs.items()}),
        ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.output) as pdf:
        for label, sec_dfs in sections:
            # Drop any variants that have no rows for this modality —
            # otherwise downstream panels crash on empty inputs.
            sec_dfs = {v: df for v, df in sec_dfs.items() if not df.empty}
            if not sec_dfs:
                print(f"  [skip section] {label}: no variants have rows")
                continue
            n_class_per_v = {
                v: df.groupby(["gene","_chan_key"], observed=True).ngroups
                for v, df in sec_dfs.items()
            }
            print(f"\n[{label}] variants × classifiers: " +
                  ", ".join(f"{v}:{n}" for v, n in n_class_per_v.items()))
            _render_modality_pages(pdf, sec_dfs, colors,
                                     top_k=args.top_k,
                                     modality_label=label)
    print(f"\nWrote PDF: {args.output}")

    # Summary CSV — one row per (modality, variant). When user picks
    # `all`, one row per variant; otherwise two rows per variant.
    all_summary_rows: list[dict] = []
    all_dominant_rows: list[dict] = []
    for label, sec_dfs in sections:
        sec_dfs = {v: df for v, df in sec_dfs.items() if not df.empty}
        all_summary_rows.extend(_summary_rows(sec_dfs, label))
        all_dominant_rows.extend(_dominant_rows(sec_dfs, label))
    summary_path = args.output.with_suffix(".summary.csv")
    pd.DataFrame(all_summary_rows).to_csv(summary_path, index=False)
    print(f"Wrote summary: {summary_path}")
    dom_path = args.output.with_suffix(".dominant.csv")
    pd.DataFrame(all_dominant_rows).to_csv(dom_path, index=False)
    print(f"Wrote dominant-features: {dom_path}")


if __name__ == "__main__":
    main()
