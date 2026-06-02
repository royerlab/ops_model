"""Cross-approach SHAP comparison.

Reads N SHAP feature CSVs (each from a different approach: top-attention
distinct/ntc/global, all-cells ntc/global, etc.) and emits a PDF +
summary CSV that surface:

  1. AUROC distribution per approach — flags saturated classifiers
     (AUROC≈1.0 across the board → trivially-separable pipeline
     fingerprint, not real biology).
  2. Top-N most-frequent features per approach — surfaces feature
     dominance (a small set hogging most SHAP top slots = low
     biological information).
  3. Pairwise top-5 feature Jaccard heatmap — quantifies which
     approaches converge on the same answer.
  4. Per-(gene, channel) feature variability — how many distinct
     features show up across approaches at the same row.
  5. Expected-hit coverage — for an optional `--expected-genes` list,
     a heatmap of best AUROC per (gene × approach) and a CSV with
     each gene's top-3 features under each approach.

Designed to be approach-agnostic: just pass `--csv name=path` pairs.
Filters by `--contrast` (for CSVs that store multiple contrasts in
one file, like ntc_shap_features.csv with contrast={ntc, global}).

Usage:
  # CHAD-level: 5 approaches
  python shap_approach_compare.py \\
      --csv top-attn-distinct=/hpc/.../top20_v4_chad/ko_shap_features.csv \\
      --csv top-attn-ntc=/hpc/.../top20_v4_chad_ntc/ko_shap_features.csv \\
      --csv top-attn-global=/hpc/.../top20_v4_chad_global/ko_shap_features.csv \\
      --csv all-cells-ntc=/hpc/.../ntc_v2_chad/ntc_shap_features.csv@ntc \\
      --csv all-cells-global=/hpc/.../ntc_v2_chad/ntc_shap_features.csv@global \\
      --output /tmp/shap_compare_chad.pdf

  # Gene-level w/ expected hits
  python shap_approach_compare.py \\
      --csv top-attn-distinct=/hpc/.../top20_v4/ko_shap_features.csv \\
      --csv all-cells-ntc=/hpc/.../ntc_v2/ntc_shap_features.csv@ntc \\
      --csv all-cells-global=/hpc/.../ntc_v2/ntc_shap_features.csv@global \\
      --expected-genes TIMM23,TIMM44,TOMM20,HSPA5,ERN1,EIF2AK3,RPL26,NOP56,...
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


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
    return name.strip(), Path(path).expanduser(), contrast


def _load_approach(name: str, path: Path, contrast: str | None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if contrast and "contrast" in df.columns:
        df = df[df["contrast"].astype(str) == contrast].copy()
    df["_approach"] = name
    # Canonical channel key for cross-approach matching. The two SHAP
    # pipelines (ko_shap_features.py vs ntc_shap_features.py) write the
    # `viz_channel` with different casing — `5xUPRE` vs `5xupre`,
    # `ER/Golgi COP-II_SEC23A` vs `er/golgi cop-ii_sec23a` etc — so
    # without normalization the intersection of approaches collapses
    # to a single channel ("Phase" — the only one that round-trips
    # identically through both pipelines).
    df["_chan_key"] = df["viz_channel"].astype(str).str.lower().str.strip()
    return df


def _top_per_row(df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """Keep top_k SHAP features per (gene, viz_channel)."""
    return (df.sort_values(["gene", "_chan_key", "shap_rank"])
              .groupby(["gene", "_chan_key"], observed=True)
              .head(top_k))


def _normalize_gene(s: str) -> str:
    """Loose canonicalization to recover from minor typos in expected lists."""
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())


def _feature_sets_by_row(df: pd.DataFrame, top_k: int) -> dict[tuple, set[str]]:
    sub = _top_per_row(df, top_k)
    return {(g, c): set(grp["feature"])
            for (g, c), grp in sub.groupby(["gene", "_chan_key"], observed=True)}


def _auroc_per_row(df: pd.DataFrame) -> dict[tuple, float]:
    return (df.groupby(["gene", "_chan_key"], observed=True)["auroc"]
              .first().to_dict())


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _violin_auroc(ax, approaches: list[str], auroc_lists: dict[str, list[float]]):
    parts = ax.violinplot(
        [auroc_lists[a] for a in approaches],
        positions=range(1, len(approaches) + 1), widths=0.7,
        showmeans=False, showextrema=False, showmedians=True,
    )
    for body in parts["bodies"]:
        body.set_alpha(0.7); body.set_edgecolor("black")
    ax.set_xticks(range(1, len(approaches) + 1))
    ax.set_xticklabels(approaches, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("AUROC per (gene × channel) classifier")
    ax.set_title("AUROC distribution per approach")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
    ax.axhline(1.0, color="red",  linestyle=":", linewidth=0.8)
    ax.set_ylim(0.4, 1.05)
    for i, a in enumerate(approaches, start=1):
        med = float(np.median(auroc_lists[a]))
        sat = float((np.asarray(auroc_lists[a]) >= 0.95).mean())
        ax.text(i, med + 0.01, f"med={med:.2f}\n≥0.95: {sat:.0%}",
                ha="center", fontsize=8)


def _bar_feature_dominance(ax, df: pd.DataFrame, name: str, top_features: int = 20):
    top5 = _top_per_row(df, 5)
    counts = top5["feature"].value_counts().head(top_features)
    n_rows = top5.groupby(["gene","viz_channel"], observed=True).ngroups
    ax.barh(range(len(counts)), counts.values[::-1], color="#4472C4", alpha=0.8)
    ax.set_yticks(range(len(counts)))
    ax.set_yticklabels(counts.index[::-1], fontsize=7)
    ax.set_xlabel("# (gene × channel) classifiers with this feature in top-5")
    ax.set_title(f"{name}\n(top {top_features} dominant features; "
                 f"{n_rows} classifiers total)", fontsize=10)
    # Annotate the top bar with its share
    top_share = counts.iloc[0] / n_rows if n_rows else 0
    ax.text(0.98, 0.02, f"top-feature share: {100*top_share:.1f}%",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            bbox=dict(facecolor="white", edgecolor="gray", alpha=0.8))


def _jaccard_heatmap(ax, sets_by_approach: dict[str, dict[tuple, set]],
                      approaches: list[str]):
    n = len(approaches)
    mat = np.zeros((n, n))
    for i, a in enumerate(approaches):
        for j, b in enumerate(approaches):
            common = set(sets_by_approach[a]) & set(sets_by_approach[b])
            if not common:
                mat[i, j] = np.nan; continue
            js = []
            for k in common:
                sa, sb = sets_by_approach[a][k], sets_by_approach[b][k]
                u = sa | sb
                if u: js.append(len(sa & sb) / len(u))
            mat[i, j] = float(np.mean(js)) if js else np.nan
    im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(approaches, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(approaches, fontsize=9)
    ax.set_title("Pairwise top-5 Jaccard (mean across shared classifiers)")
    for i in range(n):
        for j in range(n):
            v = mat[i, j]
            txt = "—" if np.isnan(v) else f"{v:.2f}"
            color = "white" if (np.isnan(v) or v < 0.5) else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    color=color, fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="mean Jaccard")


def _expected_hits_heatmap(ax, df_by_approach: dict[str, pd.DataFrame],
                            expected: list[str], approaches: list[str]):
    """Heatmap: rows=expected genes, cols=approaches, color=best AUROC across channels."""
    rows = []
    norm_lookup = {_normalize_gene(g): g for g in expected}
    for g_norm, g_orig in norm_lookup.items():
        row = []
        for a in approaches:
            df = df_by_approach[a]
            # Find matching gene rows. Match by exact then normalized.
            mask = df["gene"].astype(str).map(_normalize_gene) == g_norm
            if not mask.any():
                row.append(np.nan); continue
            best = float(df.loc[mask, "auroc"].max())
            row.append(best)
        rows.append(row)
    mat = np.array(rows, dtype=float)
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0.5, vmax=1.0, aspect="auto")
    ax.set_yticks(range(len(expected)))
    ax.set_yticklabels(expected, fontsize=8)
    ax.set_xticks(range(len(approaches)))
    ax.set_xticklabels(approaches, rotation=30, ha="right", fontsize=9)
    ax.set_title("Best AUROC per expected hit × approach\n"
                 "(saturated ≈1.0 means trivially separable; "
                 "≈0.5 means no signal)")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            txt = "—" if np.isnan(v) else f"{v:.2f}"
            color = "white" if (np.isnan(v) or v < 0.7) else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    color=color, fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="best AUROC")


# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--csv", action="append", required=True, type=_parse_csv_spec,
                    help="name=path[@contrast]. Repeat for each approach.")
    p.add_argument("--top-k", type=int, default=5,
                    help="Top-K SHAP rows per (gene × channel) for set comparisons.")
    p.add_argument("--top-features", type=int, default=20,
                    help="How many dominant features to show in bar charts.")
    p.add_argument("--expected-genes", default=None,
                    help="Comma-separated gene list to highlight; matched "
                         "loosely (case + non-alnum stripped) against CSV gene "
                         "names so minor typos still resolve.")
    p.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parents[3] /
                "outputs" / "ko_shap_analysis" / "shap_approach_compare.pdf",
        help="Output PDF path. Default: <repo>/outputs/ko_shap_analysis/. "
             "Also writes a sibling .summary.csv.",
    )
    args = p.parse_args()

    # Load
    approaches: list[str] = []
    dfs: dict[str, pd.DataFrame] = {}
    for name, path, contrast in args.csv:
        if not path.exists():
            print(f"  [skip] {name}: missing {path}", flush=True)
            continue
        try:
            df = _load_approach(name, path, contrast)
            if not len(df):
                print(f"  [skip] {name}: empty after filter", flush=True)
                continue
            dfs[name] = df
            approaches.append(name)
            print(f"  loaded {name}: {len(df):,} rows, "
                  f"{df['gene'].nunique()} genes, "
                  f"{df['viz_channel'].nunique()} channels", flush=True)
        except Exception as e:
            print(f"  [err]  {name}: {e}", flush=True)

    if not approaches:
        raise SystemExit("No approaches loaded.")

    # Per-row AUROC + feature sets
    auroc_per = {a: _auroc_per_row(dfs[a]) for a in approaches}
    sets_per  = {a: _feature_sets_by_row(dfs[a], args.top_k) for a in approaches}

    # Common (gene, channel) tuples across ALL approaches → fair comparison
    common = set.intersection(*[set(sets_per[a]) for a in approaches])
    print(f"\n(gene × channel) tuples in ALL approaches: {len(common):,}")

    auroc_lists = {
        a: [float(auroc_per[a][k]) for k in common if k in auroc_per[a]]
        for a in approaches
    }

    # ── PDF ─────────────────────────────────────────────────────────────
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.output) as pdf:
        # Page 1 — AUROC + Jaccard
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        _violin_auroc(axes[0], approaches, auroc_lists)
        _jaccard_heatmap(axes[1], sets_per, approaches)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # Page 2 — feature dominance, one panel per approach (3-col grid)
        n = len(approaches)
        ncols = min(3, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols,
                                  figsize=(6 * ncols, 5 * nrows), squeeze=False)
        for i, a in enumerate(approaches):
            ax = axes[i // ncols, i % ncols]
            _bar_feature_dominance(ax, dfs[a], a, top_features=args.top_features)
        # blank unused panels
        for j in range(len(approaches), nrows * ncols):
            axes[j // ncols, j % ncols].axis("off")
        fig.suptitle(f"Top-{args.top_features} dominant features per approach "
                     f"(out of top-{args.top_k} SHAP rows per classifier)",
                     fontsize=12, fontweight="bold", y=1.005)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # Page 3 — expected hits coverage (only if provided)
        if args.expected_genes:
            expected = [g.strip() for g in args.expected_genes.split(",") if g.strip()]
            fig, ax = plt.subplots(
                figsize=(2 + 1.2 * len(approaches), max(6, 0.32 * len(expected)))
            )
            _expected_hits_heatmap(ax, dfs, expected, approaches)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    print(f"\nWrote PDF: {args.output}")

    # ── Summary CSV ─────────────────────────────────────────────────────
    summary_rows = []
    for a in approaches:
        df = dfs[a]
        top5 = _top_per_row(df, args.top_k)
        n_class = top5.groupby(["gene","viz_channel"], observed=True).ngroups
        feat_counts = top5["feature"].value_counts()
        n_unique_feats = len(feat_counts)
        top_feat_share = float(feat_counts.iloc[0]) / n_class if n_class else 0
        # Top-10 most-frequent features for this approach
        top10 = "; ".join(f"{f}({c})" for f, c in feat_counts.head(10).items())
        summary_rows.append({
            "approach": a,
            "n_rows": len(df),
            "n_genes": df["gene"].nunique(),
            "n_channels": df["viz_channel"].nunique(),
            "n_classifiers": n_class,
            "auroc_median": float(np.median(auroc_lists[a])),
            "auroc_fraction_ge_0p95": float(
                (np.asarray(auroc_lists[a]) >= 0.95).mean()
            ),
            "unique_top_features": n_unique_feats,
            "top_feature_share_pct": round(100 * top_feat_share, 1),
            "top10_features": top10,
        })
    summary_path = args.output.with_suffix(".summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"Wrote summary: {summary_path}")

    # Expected-hit per-gene per-approach detail (optional)
    if args.expected_genes:
        expected = [g.strip() for g in args.expected_genes.split(",") if g.strip()]
        rows = []
        norm = {_normalize_gene(g): g for g in expected}
        for g_norm, g_orig in norm.items():
            for a in approaches:
                df = dfs[a]
                mask = df["gene"].astype(str).map(_normalize_gene) == g_norm
                if not mask.any():
                    rows.append({"expected_gene": g_orig, "approach": a,
                                 "matched_gene": None,
                                 "n_channels_with_data": 0,
                                 "best_auroc": np.nan,
                                 "top3_features": ""})
                    continue
                sub = df.loc[mask]
                matched = sub["gene"].astype(str).iloc[0]
                best = sub.sort_values(
                    ["auroc","shap_importance"], ascending=False
                ).head(3)
                rows.append({
                    "expected_gene": g_orig,
                    "matched_gene":  matched,
                    "approach":      a,
                    "n_channels_with_data": sub["viz_channel"].nunique(),
                    "best_auroc":    float(sub["auroc"].max()),
                    "top3_features": " | ".join(
                        f"{r['viz_channel']}:{r['feature']}"
                        for _, r in best.iterrows()
                    ),
                })
        hit_path = args.output.with_suffix(".expected_hits.csv")
        pd.DataFrame(rows).to_csv(hit_path, index=False)
        print(f"Wrote expected-hit detail: {hit_path}")


if __name__ == "__main__":
    main()
