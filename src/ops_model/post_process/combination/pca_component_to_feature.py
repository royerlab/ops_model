"""PCA loadings analysis for interpretable features (CellProfiler).

Parses CellProfiler feature names into (channel, compartment, category) and
produces per-PC and per-channel/reporter loading plots.

Feature name conventions handled:
  single_object_{channel}_{compartment}_{category}_{stat}
  coloc_{channel_a}_{channel_b}_{stat}
  {compartment}_{feature}   (shape / morphology, channel-free)

Outputs (written under output_dir/loadings/):
  top_features.csv                    — top-N loadings per PC, all PCs
  loadings_heatmap.png                — features × PCs heatmap (top features)
  {channel}/                          — one subdir per channel/reporter
    {channel}_loading_profile.png     — which PCs this channel dominates + top features

Usage
-----
  # Per-channel mode (-o must be the parent of per_channel/):
  python -m ops_model.post_process.combination.pca_loadings_analysis \\
      -o /hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized/cellprofiler

  # Downsampled mode (-o must be the parent of per_signal/, i.e. the downsampled/ subdir):
  python -m ops_model.post_process.combination.pca_loadings_analysis \\
      -o /hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized/cellprofiler/downsampled \\
      --downsampled
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TOP_N_PER_PC = 20          # features shown in per-PC bar plots
TOP_N_HEATMAP = 40         # features shown in heatmap (ranked by max |loading| across PCs)
MIN_CHANNEL_FEATURES = 3   # skip channels with fewer features than this


# ---------------------------------------------------------------------------
# Feature name parsing
# ---------------------------------------------------------------------------

def parse_cp_feature(name: str) -> dict:
    """Parse a CellProfiler feature name into components.

    Returns dict with keys: channel, compartment, category, stat, raw.
    """
    parts = name.split("_")

    # single_object_{channel}_{compartment}_{category}[_{stat}...]
    if parts[0] == "single" and len(parts) > 1 and parts[1] == "object":
        # parts: [single, object, channel, compartment, category, ...]
        if len(parts) >= 5:
            return {
                "channel":     parts[2],
                "compartment": parts[3],
                "category":    parts[4],
                "stat":        "_".join(parts[5:]) if len(parts) > 5 else "",
                "raw": name,
            }

    # coloc_{channel_a}_{channel_b}_{stat}
    if parts[0] == "coloc" and len(parts) >= 3:
        return {
            "channel":     f"coloc_{parts[1]}_{parts[2]}",
            "compartment": "coloc",
            "category":    "Colocalization",
            "stat":        "_".join(parts[3:]),
            "raw": name,
        }

    # {compartment}_{feature} — shape/morphology, no specific channel
    if len(parts) >= 2 and parts[0] in ("cell", "nucleus", "cytoplasm"):
        return {
            "channel":     "morphology",
            "compartment": parts[0],
            "category":    parts[1] if len(parts) > 1 else "unknown",
            "stat":        "_".join(parts[2:]),
            "raw": name,
        }

    # fallback
    return {"channel": "other", "compartment": "unknown",
            "category": "unknown", "stat": name, "raw": name}


def build_feature_table(feature_names: List[str]) -> pd.DataFrame:
    """Parse all feature names into a structured DataFrame."""
    rows = [parse_cp_feature(n) for n in feature_names]
    df = pd.DataFrame(rows)
    df.index = feature_names
    return df


# ---------------------------------------------------------------------------
# Category colour palette
# ---------------------------------------------------------------------------

_CATEGORY_PALETTE = {
    "Intensity":          "#E65100",
    "Texture":            "#6A1B9A",
    "RadialDistribution": "#1565C0",
    "Granularity":        "#2E7D32",
    "Colocalization":     "#F9A825",
    "AreaShape":          "#558B2F",
    "Area":               "#558B2F",
    "Morphology":         "#558B2F",
    "NormalizedMoment":   "#00838F",
    "CentralMoment":      "#00838F",
    "SpatialMoment":      "#00838F",
    "HuMoment":           "#00838F",
    "InertiaTensor":      "#00838F",
    "other":              "#9E9E9E",
    "unknown":            "#9E9E9E",
}


def _cat_colour(cat: str) -> str:
    for key in _CATEGORY_PALETTE:
        if key.lower() in cat.lower():
            return _CATEGORY_PALETTE[key]
    return _CATEGORY_PALETTE["other"]


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def analyze_pca_loadings(
    loadings: np.ndarray,
    feature_names: List[str],
    explained_variance_ratio: np.ndarray,
    output_dir: Path,
    file_prefix: str,
    top_n: int = TOP_N_PER_PC,
) -> None:
    """Generate PCA loadings analysis plots and CSV.

    Parameters
    ----------
    loadings : np.ndarray, shape (n_pcs, n_features)
        PCA components (pca_model.components_[:peak_n]).
    feature_names : list of str
        Original feature names in the same order as loadings columns.
    explained_variance_ratio : np.ndarray, shape (n_pcs,)
        Per-PC variance ratio (not cumulative).
    output_dir : Path
        Root output directory — a `loadings/` subdir is created here.
    file_prefix : str
        Prefix for saved files.
    top_n : int
        Number of top features to show per PC.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_pcs, n_feats = loadings.shape
    if n_feats != len(feature_names):
        logger.warning(f"  Loadings shape mismatch: {n_feats} vs {len(feature_names)} features")
        return

    # Cap to PCs that explain at least 2% variance
    MIN_VAR = 0.02
    if len(explained_variance_ratio) >= n_pcs:
        keep = np.where(explained_variance_ratio[:n_pcs] >= MIN_VAR)[0]
        if len(keep) == 0:
            logger.warning(f"  No PCs explain ≥{MIN_VAR:.0%} variance — using top 3")
            keep = np.arange(min(3, n_pcs))
        if len(keep) < n_pcs:
            logger.info(f"  Keeping {len(keep)}/{n_pcs} PCs with ≥{MIN_VAR:.0%} variance explained")
        loadings = loadings[keep]
        explained_variance_ratio = explained_variance_ratio[keep]
        n_pcs = len(keep)

    loadings_dir = output_dir / "loadings"
    loadings_dir.mkdir(parents=True, exist_ok=True)

    feat_df = build_feature_table(feature_names)
    abs_loadings = np.abs(loadings)  # (n_pcs, n_feats)

    # ------------------------------------------------------------------
    # 1. Top features CSV
    # ------------------------------------------------------------------
    rows = []
    for pc_i in range(n_pcs):
        order = np.argsort(abs_loadings[pc_i])[::-1][:top_n]
        for rank, feat_idx in enumerate(order):
            info = feat_df.iloc[feat_idx]
            rows.append({
                "pc": pc_i,
                "rank": rank + 1,
                "feature": feature_names[feat_idx],
                "loading": float(loadings[pc_i, feat_idx]),
                "abs_loading": float(abs_loadings[pc_i, feat_idx]),
                "channel": info["channel"],
                "compartment": info["compartment"],
                "category": info["category"],
                "var_explained": float(explained_variance_ratio[pc_i]) if pc_i < len(explained_variance_ratio) else 0.0,
            })
    top_df = pd.DataFrame(rows)
    top_df.to_csv(loadings_dir / f"{file_prefix}_top_features.csv", index=False)

    # ------------------------------------------------------------------
    # 2. Loadings heatmap (top features by max |loading| across all PCs)
    # ------------------------------------------------------------------
    max_abs = abs_loadings.max(axis=0)
    top_feat_idx = np.argsort(max_abs)[::-1][:TOP_N_HEATMAP]
    heatmap_data = loadings[:, top_feat_idx]   # (n_pcs, TOP_N_HEATMAP)
    heatmap_labels = [feature_names[i] for i in top_feat_idx]

    fig_h = max(5, n_pcs * 0.3 + 1)
    fig_w = max(10, TOP_N_HEATMAP * 0.22 + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(heatmap_data, aspect="auto", cmap="RdBu_r",
                   vmin=-abs_loadings.max(), vmax=abs_loadings.max())
    ax.set_xticks(range(len(heatmap_labels)))
    ax.set_xticklabels(heatmap_labels, rotation=90, fontsize=5)
    ax.set_yticks(range(n_pcs))
    ax.set_yticklabels(
        [f"PC{i} ({explained_variance_ratio[i]:.1%})" if i < len(explained_variance_ratio) else f"PC{i}"
         for i in range(n_pcs)],
        fontsize=7,
    )
    plt.colorbar(im, ax=ax, label="Loading", shrink=0.6)
    ax.set_title(f"{file_prefix} — PCA loadings heatmap (top {TOP_N_HEATMAP} features)",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(loadings_dir / f"{file_prefix}_loadings_heatmap.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved loadings/loadings_heatmap.png")

    # ------------------------------------------------------------------
    # 3. Per-channel subdirs
    # ------------------------------------------------------------------
    channels = feat_df["channel"].unique()
    for channel in sorted(channels):
        ch_mask = feat_df["channel"] == channel
        ch_feat_idx = np.where(ch_mask.values)[0]
        if len(ch_feat_idx) < MIN_CHANNEL_FEATURES:
            continue

        ch_dir = loadings_dir / channel
        ch_dir.mkdir(exist_ok=True)

        ch_abs = abs_loadings[:, ch_feat_idx]     # (n_pcs, n_ch_feats)
        ch_loadings = loadings[:, ch_feat_idx]
        ch_names = [feature_names[i] for i in ch_feat_idx]
        ch_feat_info = feat_df.iloc[ch_feat_idx]

        # --- Per-PC bar chart grid: one subplot per PC, top features with sign + magnitude ---
        n_cols = 4
        n_rows_grid = int(np.ceil(n_pcs / n_cols))
        fig, axes = plt.subplots(
            n_rows_grid, n_cols,
            figsize=(n_cols * 4.5, n_rows_grid * 3.2),
            gridspec_kw={"hspace": 0.55, "wspace": 0.45},
        )
        axes_flat = np.array(axes).flatten()

        for pc_i in range(n_pcs):
            ax = axes_flat[pc_i]
            # Top features for this PC from this channel, sorted by |loading|
            order = np.argsort(ch_abs[pc_i])[::-1][:top_n]
            feat_labels = [ch_names[j].replace(f"single_object_{channel}_", "")
                           for j in order]
            values = [ch_loadings[pc_i, j] for j in order]
            colours = ["#E65100" if v > 0 else "#1565C0" for v in values]

            # Plot in descending |loading| order (top at top)
            y_pos = range(len(values) - 1, -1, -1)
            ax.barh(list(y_pos), values, color=colours, alpha=0.8, height=0.7)
            ax.set_yticks(list(y_pos))
            ax.set_yticklabels(feat_labels, fontsize=5.5)
            ax.axvline(0, color="black", lw=0.7)
            ax.tick_params(axis="x", labelsize=6)

            var_str = (f" — {explained_variance_ratio[pc_i]:.1%} var"
                       if pc_i < len(explained_variance_ratio) else "")
            ax.set_title(f"PC{pc_i}{var_str}", fontsize=8, fontweight="bold")
            ax.set_xlabel("Loading", fontsize=6.5)

        # Hide unused subplots
        for ax in axes_flat[n_pcs:]:
            ax.set_visible(False)

        fig.suptitle(f"{file_prefix} — {channel} ({len(ch_feat_idx)} features, top {top_n} per PC)",
                     fontsize=11, fontweight="bold", y=1.01)
        fig.savefig(ch_dir / f"{channel}_loading_profile.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

        # Per-channel top features CSV
        ch_rows = []
        for pc_i in range(n_pcs):
            order = np.argsort(ch_abs[pc_i])[::-1][:top_n]
            for rank, local_i in enumerate(order):
                feat_i = ch_feat_idx[local_i]
                info = feat_df.iloc[feat_i]
                ch_rows.append({
                    "pc": pc_i, "rank": rank + 1,
                    "feature": feature_names[feat_i],
                    "loading": float(loadings[pc_i, feat_i]),
                    "abs_loading": float(abs_loadings[pc_i, feat_i]),
                    "compartment": info["compartment"],
                    "category": info["category"],
                })
        pd.DataFrame(ch_rows).to_csv(ch_dir / f"{channel}_top_features.csv", index=False)

    logger.info(f"  Saved loadings/ with {len([c for c in channels if (loadings_dir / c).exists()])} channel subdirs")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
# Cross-signal summary
# ---------------------------------------------------------------------------

def _plot_cross_signal_summary(
    signal_loadings: dict,   # signal -> {"loadings": ndarray, "feature_names": list, "var_ratio": ndarray}
    output_dir: Path,
) -> None:
    """Summarise PC→feature-category mapping across all signals.

    For each PC, computes the fraction of total |loading| weight contributed
    by each feature category and compartment, averaged across signals.
    Highlights consistent themes (low cross-signal variance = reliable pattern).

    Produces:
      loadings/summary_pc_categories.png   — stacked bar: PC × category
      loadings/summary_pc_compartments.png — stacked bar: PC × compartment
      loadings/summary_pc_top_features.png — heatmap: PC × top feature (weighted mean |loading|)
      loadings/summary_cross_signal.csv    — raw category fractions per signal per PC
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import get_cmap

    loadings_dir = output_dir / "loadings"
    loadings_dir.mkdir(parents=True, exist_ok=True)

    if not signal_loadings:
        return

    # Collect all category/compartment fractions per (signal, pc)
    records = []
    feature_weight_accumulator = {}  # feature_name -> array over signals of mean |loading|

    # Find max n_pcs across signals
    max_pcs = max(d["loadings"].shape[0] for d in signal_loadings.values())

    for signal, data in signal_loadings.items():
        L = np.abs(data["loadings"])          # (n_pcs, n_feats)
        feat_df = build_feature_table(data["feature_names"])
        n_pcs_s = L.shape[0]

        for pc_i in range(n_pcs_s):
            total = L[pc_i].sum() or 1.0
            for cat in feat_df["category"].unique():
                mask = feat_df["category"] == cat
                frac = L[pc_i, mask.values].sum() / total
                records.append({"signal": signal, "pc": pc_i,
                                 "dim": "category", "label": cat, "fraction": frac})
            for comp in feat_df["compartment"].unique():
                mask = feat_df["compartment"] == comp
                frac = L[pc_i, mask.values].sum() / total
                records.append({"signal": signal, "pc": pc_i,
                                 "dim": "compartment", "label": comp, "fraction": frac})

        # Accumulate per-feature mean |loading| across PCs (max across PCs = importance)
        feat_importance = L.max(axis=0)   # (n_feats,)
        for feat_name, imp in zip(data["feature_names"], feat_importance):
            if feat_name not in feature_weight_accumulator:
                feature_weight_accumulator[feat_name] = []
            feature_weight_accumulator[feat_name].append(float(imp))

    df = pd.DataFrame(records)
    df.to_csv(loadings_dir / "summary_cross_signal.csv", index=False)

    def _stacked_bar_summary(dim: str, fname: str, title: str) -> None:
        sub = df[df["dim"] == dim]
        # Mean fraction per (pc, label) across signals
        pivot = sub.groupby(["pc", "label"])["fraction"].mean().unstack(fill_value=0)
        # Limit to PCs present in majority of signals
        pivot = pivot[pivot.index < max_pcs]

        # Sort columns by total weight descending
        pivot = pivot[pivot.sum().sort_values(ascending=False).index]

        n_pcs_plot = len(pivot)
        n_cats = len(pivot.columns)
        colours = [get_cmap("tab20")(i / max(n_cats, 1)) for i in range(n_cats)]

        fig, (ax_main, ax_std) = plt.subplots(
            2, 1, figsize=(max(8, n_pcs_plot * 0.7), 8),
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.35},
        )

        # Stacked bar — mean fraction per PC
        bottom = np.zeros(n_pcs_plot)
        for i, col in enumerate(pivot.columns):
            vals = pivot[col].values
            ax_main.bar(range(n_pcs_plot), vals, bottom=bottom,
                        color=colours[i], label=col, width=0.75, alpha=0.88)
            bottom += vals

        ax_main.set_xticks(range(n_pcs_plot))
        ax_main.set_xticklabels([f"PC{i}" for i in pivot.index], fontsize=8)
        ax_main.set_ylabel("Mean fraction of |loading|", fontsize=9)
        ax_main.set_title(title, fontsize=11, fontweight="bold")
        ax_main.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left", framealpha=0.8)
        ax_main.set_ylim(0, 1)

        # Consistency bar — std across signals per PC (lower = more consistent)
        std_per_pc = sub.groupby(["pc", "label"])["fraction"].std().unstack(fill_value=0)
        std_per_pc = std_per_pc.reindex(pivot.index, fill_value=0)
        mean_std = std_per_pc.mean(axis=1)
        consistency = 1.0 - (mean_std / mean_std.max().clip(1e-6))
        ax_std.bar(range(n_pcs_plot), consistency.values,
                   color="#2E7D32", alpha=0.7, width=0.75)
        ax_std.set_xticks(range(n_pcs_plot))
        ax_std.set_xticklabels([f"PC{i}" for i in pivot.index], fontsize=8)
        ax_std.set_ylabel("Consistency\n(1 − norm.std)", fontsize=8)
        ax_std.set_ylim(0, 1)
        ax_std.axhline(0.7, color="grey", lw=0.8, ls="--", alpha=0.6)
        ax_std.set_title("Cross-signal consistency (higher = same theme across reporters)",
                          fontsize=8)

        fig.tight_layout()
        fig.savefig(loadings_dir / fname, dpi=160, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  Saved loadings/{fname}")

    _stacked_bar_summary("category",    "summary_pc_categories.png",
                          f"PC → Feature Category ({len(signal_loadings)} signals)")
    _stacked_bar_summary("compartment", "summary_pc_compartments.png",
                          f"PC → Compartment ({len(signal_loadings)} signals)")

    # Cross-signal top feature heatmap — mean |loading| across signals
    mean_importance = {f: np.mean(v) for f, v in feature_weight_accumulator.items()}
    top_feats = sorted(mean_importance, key=mean_importance.get, reverse=True)[:TOP_N_HEATMAP]

    # Build PC × top_feature matrix (mean |loading| across signals that have the feature)
    heatmap = np.zeros((max_pcs, len(top_feats)))
    for signal, data in signal_loadings.items():
        L = np.abs(data["loadings"])
        fn = data["feature_names"]
        feat_idx_map = {f: i for i, f in enumerate(fn)}
        for fi, feat in enumerate(top_feats):
            if feat in feat_idx_map:
                src_i = feat_idx_map[feat]
                for pc_i in range(min(data["loadings"].shape[0], max_pcs)):
                    heatmap[pc_i, fi] += L[pc_i, src_i]
    # Normalise by number of signals
    n_signals = len(signal_loadings)
    heatmap /= n_signals

    short_labels = [f.replace("single_object_", "")[:35] for f in top_feats]
    fig_w = max(12, len(top_feats) * 0.22 + 2)
    fig_h = max(4, max_pcs * 0.3 + 1)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(heatmap, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(top_feats)))
    ax.set_xticklabels(short_labels, rotation=90, fontsize=5)
    ax.set_yticks(range(max_pcs))
    ax.set_yticklabels([f"PC{i}" for i in range(max_pcs)], fontsize=7)
    plt.colorbar(im, ax=ax, label="Mean |loading| across signals", shrink=0.6)
    ax.set_title(f"Mean |loading| per PC across {n_signals} signals — top {len(top_feats)} features",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(loadings_dir / "summary_pc_top_features.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved loadings/summary_pc_top_features.png")


# ---------------------------------------------------------------------------

def main():
    import argparse
    import anndata as ad

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="PCA loadings analysis from existing per-channel/signal h5ads"
    )
    parser.add_argument(
        "-o", "--output-dir", type=str,
        default="/hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized/cellprofiler",
        help="Feature-type output dir (e.g. .../pca_optimized/cellprofiler)",
    )
    parser.add_argument(
        "--downsampled", action="store_true",
        help="Read from per_signal/ instead of per_channel/",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    subdir = "per_signal" if args.downsampled else "per_channel"
    per_unit_dir = output_dir / subdir

    guide_files = sorted(per_unit_dir.glob("*_guide.h5ad"))
    if not guide_files:
        print(f"ERROR: no guide h5ads found in {per_unit_dir}")
        return

    print(f"Analysing loadings for {len(guide_files)} h5ads in {per_unit_dir}")
    n_done, n_skip = 0, 0
    signal_loadings = {}  # collected for cross-signal summary

    for gf in guide_files:
        file_prefix = gf.stem.replace("_guide", "")
        g = ad.read_h5ad(gf)
        components  = g.uns.get("pca_components")
        feat_names  = g.uns.get("pca_feature_names")
        if components is None or feat_names is None:
            logger.warning(f"  {file_prefix}: no loadings in uns — re-run sweep to populate")
            n_skip += 1
            continue
        var_ratio = g.uns.get("pca", {}).get("variance_ratio")
        if var_ratio is None:
            var_ratio = np.ones(len(components)) / len(components)
        L = np.asarray(components)
        vr = np.asarray(var_ratio)
        try:
            analyze_pca_loadings(
                loadings=L,
                feature_names=feat_names,
                explained_variance_ratio=vr,
                output_dir=per_unit_dir,
                file_prefix=file_prefix,
            )
            signal_loadings[file_prefix] = {
                "loadings": L, "feature_names": feat_names, "var_ratio": vr,
            }
            n_done += 1
        except Exception as e:
            logger.warning(f"  {file_prefix}: failed — {e}")
            n_skip += 1

    print(f"Done: {n_done} analysed, {n_skip} skipped")

    if len(signal_loadings) >= 2:
        print("Generating cross-signal summary plots...")
        try:
            _plot_cross_signal_summary(signal_loadings, per_unit_dir)
        except Exception as e:
            logger.warning(f"Cross-signal summary failed: {e}")


if __name__ == "__main__":
    main()
