"""Generate per-gene violin plots of top SHAP features with caption underneath.

For each gene:
  - Up to 4 panels (Phase + 3 fluorescent channels), arranged in a 2×2 grid
  - Each panel shows split violins: gray = background cells, orange = this KO's top-attention cells
  - Caption from generate_shap_captions_combined.py printed below the grid

Usage:
    python scripts/ko_shap/generate_ko_violin_plots.py \\
        --cache-phase /path/to/phase_cache \\
        --cache-fluor /path/to/fluor_cache \\
        --features ko_shap_features.csv \\
        --captions ko_shap_captions.csv \\
        --out-dir violin_plots --zscore
"""

import argparse
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Visual constants
# ---------------------------------------------------------------------------

# Visual constants
BG_N    = 3000
KO_CLR  = "#e07b39"
BG_CLR  = "#aaaaaa"


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class ModalityCache:
    X:   np.ndarray          # memory-mapped (n_cells, n_features)
    obs: pd.DataFrame        # cell metadata
    fi:  dict[str, int]      # feature name → column index
    med: np.ndarray          # per-feature median (for z-score)
    std: np.ndarray          # per-feature global std (for z-score)

    @classmethod
    def load(cls, cache_dir: Path) -> "ModalityCache":
        print(f"Loading {cache_dir.name} ...")
        X   = np.load(cache_dir / "X.npy", mmap_mode="r")
        obs = pd.read_parquet(cache_dir / "obs.parquet")
        fnames = open(cache_dir / "features.txt").read().splitlines()
        fi  = {f: i for i, f in enumerate(fnames)}
        med = np.load(cache_dir / "median.npy")
        std = np.load(cache_dir / "global_std.npy").clip(1e-6)
        return cls(X=X, obs=obs, fi=fi, med=med, std=std)


# ---------------------------------------------------------------------------
# Feature label helpers
# ---------------------------------------------------------------------------

_ORG_PREFIXES = re.compile(
    r"^op_|^(phase2d_tubular|phase2d_vesicular(?:_dark)?|focus3d_tubular|"
    r"focus3d_vesicular(?:_dark)?|nucleoli_(?:focus3d|phase2d)|nucleoli|nuclei|cell|"
    r"gfp|fluor_unified|network_phase2d_tubular|network_focus3d_tubular|network|"
    r"cp_(?:nucleus|cell|cytoplasm))_",
    re.I,
)
_AGG_SUFFIX = re.compile(r"_(sum|mean|median|max|min|std|count)$", re.I)

_ORG_SHORT: dict[str, str] = {
    "phase2d_tubular":        "cell (ph2D)",
    "focus3d_tubular":        "cell (f3D)",
    "nucleoli_focus3d":       "nucleoli (f3D)",
    "nucleoli_phase2d":       "nucleoli (ph2D)",
    "focus3d_vesicular":      "vesicles (f3D)",
    "focus3d_vesicular_dark": "vacuoles (f3D)",
    "phase2d_vesicular":      "vesicles (ph2D)",
    "phase2d_vesicular_dark": "vacuoles (ph2D)",
    "nuclei":                 "nuclei",
    "cell":                   "cell",
    "fluor_unified":          "fluor",
    "gfp":                    "fluor",
}


def _feat_label(feat: str, organelle: str | None = None) -> str:
    """Short x-axis label: strip organelle prefixes, move agg suffix to line 2, prepend organelle."""
    m = feat
    for _ in range(4):
        prev = m
        m = _ORG_PREFIXES.sub("", m)
        if m == prev:
            break
    m = _AGG_SUFFIX.sub(r"\n(\1)", m)
    m = m.replace("_", " ")
    lines = m.split("\n")
    lines[0] = lines[0][:20]
    org_str = _ORG_SHORT.get(str(organelle), "") if organelle else ""
    if org_str:
        return org_str + "\n" + "\n".join(lines)
    return "\n".join(lines)


def _ch_label(viz_channel: str) -> str:
    """Shorten channel name: 'chaperones_HSPA1B' → 'HSPA1B'."""
    s = re.sub(r"_?(live.cell\s*dye|Live\s*Cell\s*Dye|excitation|live\s+cell\s+dye)", "",
               viz_channel, flags=re.I).strip()
    if "_" in s:
        s = s.split("_", 1)[1]
    return s[:30]


# ---------------------------------------------------------------------------
# Violin drawing helpers
# ---------------------------------------------------------------------------

def _violin(ax: plt.Axes, vals: np.ndarray, pos: float, color: str, median_color: str) -> None:
    vp = ax.violinplot([vals], positions=[pos], widths=0.38,
                       showmedians=True, showextrema=False)
    for pc in vp["bodies"]:  # type: ignore[union-attr]
        pc.set_facecolor(color)
        pc.set_alpha(0.65 if color == BG_CLR else 0.85)
        pc.set_linewidth(0.5)
    vp["cmedians"].set_color(median_color)
    vp["cmedians"].set_linewidth(1.8)


def _fetch(cache: ModalityCache, feat: str, idx: np.ndarray, zscore: bool) -> np.ndarray | None:
    fi = cache.fi.get(feat)
    if fi is None:
        return None
    v = np.asarray(cache.X[idx, fi], dtype=np.float64)
    v = v[np.isfinite(v)]
    if zscore:
        v = (v - cache.med[fi]) / cache.std[fi]
    return v if len(v) >= 5 else None


# ---------------------------------------------------------------------------
# Per-gene plot
# ---------------------------------------------------------------------------

def plot_gene(
    gene: str,
    feats_df: pd.DataFrame,
    gene_cap: dict[str, str],
    phase: ModalityCache,
    fluor: ModalityCache,
    top_n: int,
    zscore: bool,
    out_dir: Path,
    rng: np.random.Generator,
) -> Path | None:
    gf  = feats_df[feats_df["gene"] == gene]
    cap = gene_cap.get(gene, "")

    # Phase section
    ph_rows = gf[gf["modality"] == "phase"].sort_values("shap_rank").head(top_n)  # type: ignore[union-attr]
    ph_ko   = np.where(phase.obs["gene"] == gene)[0]
    ph_bg   = rng.choice(
        np.where(phase.obs["gene"] != gene)[0],
        size=min(BG_N, (phase.obs["gene"] != gene).sum()),
        replace=False,
    )

    # Fluor sections (skip noise-floor channels, effect_size < 0.2)
    fl_sections: list[tuple[str, float, pd.DataFrame, np.ndarray, np.ndarray]] = []
    for cr, grp in gf[gf["modality"] == "fluor"].groupby("channel_rank"):  # type: ignore[union-attr]
        grp = grp.sort_values("shap_rank").head(top_n)  # type: ignore[union-attr]
        if "effect_size" in grp.columns and grp["effect_size"].abs().max() < 0.2:
            continue
        auroc = float(grp["auroc"].iat[0]) if "auroc" in grp.columns else 0.0
        ch_label = _ch_label(str(grp["viz_channel"].iat[0]))
        cr = int(cr)  # type: ignore[arg-type]
        ko_mask = (fluor.obs["gene"] == gene) & (fluor.obs["channel_rank"] == cr) & (fluor.obs["rank_type"] == "top")
        bg_mask  = (fluor.obs["gene"] != gene) & (fluor.obs["channel_rank"] == cr) & (fluor.obs["rank_type"] == "top")
        f_ko = np.where(ko_mask)[0]
        f_bg = rng.choice(np.where(bg_mask)[0], size=min(BG_N, bg_mask.sum()), replace=False)
        fl_sections.append((ch_label, auroc, grp, f_ko, f_bg))

    sections = []
    if len(ph_rows):
        auroc_ph = float(ph_rows["auroc"].iat[0]) if "auroc" in ph_rows.columns else 0.0
        sections.append(("Phase", auroc_ph, ph_rows, ph_ko, ph_bg, False))
    for ch_label, auroc, grp, f_ko, f_bg in fl_sections:
        sections.append((ch_label, auroc, grp, f_ko, f_bg, True))

    if not sections:
        return None

    # Layout: 2-column grid + caption row
    n_sec  = len(sections)
    n_cols = min(n_sec, 2)
    n_rows = (n_sec + 1) // 2

    panel_w = top_n * 1.5
    fig = plt.figure(
        figsize=(n_cols * panel_w + 0.8, n_rows * 3.2 + 1.5),
        facecolor="white",
    )
    gs = gridspec.GridSpec(
        n_rows + 1, n_cols,
        figure=fig,
        height_ratios=[3.2] * n_rows + [1.5],
        hspace=0.55,
        wspace=0.35,
    )

    for si, (title, auroc, rows, ko_idx, bg_idx, is_fluor) in enumerate(sections):
        ri, ci = divmod(si, n_cols)
        ax = fig.add_subplot(gs[ri, ci])
        cache = fluor if is_fluor else phase

        x = 0
        tick_pos, tick_labels = [], []
        plotted = 0
        for _, row in rows.iterrows():
            feat = str(row["feature"])
            ko_v = _fetch(cache, feat, ko_idx, zscore)
            bg_v = _fetch(cache, feat, bg_idx, zscore)
            if ko_v is None or bg_v is None:
                continue
            _violin(ax, bg_v, x,        BG_CLR, "#777777")
            _violin(ax, ko_v, x + 0.45, KO_CLR, "#993300")
            tick_pos.append(x + 0.225)
            org = (row.get("organelle") if hasattr(row, "get") else None) if not is_fluor else None
            tick_labels.append(_feat_label(feat, str(org) if org is not None else None))
            x += 1.1
            plotted += 1

        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels, rotation=35, ha="right", fontsize=6.5)
        ax.set_ylabel("z-score" if zscore else "value", fontsize=6.5)
        ax.tick_params(axis="y", labelsize=6.5)
        ax.set_title(f"{title}  (AUROC {auroc:.2f})", fontsize=8.5, fontweight="bold", pad=4)
        ax.spines[["top", "right"]].set_visible(False)

        # Clip y-axis to 1st–99th percentile of all plotted data
        if plotted > 0:
            all_vals: list[float] = []
            for _, row in rows.iterrows():
                ko_v = _fetch(cache, str(row["feature"]), ko_idx, zscore)
                bg_v = _fetch(cache, str(row["feature"]), bg_idx, zscore)
                if ko_v is not None:
                    all_vals.extend(ko_v)
                if bg_v is not None:
                    all_vals.extend(bg_v)
            if all_vals:
                p1, p99 = np.percentile(all_vals, [1, 99])
                pad = (p99 - p1) * 0.15
                ax.set_ylim(p1 - pad, p99 + pad)
        else:
            ax.set_visible(False)

    # Legend on first visible panel
    axes = [a for a in fig.get_axes() if a.get_visible()]
    if axes:
        axes[0].legend(
            handles=[
                Patch(fc=BG_CLR, alpha=0.65, label="other KOs (background)"),
                Patch(fc=KO_CLR, alpha=0.85, label=f"{gene} KO"),
            ],
            fontsize=6.5,
            loc="upper right",
            framealpha=0.5,
            handlelength=1.2,
        )

    # Caption row
    cap_ax = fig.add_subplot(gs[n_rows, :])
    cap_ax.axis("off")
    body = cap.split(": ", 1)[1] if ": " in cap else cap
    wrapped = textwrap.fill(body, width=110)
    cap_ax.text(0.5, 0.9, wrapped, ha="center", va="top", fontsize=7.5,
                transform=cap_ax.transAxes, style="italic", color="#222222", linespacing=1.4)

    fig.suptitle(f"{gene} KO — top distinguishing features", fontsize=11, fontweight="bold", y=1.0)

    out = out_dir / f"{gene}_violin.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-phase", required=True, help="Phase feature cache directory")
    parser.add_argument("--cache-fluor", required=True, help="Fluor feature cache directory")
    parser.add_argument("--features",    required=True, help="ko_shap_features.csv")
    parser.add_argument("--captions",    required=True, help="ko_shap_captions.csv")
    parser.add_argument("--genes",    default="", help="comma-separated; empty = all")
    parser.add_argument("--top-n",   type=int, default=4, help="features per panel")
    parser.add_argument("--out-dir", required=True, help="Output directory for PNGs")
    parser.add_argument("--zscore",  action="store_true", help="z-score features before plotting")
    args = parser.parse_args()

    phase = ModalityCache.load(Path(args.cache_phase))
    fluor = ModalityCache.load(Path(args.cache_fluor))

    feats_df = pd.read_csv(Path(args.features))
    caps_df  = pd.read_csv(Path(args.captions))
    gene_cap = caps_df.set_index("gene")["caption"].to_dict()

    gene_filter = set(args.genes.split(",")) - {""} if args.genes else None
    genes = sorted(feats_df["gene"].unique())
    if gene_filter:
        genes = [g for g in genes if g in gene_filter]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    print(f"Plotting {len(genes)} genes → {out_dir}\n")
    for i, gene in enumerate(genes):
        try:
            out = plot_gene(
                gene, feats_df, gene_cap,
                phase, fluor,
                top_n=args.top_n,
                zscore=args.zscore,
                out_dir=out_dir,
                rng=rng,
            )
            status = out.name if out else "skipped"
        except Exception as e:
            status = f"ERROR: {e}"
        print(f"[{i+1:4d}/{len(genes)}] {gene:<12} {status}")


if __name__ == "__main__":
    main()
