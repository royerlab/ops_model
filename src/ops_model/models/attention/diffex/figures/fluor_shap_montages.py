"""SHAP-ranked fluor cell montages (v5/multi_rank) — top-N cells per fig-4 fluor group, rank-ordered with
rank + cell-key + SHAP annotations, marker-global normalized, inverse blue seg mask. One montage per KO
group + one per marker NTC, so cells can be hand-picked. ALSO writes per-marker SHAP rank parquets
(fluor_rank format) to a new _rankings/fluor_multirank/ dir.

Run: python fluor_shap_montages.py [GENE_substr]   # default all groups
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ops_model.models.attention.diffex.classifier.config import slugify
from _setacc_common import _materialize, seg_crop, composite, CROP_SIZE

MR = "/hpc/projects/icd.fast.ops/models/alex_lin_attention/v5/multi_rank/shap_screen/shap_screen_fluor_all.compact.parquet"
OUT = "/hpc/projects/icd.fast.ops/analysis/figure4_shap_montages"
PQ = "/hpc/projects/icd.fast.ops/models/diffex/viewer_assets_v5/_rankings/fluor_multirank/geneKO"
plt.rcParams["pdf.fonttype"] = 42

# fig-4 fluor groups: multi_rank channel_name -> (gene, zarr channel)
GROUPS = [
    ("autophagosome_MAP1LC3B", "ATG9A", "GFP"),
    ("actin filament_FastAct_SPY555 Live Cell Dye", "CAPZB", "mCherry"),
    ("ER/Golgi COP-II_SEC23A", "GBF1", "GFP"),
    ("lysosome_LysoTracker live-cell dye", "LAMTOR2", "GFP"),
    ("lipid droplet_BODIPY live cell dye", "RAB7A", "GFP"),
    ("clathrin vesicles_CLTA", "AP2M1", "GFP"),
    ("stress granule_G3BP1", "EIF2S2", "GFP"),
    ("chromatin_H2BC21", "AURKB", "mCherry"),
    ("nucleolus-DFC_FBL", "NOP56", "GFP"),
    ("lysosome_LAMP1", "ATP6V1B2", "GFP"),
    ("proteasome_PSMB7", "PSMB6", "GFP"),
    ("nucleolus-GC_NPM3", "POLR1B", "GFP"),
    ("nucleus_NucleoLIVE Live Cell dye", "KIF23", "mCherry"),
    ("mitochondria_ChromaLIVE 561 excitation", "TOMM20", "mCherry"),
    ("5xUPRE", "HSPA5", "GFP"),                                  # UPR reporter (set-acc panel group)
]
_MR = None


def _rows(channel_name, gene, n):
    """Top-n SHAP cells for (channel, gene) from the 'top' pool, rank-ordered. score = shap."""
    global _MR
    if _MR is None:
        _MR = pd.read_parquet(MR, columns=["gene", "channel_name", "rank", "shap", "_pool",
                                           "experiment", "well", "x_pheno", "y_pheno", "segmentation_id"])
    d = _MR[(_MR["channel_name"] == channel_name) & (_MR["gene"] == gene) & (_MR["_pool"] == "top")]
    d = d.sort_values("rank").head(n).copy()
    d["score"] = d["shap"]
    d["segmentation"] = d["segmentation_id"]                # make_labels_df expects 'segmentation'
    return d


def render_montage(raw, recs, title, out_stem):
    n = len(recs)
    lo, hi = np.percentile(raw, (1, 99))
    if hi - lo < 1e-6:
        hi = lo + 1
    half = CROP_SIZE // 2
    ncols = 10
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.9, nrows * 2.05), facecolor="white")
    axes = np.atleast_2d(axes)
    for k in range(nrows * ncols):
        ax = axes.flat[k]; ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        if k >= n:
            ax.axis("off"); continue
        r = recs.iloc[k]
        gray = np.clip((raw[k, 0] - lo) / (hi - lo), 0, 1) * 255
        seg = seg_crop(r["experiment"], r["well"], r["x_pheno"], r["y_pheno"], half)
        ax.imshow(composite(gray, seg, half))
        ax.text(0.03, 0.97, f"#{int(r['rank'])}", transform=ax.transAxes, fontsize=13, fontweight="bold",
                color="white", va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.15", fc="#c1272d", ec="none", alpha=0.9))
        key = f"{r['experiment']}/{r['well']} x{int(round(r['x_pheno']))} y{int(round(r['y_pheno']))}"
        ax.text(0.5, -0.03, f"{key}\nshap={float(r['score']):.3f}", transform=ax.transAxes,
                fontsize=6.0, color="#222", va="top", ha="center")
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.997)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.965, bottom=0.01, wspace=0.05, hspace=0.32)
    os.makedirs(OUT, exist_ok=True)
    out = f"{OUT}/{out_stem}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {out}  ({n} cells)", flush=True)


def montage(channel_name, gene, ch, title, out_stem, top_n=100):
    rows = _rows(channel_name, gene, top_n)
    if rows.empty:
        print(f"skip {gene} ({channel_name}): no SHAP rows"); return
    raw, recs = _materialize(rows, channel_name, ch, gene)
    render_montage(raw, recs, title, out_stem)


def write_parquet(channel_name, gene, ch):
    """Per-marker SHAP rank parquet (fluor_rank format: gene + base cols + channel + rank_type + rank), KO + NTC."""
    os.makedirs(PQ, exist_ok=True)
    ko = _rows(channel_name, gene, 200); ntc = _rows(channel_name, "NTC", 200)
    df = pd.concat([ko, ntc], ignore_index=True)
    df["channel"] = channel_name; df["rank_type"] = "top"
    cols = ["gene", "experiment", "well", "x_pheno", "y_pheno", "segmentation_id", "channel", "rank_type", "rank", "score"]
    out = f"{PQ}/{slugify(channel_name)}.parquet"
    df[cols].to_parquet(out)
    print(f"parquet {out}  (KO {len(ko)} + NTC {len(ntc)})", flush=True)


def main():
    filt = sys.argv[1] if len(sys.argv) > 1 else None
    ntc_done = set()
    for channel_name, gene, ch in GROUPS:
        if filt and filt.lower() not in gene.lower() and filt.lower() not in channel_name.lower():
            continue
        try:
            write_parquet(channel_name, gene, ch)
        except Exception as e:
            print(f"parquet skip {gene}: {type(e).__name__}: {e}")
        try:
            montage(channel_name, gene, ch, f"KO — {gene} ({channel_name})   SHAP-ranked (multi_rank)",
                    f"shap_KO_{slugify(channel_name)}_{gene}")
        except Exception as e:
            print(f"KO montage skip {gene}: {type(e).__name__}: {e}")
        if channel_name not in ntc_done:
            ntc_done.add(channel_name)
            try:
                montage(channel_name, "NTC", ch, f"NTC — {channel_name} marker   SHAP-ranked (multi_rank)",
                        f"shap_NTC_{slugify(channel_name)}")
            except Exception as e:
                print(f"NTC montage skip {channel_name}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
