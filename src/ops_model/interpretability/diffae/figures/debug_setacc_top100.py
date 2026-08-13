"""Debug montages — top-N set-accuracy cells per figure group, rank-ordered with rank + cell-key
annotations, marker-global normalized, inverse blue mask. One montage per KO class and one per
(marker, block) NTC so specific KO/NTC cells can be picked into GENE_COLS/COMPLEX_COLS in
_setacc_common.py. Genes have up to ~1200 ranked cells; complexes only top-30.

Run: python debug_setacc_top100.py            # all groups (KO + NTC)
     python debug_setacc_top100.py TOMM20     # only groups whose top_label matches (arg substring)
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ops_model.interpretability.diffae.classifier.config import slugify
from _setacc_common import (COMPLEX_COLS, GENE_COLS, OUT, CROP_SIZE, materialize_class, seg_crop, composite)

plt.rcParams["pdf.fonttype"] = 42


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
        ax = axes.flat[k]
        ax.set_xticks([]); ax.set_yticks([])
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
        ax.text(0.5, -0.03, f"{key}\nconf={float(r['score']):.3f}", transform=ax.transAxes,
                fontsize=6.0, color="#222", va="top", ha="center")
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.997)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.965, bottom=0.01, wspace=0.05, hspace=0.32)
    os.makedirs(OUT, exist_ok=True)
    out = f"{OUT}/{out_stem}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {out}  ({n} cells)")


def montage(mc, ch, block, cls, title, out_stem, top_n=100):
    raw, recs = materialize_class(mc, ch, block, cls, top_n)
    render_montage(raw, recs, title, out_stem)


def main():
    filt = sys.argv[1] if len(sys.argv) > 1 else None
    cols = [c for c in GENE_COLS + COMPLEX_COLS if not filt or filt.lower() in c["top_label"].lower()]
    ntc_done = set()
    for c in cols:
        try:
            montage(c["mc"], c["ch"], c["block"], c["key"],
                    f"KO — {c['top_label']} ({c['mc']})   rank-ordered set-accuracy",
                    f"debug_KO_{c['slug']}_{slugify(c['key'])[:30]}")
        except Exception as e:
            print(f"skip KO {c['top_label']}: {e}")
        nk = (c["slug"], c["block"])
        if nk not in ntc_done:
            ntc_done.add(nk)
            try:
                montage(c["mc"], c["ch"], c["block"], "NTC",
                        f"NTC — {c['top_label']} marker ({c['mc']})   rank-ordered set-accuracy",
                        f"debug_NTC_{c['slug']}_{c['block']}")
            except Exception as e:
                print(f"skip NTC {c['slug']}: {e}")


if __name__ == "__main__":
    main()
