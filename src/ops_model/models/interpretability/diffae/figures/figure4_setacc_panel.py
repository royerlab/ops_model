"""Figure 4 C/D-style panel — top *set-accuracy* real cells (fluorescence), KO vs NTC grid.

Mimics the published fig-4 C/D layout (gene-KO panel + protein-complex panel, KO row over NTC row,
marker label under each column) but the cells shown are hand-picked from the v5 SetTransformer
set-accuracy rankings (ko_rank / ntc_rank per column in _setacc_common.GENE_COLS/COMPLEX_COLS;
pick them from the debug_setacc_top100.py montages). Real cells, cropped on demand, marker-global
normalized so KO vs NTC brightness is comparable, with the inverse blue seg mask (C/D look).
Vector output (SVG + PNG).

Run: python figure4_setacc_panel.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _setacc_common import COMPLEX_COLS, GENE_COLS, OUT, column_tiles

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"


def make_panel(columns, panel_title, out_stem, tile=1.6, tiles_fn=column_tiles, bottom_caption=None,
               title_in=0.55, pert_bottom=False, top_caption=None):
    cols = []
    for c in columns:
        try:
            ko_im, ntc_im, kc, nc = tiles_fn(c)
            cols.append((c, ko_im, ntc_im, kc, nc))
        except Exception as e:
            print(f"skip col {c.get('top_label')} ({c['slug']}): {e}")
    if not cols:
        print(f"no columns for {out_stem}")
        return
    n = len(cols)

    left, right = 0.05, 0.997
    bot_in = 0.5 if pert_bottom else (0.40 if bottom_caption else 0.10)   # room for the perturbation label under the KO row
    W = n * tile / (right - left)
    H = 2 * tile + title_in + bot_in                   # square cells (tile x tile) → images tile tight
    fig = plt.figure(figsize=(W, H), facecolor="white")
    gs = fig.add_gridspec(2, n, hspace=0.02, wspace=0.02, left=left, right=right,
                          top=1 - title_in / H, bottom=bot_in / H)
    for j, (c, ko_im, ntc_im, kc, nc) in enumerate(cols):
        for i, im in enumerate((ntc_im, ko_im)):            # NTC on top, KO on bottom
            ax = fig.add_subplot(gs[i, j])
            ax.imshow(im)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor("#888"); s.set_linewidth(0.5)
            if pert_bottom:                                  # phase: no top title, perturbation label under the KO row
                if i == 1 and c.get("top_label"):
                    ax.set_xlabel(c["top_label"], fontsize=11, fontweight="bold")
            elif i == 0:
                ax.set_title(c.get("marker_label") or c["top_label"], fontsize=11, fontweight="bold", pad=4)   # marker on top
            elif c.get("marker_label"):
                ax.set_xlabel(c["top_label"], fontsize=11, fontweight="bold")                                  # KO/gene name on bottom
            if j == 0:
                ax.set_ylabel("NTC" if i == 0 else "KO", fontsize=11, fontweight="bold", rotation=0,
                              labelpad=14, va="center")
    fig.suptitle(panel_title, fontsize=13, fontweight="bold", x=left, ha="left", va="top", y=0.995)
    if top_caption:
        fig.text(0.5, 1 - title_in / H + 0.012, top_caption, fontsize=11, ha="center", va="bottom", style="italic")
    if bottom_caption:
        fig.text(0.5, 0.4 * bot_in / H, bottom_caption, fontsize=11, ha="center", style="italic")
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(f"{OUT}/{out_stem}.{ext}", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {OUT}/{out_stem}.png/.svg  (" +
          ", ".join(f"{c['top_label']}: KO#{c['ko_rank']}={kc:.2f}/NTC#{c['ntc_rank']}={nc:.2f}"
                    for c, _, _, kc, nc in cols) + ")")


if __name__ == "__main__":
    make_panel(GENE_COLS, "Gene KO  top-predictive cells (fluorescence)", "panelC_geneKO_setacc", title_in=0.66)
    make_panel(COMPLEX_COLS, "Protein complex  top-predictive cells (fluorescence)", "panelD_complex_setacc", title_in=0.66)
