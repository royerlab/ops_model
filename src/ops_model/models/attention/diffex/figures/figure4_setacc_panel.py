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


def make_panel(columns, panel_title, out_stem, tile=1.6, tiles_fn=column_tiles, bottom_caption=None):
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

    fig = plt.figure(figsize=(n * tile + 0.7, 2 * tile + 1.1), facecolor="white")
    gs = fig.add_gridspec(2, n, hspace=0.04, wspace=0.04, left=0.06, right=0.995, top=0.86, bottom=0.11)
    for j, (c, ko_im, ntc_im, kc, nc) in enumerate(cols):
        for i, im in enumerate((ko_im, ntc_im)):
            ax = fig.add_subplot(gs[i, j])
            ax.imshow(im)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor("#888"); s.set_linewidth(0.5)
            if i == 0:
                ax.set_title(c["top_label"], fontsize=11, fontweight="bold", pad=4)
            elif c.get("marker_label"):
                ax.set_xlabel(c["marker_label"], fontsize=8.5)
            if j == 0:
                ax.set_ylabel("KO" if i == 0 else "NTC", fontsize=11, fontweight="bold", rotation=0,
                              labelpad=14, va="center")
    fig.suptitle(panel_title, fontsize=13, fontweight="bold", x=0.06, ha="left", y=0.965)
    if bottom_caption:
        fig.text(0.5, 0.045, bottom_caption, fontsize=11, ha="center", style="italic")
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(f"{OUT}/{out_stem}.{ext}", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {OUT}/{out_stem}.png/.svg  (" +
          ", ".join(f"{c['top_label']}: KO#{c['ko_rank']}={kc:.2f}/NTC#{c['ntc_rank']}={nc:.2f}"
                    for c, _, _, kc, nc in cols) + ")")


if __name__ == "__main__":
    make_panel(GENE_COLS, "Gene KO  top set-accuracy cells (fluorescence)", "panelC_geneKO_setacc")
    make_panel(COMPLEX_COLS, "Protein complex  top set-accuracy cells (fluorescence)", "panelD_complex_setacc")
