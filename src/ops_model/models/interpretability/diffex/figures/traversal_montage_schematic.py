"""Figure 4 schematic: DiffAE counterfactual phenotype traversal (single row, 3 steps).

  1. Train DiffAE — semantic encoder -> semantic latent z (Cell-DINO); forward noise -> x_T;
     conditional U-Net denoiser reverses it (DDIM) back to the cell, conditioned on z.
  2. Pick the traversal destination — supervised NTC -> gene-KO direction from the top set-accuracy
     cells (kept cells solid; excluded cells faded).
  3. Traverse — decode the control cell along that direction (alpha 0 -> +5).

The z heat-strip shows the *values* of the semantic embedding vector (not categories).

  python traversal_montage_schematic.py
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Ellipse, FancyBboxPatch, Rectangle, Circle

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

INK = "#1a1a1a"; GREY = "#7a7a7a"
TEAL = "#0b6b73"; DORANGE = "#b35900"; PINK = "#d94f9a"
CELL_FILL = "#d7d7d7"; NUC = "#4a4a4a"


def _arrow(ax, x0, y0, x1, y1, lw=1.6, color=INK, mut=11):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=mut,
                                 lw=lw, color=color, shrinkA=0, shrinkB=0, zorder=7))


def _cell(ax, cx, cy, s, morph=0.0, edge=INK, lw=1.5):
    """Nucleus size (small→large) and speckle roundness (streaky→round) illustrate the phenotype shift."""
    ax.add_patch(Rectangle((cx - s, cy - s), 2 * s, 2 * s, facecolor=CELL_FILL, edgecolor=edge, lw=lw, zorder=4))
    nr = s * (0.30 + 0.36 * morph)
    ax.add_patch(Circle((cx, cy), nr, facecolor=NUC, edgecolor="none", zorder=5))
    rnd = 0.28 + 0.72 * morph; dw = s * 0.15
    for k in range(3):
        ang = 2.2 * k + 0.7; rr = nr * 0.5
        ax.add_patch(Ellipse((cx + rr * np.cos(ang), cy + rr * np.sin(ang)), 2 * dw, 2 * dw * rnd,
                            angle=35, facecolor="white", edgecolor="none", zorder=6))


def _noise_tile(ax, cx, cy, s, level=1.0, edge=INK, lw=1.5):
    ax.add_patch(Rectangle((cx - s, cy - s), 2 * s, 2 * s, facecolor=CELL_FILL, edgecolor="none", zorder=3))
    if level < 0.99:
        ax.add_patch(Circle((cx, cy), s * 0.5, facecolor=NUC, edgecolor="none", alpha=1 - level, zorder=4))
    ax.imshow(np.random.rand(14, 14), extent=[cx - s, cx + s, cy - s, cy + s], cmap="gray", vmin=0, vmax=1,
              alpha=level, zorder=5, aspect="auto", interpolation="nearest")
    ax.add_patch(Rectangle((cx - s, cy - s), 2 * s, 2 * s, fill=False, edgecolor=edge, lw=lw, zorder=6))


def _vec(ax, x, y, w=1.0, h=0.15, n=9):
    """Semantic embedding vector: cells shaded by value (grey shades = different numbers, not categories)."""
    for i, v in enumerate(np.random.rand(n)):
        ax.add_patch(Rectangle((x + i * w / n, y), w / n * 0.9, h, facecolor=str(0.22 + 0.6 * v),
                               edgecolor="white", lw=0.5, zorder=7))


def _cluster(ax, cx, cy, color, n_sel=6, n_un=8):
    """Kept (top set-accuracy) cells solid; excluded cells faded/hollow. Centroid = kept cells."""
    un = np.random.normal([cx, cy], 0.34, (n_un, 2))
    ax.scatter(un[:, 0], un[:, 1], s=20, facecolors="none", edgecolors=color, linewidths=1.1, alpha=0.4, zorder=3)
    sel = np.random.normal([cx, cy], 0.19, (n_sel, 2))
    ax.scatter(sel[:, 0], sel[:, 1], s=24, c=color, edgecolors="white", linewidths=0.4, zorder=4)
    ax.add_patch(Circle((cx, cy), 0.09, facecolor=color, edgecolor=INK, lw=1.3, zorder=6))


def build(outstem="traversal_montage_schematic"):
    fig, ax = plt.subplots(figsize=(15.2, 4.4))
    ax.set_xlim(0, 15.2); ax.set_ylim(0, 4.4); ax.axis("off")
    ax.text(0.12, 4.28, "B", fontsize=28, fontweight="bold", va="top")
    ax.text(0.68, 4.26, "Counterfactual phenotype traversal", fontsize=20, va="top")
    np.random.seed(1)
    hy = 3.55; yc = 2.25

    # ===== 1. Train DiffAE =====
    ax.text(0.3, hy, "1   Train DiffAE", fontsize=14, fontweight="bold", va="center")
    _cell(ax, 0.7, yc, 0.26, morph=1.0)
    ax.text(0.7, yc - 0.4, "real cell x0", fontsize=8.5, color=GREY, ha="center", va="top")
    _arrow(ax, 0.85, yc + 0.24, 1.22, yc + 0.5, lw=1.2, mut=8)
    ax.text(1.28, yc + 0.53, "z =", fontsize=11, ha="left", va="center", fontweight="bold", color="#a3266f", zorder=8)
    _vec(ax, 1.68, yc + 0.46, w=0.85)
    ax.text(2.1, yc + 0.9, "semantic latent (Cell-DINO)", fontsize=8.5, ha="center", color="#a3266f")
    _arrow(ax, 0.98, yc, 1.45, yc, lw=1.2, mut=8)
    ax.text(1.2, yc + 0.15, "noise", fontsize=7.5, color=GREY, ha="center")
    _noise_tile(ax, 1.72, yc, 0.26, level=1.0)
    ax.text(1.72, yc - 0.4, "x_T", fontsize=8.5, color=GREY, ha="center", va="top")
    _arrow(ax, 1.98, yc, 2.36, yc, lw=1.2, mut=8)
    _noise_tile(ax, 2.66, yc, 0.26, level=0.5)
    _arrow(ax, 2.92, yc, 3.3, yc, lw=1.2, mut=8)
    _cell(ax, 3.6, yc, 0.26, morph=1.0)                   # = α=0 control reconstruction in step 3
    ax.text(3.6, yc - 0.4, "generated cell", fontsize=8.5, color=GREY, ha="center", va="top")
    ax.add_patch(FancyArrowPatch((2.35, yc + 0.46), (2.64, yc + 0.32), arrowstyle="-|>", mutation_scale=8,
                                 lw=1.0, color=GREY, ls="--", shrinkA=0, shrinkB=2, zorder=6))
    ax.text(0.3, 1.28, "conditional U-Net denoises x_T → cell\n(reverse diffusion, DDIM), conditioned on z;\nfixed noise x_T carries cell identity",
            fontsize=8.5, color=GREY, ha="left", va="top")

    _arrow(ax, 4.15, yc, 4.75, yc, lw=1.8, mut=13)

    # ===== 2. Pick destination =====
    ax.text(5.0, hy, "2   Pick destination", fontsize=14, fontweight="bold", va="center")
    _cluster(ax, 5.7, yc - 0.05, GREY)
    _cluster(ax, 7.15, yc + 0.2, PINK)
    _arrow(ax, 5.88, yc, 6.95, yc + 0.17, lw=1.9)
    ax.text(5.7, yc - 0.52, "NTC", fontsize=9, ha="center", va="top", color=GREY)
    ax.text(7.62, yc + 0.2, "gene-KO", fontsize=9, ha="left", va="center", color="#a3266f")
    # legend: kept vs excluded
    ax.scatter([5.35], [1.5], s=24, c=PINK, edgecolors="white", linewidths=0.4, zorder=5)
    ax.text(5.5, 1.5, "top set-accuracy cells (kept → centroid)", fontsize=8, ha="left", va="center")
    ax.scatter([5.35], [1.22], s=20, facecolors="none", edgecolors=PINK, linewidths=1.1, alpha=0.6, zorder=5)
    ax.text(5.5, 1.22, "other cells (excluded)", fontsize=8, ha="left", va="center", color=GREY)
    ax.text(5.35, 0.9, "→ supervised NTC → gene-KO direction", fontsize=8.5, ha="left", color=INK)

    _arrow(ax, 8.5, yc, 9.1, yc, lw=1.8, mut=13)

    # ===== 3. Traverse =====
    ax.text(9.35, hy, "3   Traverse", fontsize=14, fontweight="bold", va="center")
    axs = np.linspace(9.9, 14.2, 6)
    for xx, a in zip(axs, [0, 1, 2, 3, 4, 5]):
        edge = PINK if a == 1 else DORANGE if a == 5 else INK
        _cell(ax, xx, yc, 0.34, morph=1 - a / 5.0, edge=edge, lw=2.6 if a in (1, 5) else 1.5)
        ax.text(xx, yc - 0.6, f"α = {a}" if a == 0 else f"+{a}", fontsize=10, ha="center")
    ax.text(axs[1], yc + 0.52, "gene-KO\ncentroid", fontsize=8.5, ha="center", va="bottom",
            color="#a3266f", fontweight="bold")
    ax.text(axs[5], yc + 0.52, "exaggerated\nphenotype", fontsize=8.5, ha="center", va="bottom",
            color=DORANGE, fontweight="bold")
    ax.annotate("", (axs[-1] + 0.42, 1.42), (axs[0] - 0.42, 1.42),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.7))
    ax.text(axs[0] - 0.42, 1.16, "NTC", fontsize=11, ha="left", color=GREY, fontweight="bold")
    ax.text(axs[-1] + 0.42, 1.16, "+KO", fontsize=11, ha="right", color=DORANGE, fontweight="bold")
    ax.text(12.05, 0.72, "α interpolates the control cell toward the knockdown state;  |α| > 1 exaggerates",
            fontsize=9.5, ha="center", color=GREY)

    for ext in ("png", "svg"):
        fig.savefig(f"{outstem}.{ext}", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[schematic] -> {outstem}.png / .svg")


if __name__ == "__main__":
    build()
