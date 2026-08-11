"""Figure schematic: spatially-conditioned DiffAE for multi-channel virtual staining (phase → marker).

Variant of traversal_montage_schematic.py (same visual language) for the cross-channel task:

  1. Spatially-conditioned DiffAE — a label-free Phase2D crop conditions the generator TWO ways:
       - semantic z: phase → frozen Cell-DINO (ViT, patch-tokenise → pooled 1024-d) → global FiLM
         conditioning (+ learned marker id) — content, but NO spatial layout;
       - dense pixels: the raw phase image concatenated as an extra U-Net input channel (in_channels 1→2)
         — keeps the layout, so the output stays pixel-registered to the input (the lift 0.13 → 0.78).
     The conditional U-Net denoises x_T → the chosen fluorescent marker.
  2. Render any marker — from the SAME phase cell, switching the marker id renders every trained marker,
     each co-registered to the input; compared to the real marker (held-out Pearson).

One model over 42 live markers; trained on paired (phase, marker) crops from phenotyping_v3.zarr (both
channels in one stitched frame → registered). 4i/CP markers excluded (fixed later → misregistered).

  python virtual_staining_schematic.py
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, Circle, Ellipse

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

INK = "#1a1a1a"; GREY = "#7a7a7a"; PURPLE = "#a3266f"; TEAL = "#0b6b73"
PHASE_FILL = "#d7d7d7"; NUC = "#4a4a4a"
# marker channel colours (distinct organelles rendered from the same phase)
MARKERS = [("mito", "#e8791a"), ("ER", "#2ca089"), ("nucleus", "#3a6fd8"), ("lysosome", "#d94f4f")]


def _arrow(ax, x0, y0, x1, y1, lw=1.6, color=INK, mut=11, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=mut,
                                 lw=lw, color=color, ls=ls, shrinkA=0, shrinkB=0, zorder=7))


def _phase_cell(ax, cx, cy, s, edge=INK, lw=1.5):
    """Label-free phase cell: grey square, dark nucleus, faint internal texture (no fluor signal)."""
    ax.add_patch(Rectangle((cx - s, cy - s), 2 * s, 2 * s, facecolor=PHASE_FILL, edgecolor=edge, lw=lw, zorder=4))
    ax.add_patch(Circle((cx, cy), s * 0.42, facecolor=NUC, edgecolor="none", alpha=0.85, zorder=5))
    for k in range(4):
        ang = 1.7 * k + 0.4; rr = s * 0.55
        ax.add_patch(Ellipse((cx + rr * np.cos(ang), cy + rr * np.sin(ang)), s * 0.34, s * 0.16,
                             angle=40 * k, facecolor="#c2c2c2", edgecolor="none", zorder=5))


def _marker_tile(ax, cx, cy, s, kind, color, edge=INK, lw=1.5):
    """Stylised fluorescent marker on black: a characteristic organelle pattern in the channel colour."""
    ax.add_patch(Rectangle((cx - s, cy - s), 2 * s, 2 * s, facecolor="black", edgecolor=edge, lw=lw, zorder=4))
    rng = np.random.default_rng(abs(hash(kind)) % 2**32)
    if kind == "mito":                                   # tubular streaks
        for _ in range(9):
            a = rng.uniform(0, np.pi); p = rng.uniform(-0.6, 0.6, 2) * s
            ax.add_patch(Ellipse((cx + p[0], cy + p[1]), s * 0.5, s * 0.13, angle=np.degrees(a),
                                 facecolor=color, edgecolor="none", alpha=0.9, zorder=5))
    elif kind == "ER":                                   # reticular mesh
        for _ in range(11):
            p = rng.uniform(-0.7, 0.7, 2) * s
            ax.add_patch(Circle((cx + p[0], cy + p[1]), s * 0.18, facecolor="none",
                                edgecolor=color, lw=1.0, alpha=0.85, zorder=5))
    elif kind == "nucleus":                              # filled nuclear blob
        ax.add_patch(Ellipse((cx, cy), s * 1.1, s * 0.9, facecolor=color, edgecolor="none", alpha=0.85, zorder=5))
    else:                                                # puncta (lysosome)
        for _ in range(13):
            p = rng.uniform(-0.75, 0.75, 2) * s
            ax.add_patch(Circle((cx + p[0], cy + p[1]), s * 0.09, facecolor=color, edgecolor="none", zorder=5))


def _noise_tile(ax, cx, cy, s, level=1.0, edge=INK, lw=1.5):
    ax.add_patch(Rectangle((cx - s, cy - s), 2 * s, 2 * s, facecolor="black", edgecolor="none", zorder=3))
    ax.imshow(np.random.rand(14, 14), extent=[cx - s, cx + s, cy - s, cy + s], cmap="gray", vmin=0, vmax=1,
              alpha=level, zorder=5, aspect="auto", interpolation="nearest")
    ax.add_patch(Rectangle((cx - s, cy - s), 2 * s, 2 * s, fill=False, edgecolor=edge, lw=lw, zorder=6))


def _vec(ax, x, y, w=1.0, h=0.15, n=9, seed=0):
    """Semantic embedding vector: cells shaded by value."""
    rng = np.random.default_rng(seed)
    for i, v in enumerate(rng.random(n)):
        ax.add_patch(Rectangle((x + i * w / n, y), w / n * 0.9, h, facecolor=str(0.22 + 0.6 * v),
                               edgecolor="white", lw=0.5, zorder=7))


def _marker_selector(ax, x, y, w, sel_i):
    """Row of marker-id tokens; the selected one highlighted (which channel to render)."""
    n = len(MARKERS); cw = w / n
    for i, (name, col) in enumerate(MARKERS):
        on = i == sel_i
        ax.add_patch(FancyBboxPatch((x + i * cw, y), cw * 0.82, 0.22, boxstyle="round,pad=0.01,rounding_size=0.03",
                                    facecolor=col if on else "white", edgecolor=col, lw=1.6 if on else 1.0, zorder=7))
        ax.text(x + i * cw + cw * 0.41, y + 0.11, name, fontsize=6.8, ha="center", va="center",
                color="white" if on else col, fontweight="bold" if on else "normal", zorder=8)


def _patch_cell(ax, cx, cy, s, edge=INK):
    """Phase cell overlaid with a faint patch grid — the ViT tokenisation used only for the semantic branch."""
    _phase_cell(ax, cx, cy, s, edge=edge, lw=1.3)
    for k in range(1, 4):
        g = -s + 2 * s * k / 4
        ax.plot([cx - s, cx + s], [cy + g, cy + g], color=PURPLE, lw=0.5, alpha=0.55, zorder=7)
        ax.plot([cx + g, cx + g], [cy - s, cy + s], color=PURPLE, lw=0.5, alpha=0.55, zorder=7)


def build(outstem="/hpc/projects/icd.fast.ops/analysis/figure4_schematic/virtual_staining_schematic"):
    fig, ax = plt.subplots(figsize=(14.6, 4.9))
    ax.set_xlim(0, 14.6); ax.set_ylim(0, 4.9); ax.axis("off")
    ax.text(0.12, 4.72, "C", fontsize=28, fontweight="bold", va="top")
    ax.text(0.68, 4.70, "Multi-channel virtual staining  (label-free phase → fluorescent marker)",
            fontsize=19, va="top")
    np.random.seed(3)
    hy = 3.95; yc = 2.15

    # ===== 1. Spatially-conditioned DiffAE (encode + generate) =====
    ax.text(0.3, hy, "1   Spatially-conditioned DiffAE", fontsize=14, fontweight="bold", va="center")

    # phase cell (shared input to BOTH conditioning paths)
    _phase_cell(ax, 0.8, yc, 0.30)
    ax.text(0.8, yc - 0.46, "phase cell", fontsize=8.5, color=GREY, ha="center", va="top")

    # --- semantic branch (up): phase -> Cell-DINO ViT -> pooled z (global, no layout) ---
    _arrow(ax, 1.0, yc + 0.28, 1.45, yc + 0.72, lw=1.1, mut=8, color=PURPLE)
    _patch_cell(ax, 1.75, yc + 0.95, 0.19)
    _arrow(ax, 1.98, yc + 0.95, 2.35, yc + 0.95, lw=1.0, mut=7, color=PURPLE)
    ax.add_patch(FancyBboxPatch((2.4, yc + 0.72), 0.95, 0.46, boxstyle="round,pad=0.02,rounding_size=0.05",
                                facecolor="white", edgecolor=PURPLE, lw=1.3, zorder=6))
    ax.text(2.88, yc + 0.95, "Cell-DINO\n(ViT, frozen)", fontsize=7.6, ha="center", va="center", color=PURPLE, zorder=7)
    _arrow(ax, 3.35, yc + 0.95, 3.65, yc + 0.95, lw=1.0, mut=7, color=PURPLE)
    _vec(ax, 3.7, yc + 0.87, w=0.72, seed=2)
    ax.text(4.06, yc + 1.32, "z: pooled 1024-d (semantic, no spatial layout)", fontsize=7.8, ha="center", color=PURPLE)
    ax.text(1.68, yc + 0.62, "patch-tokenise → pool", fontsize=6.8, ha="center", color=PURPLE, style="italic")

    # --- spatial branch (down): phase raw pixels concatenated to the noisy input = SPATIAL CONDITIONING ---
    _arrow(ax, 1.02, yc - 0.26, 1.42, yc - 0.5, lw=1.1, mut=8, color=TEAL)
    ax.text(1.28, yc - 0.62, "raw pixels", fontsize=7.0, color=TEAL, ha="center", va="top", style="italic")
    # boxed, labelled callout so it's unmistakable which element is the spatial conditioning:
    # the U-Net INPUT = [ noisy x_T  ⊕  phase image ] (2 channels)
    ax.add_patch(FancyBboxPatch((1.78, yc - 1.02), 1.02, 1.16, boxstyle="round,pad=0.02,rounding_size=0.05",
                                facecolor="#e7f2f1", edgecolor=TEAL, lw=1.9, zorder=3))
    ax.text(2.29, yc + 0.30, "SPATIAL CONDITIONING", fontsize=7.8, color=TEAL, ha="center", va="center",
            fontweight="bold", zorder=8)
    _noise_tile(ax, 2.14, yc - 0.32, 0.185); ax.text(2.44, yc - 0.32, "x_T", fontsize=6.8, color=GREY, ha="left", va="center", zorder=8)
    # circled-plus (concat) between the two channel tiles
    ax.add_patch(Circle((2.14, yc - 0.58), 0.058, facecolor="white", edgecolor=TEAL, lw=1.1, zorder=8))
    ax.plot([2.104, 2.176], [yc - 0.58, yc - 0.58], color=TEAL, lw=1.0, zorder=9)
    ax.plot([2.14, 2.14], [yc - 0.616, yc - 0.544], color=TEAL, lw=1.0, zorder=9)
    _phase_cell(ax, 2.14, yc - 0.84, 0.185, edge=TEAL, lw=1.3); ax.text(2.44, yc - 0.84, "phase", fontsize=6.8, color=TEAL, ha="left", va="center", zorder=8)
    ax.text(2.29, yc + 0.05, "2-channel U-Net input", fontsize=6.6, color=TEAL, ha="center", va="center", zorder=8)
    _arrow(ax, 2.82, yc - 0.5, 3.2, yc - 0.42, lw=1.3, mut=9, color=TEAL)

    # U-Net
    ax.add_patch(FancyBboxPatch((3.25, yc - 0.68), 1.2, 1.02, boxstyle="round,pad=0.02,rounding_size=0.06",
                                facecolor="#f2f2f2", edgecolor=INK, lw=1.7, zorder=5))
    ax.text(3.85, yc - 0.02, "conditional\nU-Net", fontsize=9, ha="center", va="center", fontweight="bold", zorder=6)
    ax.text(3.85, yc - 0.46, "DDIM", fontsize=7.5, ha="center", va="center", color=GREY, zorder=6)

    # z (+ marker id) -> U-Net as FiLM class-embedding (from top)
    _arrow(ax, 4.05, yc + 0.82, 3.95, yc + 0.36, lw=1.1, mut=8, color=PURPLE, ls="--")
    ax.text(4.2, yc + 0.6, "z + marker id\n(FiLM)", fontsize=7.4, color=PURPLE, ha="left", va="center", fontweight="bold")

    # denoise -> predicted marker
    _arrow(ax, 4.5, yc - 0.02, 4.95, yc - 0.02, lw=1.3, mut=10)
    _marker_tile(ax, 5.3, yc, 0.30, "mito", MARKERS[0][1])
    ax.text(5.3, yc - 0.48, "predicted marker", fontsize=8.5, color=GREY, ha="center", va="top")

    # marker-id selector
    ax.text(0.35, 0.55, "marker id:", fontsize=8.5, ha="left", va="center", fontweight="bold")
    _marker_selector(ax, 1.2, 0.43, 2.4, sel_i=0)
    # the two conditioning roles, spelled out
    ax.text(6.0, yc + 0.95, "semantic z  →  global FiLM conditioning\n(what to render — content, no layout)",
            fontsize=8.1, color=PURPLE, ha="left", va="center")
    ax.text(6.0, yc - 0.55, "phase pixels  →  dense concat into U-Net input\n(keeps layout → output pixel-registered to input)",
            fontsize=8.1, color=TEAL, ha="left", va="center", fontweight="bold")

    _arrow(ax, 8.7, yc, 9.35, yc, lw=1.8, mut=13)

    # ===== 2. Render any marker =====
    ax.text(9.55, hy, "2   Render any marker", fontsize=14, fontweight="bold", va="center")
    ax.text(9.55, hy - 0.36, "same phase cell → switch marker id", fontsize=8.3, color=GREY, va="center")
    xs = np.linspace(10.1, 14.2, 4)
    rs = [0.78, 0.74, 0.81, 0.69]
    for i, (xx, (name, col)) in enumerate(zip(xs, MARKERS)):
        _marker_tile(ax, xx, yc + 0.15, 0.33, name, col, edge=col, lw=2.2)
        ax.text(xx, yc + 0.62, name, fontsize=9, ha="center", color=col, fontweight="bold")
        ax.text(xx, yc - 0.32, f"r = {rs[i]:.2f}", fontsize=8.2, ha="center", color=INK)
    ax.text(12.15, yc - 0.62, "held-out Pearson(pred, real)", fontsize=8, ha="center", color=GREY)
    ax.annotate("", (xs[-1] + 0.45, 1.15), (xs[0] - 0.45, 1.15),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.5))
    ax.text(12.15, 0.85, "one model over 42 live markers (dyes + FP tags), all from a single phase image",
            fontsize=8.6, ha="center", color=INK)

    import os
    os.makedirs(os.path.dirname(outstem), exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(f"{outstem}.{ext}", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[schematic] -> {outstem}.png / .svg")


if __name__ == "__main__":
    build()
