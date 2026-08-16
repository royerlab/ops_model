"""FINAL Figure-4 deliverable (for review before Confluence upload): 3 groups (mTOR/POLR1B/TIM23), each with
raw + normalized violins (α0,0.5,1,1.5,2,2.5,3) and a remade cell-grid image panel (real KO/NTC + gen α0,1,2 — α3 dropped there).

Normalization is now a SINGLE real-NTC baseline shared by every category (real NTC, real KO, gen α0/α1/α2) —
NOT the old per-modality scheme (gen normalized to its own α0). Valid now because we've validated gen α0 ≈
real NTC for these exact (group, feature) pairs — so one shared ruler is meaningful, not just a workaround.

Reads already-computed stats.npz/panel.npz from _native/<dir>/ — no SLURM, no re-measurement, pure rendering.

Run: python bruno_final.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import find_boundaries

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

NAT = "/hpc/projects/icd.fast.ops/analysis/figure4_traversals_violin/_native"
OUT = "/hpc/projects/icd.fast.ops/analysis/figure4_traversals_violin/bruno"
C = {"real": "#999999", "KO": "#2e8b57", "α=0": "#c6dbef", "α=0.5": "#9ecae1", "α=1": "#6baed6",
    "α=1.5": "#4292c6", "α=2": "#3182bd", "α=2.5": "#1c5c94", "α=3": "#08519c"}
LABS = ["real NTC", "real KO", "α=0", "α=0.5", "α=1", "α=1.5", "α=2", "α=2.5", "α=3"]
ALPHAS = (0, 0.5, 1, 1.5, 2, 2.5, 3)
PX_UM = 0.325                                          # native phenotyping_v3 pixel size (shared by phase + fluor channels, same instrument/FOV)
AREA_FEATS = {"area"}                                  # features measured in px² (spacing=(1,1) in _measure) → convert to µm² for raw display

# (group, source_dir, [(feature, display_label, unit_label), ...])
GROUPS = [
    ("mTOR", "mtor_mo_hm_100", [("location", "Radial position", "(normalized radial position)"),
                                 ("area", "Lysosome area", "(µm²)")]),
    ("POLR1B", "polr1b_vsnpm3_100cpu", [("circularity", "Nucleolar circularity", ""),
                                        ("aspect_ratio", "Nucleolar aspect ratio", "(major/minor axis)"),
                                        ("area", "Nucleolar area", "(µm²)")]),
    ("TIM23", "tim23_100", [("degree", "Network degree", "(mean node degree)"),
                            ("count", "Mitochondrial fragment count", "(objects/cell)"),
                            ("area", "Mitochondrial total area", "(µm²)"),
                            ("connectivity", "Largest connected component", "(px)"),
                            ("branches", "Branch count", "(num branches/cell)"),
                            ("nodes", "Node count", "(num nodes/cell)")]),
]


def _pct(vals, base):
    b = abs(base) or 1e-9
    return (np.asarray(vals, float) - base) / b * 100.0


def _load(dirname):
    return np.load(f"{NAT}/{dirname}/stats.npz")


def _series(z, feat, normalize):
    rn, rk = z[f"rn_{feat}"], z[f"rk_{feat}"]
    gens = [z[f"gen_{feat}_a{a}"] for a in ALPHAS]
    if feat in AREA_FEATS:                                            # px² → µm² (spacing=(1,1) in _measure → native px units)
        rn, rk, *gens = (v * PX_UM ** 2 for v in (rn, rk, *gens))
    if not normalize:
        return [rn, rk, *gens]
    base = float(np.nanmean(rn))                                     # SINGLE shared baseline — real NTC mean (matches the median line drawn below)
    return [_pct(rn, base), _pct(rk, base), *(_pct(g, base) for g in gens)]


def render_violin(group, dirname, feat, disp, unit, normalize):
    z = _load(dirname)
    data = [np.asarray(d, float) for d in _series(z, feat, normalize)]
    data = [d[np.isfinite(d)] for d in data]
    keep = [i for i, d in enumerate(data) if len(d)]
    fig, ax = plt.subplots(figsize=(7.8, 5.4), facecolor="white")
    parts = ax.violinplot([data[i] for i in keep], positions=keep, showmeans=False, showextrema=False, showmedians=False, widths=0.82)
    for pc, i in zip(parts["bodies"], keep):
        pc.set_facecolor(C[LABS[i].replace("real NTC", "real").replace("real KO", "KO")]); pc.set_alpha(0.6)
        pc.set_edgecolor(C[LABS[i].replace("real NTC", "real").replace("real KO", "KO")]); pc.set_linewidth(1.5)
    for i in keep:
        ax.hlines(np.median(data[i]), i - 0.34, i + 0.34, color="#222", lw=3, zorder=5)
    if normalize:
        ax.axhline(0, color="#999", lw=1.5, zorder=1)
        from matplotlib.ticker import FuncFormatter
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.0f}%" if abs(v) >= 0.5 else "0%"))
        ylab = f"{disp}\n(% change vs real NTC)"
    else:
        ylab = f"{disp} {unit}"
    ax.set_xticks(range(len(LABS))); ax.set_xticklabels(["real\nNTC", "real\nKO"] + [f"α={a:g}" for a in ALPHAS], fontsize=15)
    longest = max(len(line) for line in ylab.split("\n"))
    ax.set_ylabel(ylab, fontsize=(20 if longest <= 30 else 16))
    ax.tick_params(axis="y", labelsize=18, width=2.5, length=8); ax.tick_params(axis="x", length=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(2.5)
    tag = "normalized" if normalize else "raw"
    stem = f"{OUT}/{group}_{feat}_{tag}"
    for ext in ("png", "svg"):
        fig.savefig(f"{stem}.{ext}", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {stem}.png/svg")
    return stem


PROD_VA = "/hpc/projects/icd.fast.ops/models/diffex/viewer_assets_v5"
# group → (marker_dir/grain/target, cell_offset) for sourcing the RAW (non-histogram-matched) display frame —
# panel.npz's own `img` is whatever the group's measurement pipeline stored (e.g. mtor_mo_hm's is HISTOGRAM-
# MATCHED for segmentation fairness, never meant for display) — always render the true raw model output instead.
_RAW_SRC = {
    "mtor_mo_hm_100": ("lysosome_LysoTracker_live_cell_dye/geneKO/MTOR", 0),
    "tim23_100": ("mitochondria_ChromaLIVE_561_excitation/complex/TIM23_mitochondrial_inner_membrane_pre_sequence_translocase_complex__TIM17A_variant", 0),
    "polr1b_vsnpm3_100cpu": ("phase/geneKO/POLR1B", 200),
}
Z0, Z1, Z2 = 8, 10, 12


def render_cellgrid(group, dirname, n=6):
    """Rows: real KO, real NTC, gen α0, α1, α2 (α3 DROPPED) — rebuilt from cached panel.npz masks + RAW frames."""
    pf = f"{NAT}/{dirname}/panel.npz"
    if not os.path.exists(pf):
        print(f"skip {group}: no panel.npz at {pf}"); return None
    d = np.load(pf, allow_pickle=True)
    gpanel = d["gpanel"].item()
    rn, rk = list(d["rn"]), list(d["rk"])
    rows = [("real KO", rk), ("real NTC", rn), ("α=0", 0), ("α=1", 1), ("α=2", 2)]
    raw_dir, raw_off = _RAW_SRC.get(dirname, (None, 0))
    fig, axes = plt.subplots(len(rows), n, figsize=(n * 2.0, len(rows) * 2.0), facecolor="white")
    for ri, (label, src) in enumerate(rows):
        for ci in range(n):
            ax = axes[ri, ci]; ax.axis("off")
            if ri == 0:
                ax.set_title(f"cell {ci}", fontsize=11)
            if ci == 0:
                ax.text(-0.2, 0.5, label, transform=ax.transAxes, rotation=90, va="center", ha="center", fontsize=15, fontweight="bold")
            if isinstance(src, list):
                if ci >= len(src):
                    continue
                img, lc, mask = src[ci]
                img = np.asarray(img, np.float64)                    # real crop is RAW intensity (not [0,1]) — percentile-normalize for display
                p1, p99 = np.percentile(img, [1, 99])
                img = np.clip((img - p1) / max(p99 - p1, 1e-6), 0, 1).astype(np.float32)
            else:
                key = f"gen_a{src}"
                if ci not in gpanel or key not in gpanel[ci]:
                    continue
                _, lc, mask = gpanel[ci][key]
                if raw_dir is not None:                                    # RAW model output, not panel.npz's (possibly HM'd) img
                    zi = {0: Z0, 1: Z1, 2: Z2}[src]
                    raw = np.load(f"{PROD_VA}/{raw_dir}/cell{ci + raw_off}/frames_f32.npz")["gen"][zi]
                    img = np.clip((raw + 1) / 2, 0, 1).astype(np.float32)
                else:
                    img = np.asarray(gpanel[ci][key][0], np.float32)
            lc = np.asarray(lc); mask = np.asarray(mask) > 0
            ax.imshow(np.clip(img, 0, 1), cmap="gray", vmin=0, vmax=1)
            b = find_boundaries(np.where(mask, lc, 0) > 0, mode="outer")
            ov = np.zeros((*b.shape, 4)); ov[b] = [1, 0.3, 0, 1]; ax.imshow(ov)
            mb = find_boundaries(mask, mode="inner")
            ov2 = np.zeros((*mb.shape, 4)); ov2[mb] = [0.2, 0.8, 1, 0.7]; ax.imshow(ov2)
    fig.suptitle(f"{group} — real KO / real NTC / gen α0,1,2", fontsize=15)
    fig.tight_layout(rect=[0.02, 0, 1, 0.96])
    stem = f"{OUT}/{group}_cellgrid"
    for ext in ("png", "svg"):
        fig.savefig(f"{stem}.{ext}", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {stem}.png/svg")
    return stem


def build():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for group, dirname, feats in GROUPS:
        for feat, disp, unit in feats:
            render_violin(group, dirname, feat, disp, unit, normalize=False)
            render_violin(group, dirname, feat, disp, unit, normalize=True)
            rows.append((group, disp))
    _write_index(rows)


def _write_index(rows):
    lines = ["# Figure 4 — final panels (bruno review, not yet on Confluence)\n",
             f"Source: `_native/` stats.npz + panel.npz (no re-measurement — pure rendering).\n",
             "Normalization: SINGLE real-NTC baseline shared by real NTC / real KO / gen α0,1,2 (not per-modality).\n",
             "Violins show α0, 0.5, 1, 1.5, 2, 2.5, 3; the cell grid shows α0, α1, α2 (α3+ dropped there).\n",
             "\n| Group | Feature | Raw | Normalized |",
             "|---|---|---|---|"]
    for group, dirname, feats in GROUPS:
        for feat, disp, unit in feats:
            lines.append(f"| {group} | {disp} | [{feat}_raw.svg]({group}_{feat}_raw.svg) | [{feat}_normalized.svg]({group}_{feat}_normalized.svg) |")
    lines.append("\n| Group | Cell grid |\n|---|---|")
    for group, dirname, feats in GROUPS:
        lines.append(f"| {group} | [{group}_cellgrid.svg]({group}_cellgrid.svg) |")
    open(f"{OUT}/index.md", "w").write("\n".join(lines))
    print(f"saved {OUT}/index.md")


if __name__ == "__main__":
    build()
