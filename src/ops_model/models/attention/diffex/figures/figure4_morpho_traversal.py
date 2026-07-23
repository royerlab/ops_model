"""Figure 4 (paper) — morpho traversal panel: image row + seg-overlay row + %-change plot.

Reads a morpho target's full_features.json (per-α mean trajectory + real NTC→KO reference)
and its traversal frames / per-frame seg labels + per-object features (same assets the
morpho_demo.html viewer uses). Three stacked rows:
  1. grayscale traversal images (original real NTC, then generated cell at each shown α)
  2. the same panels with the org-seg mask overlaid, objects colored by the feature
     (inferno heatmap, per-object value normalized over the cell trajectory + NTC) —
     identical mapping to morpho_demo.html (overlayBase → per-object key, infernoRGB).
  3. generated %-change vs its own α=0 baseline (starts at α=0), real KO reference ±SEM.

Usage: python figure4_morpho_traversal.py   (edit the __main__ block for a different target).
"""
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from PIL import Image

KEYLABEL = {"area": "object area (px²)", "mean_int": "object intensity",
            "ecc": "eccentricity", "skel": "skeleton length"}

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

VA = f"/hpc/projects/icd.fast.ops/models/diffex/{os.environ.get('OPS_DIFFEX_ASSETS', 'viewer_assets')}"
OUT = "/hpc/projects/icd.fast.ops/analysis/figure4_traversals"


def _objkey(feature, avail):
    """Map a full feature name → per-object key used for overlay coloring (morpho_demo.overlayBase)."""
    if "intensity" in feature and "mean_int" in avail:
        return "mean_int"
    if "eccentric" in feature and "ecc" in avail:
        return "ecc"
    if ("branch" in feature or "skeleton" in feature) and "skel" in avail:
        return "skel"
    if "area" in feature and "area" in avail:
        return "area"
    return None


NTC_RGB = (0.18, 0.72, 0.70)   # flat single color for the NTC reference seg overlay (teal)


def _uniform_rgba(labels, rgb, op=0.75):
    """Flat single-color RGBA over every labeled object (bg transparent)."""
    rgba = np.zeros((*labels.shape, 4))
    m = labels > 0
    rgba[m, :3] = rgb
    rgba[m, 3] = op
    return rgba


def _overlay_rgba(labels, feats, key, lo, hi, op=0.75):
    """Per-object inferno RGBA over the label mask; bg / valueless objects transparent."""
    maxid = int(labels.max())
    tlut = np.full(maxid + 1, np.nan, float)
    for sid, props in feats.items():
        i = int(sid)
        v = props.get(key)
        if i <= maxid and v is not None:
            tlut[i] = (v - lo) / (hi - lo + 1e-9)
    t = tlut[labels]
    rgba = cm.inferno(np.clip(np.nan_to_num(t), 0, 1))
    rgba[..., 3] = np.where(np.isfinite(t) & (labels > 0), op, 0.0)
    return rgba


def make_figure(morpho_dir, traversal_dir, feature, cell, alphas_show, label, simple, out_stem, op=0.75):
    md = f"{VA}/_morphometrics/{morpho_dir}"
    ff = json.load(open(f"{md}/full_features.json"))
    alphas = ff["alphas"]
    n = len(alphas)
    z = n >> 1  # α=0 baseline index
    raw = ff["agg"][feature]
    gb = abs(raw[z]) or 1e-9
    gen = [(v - raw[z]) / gb * 100 for v in raw]
    asem = (ff.get("agg_sem") or {}).get(feature) or [0.0] * n
    sem = [e / gb * 100 for e in asem]

    rr = (ff.get("real_ref") or {}).get(feature)
    koV = koS = None
    if rr and rr.get("ntc") and rr["ntc"][0] is not None and rr.get("ko") and rr["ko"][0] is not None:
        nm = rr["ntc"][0]
        nmp = abs(nm) or 1e-9
        koV = (rr["ko"][0] - nm) / nmp * 100  # real KO: % change vs real NTC
        koS = (rr["ko"][1] or 0) / nmp * 100

    idxs = [min(range(n), key=lambda i: abs(alphas[i] - a)) for a in alphas_show]

    # per-object overlay key + color limits over this cell's whole trajectory + NTC ref
    f0 = json.load(open(f"{md}/cell{cell}/a{idxs[0]:02d}_feats.json"))
    avail = set(next(iter(f0.values())).keys()) if f0 else set()
    okey = _objkey(feature, avail)
    if okey is None and "area" in avail:        # feature not stored per-object → color overlay by object area
        okey = "area"

    ref = json.load(open(f"{md}/_ref.json"))
    ntc_cells = ref.get("cells", {}).get("NTC", [])
    ntc_entry = next((c for c in ntc_cells if c.get("cell") == cell), ntc_cells[0] if ntc_cells else None)
    ntc_feats = (ntc_entry or {}).get("feats", {})

    vals = []
    if okey:                                   # clim over generated frames only (NTC excluded — its
        for i in range(n):                     # filled-object seg would otherwise dominate the scale)
            try:
                fj = json.load(open(f"{md}/cell{cell}/a{i:02d}_feats.json"))
                vals += [p[okey] for p in fj.values() if p.get(okey) is not None]
            except FileNotFoundError:
                pass
    # robust color scale: percentile-clip so a few large objects don't wash out the rest.
    # CLIP=(0,100) = raw min/max.
    lo, hi = (tuple(np.percentile(vals, CLIP)) if vals else (0.0, 1.0))

    # panels: (title, grayscale, label mask, per-object feats)
    panels = []
    ntc_img = f"{md}/_ref/NTC/cell{cell}.webp"
    if not os.path.exists(ntc_img):
        alt = sorted(glob.glob(f"{md}/_ref/NTC/cell*.webp"))
        ntc_img = alt[0] if alt else None
    if ntc_img:
        lbp = ntc_img.replace(".webp", "_labels.png")
        lab = np.asarray(Image.open(lbp)) if os.path.exists(lbp) else None
        panels.append(("original NTC", np.asarray(Image.open(ntc_img).convert("L")), lab, ntc_feats))
    for i in idxs:
        gray = np.asarray(Image.open(f"{VA}/{traversal_dir}/cell{cell}/frame_{i:02d}.webp").convert("L"))
        lab = np.asarray(Image.open(f"{md}/cell{cell}/a{i:02d}_labels.png"))
        fj = json.load(open(f"{md}/cell{cell}/a{i:02d}_feats.json"))
        panels.append((f"α={alphas[i]:+.0f}", gray, lab, fj))
    nc = len(panels)

    fig = plt.figure(figsize=(nc * 2.3, 8.6), facecolor="white")
    gs = fig.add_gridspec(3, nc, height_ratios=[1.0, 1.0, 1.55], hspace=0.14, wspace=0.04)

    oaxes = []
    for j, (t, gray, lab, fj) in enumerate(panels):
        axi = fig.add_subplot(gs[0, j])              # row 1: image
        axi.imshow(gray, cmap="gray")
        axi.set_xticks([]); axi.set_yticks([])
        axi.set_title(t, fontsize=22)
        for s in axi.spines.values():
            s.set_visible(False)
        axo = fig.add_subplot(gs[1, j])              # row 2: seg overlay colored by feature
        axo.imshow(gray, cmap="gray")
        if lab is None:
            pass
        elif t == "original NTC":                              # NTC: flat single-color seg (not value-colored)
            axo.imshow(_uniform_rgba(lab, NTC_RGB, op))
        elif okey:                                             # generated: value heatmap
            axo.imshow(_overlay_rgba(lab, fj, okey, lo, hi, op))
        axo.set_xticks([]); axo.set_yticks([])
        for s in axo.spines.values():
            s.set_visible(False)
        oaxes.append(axo)

    if okey:                                         # inferno colorbar for the row-2 overlay
        p = oaxes[-1].get_position()
        cax = fig.add_axes([p.x1 + 0.012, p.y0, 0.013, p.height])
        cb = fig.colorbar(cm.ScalarMappable(Normalize(lo, hi), cmap="inferno"), cax=cax)
        cb.set_label(KEYLABEL.get(okey, okey), fontsize=18)
        cb.set_ticks([lo, hi])
        cb.ax.set_yticklabels([f"{lo:.0f}", f"{hi:.0f}"])
        cb.ax.tick_params(labelsize=16)
        cb.outline.set_visible(False)

    ax = fig.add_subplot(gs[2, :])                   # row 3: %-change plot (α ≥ 0)
    ax.set_facecolor("white")
    a0, g0, s0 = alphas[z:], gen[z:], sem[z:]
    if koV is not None:
        ax.axhspan(koV - koS, koV + koS, color="#5ad17a", alpha=0.18, lw=0)
        ax.axhline(koV, color="#2e8b57", ls="--", lw=3, label=f"real KO ({koV:+.0f}%)")
    ax.axhline(0, color="#999", lw=1.5)
    ax.errorbar(a0, g0, yerr=s0, fmt="-o", color="#1f77b4", lw=4, ms=9,
                capsize=5, elinewidth=2, ecolor="#1f77b4", label="generated (mean ± SEM)")
    ax.set_xlabel("α (traversal strength)", fontsize=28)
    ax.set_ylabel(f"{simple}\n(% change vs NTC)", fontsize=28)
    ax.set_xlim(-0.2, max(a0) + 0.2)
    ax.set_xticks([a for a in a0 if a == int(a)])
    ax.tick_params(labelsize=24, width=2.5, length=9)
    ax.legend(frameon=False, fontsize=22, loc="best")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(2.5)

    os.makedirs(os.path.dirname(f"{OUT}/{out_stem}"), exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(f"{OUT}/{out_stem}.{ext}", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {OUT}/{out_stem}  (overlay key={okey}, clim=[{lo:.1f},{hi:.1f}], gen {g0[-1]:+.0f}% @α+5, "
          f"real KO {None if koV is None else round(koV)}%)")


FIGURES = [
    dict(group="KIF23_nucleolive", dir="nucleus_NucleoLIVE_Live_Cell_dye/geneKO/KIF23",
         feature="obj_area_filled_sum", cell=0, simple="Nuclear size",
         label="KIF23 (NucleoLIVE) · object area filled sum", out_stem="KIF23_nucleolive_obj_area_filled_sum"),
    dict(group="TOMM20_phase", dir="phase/geneKO/TOMM20",
         feature="obj_area_max", cell=0, simple="Mitochondrial size",
         label="TOMM20 (phase) · object area max", out_stem="TOMM20_phase_obj_area_max"),
    dict(group="CCT_npm3", dir="nucleolus_GC_NPM3/complex/Chaperonin_containing_T_complex",
         feature="obj_circularity_sum", cell=0, label="CCT complex (NPM3 nucleoli) · object circularity sum", out_stem="CCT_npm3_obj_circularity_sum"),
    dict(group="TOMM20_chromalive561", dir="mitochondria_ChromaLIVE_561_excitation/geneKO/TOMM20",
         feature="network_mitochondria_chromalive_561_excitation_tubular_seg_num_endpoints", cell=0,
         label="TOMM20 (ChromaLIVE561) · network num endpoints", out_stem="TOMM20_chromalive561_num_endpoints"),
    dict(group="GBF1_sec23a", dir="ER_Golgi_COP_II_SEC23A/geneKO/GBF1",
         feature="obj_area_sum", cell=0, label="GBF1 (ER/Golgi COP-II SEC23A) · object area sum", out_stem="GBF1_sec23a_obj_area_sum"),
    dict(group="POLR1B_npm3", dir="nucleolus_GC_NPM3/geneKO/POLR1B",
         feature="obj_extent_sum", cell=0, simple="Nucleolar extent",
         label="POLR1B (NPM3 nucleoli) · object extent sum", out_stem="POLR1B_npm3_obj_extent_sum"),
    dict(group="TIM23_chromalive561", dir="mitochondria_ChromaLIVE_561_excitation/complex/TIM23_mitochondrial_inner_membrane_pre_sequence_translocase_complex__TIM17A_variant",
         feature="network_mitochondria_chromalive_561_excitation_tubular_seg_average_degree", cell=0,
         label="TIM23 complex (ChromaLIVE561) · network average degree", out_stem="TIM23_chromalive561_network_average_degree"),
    dict(group="CAPZB_fastact", dir="actin_filament_FastAct_SPY555_Live_Cell_Dye/geneKO/CAPZB",
         feature="obj_intensity_integrated_median", cell=0, simple="Intensity",
         label="CAPZB (FastAct actin) · object intensity integrated median", out_stem="CAPZB_fastact_obj_intensity_integrated_median"),
]

CELLS = [0, 1, 2, 3, 4, 5]              # render a PNG per cell so the best one can be picked
ALPHAS_SHOW = [0, 1, 5]                 # generated α panels (original NTC is prepended automatically)
CLIP = (2, 98)                          # overlay color-scale percentiles ((0,100)=raw min/max, demo-style)

if __name__ == "__main__":
    for f in FIGURES:
        for c in CELLS:
            try:
                make_figure(morpho_dir=f["dir"], traversal_dir=f["dir"], feature=f["feature"], cell=c,
                            alphas_show=ALPHAS_SHOW, label=f["label"], simple=f.get("simple", f["label"]),
                            out_stem=f"{f['group']}/{f['out_stem']}_cell{c}")
            except (FileNotFoundError, IndexError) as e:
                print(f"skip {f['out_stem']}_cell{c}: {e}")
