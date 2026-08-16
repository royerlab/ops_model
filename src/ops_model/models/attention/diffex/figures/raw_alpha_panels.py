"""Render the FINAL single-cell traversal image panel using the ORIGINAL, unmodified `image_panels`/
`render_images` code from figure4_morpho_traversal.py (the exact code used for the real paper figures) —
fed with data from our NEW validated segmentation (native-160, VS-NPM3 for POLR1B, etc).

Mechanism: populate a PRIVATE scratch asset tree (never touches production viewer_assets_v5) with the
files image_panels expects (full_features.json, cell{c}/a{i:02d}_labels.png, a{i:02d}_feats.json), sourced
from panel.npz (already-computed label arrays from our validated pipeline) — then monkeypatch the `VA`
module constant so image_panels/render_images run completely unmodified against this scratch tree.

Run: python raw_alpha_panels.py
"""
import json
import os

import numpy as np
from PIL import Image
from skimage.measure import regionprops
from skimage.transform import resize

import figure4_morpho_traversal as T   # the ORIGINAL, unmodified render code
from ops_model.models.attention.diffex.viewer.morpho_pipeline import _clip_border   # SAME border policy _measure() uses

NAT = "/hpc/projects/icd.fast.ops/analysis/figure4_traversals_violin/_native"
PROD_VA = "/hpc/projects/icd.fast.ops/models/diffex/viewer_assets_v5"
SCRATCH = "/hpc/projects/icd.fast.ops/analysis/figure4_traversals_violin/bruno/_scratch_assets"
OUT = "/hpc/projects/icd.fast.ops/analysis/figure4_traversals_violin/bruno"
ALPHAS = [-5.0, -4.0, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
Z0, Z1, Z2 = 8, 10, 12                                  # α=0,1,2 indices (native panel.npz stores these keys)

# group → (native_dir=source panel.npz, cell=GLOBAL cell index used for BOTH anchor + scratch frame paths,
#          local_cell=panel.npz rank position (locked-in pick), marker_dir/grain/target=identity for the scratch traversal_dir)
GROUPS = {
    "mTOR": dict(native_dir="mtor_mo_hm_100", cell=0, local_cell=0, marker_dir="lysosome_LysoTracker_live_cell_dye",
                grain="geneKO", target="MTOR"),
    "POLR1B": dict(native_dir="polr1b_vsnpm3_100cpu", cell=211, local_cell=11, marker_dir="phase", grain="geneKO", target="POLR1B"),  # picked from candidates 0-20
    "TIM23": dict(native_dir="tim23_100", cell=0, local_cell=0, marker_dir="mitochondria_ChromaLIVE_561_excitation",
                  grain="complex", target="TIM23_mitochondrial_inner_membrane_pre_sequence_translocase_complex__TIM17A_variant"),
    "TAF1B": dict(native_dir="taf1b_vsnpm3_stringent", cell=200, local_cell=0, marker_dir="phase", grain="geneKO", target="TAF1B"),
    "SAMM50": dict(native_dir="samm50_chromalive", cell=1, local_cell=1, marker_dir="mitochondria_ChromaLIVE_561_excitation",
                   grain="geneKO", target="SAMM50"),   # swapped primary candidate 0 -> 1; old candidate 0 now lives in the backups as "_candidate1"
    "MICOS13": dict(native_dir="micos13_chromalive", cell=0, local_cell=0, marker_dir="mitochondria_ChromaLIVE_561_excitation",
                    grain="geneKO", target="MICOS13"),
}


_PALETTE = [                                                            # 18 hues stepped by the golden angle (not swept around the wheel in order) —
    (1.00, 0.15, 0.15), (0.05, 0.29, 0.90), (0.65, 1.00, 0.15), (0.90, 0.05, 0.79),   # consecutive palette slots land far apart on the hue wheel, so
    (0.15, 1.00, 0.86), (0.90, 0.51, 0.05), (0.36, 0.15, 1.00), (0.08, 0.90, 0.05),   # adjacent-ranked objects (which get adjacent slots) never look similar
    (1.00, 0.15, 0.43), (0.05, 0.58, 0.90), (0.93, 1.00, 0.15), (0.72, 0.05, 0.90),
    (0.15, 1.00, 0.57), (0.90, 0.22, 0.05), (0.15, 0.22, 1.00), (0.37, 0.90, 0.05),
    (1.00, 0.15, 0.72), (0.05, 0.87, 0.90),
]


def _qualitative_rgba(lab, op=0.75):
    """Distinct flat color per object, size-ranked (largest object -> palette[0], 2nd-largest -> palette[1], ...)
    — NO encoded measurement, just visual separation. Label IDs aren't persistent across the traversal (each
    frame is independently connected-component-labeled), so ranking by size instead of raw ID is what keeps
    the "same" object (usually the dominant one) getting the same color across the NTC/α0/α1/α2 panels."""
    ids, counts = np.unique(lab[lab > 0], return_counts=True)
    order = ids[np.argsort(-counts)]
    rgba = np.zeros((*lab.shape, 4))
    for j, i in enumerate(order):
        m = lab == i
        rgba[m, :3] = _PALETTE[j % len(_PALETTE)]
        rgba[m, 3] = op
    return rgba


def _regionprops_dict(lc):
    """Per-object {id: {"area":..., "ecc":..., "mean_int":0.0, "circularity":...}} — matches the original
    feats.json schema, plus circularity computed with the SAME cpu-path formula as morphology_features.py's
    circularity_approx (Ramanujan ellipse-perimeter approx from axis lengths, not real boundary tracing)."""
    out = {}
    for r in regionprops(lc.astype(np.int32)):
        a, b = r.axis_major_length / 2, max(r.axis_minor_length / 2, 0.1)
        perim = np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b))) or 1.0
        out[str(r.label)] = {"area": float(r.area), "ecc": float(r.eccentricity), "mean_int": 0.0,
                              "circularity": float(4 * np.pi * r.area / perim ** 2)}
    return out


def _build_scratch(group, cfg, scratch_cell=None, local_cell=None):
    """Populate the private scratch tree for one group from its panel.npz (once, shared across that group's
    features). scratch_cell = the cell number used for scratch tree paths (defaults to cfg['cell']);
    local_cell = which panel.npz cell to source from (defaults to the same number — mTOR/TIM23 have no offset)."""
    scratch_cell = cfg["cell"] if scratch_cell is None else scratch_cell
    dir_ = f"{cfg['marker_dir']}/{cfg['grain']}/{cfg['target']}"
    md = f"{SCRATCH}/_morphometrics/{dir_}"
    os.makedirs(f"{md}/cell{scratch_cell}", exist_ok=True)
    os.makedirs(f"{SCRATCH}/{dir_}/cell{scratch_cell}", exist_ok=True)
    json.dump({"alphas": ALPHAS}, open(f"{md}/full_features.json", "w"))

    d = np.load(f"{NAT}/{cfg['native_dir']}/panel.npz", allow_pickle=True)
    gpanel = d["gpanel"].item()
    local_cell = sorted(gpanel.keys())[0] if local_cell is None else local_cell
    raw = np.load(f"{PROD_VA}/{dir_}/cell{scratch_cell}/frames_f32.npz")["gen"]   # RAW model output — NOT panel.npz's `img`,
    for a, zi in ((0, Z0), (1, Z1), (2, Z2)):                                      # which for hist_match=True groups (e.g. mtor_mo_hm) is the
        _, lc, mask = gpanel[local_cell][f"gen_a{a}"]                              # histogram-MATCHED image (a measurement-only transform, never meant for display)
        lc = _clip_border(np.where(np.asarray(mask), np.asarray(lc), 0)).astype(np.int32)   # same border policy _measure() applies — don't display objects that were clipped/excluded from the real measurement
        img01 = np.clip((raw[zi] + 1) / 2, 0, 1).astype(np.float32)                # SAME mapping _save_webp uses on the raw [-1,1] model output
        img256 = resize(img01, (256, 256), preserve_range=True)
        Image.fromarray((np.clip(img256, 0, 1) * 255).astype(np.uint8)).save(
            f"{SCRATCH}/{dir_}/cell{scratch_cell}/frame_{zi:02d}.webp", quality=90)
        feats = _regionprops_dict(lc)                                   # regionprops on NATIVE res → correct area/ecc values
        json.dump(feats, open(f"{md}/cell{scratch_cell}/a{zi:02d}_feats.json", "w"))
        lc256 = resize(lc, (256, 256), order=0, preserve_range=True, anti_aliasing=False).astype(np.uint16)  # nearest (no blending); 16-bit — TIM23 has up to ~1800 fragments, uint8 wraps IDs >255 and _overlay_rgba then drops them transparent
        Image.fromarray(lc256, mode="I;16").save(f"{md}/cell{scratch_cell}/a{zi:02d}_labels.png")
    os.makedirs(f"{SCRATCH}/{cfg['marker_dir']}/_anchors/NTC/cell{scratch_cell}", exist_ok=True)
    print(f"[scratch] built {group} (panel.npz cell {local_cell} -> scratch cell{scratch_cell}) -> {md}/cell{scratch_cell}")


def render(group, scratch_cell=None, local_cell=None, suffix="", op=0.75):
    """One panel per group (not per feature/metric) — objects get distinct qualitative colors, no encoded value."""
    cfg = GROUPS[group]
    scratch_cell = cfg["cell"] if scratch_cell is None else scratch_cell
    dir_ = f"{cfg['marker_dir']}/{cfg['grain']}/{cfg['target']}"
    md = f"{SCRATCH}/_morphometrics/{dir_}"
    real_anchor = f"/hpc/projects/icd.fast.ops/models/diffex/viewer_assets_v5/{cfg['marker_dir']}/_anchors/NTC/cell{scratch_cell}/real.webp"
    scratch_anchor = f"{SCRATCH}/{cfg['marker_dir']}/_anchors/NTC/cell{scratch_cell}/real.webp"
    if not os.path.exists(scratch_anchor) and os.path.exists(real_anchor):
        import shutil; shutil.copy(real_anchor, scratch_anchor)        # real anchor image (unmodified) — just made visible under the scratch tree
    T.VA = SCRATCH                                                     # monkeypatch: image_panels/render_images now read ONLY our scratch tree
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels, okey, lo, hi = T.image_panels(md, dir_, dir_, "", scratch_cell, [0, 1, 2])
    if panels and panels[0][0] == "original NTC":                     # image_panels reuses gen-α0's OWN mask here (shared-file behavior) — that
        d = np.load(f"{NAT}/{cfg['native_dir']}/panel.npz", allow_pickle=True)   # falsely implies it's the real measurement's segmentation. Swap in the
        rn = list(d["rn"])                                             # ACTUAL independently-measured real-NTC photo+mask (panel.npz's own "rn")
        lc0 = (local_cell if local_cell is not None else 0)             # so this panel shows what real NTC really measures as, not gen's mask.
        if lc0 < len(rn):
            rimg, rlab, rmask = rn[lc0]
            rimg = np.asarray(rimg, np.float64)
            p1, p99 = np.percentile(rimg, [1, 99])
            rimg01 = np.clip((rimg - p1) / max(p99 - p1, 1e-6), 0, 1)
            gray256 = (resize(rimg01, (256, 256), preserve_range=True) * 255).astype(np.uint8)
            rlab_clip = _clip_border(np.where(np.asarray(rmask), np.asarray(rlab, np.int32), 0))   # same border policy _measure() applies
            lab256 = resize(rlab_clip, (256, 256), order=0, preserve_range=True, anti_aliasing=False).astype(np.int32)
            panels[0] = ("original NTC", gray256, lab256, {})
        else:
            print(f"  [warn] no rn[{lc0}] for {group} — original NTC panel keeps gen-α0's mask (mismatched)")
    nc = len(panels)
    fig = plt.figure(figsize=(nc * 2.6, 5.4), facecolor="white")
    oaxes = T.render_images(fig, fig.add_gridspec(1, 1)[0], panels, None, lo, hi, title_fs=20, hspace=0.05, cbar=False)
    for ax in fig.get_axes():                                            # imshow's default aspect='equal' pads each axis to preserve
        ax.set_aspect("auto")                                            # the square image ratio, independent of hspace — this closes that gap
    for ax, (t, gray, lab, fj) in zip(oaxes, panels):                    # override EVERY panel (incl. "original NTC") with the qualitative palette
        if lab is not None:
            ax.imshow(_qualitative_rgba(lab, op=op), aspect="auto")
    fig.suptitle(f"{group}{suffix}", fontsize=16)
    stem = f"{OUT}/{group}_cellpanel{suffix}"
    for ext in ("png", "svg"):
        fig.savefig(f"{stem}.{ext}", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {stem}.png/svg (qualitative)")


def build():
    for group, cfg in GROUPS.items():
        lc = cfg["local_cell"]
        _build_scratch(group, cfg, local_cell=lc)
        render(group, local_cell=lc)


def render_candidates(group, local_cells, offset=0):
    """Render the SAME panel for several candidate cells — for picking. `local_cells` are panel.npz gpanel
    keys; scratch_cell (real global cell number, used for anchor/frames_f32 lookups) = local_cell + offset
    (POLR1B's global numbering is offset +200 from its local panel.npz rank; mTOR/TIM23 have no offset)."""
    cfg = GROUPS[group]
    for lc in local_cells:
        sc = lc + offset
        _build_scratch(group, cfg, scratch_cell=sc, local_cell=lc)
        render(group, scratch_cell=sc, local_cell=lc, suffix=f"_candidate{lc}")


if __name__ == "__main__":
    build()
