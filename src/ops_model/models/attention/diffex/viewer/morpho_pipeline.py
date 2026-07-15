"""Morphometrics on generated cells by REUSING the real organelle_profiler pipeline (no reimplementation).

The production org-seg (`segment_single_position_channel`) + feature extraction are zarr-bound, so we
stage the generated α-frames as a phenotyping-v3-style OME-Zarr under a REAL experiment name (via
OPS_OUTPUT_BASE_DIR override for the store path, default OPS_CONFIGS_DIR so the real channel-map/seg-params
resolve). Then the exact production seg runs on our crops and writes real organelle labels.

Layout: one position per α (`A/<α_idx>/0`); each position is a horizontal strip of the n_cells crops
(padded) in the marker channel. Seg runs per position → labels/<organelle>_seg, read back per crop.
"""
from __future__ import annotations

import json
import os

import numpy as np

CACHE = "/hpc/projects/icd.fast.ops/models/diffex/viewer_assets"
SYNTH_BASE = "/hpc/projects/icd.fast.ops/models/diffex/morpho_synth"
PAD = 24
CROP = 256


def build_mini_zarr(marker_dir, target, grain, real_exp, channel_names, n_cells=6, base_dir=SYNTH_BASE):
    """Stage a traversal's generated α-frames as a phenotyping_v3-style zarr under <base>/<real_exp>/.
    Generated crop → the LAST channel (the marker channel, e.g. GFP); others zero. Returns (zpath, n_alpha)."""
    from iohub import open_ome_zarr
    from PIL import Image
    src = f"{CACHE}/{marker_dir}/{grain}/{target}"
    mp = f"{src}/cell0/meta.json" if os.path.exists(f"{src}/cell0/meta.json") else f"{src}/meta.json"
    alphas = json.load(open(mp))["alphas"]
    na = len(alphas)
    W = n_cells * CROP + (n_cells - 1) * PAD
    zpath = f"{base_dir}/{real_exp}/3-assembly/phenotyping_v3.zarr"
    os.makedirs(os.path.dirname(zpath), exist_ok=True)
    import shutil
    if os.path.exists(zpath):
        shutil.rmtree(zpath)
    C = len(channel_names)
    # version="0.5" → NGFF 0.5 = zarr v3 (matches real phenotyping_v3; the org-seg label writer needs v3 sharding)
    with open_ome_zarr(zpath, layout="hcs", mode="w", channel_names=channel_names, version="0.5") as ds:
        for ai in range(na):
            strip = np.zeros((C, CROP, W), np.float32)
            for c in range(n_cells):
                f = f"{src}/cell{c}/frame_{ai:02d}.webp"
                if not os.path.exists(f):
                    continue
                im = np.asarray(Image.open(f).convert("L"), np.float32) / 255.0
                x0 = c * (CROP + PAD)
                strip[-1, :, x0:x0 + CROP] = im            # marker channel = last
            pos = ds.create_position("A", str(ai), "0")
            pos.create_image("0", strip[None, :, None])    # (T=1,C,Z=1,Y,X)
    print(f"[mini-zarr] {zpath}  {na} α × {n_cells} cells, channels={channel_names}")
    return zpath, na


def run_seg(real_exp, marker_channel, n_alpha, structure_type=None, base_dir=SYNTH_BASE, frangi_params=None):
    """Run the REAL production org-seg on each α position of the mini-zarr (config resolved from the
    real experiment's channel map). `frangi_params` overrides the resolved frangi config (e.g. to switch
    to the ADAPTIVE dynamic threshold on the generated images). Writes labels; returns per-α results."""
    os.environ["OPS_OUTPUT_BASE_DIR"] = base_dir
    from organelle_profiler.organelle_seg.organelle_segmentation import segment_single_position_channel
    out = []
    for ai in range(n_alpha):
        r = segment_single_position_channel(experiment=real_exp, position=f"A/{ai}/0",
                                            channel_key=marker_channel, structure_type=structure_type,
                                            use_clahe=True, frangi_params=frangi_params)
        out.append((ai, r.get("success"), r.get("num_objects"), r.get("output_label"), r.get("error")))
        print(f"  α_idx {ai}: success={r.get('success')} n_obj={r.get('num_objects')} "
              f"label={r.get('output_label')} {r.get('error') or ''}")
    return out


OPCP = "/hpc/projects/icd.fast.ops/analysis/op_cp_features/op_cp_features_{store}.h5ad"
REF_SUFFIX = {"count": "count", "total_area": "area_sum", "mean_area": "area_mean",
              "mean_int": "intensity_mean_mean", "mean_ecc": "eccentricity_mean"}


def _real_ref(store_marker, org_prefix, target, ref_map=None):
    """NTC(empty gene) + target mean±SEM for each aggregate feature, from the precomputed op_cp_features
    store (per-cell, 60M-cell scale). → {agg: {ntc:[mean,sem], ko:[mean,sem]}}.
    ref_map (optional) = {agg: full store feature name}, overrides the default op_<prefix>_<suffix>."""
    import anndata as ad
    st = ad.read_h5ad(OPCP.format(store=store_marker), backed="r")
    gn = st.obs["gene_name"].astype(str).values
    ntc, ko = gn == "", gn == target
    items = ref_map.items() if ref_map else {ak: f"op_{org_prefix}_{suf}" for ak, suf in REF_SUFFIX.items()}.items()
    ref = {}
    for ak, fn in items:
        if fn not in st.var_names:
            continue
        col = np.asarray(st[:, fn].X).ravel().astype(float)
        def ms(m):
            v = col[m]; v = v[np.isfinite(v)]
            return [float(v.mean()), float(v.std() / max(len(v) ** 0.5, 1.0))] if len(v) else [None, None]
        ref[ak] = {"ntc": ms(ntc), "ko": ms(ko)}
    return ref


def readback_cache(marker_dir, target, grain, real_exp, label_name, n_cells, base_dir=SYNTH_BASE,
                   store_marker=None, org_prefix=None, network=False, ref_map=None):
    """Read the REAL org-seg labels the pipeline wrote into the mini-zarr, split per (cell, α) crop,
    measure per-organelle features (regionprops on the real labels) + per-α aggregates, and cache for
    the viewer overlay/plot. → viewer_assets/_morphometrics/<marker_dir>/<grain>/<target>/"""
    import json
    import zarr
    from PIL import Image
    from skimage.measure import label as relabel, regionprops
    zpath = f"{base_dir}/{real_exp}/3-assembly/phenotyping_v3.zarr"
    src = f"{CACHE}/{marker_dir}/{grain}/{target}"
    mp = f"{src}/cell0/meta.json" if os.path.exists(f"{src}/cell0/meta.json") else f"{src}/meta.json"
    alphas = json.load(open(mp))["alphas"]
    root = zarr.open(zpath, mode="r")
    out = f"{CACHE}/_morphometrics/{marker_dir}/{grain}/{target}"; os.makedirs(out, exist_ok=True)
    # network markers (mito/tubular): count = num fragments (num_objects), skel = per-component skeleton
    # length (branch-length proxy). Else vesicular morphology.
    AGG = (["count", "total_area", "total_skel", "mean_skel", "mean_int"] if network
           else ["count", "total_area", "mean_area", "mean_int", "mean_ecc"])
    agg = {k: [] for k in AGG}
    if network:
        from skimage.morphology import skeletonize
    for ai in range(len(alphas)):
        lab = np.asarray(root[f"A/{ai}/0/labels/{label_name}/0"][0, 0, 0])   # (Y, W) real labels
        img = np.asarray(root[f"A/{ai}/0/0"][0, -1, 0])                       # marker channel strip
        per_cell = []
        for c in range(n_cells):
            x0 = c * (CROP + PAD)
            lc = relabel(lab[:, x0:x0 + CROP] > 0)                           # relabel within crop
            ic = img[:, x0:x0 + CROP]
            EM = 6                                                            # drop border-touching objects
            rp = [r for r in regionprops(lc, intensity_image=ic)
                  if r.area >= 3 and r.bbox[0] >= EM and r.bbox[1] >= EM
                  and r.bbox[2] <= CROP - EM and r.bbox[3] <= CROP - EM]
            rp = rp[:255]                                                     # 8-bit mask cap (index 1..255)
            remap = {r.label: i + 1 for i, r in enumerate(rp)}                # sequential 1..K for the 8-bit mask
            mask8 = np.zeros(lc.shape, np.uint8)
            feats = {}
            for r in rp:
                nl = remap[r.label]; mask8[lc == r.label] = nl
                d = {"area": float(r.area), "mean_int": float(r.intensity_mean), "ecc": float(r.eccentricity)}
                if network:
                    d["skel"] = float(skeletonize(lc == r.label).sum())     # per-component branch-length proxy
                feats[str(nl)] = d                                          # keyed by the mask's pixel value
            cdir = f"{out}/cell{c}"; os.makedirs(cdir, exist_ok=True)
            Image.fromarray(mask8).save(f"{cdir}/a{ai:02d}_labels.png")     # 8-bit component-index mask (real shape)
            json.dump(feats, open(f"{cdir}/a{ai:02d}_feats.json", "w"))
            A = [f["area"] for f in feats.values()]; I = [f["mean_int"] for f in feats.values()]
            pc = {"count": len(feats), "total_area": float(np.sum(A) if A else 0),
                  "mean_int": float(np.mean(I) if I else 0)}
            if network:
                S = [f["skel"] for f in feats.values()]
                pc["total_skel"] = float(np.sum(S) if S else 0); pc["mean_skel"] = float(np.mean(S) if S else 0)
            else:
                E = [f["ecc"] for f in feats.values()]
                pc["mean_area"] = float(np.mean(A) if A else 0); pc["mean_ecc"] = float(np.mean(E) if E else 0)
            per_cell.append(pc)
        for k in AGG:
            agg[k].append(float(np.mean([pc[k] for pc in per_cell])))
    real_ref = _real_ref(store_marker, org_prefix, target, ref_map) if (store_marker and (org_prefix or ref_map)) else {}
    json.dump({"marker_dir": marker_dir, "target": target, "grain": grain, "alphas": alphas,
               "n_cells": n_cells, "label_name": label_name, "agg": agg, "features": AGG,
               "real_ref": real_ref}, open(f"{out}/morpho.json", "w"))
    if real_ref:
        print(f"  real_ref (NTC vs {target}): " +
              ", ".join(f"{k}={v['ntc'][0]:.1f}/{v['ko'][0]:.1f}" for k, v in real_ref.items() if v['ntc'][0]))
    print(f"[readback] {marker_dir}/{target}: REAL-seg aggregates cached -> {out}")
    print(f"  count α-series:      {[round(v,1) for v in agg['count']]}")
    print(f"  total_area α-series: {[round(v,0) for v in agg['total_area']]}")
    return out


def reference_cells(marker_dir, target, real_exp, marker_channel, structure_type=None,
                    groups=("NTC",), n_cells=6, base_dir=SYNTH_BASE):
    """Segment REAL reference cells (the traversal's cached `_anchors/<group>/cell*/real.webp`) through
    the SAME org-seg pipeline → per-organelle feats, so the viewer can show real cells with the identical
    overlay next to the generated ones. groups e.g. ("NTC","AP2M1"). → _morphometrics/.../<target>/_ref.json"""
    import json
    import shutil
    import zarr
    from iohub import open_ome_zarr
    from PIL import Image
    from skimage.measure import label as relabel, regionprops
    grp_cells = {}                                          # {group: [crop arrays]}
    for g in groups:
        crops = []
        for c in range(n_cells):
            f = f"{CACHE}/{marker_dir}/_anchors/{g}/cell{c}/real.webp"
            if os.path.exists(f):
                crops.append(np.asarray(Image.open(f).convert("L"), np.float32) / 255.0)
        if crops:
            grp_cells[g] = crops
    if not grp_cells:
        print("[ref] no real cells cached for", groups); return
    # stage: one position per group, strip of its cells
    zpath = f"{base_dir}/{real_exp}/3-assembly/phenotyping_v3.zarr"
    if os.path.exists(zpath):
        shutil.rmtree(zpath)
    os.makedirs(os.path.dirname(zpath), exist_ok=True)
    glist = list(grp_cells)
    with open_ome_zarr(zpath, layout="hcs", mode="w", channel_names=["Phase2D", marker_channel], version="0.5") as ds:
        for gi, g in enumerate(glist):
            crops = grp_cells[g]; W = len(crops) * CROP + (len(crops) - 1) * PAD
            strip = np.zeros((2, CROP, W), np.float32)
            for c, cr in enumerate(crops):
                strip[-1, :, c * (CROP + PAD):c * (CROP + PAD) + CROP] = cr
            ds.create_position("A", str(gi), "0").create_image("0", strip[None, :, None])
    os.environ["OPS_OUTPUT_BASE_DIR"] = base_dir
    from organelle_profiler.organelle_seg.organelle_segmentation import segment_single_position_channel
    root = None
    ref = {}
    for gi, g in enumerate(glist):
        r = segment_single_position_channel(experiment=real_exp, position=f"A/{gi}/0",
                                             channel_key=marker_channel, structure_type=structure_type, use_clahe=True)
        if not r.get("success"):
            print(f"[ref] {g} seg failed: {r.get('error')}"); continue
        if root is None:
            root = zarr.open(zpath, mode="r")
        lab = np.asarray(root[f"A/{gi}/0/labels/{r['output_label']}/0"][0, 0, 0])
        img = np.asarray(root[f"A/{gi}/0/0"][0, -1, 0])
        cells = []
        for c in range(len(grp_cells[g])):
            x0 = c * (CROP + PAD); lc = relabel(lab[:, x0:x0 + CROP] > 0); ic = img[:, x0:x0 + CROP]
            EM = 6
            orgs = [{"cx": float(rr.centroid[1]), "cy": float(rr.centroid[0]), "r": float((rr.area / 3.14159) ** 0.5),
                     "area": float(rr.area), "mean_int": float(rr.intensity_mean), "ecc": float(rr.eccentricity)}
                    for rr in regionprops(lc, intensity_image=ic)
                    if rr.area >= 3 and rr.bbox[0] >= EM and rr.bbox[1] >= EM and rr.bbox[2] <= CROP - EM and rr.bbox[3] <= CROP - EM]
            cells.append({"cell": c, "organelles": orgs})
        ref[g] = cells
    out = f"{CACHE}/_morphometrics/{marker_dir}/geneKO/{target}"
    os.makedirs(out, exist_ok=True)
    json.dump({"groups": glist, "n_cells": n_cells, "cells": ref}, open(f"{out}/_ref.json", "w"))
    print(f"[ref] {marker_dir}/{target}: real cells {[(g, len(ref.get(g,[]))) for g in glist]} -> {out}/_ref.json")


def reference_from_direction(marker_dir, target, real_exp, marker_channel, structure_type=None,
                             network=False, n_cells=4, base_dir=SYNTH_BASE):
    """Real reference cells for targets with NO cached anchors (e.g. MICOS13): pull the direction-build's
    real crops (`directions/<marker_dir>/geneKO/<target>/cache/crops_<target>_*.npz` — labels 0=control,
    1=KD), seg them through the SAME pipeline, save crop webps + per-organelle feats. → <target>/_ref."""
    import glob
    import json
    import shutil
    import zarr
    from iohub import open_ome_zarr
    from PIL import Image
    from skimage.measure import label as relabel, regionprops
    from skimage.transform import resize as skresize
    npz = glob.glob(f"{OUT}/directions/{marker_dir}/geneKO/{target}/cache/crops_{target}_*.npz") if False else \
        glob.glob(f"/hpc/projects/icd.fast.ops/models/diffex/directions/{marker_dir}/geneKO/{target}/cache/crops_{target}_*.npz")
    if not npz:
        print(f"[ref-dir] no crops npz for {marker_dir}/{target}"); return
    d = np.load(npz[0], allow_pickle=True)
    imgs = d["images"]; labs = d["labels"]
    imgs = imgs[:, 0] if imgs.ndim == 4 else imgs                # (N,H,W)
    def norm256(a):
        a = skresize(a.astype(np.float32), (CROP, CROP), preserve_range=True, anti_aliasing=True)
        lo, hi = np.percentile(a, [1, 99]); return np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)
    grp_cells = {"NTC": [norm256(x) for x in imgs[labs == 0][:n_cells]],
                 target: [norm256(x) for x in imgs[labs == 1][:n_cells]]}
    grp_cells = {g: v for g, v in grp_cells.items() if v}
    zpath = f"{base_dir}/{real_exp}/3-assembly/phenotyping_v3.zarr"
    if os.path.exists(zpath):
        shutil.rmtree(zpath)
    os.makedirs(os.path.dirname(zpath), exist_ok=True)
    glist = list(grp_cells)
    with open_ome_zarr(zpath, layout="hcs", mode="w", channel_names=["Phase2D", marker_channel], version="0.5") as ds:
        for gi, g in enumerate(glist):
            cr = grp_cells[g]; W = len(cr) * CROP + (len(cr) - 1) * PAD; strip = np.zeros((2, CROP, W), np.float32)
            for c, x in enumerate(cr):
                strip[-1, :, c * (CROP + PAD):c * (CROP + PAD) + CROP] = x
            ds.create_position("A", str(gi), "0").create_image("0", strip[None, :, None])
    os.environ["OPS_OUTPUT_BASE_DIR"] = base_dir
    from organelle_profiler.organelle_seg.organelle_segmentation import segment_single_position_channel
    fp = None
    if network:                                                  # match the generated: adaptive dynamic threshold
        from skimage.morphology import skeletonize
        _, dp = _resolve_seg(real_exp, marker_channel); fp = dict(dp)
        fp["threshold"] = None; fp["threshold_mult"] = 0.1; fp["pixel_size_um"] = 0.065   # match generated seg
    out = f"{CACHE}/_morphometrics/{marker_dir}/geneKO/{target}"; os.makedirs(out, exist_ok=True)
    root, ref = None, {}
    for gi, g in enumerate(glist):
        r = segment_single_position_channel(experiment=real_exp, position=f"A/{gi}/0",
                                             channel_key=marker_channel, structure_type=structure_type,
                                             use_clahe=True, frangi_params=fp)
        if not r.get("success"):
            print(f"[ref-dir] {g} seg failed: {r.get('error')}"); continue
        root = root or zarr.open(zpath, mode="r")
        lab = np.asarray(root[f"A/{gi}/0/labels/{r['output_label']}/0"][0, 0, 0]); img = np.asarray(root[f"A/{gi}/0/0"][0, -1, 0])
        gd = f"{out}/_ref/{g}"; os.makedirs(gd, exist_ok=True); cells = []
        for c in range(len(grp_cells[g])):
            x0 = c * (CROP + PAD); lc = relabel(lab[:, x0:x0 + CROP] > 0); ic = img[:, x0:x0 + CROP]; EM = 6
            Image.fromarray((grp_cells[g][c] * 255).astype(np.uint8)).save(f"{gd}/cell{c}.webp")
            keep = [rr for rr in regionprops(lc, intensity_image=ic)
                    if rr.area >= 3 and rr.bbox[0] >= EM and rr.bbox[1] >= EM and rr.bbox[2] <= CROP - EM and rr.bbox[3] <= CROP - EM][:255]
            remap = {rr.label: i + 1 for i, rr in enumerate(keep)}
            mask8 = np.zeros(lc.shape, np.uint8); feats = {}
            for rr in keep:
                nl = remap[rr.label]; mask8[lc == rr.label] = nl
                o = {"area": float(rr.area), "mean_int": float(rr.intensity_mean), "ecc": float(rr.eccentricity)}
                if network:
                    o["skel"] = float(skeletonize(lc == rr.label).sum())
                feats[str(nl)] = o
            Image.fromarray(mask8).save(f"{gd}/cell{c}_labels.png")
            cells.append({"cell": c, "feats": feats})
        ref[g] = cells
    json.dump({"groups": glist, "n_cells": n_cells, "cells": ref, "img_dir": "_ref"}, open(f"{out}/_ref.json", "w"))
    print(f"[ref-dir] {marker_dir}/{target}: real cells {[(g, len(ref.get(g, []))) for g in glist]} + webps -> {out}/_ref")


def sweep_grid(marker_dir, target, real_exp, marker_channel, pix=(0.065, 0.13, 0.185, 0.37),
               thr=(0.1, 0.5, 1.0), ai=8, out="/hpc/projects/icd.fast.ops/models/diffex/morpho_grid_sweep.png"):
    """Sweep the two frangi knobs — pixel_size_um (rows) × threshold_mult (cols) — on one generated crop,
    render seg boundaries + object counts, so we can pick the cleanest (signal, no noise). No postprocess."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import zarr
    from skimage.segmentation import find_boundaries
    plt.rcParams["pdf.fonttype"] = 42
    zp, _ = build_mini_zarr(marker_dir, target, "geneKO", real_exp, ["Phase2D", marker_channel], 1)
    os.environ["OPS_OUTPUT_BASE_DIR"] = SYNTH_BASE
    from organelle_profiler.organelle_seg.organelle_segmentation import segment_single_position_channel
    _, dp = _resolve_seg(real_exp, marker_channel)
    root = zarr.open(zp, mode="r"); img = np.asarray(root[f"A/{ai}/0/0"][0, -1, 0])[:, :CROP]
    fig, ax = plt.subplots(len(pix), len(thr) + 1, figsize=((len(thr) + 1) * 2.3, len(pix) * 2.3))
    for ri, px in enumerate(pix):
        ax[ri, 0].imshow(img, cmap="gray"); ax[ri, 0].set_ylabel(f"px={px}", fontsize=9)
        ax[ri, 0].set_xticks([]); ax[ri, 0].set_yticks([])
        if ri == 0: ax[ri, 0].set_title("raw", fontsize=9)
        for cj, t in enumerate(thr):
            fp = dict(dp); fp["threshold"] = None; fp["threshold_mult"] = t; fp["pixel_size_um"] = px
            r = segment_single_position_channel(experiment=real_exp, position=f"A/{ai}/0", channel_key=marker_channel,
                                                structure_type=None, use_clahe=True, frangi_params=fp)
            lab = np.asarray(root[f"A/{ai}/0/labels/{r['output_label']}/0"][0, 0, 0])[:, :CROP]
            n = len(np.unique(lab)) - 1; b = find_boundaries(lab)
            a = ax[ri, cj + 1]; a.imshow(img, cmap="gray"); a.imshow(np.ma.masked_where(~b, b), cmap="autumn"); a.axis("off")
            if ri == 0: a.set_title(f"thr={t}", fontsize=9)
            a.text(4, 20, str(n), color="cyan", fontsize=8)
    fig.suptitle(f"{target} frangi: pixel_size (rows) × threshold_mult (cols) — obj count cyan", fontsize=11)
    fig.tight_layout(); fig.savefig(out, dpi=125, bbox_inches="tight"); print(f"[sweep] {out}")


def _resolve_seg(real_exp, marker_channel):
    """Config-matched (method, detection_params) for this channel — from org_seg_params, no hardcoding."""
    os.environ.setdefault("OPS_OUTPUT_BASE_DIR", SYNTH_BASE)
    from organelle_profiler.organelle_seg.channel_processor import resolve_single_channel_info
    from organelle_profiler.organelle_seg.metadata import _determine_processing_params
    ci = resolve_single_channel_info(marker_channel, ["Phase2D", marker_channel], experiment=real_exp)
    dp, _, m = _determine_processing_params(organelle_key=ci.get("organelle_key", marker_channel),
        source_channel=marker_channel, structure_type=None, ch_info=ci, frangi_params=None,
        clahe_params=None, post_clahe_smoothing_sigma=None)
    return m, (dp or {})


def _auto_ref_map(network, channel):
    """Store (op_cp_features) feature names for the aggregates — network markers use op_network_<ch>_*."""
    ch = channel.lower()
    if network:
        return {"count": f"op_network_{ch}_num_skeleton_components", "total_skel": f"op_network_{ch}_total_branch_length",
                "mean_skel": f"op_network_{ch}_branch_length_mean", "total_area": f"op_{ch}_area_sum",
                "mean_int": f"op_{ch}_intensity_mean_mean"}
    return {"count": f"op_{ch}_count", "total_area": f"op_{ch}_area_sum", "mean_area": f"op_{ch}_area_mean",
            "mean_int": f"op_{ch}_intensity_mean_mean", "mean_ecc": f"op_{ch}_eccentricity_mean"}


def run_target(marker_dir, target, real_exp, marker_channel, store_marker, grain="geneKO", n_cells=6,
               refs=True, adaptive_mult=0.1, fake_pixel_um=0.065):
    """Fully config-driven: seg method + params AUTO from org_seg_params; network-vs-vesicular feature set
    + store ref-map AUTO from the resolved method. For frangi on the GENERATED (fake) images, switch to the
    ADAPTIVE dynamic threshold (compute_frangi_threshold) — the config's fixed threshold mis-fits the fake
    intensity/noise. mini-zarr → REAL seg → readback + real_ref (+ ref cells)."""
    method, dp = _resolve_seg(real_exp, marker_channel)
    network = method == "frangi"                                  # frangi/tubular → network features (skel), else vesicular
    ref_map = _auto_ref_map(network, marker_channel)
    fp = None
    if network:                                                   # adapt frangi to the fake resolution
        fp = dict(dp); fp["threshold"] = None; fp["threshold_mult"] = adaptive_mult
        fp["pixel_size_um"] = fake_pixel_um                       # smaller px → larger sigmas → coarse network, not noise
    print(f"[run_target] {marker_dir}/{target}: method={method} network={network} px={fake_pixel_um if network else '-'}")
    zpath, na = build_mini_zarr(marker_dir, target, grain, real_exp, ["Phase2D", marker_channel], n_cells)
    res = run_seg(real_exp, marker_channel, na, structure_type=None, frangi_params=fp)   # config auto-resolves; fp=adaptive
    label_name = next((r[3] for r in res if r[1] and r[3]), None)
    if not label_name:
        print("[run_target] seg produced no label — abort"); return
    readback_cache(marker_dir, target, grain, real_exp, label_name, n_cells,
                   store_marker=store_marker, network=network, ref_map=ref_map)
    if refs:
        reference_from_direction(marker_dir, target, real_exp, marker_channel, structure_type=None,
                                 network=network, n_cells=min(4, n_cells))


def validate(marker_dir="lysosome_LAMP1", target="ABCE1", grain="geneKO",
             real_exp="ops0047_20250612", marker_channel="GFP", structure_type="vesicular", n_cells=6):
    """Build the mini-zarr + run the REAL org-seg on a demo traversal → per-α num_objects (should
    amplify with α if the phenotype does). Proves the reuse works end-to-end."""
    zpath, na = build_mini_zarr(marker_dir, target, grain, real_exp, ["Phase2D", marker_channel], n_cells)
    res = run_seg(real_exp, marker_channel, na, structure_type)
    nobj = [r[2] for r in res if r[1]]
    print(f"\n[validate] {marker_dir}/{target}: real org-seg num_objects per α = {nobj}")


def full_features(marker_dir, target, real_exp, marker_channel, grain="geneKO", n_cells=6,
                  fake_pixel_um=0.05, adaptive_mult=0.1, out_root=None):
    """REAL org-profiler feature extraction on the generated (cell, α) crops — NO skimage shortcut.
    Stage traversal (mini-zarr) → production org-seg → per crop run `process_single_cell`
    (extract_organelle_features + calculate_network_features), aggregate objects→cell with the pipeline's
    AGGREGATION_FUNCTIONS. Writes a full per-(cell,α) feature table (op_cp-comparable) → cache parquet.
    Each generated crop is treated as one cell (whole-crop cell mask, per design)."""
    import pandas as pd
    import zarr
    from organelle_profiler.feature_extraction.fe_workers import process_single_cell
    from organelle_profiler.feature_extraction.fe_constants import AGGREGATION_FUNCTIONS
    method, dp = _resolve_seg(real_exp, marker_channel)
    network = method == "frangi"
    fp = None
    if network:                                                   # adaptive frangi on the fake images (as in run_target)
        fp = dict(dp); fp["threshold"] = None; fp["threshold_mult"] = adaptive_mult; fp["pixel_size_um"] = fake_pixel_um
    zpath, na = build_mini_zarr(marker_dir, target, grain, real_exp, ["Phase2D", marker_channel], n_cells)
    res = run_seg(real_exp, marker_channel, na, structure_type=None, frangi_params=fp)
    label_name = next((r[3] for r in res if r[1] and r[3]), None)
    if not label_name:
        print("[full] seg produced no label — abort"); return
    organelle, chans, sp = label_name, ["Phase2D", marker_channel], (fake_pixel_um, fake_pixel_um)
    netorg = [organelle] if network else []
    orgmap = {organelle: marker_channel}
    root = zarr.open(zpath, mode="r")
    rows = []
    for ai in range(na):
        lab = np.asarray(root[f"A/{ai}/0/labels/{label_name}/0"][0, 0, 0]).astype(np.int32)   # (Y, W)
        img = np.asarray(root[f"A/{ai}/0/0"][0, :, 0])                                          # (C, Y, W)
        for c in range(n_cells):
            x0 = c * (CROP + PAD)
            org_crop = lab[:, x0:x0 + CROP]
            if org_crop.max() == 0:
                continue
            inten = img[:, :, x0:x0 + CROP].astype(np.float32)                                  # (C, Y, CROP)
            cf, of, _nf = process_single_cell(
                cell_info={"global_cell_id": f"{target}_a{ai:02d}_c{c}", "well": f"A/{ai}/0"},
                cell_specific_mask=np.ones((CROP, CROP), np.uint8),
                organelle_mask_arrays={organelle: org_crop}, intensity_image=inten,
                frangi_image_arrays={}, organelles_to_process=[organelle], network_organelles=netorg,
                spacing=sp, channel_names=chans, organelle_map=orgmap, full_features=True)
            row = {k: v for k, v in cf.items()}
            row["alpha_idx"], row["cell"] = ai, c
            odf = of.get(organelle) if of else None
            if odf is not None and len(odf):                       # aggregate per-object features → cell (pipeline AGG_FUNCS)
                num = odf.select_dtypes(include="number")
                for feat in num.columns:
                    for fn in AGGREGATION_FUNCTIONS:
                        row[f"obj_{feat}_{fn}"] = int(num[feat].count()) if fn == "count" else float(getattr(num[feat], fn)())
            rows.append(row)
    df = pd.DataFrame(rows)
    out = out_root or f"{CACHE}/_morphometrics/{marker_dir}/{grain}/{target}"
    os.makedirs(out, exist_ok=True)
    fp_out = f"{out}/full_features.parquet"
    df.to_parquet(fp_out)
    print(f"[full] {marker_dir}/{grain}/{target}: {len(df)} (cell,α) rows × {df.shape[1]} cols -> {fp_out}")
    print(f"  sample feature cols: {[c for c in df.columns if 'network' in c or c.startswith('obj_area')][:8]}")
    return fp_out


def full_features_cache(marker_dir, target, grain, store_marker):
    """Turn full_features.parquet into a demo-navigable JSON: per-α mean trajectory for EVERY feature +
    the real NTC→KO mean±SEM reference (from op_cp_features) for each feature that maps to a store column +
    grouped feature lists for the dropdown. → <base>/full_features.json (read by morpho_demo.html)."""
    import json
    import anndata as ad
    import numpy as np
    import pandas as pd
    base = f"{CACHE}/_morphometrics/{marker_dir}/{grain}/{target}"
    df = pd.read_parquet(f"{base}/full_features.parquet")
    src = f"{CACHE}/{marker_dir}/{grain}/{target}"
    mp = f"{src}/cell0/meta.json" if os.path.exists(f"{src}/cell0/meta.json") else f"{src}/meta.json"
    alphas = json.load(open(mp))["alphas"]; na = len(alphas)
    num = df.select_dtypes(include="number")
    feats = [c for c in num.columns if c not in ("alpha_idx", "cell")]
    per = df.groupby("alpha_idx")
    agg = {f: [float(per[f].mean().get(ai, np.nan)) for ai in range(na)] for f in feats}

    import re
    CH = "mcherry"                                                 # infer the channel from the network_<ch>_seg_ columns
    for f in feats:
        mm = re.match(r"network_(\w+?)_seg_", f)
        if mm:
            CH = mm.group(1); break

    def store_col(f):                                              # generated feature name → op_cp store column (channel-aware)
        mm = re.match(r"network_(\w+?)_seg_(.+)", f)
        if mm:
            return f"op_network_{mm.group(1)}_{mm.group(2)}"
        if f.startswith("obj_"):
            return f"op_{CH}_{f[len('obj_'):]}"
        return None

    a = ad.read_h5ad(OPCP.format(store=store_marker), backed="r")
    var = set(map(str, a.var_names))
    mapped = {f: store_col(f) for f in feats}
    scols = sorted({v for v in mapped.values() if v in var})
    gn = a.obs["gene_name"].astype(str).values
    i_ntc = np.where(gn == "")[0]
    if grain == "complex":                                        # complex KO ref = its member genes' real cells (op_cp is gene-keyed)
        import yaml
        from ..classifier.config import slugify
        y = yaml.safe_load(open("/hpc/projects/icd.fast.ops/configs/gene_clusters/EBI_complexes_v1_updated_gene_names.yaml"))
        members = next((e["genes"] for e in y.values() if slugify(e["name"]) == target), [])
        i_ko = np.where(np.isin(gn, members))[0]
        print(f"  [complex] {target}: {len(members)} member genes, {len(i_ko)} real cells")
    else:
        i_ko = np.where(gn == target)[0]
    # 'centroid' = the real population mean over ALL cells (all genes) — the average-cell reference (sampled for speed)
    n_all = len(gn); i_cen = np.random.default_rng(0).choice(n_all, min(50000, n_all), replace=False)
    rows_idx = np.concatenate([i_ntc, i_ko, i_cen])
    sub = a[rows_idx, scols].to_memory().to_df()
    lab = np.array([""] * len(i_ntc) + [target] * len(i_ko) + ["_CEN"] * len(i_cen))

    def msem(v):
        v = v[~np.isnan(v)]
        return [float(v.mean()), float(v.std() / max(1, len(v)) ** 0.5)] if len(v) else None
    real_ref = {}
    for f, sc in mapped.items():
        if sc in scols:
            n, k = msem(sub.loc[lab == "", sc].values), msem(sub.loc[lab == target, sc].values)
            cen = msem(sub.loc[lab == "_CEN", sc].values)
            if n and k:
                real_ref[f] = {"ntc": n, "ko": k, "cen": cen}

    def grp(f):
        if f.startswith("network_"):
            return "Network"
        if any(k in f.lower() for k in ["haralick", "glcm", "zernike", "moment", "texture"]):
            return "Texture"
        if "intensity" in f:
            return "Intensity"
        if f.startswith("obj_"):
            return "Object morphology"
        return "Cell"
    groups = {}
    for f in feats:
        groups.setdefault(grp(f), []).append(f)
    json.dump({"alphas": alphas, "agg": agg, "real_ref": real_ref, "groups": groups, "target": target,
               "marker_dir": marker_dir, "n_features": len(feats), "n_with_ref": len(real_ref)},
              open(f"{base}/full_features.json", "w"))
    print(f"[full-cache] {marker_dir}/{grain}/{target}: {len(feats)} features ({len(real_ref)} with real NTC→KO ref) "
          f"-> {base}/full_features.json")
    return f"{base}/full_features.json"


if __name__ == "__main__":
    validate()
