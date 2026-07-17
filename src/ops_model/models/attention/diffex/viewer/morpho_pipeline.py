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
GEN_CROP = 160   # DiffEx cfg.crop_size: the generated crops are 160 px (native) upsized to 256 — real ref crops must match this window


def _json_safe(o):   # non-finite floats (NaN/±Inf) → None so the browser's strict JSON.parse accepts the file
    import math
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    return o


def _clip_border(lab, m=5):
    """Zero the m-px border band: segmentations touching the crop edge are shrunk to end ~m px inside it, and
    we measure on the clipped objects. Simpler than per-object edge/reach filtering — no whole-object drops."""
    if m <= 0:
        return lab
    out = lab.copy()
    out[:m, :] = 0; out[-m:, :] = 0; out[:, :m] = 0; out[:, -m:] = 0
    return out


# Masked-Object (MO) nucleoli seg — copied from coding_exps/nucleoli_roundness (_segment_threshold),
# reusing the real apply_intensity_threshold. NPM3 nucleoli are round blobs → frangi vesselness under-
# detects them; MO (intensity threshold + per-object local adjust) is the right detector. No nucleus
# tile_mask here (the generated crop is already a single cell).
MO_PARAMS = {"threshold_method": "masked_object", "threshold_factor": 1.0,
             "mo_global_method": "triangle", "mo_local_adjust": 1.3,
             "mo_object_min_area_px": 15, "min_object_size": 15}


def _nucleus_mask(img):
    """Central-cell nucleus mask for NPM3 crops (no separate nucleus channel): blurred Otsu, flood-guarded
    (if it covers >45% of the crop, fall back to the 88th percentile), keep only the central component —
    kills off-nucleus background specks that plain MO would otherwise label on noisy high-α frames."""
    from skimage.filters import threshold_otsu, gaussian
    from skimage.morphology import binary_closing, disk
    from scipy import ndimage as ndi
    g = gaussian(img, 2)
    v = g[g > 0]
    if v.size == 0:
        return np.zeros(img.shape, bool)
    nuc = ndi.binary_fill_holes(binary_closing(g > threshold_otsu(v), disk(3)))
    if nuc.mean() > 0.45:
        nuc = ndi.binary_fill_holes(binary_closing(g > np.percentile(g, 88), disk(3)))
    lab, n = ndi.label(nuc)
    if n == 0:
        return nuc
    cy, cx = np.array(img.shape) // 2
    cen = lab[cy, cx] or (1 + int(np.argmax(np.bincount(lab.ravel())[1:])))
    return lab == cen


def _seg_masked_object(img, tp=MO_PARAMS, nucleus=False):
    """MO intensity seg on one 2D crop → int32 labels (fill_holes → CC label → min-size).
    nucleus=True → constrain to the central nucleus (NPM3 nucleoli within the nucleus)."""
    from scipy import ndimage as ndi
    from skimage.morphology import binary_erosion, disk
    from organelle_profiler.organelle_seg.thresholding import apply_intensity_threshold
    binary = apply_intensity_threshold(
        img, method=tp["threshold_method"], threshold_factor=tp.get("threshold_factor", 1.0),
        mo_global_method=tp.get("mo_global_method", "triangle"),
        mo_local_adjust=tp.get("mo_local_adjust", 0.98),
        mo_object_min_area_px=tp.get("mo_object_min_area_px", 100))
    if nucleus:
        nuc = _nucleus_mask(img)
        if not nuc.any():
            return np.zeros(img.shape, np.int32)
        binary = binary & binary_erosion(nuc, disk(2))
    binary = ndi.binary_fill_holes(binary)
    fp = ndi.generate_binary_structure(binary.ndim, 1)
    objs, _ = ndi.label(binary, structure=fp)
    ms = tp.get("min_object_size", 0)
    if ms > 0 and objs.max() > 0:
        ids, cnt = np.unique(objs, return_counts=True)
        small = ids[(ids > 0) & (cnt < ms)]
        if small.size:
            objs[np.isin(objs, small)] = 0
            objs, _ = ndi.label(objs > 0, structure=fp)
    return objs.astype(np.int32)


def _seg_strip_mo(img, n_cells, nucleus=False):
    """Run MO per generated crop within the α strip (percentile-normalized per crop, like the standalone
    _prep_npm3), assembling a strip-wide int32 label array with globally-unique ids."""
    Y, W = img.shape
    strip = np.zeros((Y, W), np.int32)
    for c in range(n_cells):
        x0 = c * (CROP + PAD)
        crop = img[:, x0:x0 + CROP]
        if crop.max() <= 0:
            continue
        lo, hi = np.percentile(crop, [1, 99.5])
        cn = np.clip((crop - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)
        objs = _seg_masked_object(cn, nucleus=nucleus)
        m = objs > 0
        if m.any():
            objs[m] += int(strip.max())
            strip[:, x0:x0 + CROP][m] = objs[m]
    return strip


def _run_seg_masked_object(zpath, n_alpha, label_name, nucleus=False):
    """MO seg branch of run_seg: read each α strip (marker=last channel), segment per crop, and write the
    labels into the mini-zarr via the production label writer so readback/full_features read them unchanged."""
    import zarr
    from ops_utils.io.zarr_labels import _init_organelle_label_array, _update_labels_metadata
    root = zarr.open(zpath, mode="r")
    W = int(np.asarray(root["A/0/0/0"]).shape[-1])
    n_cells = round((W + PAD) / (CROP + PAD))
    out = []
    for ai in range(n_alpha):
        img = np.asarray(root[f"A/{ai}/0/0"][0, -1, 0]).astype(np.float32)   # marker channel = last
        lab = _seg_strip_mo(img, n_cells, nucleus=nucleus)
        Y, Wx = lab.shape
        _init_organelle_label_array(zpath, f"A/{ai}/0", label_name, shape=(1, 1, 1, Y, Wx))
        store = zarr.open(zpath, mode="r+")
        store[f"A/{ai}/0/labels/{label_name}/0"][0, 0, 0] = lab
        _update_labels_metadata(zpath, f"A/{ai}/0", label_name)
        out.append((ai, True, int(lab.max()), label_name, None))
        print(f"  α_idx {ai}: MO n_obj={int(lab.max())} label={label_name}")
    return out


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


def run_seg(real_exp, marker_channel, n_alpha, structure_type=None, base_dir=SYNTH_BASE, frangi_params=None, method=None, label_name=None, mo_nucleus=False):
    """Run the REAL production org-seg on each α position of the mini-zarr (config resolved from the
    real experiment's channel map). `frangi_params` overrides the resolved frangi config (e.g. to switch
    to the ADAPTIVE dynamic threshold on the generated images). Writes labels; returns per-α results.
    method="masked_object" → the MO intensity-threshold path (not wired through segment_single_position_channel);
    writes labels under `label_name` directly. mo_nucleus=True → nucleus-constrained MO (NPM3 nucleoli)."""
    os.environ["OPS_OUTPUT_BASE_DIR"] = base_dir
    if method == "masked_object":
        return _run_seg_masked_object(f"{base_dir}/{real_exp}/3-assembly/phenotyping_v3.zarr",
                                      n_alpha, label_name or "organelle_seg", nucleus=mo_nucleus)
    from organelle_profiler.organelle_seg.organelle_segmentation import segment_single_position_channel
    out = []
    for ai in range(n_alpha):
        r = segment_single_position_channel(experiment=real_exp, position=f"A/{ai}/0",
                                            channel_key=marker_channel, structure_type=structure_type,
                                            use_clahe=True, frangi_params=frangi_params, method=method)
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
            lc = relabel(_clip_border(lab[:, x0:x0 + CROP]) > 0)             # clip border band, then relabel within crop
            ic = img[:, x0:x0 + CROP]
            rp = [r for r in regionprops(lc, intensity_image=ic) if r.area >= 3]   # border already clipped; just drop tiny
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
                             network=False, n_cells=4, base_dir=SYNTH_BASE, adaptive=True, grain="geneKO"):
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
    npz = glob.glob(f"/hpc/projects/icd.fast.ops/models/diffex/directions/{marker_dir}/{grain}/{target}/cache/crops_{target}_*.npz")
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
    chans0 = ["Phase2D"] if marker_channel == "Phase2D" else ["Phase2D", marker_channel]   # phase: single channel (crop IS phase; avoid empty-slot collision)
    with open_ome_zarr(zpath, layout="hcs", mode="w", channel_names=chans0, version="0.5") as ds:
        for gi, g in enumerate(glist):
            cr = grp_cells[g]; W = len(cr) * CROP + (len(cr) - 1) * PAD; strip = np.zeros((len(chans0), CROP, W), np.float32)
            for c, x in enumerate(cr):
                strip[-1, :, c * (CROP + PAD):c * (CROP + PAD) + CROP] = x
            ds.create_position("A", str(gi), "0").create_image("0", strip[None, :, None])
    os.environ["OPS_OUTPUT_BASE_DIR"] = base_dir
    from organelle_profiler.organelle_seg.organelle_segmentation import segment_single_position_channel
    fp = None
    if network:
        from skimage.morphology import skeletonize   # used below for per-object skel features
        if adaptive:                                             # match the generated: adaptive dynamic threshold (adaptive=False → perfected config)
            _, dp = _resolve_seg(real_exp, marker_channel); fp = dict(dp)
            fp["threshold"] = None; fp["threshold_mult"] = 0.1; fp["pixel_size_um"] = 0.065   # match generated seg
    out = f"{CACHE}/_morphometrics/{marker_dir}/{grain}/{target}"; os.makedirs(out, exist_ok=True)
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


def reference_from_store(marker_dir, target, grain, image_channel, org_label, n_cells=4, network=True, cache=CACHE, store_exps=None):
    """Cache real-cell reference thumbnails + PRODUCTION org-label overlays for the morpho demo. Each cell is
    cropped ONCE (build-time) from its experiment's phenotyping_v3.zarr — marker image channel + the on-disk
    production seg label — and written as static webp/png (the _v3 store is NOT on S3; the web reads the cache).
    Cells = top-1k ACCURACY KO (member genes for a complex) + top-attention NTC, matching the plot's real_ref."""
    import json
    import re
    import pandas as pd
    import zarr
    from PIL import Image
    from skimage.measure import label as relabel, regionprops
    from ..classifier.config import GRAINS
    from skimage.morphology import skeletonize
    GK = GRAINS["geneKO"]["parquet"]                                          # attention parquet (has NTC + coords)
    ACC = "/hpc/projects/icd.fast.ops/models/diffex/accuracy_ranking/phase_geneKO_topacc_ALL_top1000.parquet"
    COLS = ["gene", "experiment", "well", "segmentation", "x_pheno", "y_pheno", "rank", "rank_type"]

    def _cells(pq, genes, n):                                                 # top-n ranked 'top' cells for gene(s)
        d = pd.read_parquet(pq, columns=COLS)
        d = d[(d["gene"].astype(str).isin([str(x) for x in genes])) & (d["rank_type"] == "top")]
        if store_exps:                                                        # only experiments where the marker channel IS this reporter (fluor markers are experiment-specific)
            d = d[d["experiment"].astype(str).isin(set(store_exps))]
        return d.sort_values("rank").head(n)
    if grain == "complex":
        import yaml
        from ..classifier.config import slugify
        y = yaml.safe_load(open("/hpc/projects/icd.fast.ops/configs/gene_clusters/EBI_complexes_v1_updated_gene_names.yaml"))
        members = next((e["genes"] for e in y.values() if slugify(e["name"]) == target), [])
        ko = _cells(ACC, members, n_cells)
    else:
        ko = _cells(ACC, [target], n_cells)
    ntc = _cells(GK, ["NTC"], n_cells)
    groups = {"NTC": ntc, target: ko}
    half = CROP // 2
    import shutil
    out = f"{cache}/_morphometrics/{marker_dir}/{grain}/{target}"; os.makedirs(out, exist_ok=True)
    shutil.rmtree(f"{out}/_ref", ignore_errors=True)                         # clear stale ref cells (e.g. a prior recompute build)
    ref, glist = {}, []
    for g, df in groups.items():
        gd = f"{out}/_ref/{g}"; os.makedirs(gd, exist_ok=True); cells = []
        for c, (_, row) in enumerate(df.iterrows()):
            exp = str(row["experiment"]); w = str(row["well"]).strip()
            m = re.match(r"^([A-Za-z]+)(\d+)$", w); pos = w if w.count("/") == 2 else (f"{m.group(1)}/{m.group(2)}/0" if m else w)
            zp = f"/hpc/projects/icd.fast.ops/{exp}/3-assembly/phenotyping_v3.zarr"
            if not os.path.exists(zp):
                continue
            try:
                z = zarr.open(zp, mode="r"); P = z[pos]
                chans = [ch.get("label") for ch in dict(P.attrs).get("ome", {}).get("omero", {}).get("channels", [])]
                ci = chans.index(image_channel) if image_channel in chans else 0
                from skimage.transform import resize as _rs
                y, x = int(round(float(row["y_pheno"]))), int(round(float(row["x_pheno"])))
                h = GEN_CROP // 2                                             # SAME fixed window as the generated crops (cfg.crop_size=160), then upsize to CROP — matches traversal framing exactly
                imc = np.asarray(P["0"][0, ci, 0, max(0, y - h):y + h, max(0, x - h):x + h]).astype(np.float32)
                lbc = np.asarray(P["labels"][org_label]["0"][0, 0, 0, max(0, y - h):y + h, max(0, x - h):x + h]).astype(np.int32)
                im = _rs(imc, (CROP, CROP), preserve_range=True, anti_aliasing=True).astype(np.float32)
                lb = _rs(lbc.astype(np.float32), (CROP, CROP), order=0, preserve_range=True, anti_aliasing=False).astype(np.int32)
            except Exception as e:
                print(f"[ref-store] {g} cell{c} {exp} {pos} crop failed: {repr(e)[:100]}"); continue
            if im.shape != (CROP, CROP):                                      # pad edge crops to full tile
                im = np.pad(im, [(0, CROP - im.shape[0]), (0, CROP - im.shape[1])]); lb = np.pad(lb, [(0, CROP - lb.shape[0]), (0, CROP - lb.shape[1])])
            lo, hi = np.percentile(im, [1, 99]); imn = np.clip((im - lo) / (hi - lo + 1e-6), 0, 1)
            Image.fromarray((imn * 255).astype(np.uint8)).save(f"{gd}/cell{c}.webp")
            lc = relabel(_clip_border(lb) > 0)                               # clip border band, then measure clipped objects
            keep = [r for r in regionprops(lc, intensity_image=im) if r.area >= 3][:255]
            remap = {r.label: i + 1 for i, r in enumerate(keep)}; mask8 = np.zeros(lc.shape, np.uint8); feats = {}
            for r in keep:
                mask8[lc == r.label] = remap[r.label]
                o = {"area": float(r.area), "mean_int": float(r.intensity_mean), "ecc": float(r.eccentricity)}
                if network:
                    o["skel"] = float(skeletonize(lc == r.label).sum())
                feats[str(remap[r.label])] = o
            Image.fromarray(mask8).save(f"{gd}/cell{c}_labels.png")
            cells.append({"cell": c, "feats": feats})
        if cells:
            ref[g] = cells; glist.append(g)
    json.dump({"groups": glist, "n_cells": n_cells, "cells": ref, "img_dir": "_ref"}, open(f"{out}/_ref.json", "w"))
    print(f"[ref-store] {marker_dir}/{grain}/{target}: real cells {[(g, len(ref.get(g, []))) for g in glist]} (production {org_label}) -> {out}/_ref")


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
               refs=True, adaptive_mult=0.1, fake_pixel_um=0.065, adaptive=True, frangi_override=None, structure_type=None, seg_method=None, base_dir=SYNTH_BASE, org_label=None, mo_nucleus=False):
    """Fully config-driven: seg method + params AUTO from org_seg_params; network-vs-vesicular feature set
    + store ref-map AUTO from the resolved method. For frangi on the GENERATED (fake) images, switch to the
    ADAPTIVE dynamic threshold (compute_frangi_threshold) — the config's fixed threshold mis-fits the fake
    intensity/noise. mini-zarr → REAL seg → readback + real_ref (+ ref cells)."""
    method, dp = _resolve_seg(real_exp, marker_channel)
    network = (structure_type == "tubular") if structure_type else (method == "frangi")   # only tubular has a skeleton (vesicular = blob features)
    ref_map = _auto_ref_map(network, marker_channel)
    fp = None
    if network and adaptive:                                      # adapt frangi to the fake resolution (adaptive=False → perfected config params)
        fp = dict(dp); fp["threshold"] = None; fp["threshold_mult"] = adaptive_mult
        fp["pixel_size_um"] = fake_pixel_um                       # smaller px → larger sigmas → coarse network, not noise
    if frangi_override:                                           # per-target tweaks on the resolved config (e.g. lower pixel_size for NPM3 nucleoli)
        fp = dict(fp if fp else dp); fp.update(frangi_override)
    print(f"[run_target] {marker_dir}/{target}: method={method} network={network} adaptive={adaptive} override={frangi_override}")
    chans0 = ["Phase2D"] if marker_channel == "Phase2D" else ["Phase2D", marker_channel]   # phase: single channel (avoid empty-slot collision)
    zpath, na = build_mini_zarr(marker_dir, target, grain, real_exp, chans0, n_cells, base_dir=base_dir)
    res = run_seg(real_exp, marker_channel, na, structure_type=structure_type, frangi_params=fp, method=seg_method, base_dir=base_dir, label_name=org_label, mo_nucleus=mo_nucleus)   # config auto-resolves; fp=adaptive/override
    label_name = next((r[3] for r in res if r[1] and r[3]), None)
    if not label_name:
        print("[run_target] seg produced no label — abort"); return
    readback_cache(marker_dir, target, grain, real_exp, label_name, n_cells, base_dir=base_dir,
                   store_marker=store_marker, network=network, ref_map=ref_map)
    if refs:
        reference_from_direction(marker_dir, target, real_exp, marker_channel, structure_type=None,
                                 network=network, n_cells=min(4, n_cells), adaptive=adaptive, grain=grain)


def validate(marker_dir="lysosome_LAMP1", target="ABCE1", grain="geneKO",
             real_exp="ops0047_20250612", marker_channel="GFP", structure_type="vesicular", n_cells=6):
    """Build the mini-zarr + run the REAL org-seg on a demo traversal → per-α num_objects (should
    amplify with α if the phenotype does). Proves the reuse works end-to-end."""
    zpath, na = build_mini_zarr(marker_dir, target, grain, real_exp, ["Phase2D", marker_channel], n_cells)
    res = run_seg(real_exp, marker_channel, na, structure_type)
    nobj = [r[2] for r in res if r[1]]
    print(f"\n[validate] {marker_dir}/{target}: real org-seg num_objects per α = {nobj}")


def full_features(marker_dir, target, real_exp, marker_channel, grain="geneKO", n_cells=6,
                  fake_pixel_um=0.05, adaptive_mult=0.1, adaptive=True, out_root=None, frangi_override=None, structure_type=None, seg_method=None, base_dir=SYNTH_BASE, org_label=None, mo_nucleus=False):
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
    network = (structure_type == "tubular") if structure_type else (method == "frangi")   # only tubular has a skeleton (vesicular = blob features)
    fp = None
    if network and adaptive:                                      # adaptive frangi override for fake images
        fp = dict(dp); fp["threshold"] = None; fp["threshold_mult"] = adaptive_mult; fp["pixel_size_um"] = fake_pixel_um
    # adaptive=False → fp stays None → run_seg uses the perfected config params (e.g. Phase2D tubular) as-is
    if frangi_override:                                           # per-target tweaks on the resolved config (e.g. real NPM3/NucleoLIVE settings)
        fp = dict(fp if fp else dp); fp.update(frangi_override)
    chans0 = ["Phase2D"] if marker_channel == "Phase2D" else ["Phase2D", marker_channel]   # phase: single channel (generated frame IS phase; avoids the empty Phase2D-slot collision)
    zpath, na = build_mini_zarr(marker_dir, target, grain, real_exp, chans0, n_cells, base_dir=base_dir)
    res = run_seg(real_exp, marker_channel, na, structure_type=structure_type, frangi_params=fp, method=seg_method, base_dir=base_dir, label_name=org_label, mo_nucleus=mo_nucleus)
    label_name = next((r[3] for r in res if r[1] and r[3]), None)
    if not label_name:
        print("[full] seg produced no label — abort"); return
    px = (fp or dp).get("pixel_size_um", fake_pixel_um)                                    # match feature spacing to the seg's pixel size
    organelle, chans, sp = label_name, chans0, (px, px)
    netorg = [organelle] if network else []
    orgmap = {organelle: marker_channel}
    root = zarr.open(zpath, mode="r")
    rows = []; n_empty = 0
    for ai in range(na):
        lab = np.asarray(root[f"A/{ai}/0/labels/{label_name}/0"][0, 0, 0]).astype(np.int32)   # (Y, W)
        img = np.asarray(root[f"A/{ai}/0/0"][0, :, 0])                                          # (C, Y, W)
        for c in range(n_cells):
            x0 = c * (CROP + PAD)
            org_crop = _clip_border(lab[:, x0:x0 + CROP])                    # clip border band → measure clipped objects
            if org_crop.max() == 0:
                n_empty += 1; continue                                                          # empty seg → skip (logged below)
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
    print(f"[full] {marker_dir}/{grain}/{target}: {len(df)} (cell,α) rows × {df.shape[1]} cols "
          f"({n_empty} empty-seg cell×α dropped) -> {fp_out}")
    print(f"  sample feature cols: {[c for c in df.columns if 'network' in c or c.startswith('obj_area')][:8]}")
    return fp_out


def full_features_cache(marker_dir, target, grain, store_marker, store_channel=None):
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
    gsem = per[feats].std(ddof=1) / per[feats].count() ** 0.5   # per-α SEM over the generated cells (for demo/figure error bars)
    agg_sem = {f: [float(gsem[f].get(ai, 0.0)) for ai in range(na)] for f in feats}

    import re
    CH = "mcherry"                                                 # infer the channel from the network_<ch>_seg_ columns
    for f in feats:
        mm = re.match(r"network_(\w+?)_seg_", f)
        if mm:
            CH = mm.group(1); break

    def store_col(f):                                              # generated feature name → op_cp store column (channel-aware)
        mm = re.match(r"network_(\w+?)_seg_(.+)", f)
        if store_marker == "phase":                                # phase op_cp store uses the phase2d_tubular organelle naming
            if mm:
                return f"op_network_phase2d_tubular_{mm.group(2)}"
            if f.startswith("obj_"):
                return f"op_phase2d_tubular_{f[len('obj_'):]}"
            return None
        if mm:
            return f"op_network_{store_channel or mm.group(1)}_{mm.group(2)}"      # store_channel: seg organelle name ≠ store's physical channel (e.g. NPM3 nucleoli → 'gfp')
        if f.startswith("obj_"):
            return f"op_{store_channel or CH}_{f[len('obj_'):]}"
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
    if store_marker == "phase" and grain == "geneKO":             # restrict real ref: top-1k ACCURACY KO + top-attention NTC (rankings cover all phase experiments)
        import pandas as pd
        obs = a.obs; ekey = obs["experiment"].astype(str).values; wkey = obs["well"].astype(str).values; skey = obs["segmentation"].astype(str).values
        def _nw(w):
            w = str(w).strip()
            if w.count("/") == 2:
                return w
            m = re.match(r"^([A-Za-z]+)(\d+)$", w); return f"{m.group(1)}/{m.group(2)}/0" if m else w
        def _restrict(idx, pq, val, nkeep):                       # keep store rows among the top-nkeep ranked cells of `val`
            d = pd.read_parquet(pq, columns=["gene", "experiment", "well", "segmentation", "rank", "rank_type"])
            d = d[(d["gene"].astype(str) == str(val)) & (d["rank_type"] == "top")].sort_values("rank").head(nkeep)
            keys = set(zip(d["experiment"].astype(str), d["well"].map(_nw), d["segmentation"].astype(str)))
            return np.array([i for i in idx if (ekey[i], _nw(wkey[i]), skey[i]) in keys], dtype=int)
        from ..classifier.config import GRAINS
        ACC = "/hpc/projects/icd.fast.ops/models/diffex/accuracy_ranking/phase_geneKO_topacc_ALL_top1000.parquet"
        ko2, ntc2 = _restrict(i_ko, ACC, target, 1000), _restrict(i_ntc, GRAINS["geneKO"]["parquet"], "NTC", 1000)
        print(f"  [phase top-acc] KO {len(i_ko)}->{len(ko2)}, NTC(top-attn) {len(i_ntc)}->{len(ntc2)}")
        if len(ko2):
            i_ko = ko2
        if len(ntc2):
            i_ntc = ntc2
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
    ncell = int(df["cell"].max()) + 1 if "cell" in df.columns and len(df) else 0
    json.dump(_json_safe({"alphas": alphas, "agg": agg, "agg_sem": agg_sem, "real_ref": real_ref, "groups": groups, "target": target,
               "marker_dir": marker_dir, "n_cells": ncell, "n_features": len(feats), "n_with_ref": len(real_ref)}),
              open(f"{base}/full_features.json", "w"))
    print(f"[full-cache] {marker_dir}/{grain}/{target}: {len(feats)} features ({len(real_ref)} with real NTC→KO ref) "
          f"-> {base}/full_features.json")
    return f"{base}/full_features.json"


# morpho demo targets (SLURM-friendly, picklable). adaptive=False → the perfected org_seg config params.
MORPHO_TARGETS = {
    "MICOS13": dict(marker_dir="phase", target="MICOS13", real_exp="ops0047_20250612", marker_channel="Phase2D",
                    store_marker="phase", grain="geneKO", image_channel="Phase2D", org_label="phase2d_tubular_seg"),
    "TOMM20": dict(marker_dir="phase", target="TOMM20", real_exp="ops0047_20250612", marker_channel="Phase2D",
                   store_marker="phase", grain="geneKO", image_channel="Phase2D", org_label="phase2d_tubular_seg"),
    # CCT/NPM3: nucleoli are round blobs → frangi vesselness under-/over-detects them; use the MASKED-OBJECT
    # intensity path (from coding_exps/nucleoli_roundness, tuned NPM3 MO_PARAMS). pixel_size only sets feature spacing.
    "CCT": dict(marker_dir="nucleolus_GC_NPM3", target="Chaperonin_containing_T_complex", real_exp="ops0092_20251027",
                marker_channel="nucleolus-GC_NPM3", store_marker="nucleolus-gc_npm3", grain="complex",
                image_channel="GFP", org_label="gfp_seg", store_channel="gfp", structure_type="vesicular",
                seg_method="masked_object", mo_nucleus=True, frangi_override={"pixel_size_um": 0.825}),
    # POLR1B geneKO, same NPM3 nucleoli marker + nucleus-constrained MO seg as CCT
    "POLR1B": dict(marker_dir="nucleolus_GC_NPM3", target="POLR1B", real_exp="ops0092_20251027",
                marker_channel="nucleolus-GC_NPM3", store_marker="nucleolus-gc_npm3", grain="geneKO",
                image_channel="GFP", org_label="gfp_seg", store_channel="gfp", structure_type="vesicular",
                seg_method="masked_object", mo_nucleus=True, frangi_override={"pixel_size_um": 0.825}),
    # GBF1 geneKO on ER/Golgi COP-II (SEC23A) puncta — MO intensity seg (punctate blobs, like nucleoli)
    "GBF1": dict(marker_dir="ER_Golgi_COP_II_SEC23A", target="GBF1", real_exp="ops0081_20250924",
                marker_channel="ER_Golgi_COP-II_SEC23A", store_marker="er_golgi_cop-ii_sec23a", grain="geneKO",
                image_channel="GFP", org_label="gfp_seg", store_channel="gfp", structure_type="vesicular",
                seg_method="masked_object"),
    # NucleoLIVE (org_seg_params NucleoLIVE variant): tubular on mCherry; lower pixel than real 0.185 to suppress generated background
    "KIF23_NUCLEOLIVE": dict(marker_dir="nucleus_NucleoLIVE_Live_Cell_dye", target="KIF23", real_exp="ops0120_20260204",
                marker_channel="nucleus_NucleoLIVE Live Cell dye", store_marker="nuclei_nucleolive_live_cell_dye", grain="geneKO",
                image_channel="mCherry", org_label="mcherry_seg", store_channel="mcherry", structure_type="tubular",
                frangi_override={"pixel_size_um": 0.09, "min_object_size": 2}),
    "TIM23_PHASE": dict(marker_dir="phase", target="TIM23_mitochondrial_inner_membrane_pre_sequence_translocase_complex__TIM17A_variant",
                real_exp="ops0047_20250612", marker_channel="Phase2D", store_marker="phase", grain="complex",
                image_channel="Phase2D", org_label="phase2d_tubular_seg", structure_type="tubular"),
    "CHROMALIVE_TIM23": dict(marker_dir="mitochondria_ChromaLIVE_561_excitation",
                target="TIM23_mitochondrial_inner_membrane_pre_sequence_translocase_complex__TIM17A_variant", real_exp="ops0122_20260211",
                marker_channel="mitochondria_ChromaLIVE 561 excitation", store_marker="mitochondria_chromalive_561_excitation",
                grain="complex", image_channel="mCherry", org_label="mcherry_seg", store_channel="mcherry", structure_type="tubular"),
    "CHROMALIVE_MICOS13": dict(marker_dir="mitochondria_ChromaLIVE_561_excitation", target="MICOS13", real_exp="ops0122_20260211",
                marker_channel="mitochondria_ChromaLIVE 561 excitation", store_marker="mitochondria_chromalive_561_excitation",
                grain="geneKO", image_channel="mCherry", org_label="mcherry_seg", store_channel="mcherry", structure_type="tubular"),
    "CHROMALIVE_TOMM20": dict(marker_dir="mitochondria_ChromaLIVE_561_excitation", target="TOMM20", real_exp="ops0122_20260211",
                marker_channel="mitochondria_ChromaLIVE 561 excitation", store_marker="mitochondria_chromalive_561_excitation",
                grain="geneKO", image_channel="mCherry", org_label="mcherry_seg", store_channel="mcherry", structure_type="tubular"),
    # FastAct actin filaments (mCherry, tubular → real config = frangi, like ChromaLIVE mito)
    "ARP23_FASTACT": dict(marker_dir="actin_filament_FastAct_SPY555_Live_Cell_Dye",
                target="Actin_related_protein_2_3_complex__ARPC1A_ACTR3B_ARPC5_variant", real_exp="ops0076_20250917",
                marker_channel="actin filament_FastAct SPY555 Live Cell Dye", store_marker="actin_filament_fastact_spy555_live_cell_dye",
                grain="complex", image_channel="mCherry", org_label="mcherry_seg", store_channel="mcherry", structure_type="tubular",
                frangi_override={"pixel_size_um": 0.06}),   # lower px → coarser frangi → traces filaments, drops generated-image noise
    "CAPZB_FASTACT": dict(marker_dir="actin_filament_FastAct_SPY555_Live_Cell_Dye", target="CAPZB", real_exp="ops0076_20250917",
                marker_channel="actin filament_FastAct SPY555 Live Cell Dye", store_marker="actin_filament_fastact_spy555_live_cell_dye",
                grain="geneKO", image_channel="mCherry", org_label="mcherry_seg", store_channel="mcherry", structure_type="tubular",
                frangi_override={"pixel_size_um": 0.06}),
}


def build_morpho(marker_dir, target, real_exp, marker_channel, store_marker, grain="geneKO", n_cells=12,
                 image_channel=None, org_label=None, adaptive=False, store_channel=None, frangi_override=None, structure_type=None, seg_method=None, mo_nucleus=False):
    """One morpho-demo target end to end (the wrapper that bundles EVERYTHING into one target dir, so nothing is
    scattered): generated seg-mask overlays (a*_labels.png) + per-object feats (a*_feats.json) + full org-profiler
    feature table (full_features.parquet) + demo JSON w/ per-α trajectory + top-accuracy store real-ref stats
    (full_features.json) + CACHED production-label real-cell images & seg overlays (_ref/). adaptive=False = perfected config."""
    from ..classifier.config import slugify
    base_dir = f"{SYNTH_BASE}/job_{grain}_{slugify(target)}"   # per-target staging dir → jobs can run fully in parallel (no shared-zarr race)
    os.environ["OPS_OUTPUT_BASE_DIR"] = base_dir
    run_target(marker_dir, target, real_exp, marker_channel, store_marker=store_marker, grain=grain, n_cells=n_cells,
               refs=False, adaptive=adaptive, frangi_override=frangi_override, structure_type=structure_type, seg_method=seg_method, base_dir=base_dir, org_label=org_label, mo_nucleus=mo_nucleus)   # generated seg-mask overlays + per-object feats
    full_features(marker_dir, target, real_exp, marker_channel, grain=grain, n_cells=n_cells, adaptive=adaptive,
                  frangi_override=frangi_override, structure_type=structure_type, seg_method=seg_method, base_dir=base_dir, org_label=org_label, mo_nucleus=mo_nucleus)
    full_features_cache(marker_dir, target, grain, store_marker=store_marker, store_channel=store_channel)
    if image_channel and org_label:                                          # cached real-cell images w/ PRODUCTION org labels (built-time crop from _v3)
        exps = None
        if store_marker != "phase":                                          # fluor markers: restrict real cells to experiments where this channel IS the reporter
            import anndata as ad
            exps = set(ad.read_h5ad(OPCP.format(store=store_marker), backed="r").obs["experiment"].astype(str).unique())
        reference_from_store(marker_dir, target, grain, image_channel, org_label,
                             n_cells=min(4, n_cells), network=("tubular" in org_label), store_exps=exps)
    return f"{CACHE}/_morphometrics/{marker_dir}/{grain}/{target}"


def build_morpho_target(key, n_cells=12):
    return build_morpho(n_cells=n_cells, **MORPHO_TARGETS[key])


def build_phase_morpho(target, real_exp="ops0047_20250612", n_cells=12, grain="geneKO"):   # back-compat wrapper
    return build_morpho("phase", target, real_exp, "Phase2D", "phase", grain=grain, n_cells=n_cells,
                        image_channel="Phase2D", org_label="phase2d_tubular_seg")


if __name__ == "__main__":
    validate()
