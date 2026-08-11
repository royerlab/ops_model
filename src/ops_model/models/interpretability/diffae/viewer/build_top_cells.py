"""Build the PHASE 'Top Cells' tab assets: per-class top-N phenotype cells from the shap_screen rankings
(pma_shap_phase_{geneKO,complex}.parquet) — the EXACT cells the traversal directions are built from, so the
tab stays consistent with the traversals. Accuracy-only (no attention head), top_n per class by rank.

Each ranked cell has experiment/well/x_pheno/y_pheno/segmentation → we crop it from phenotyping_v3.zarr with
the SAME 150px + blue negative cell-mask overlay as the PC tab (reuses build_pc_crops_masked) and emit:
  viewer_assets_v5/top_cells/index.json          {"top_n", "genes"|"complexes": {CLASS: {"accuracy": [rec...]}}}
  viewer_assets_v5/top_cells/crops/<pos>.png

  python -m ops_model.models.interpretability.diffae.viewer.build_top_cells geneKO    # SLURM crop shards + finalize
  python -m ops_model.models.interpretability.diffae.viewer.build_top_cells complex --finalize   # rebuild index only
"""
from __future__ import annotations

import argparse
import json
import os

from . import catalog as C
from .build_pc_crops_masked import BASE, CROP_SIZE, PHASE_CHANNEL, _crop, _is_blank, _render, _render_gray, _overlay_rgba, _zarr_patch

TOP_N = 40


def _pos_key(exp, well, x, y):
    return f"{exp}__{well}__{int(round(float(x)))}__{int(round(float(y)))}"


def _crop_cells(attn, acc, crops_dir):
    """Crop every unique cell (dedup by position across both rankings) from the zarr with the blue mask."""
    import zarr
    from PIL import Image
    _zarr_patch()
    uniq = {}
    for d in (attn, acc):
        for recs in d.values():
            for c in recs:
                uniq[_pos_key(c["exp"], c["well"], c["x"], c["y"])] = c
    print(f"[topcells] {len(uniq)} unique cells to crop")
    os.makedirs(crops_dir, exist_ok=True)
    ov_dir = os.path.join(os.path.dirname(crops_dir), "overlays"); os.makedirs(ov_dir, exist_ok=True)
    half = CROP_SIZE // 2
    cache, ok, blank, fail, valid = {}, 0, 0, 0, set()
    for i, (pk, c) in enumerate(uniq.items()):
        exp, well = c["exp"], c["well"]
        key = (exp, well)
        if key not in cache:
            pos = f"{BASE}/{exp}/3-assembly/phenotyping_v3.zarr/{well[0]}/{well[1:]}/0"
            try:
                cache[key] = (zarr.open(f"{pos}/0", mode="r"), zarr.open(f"{pos}/labels/cell_seg/0", mode="r"))
            except Exception as e:
                cache[key] = None; print(f"[topcells] open failed {exp}/{well}: {e}")
        if cache[key] is None:
            fail += 1; continue
        img, seg = cache[key]
        x, y = int(round(c["x"])), int(round(c["y"]))
        try:
            phase = _crop(img, PHASE_CHANNEL, x, y, half)
            if _is_blank(phase):
                blank += 1; continue
            Image.fromarray(_render_gray(phase)).save(f"{crops_dir}/{pk}.png")                     # raw grayscale (toggle-off)
            Image.fromarray(_overlay_rgba(_crop(seg, None, x, y, half), half)).save(f"{ov_dir}/{pk}.png")   # blue-outside overlay
            ok += 1; valid.add(pk)
        except Exception as e:
            fail += 1
            if fail <= 5:
                print(f"[topcells] crop failed {pk}: {e}")
        if (i + 1) % 2000 == 0:
            print(f"[topcells] {i + 1}/{len(uniq)}  ok={ok} blank={blank} fail={fail}")
    print(f"[topcells] crops: {ok} written, {blank} blank, {fail} failed")
    return valid


def _recs(d, extra, valid):   # attach crop filename; drop cells without a valid crop
    out_d = {}
    for g, cells in d.items():
        lst = [{"img": f"{_pos_key(c['exp'], c['well'], c['x'], c['y'])}.png", "ov": f"{_pos_key(c['exp'], c['well'], c['x'], c['y'])}.png",
                "gene": g, "exp": c["exp"],
                "well": c["well"], "x": round(c["x"], 1), "y": round(c["y"], 1), "rank": c["rank"], extra: c["score"]}
               for c in cells if _pos_key(c["exp"], c["well"], c["x"], c["y"]) in valid]
        if lst:
            out_d[g] = lst
    return out_d


def _merge_index(out, top_n, key, entries):
    """Merge entries under index[key] ('genes' or 'complexes'), preserving the other key."""
    path = f"{out}/index.json"
    idx = json.load(open(path)) if os.path.exists(path) else {"marker": "phase"}
    idx["marker"] = "phase"; idx["top_n"] = top_n
    idx.setdefault(key, {}).update(entries)
    with open(path, "w") as f:
        json.dump(idx, f)
    print(f"[topcells] {key}: +{len(entries)} → {len(idx[key])} total; {os.path.getsize(path) / 1024:.0f} KB")
    return path


def _cap(cells, top_n):
    """Base-name aggregation can merge several variant labels → dedup by position, keep the top_n by score, re-rank."""
    seen, uniq = set(), []
    for c in sorted(cells, key=lambda d: -d["score"]):
        pk = _pos_key(c["exp"], c["well"], c["x"], c["y"])
        if pk in seen:
            continue
        seen.add(pk); uniq.append(c)
        if len(uniq) >= top_n:
            break
    return [{**c, "rank": i} for i, c in enumerate(uniq, 1)]


# ---------------------------------------------------------------------------
# Top-Accuracy cells (shap_screen rankings).
#
# NOTE ON BAG SIZE (read before touching this): there is NO single bag size for the v5 top
# cells. Alex's v5 set-accuracy ranking assigns ONE bag size PER CLASS — the bag at which that
# class saturates. Strong classes (HSPA5/CAPZB and every complex) rank at bag=10; a weak single
# gene KO like AACS only produces any signal at bag=500 (and even there rank-1 score ≈ 0.07).
# So each perturbation's cells come from its own bag. We take them straight from the slim viewer
# parquets (top-N by rank, already per-class-bag), which are the EXACT cells the traversal
# centroids are built from — keeping the Top-Cells tab and the traversals consistent. v5 has an
# accuracy ranking only (no attention head), so "attention" is left empty.
# ---------------------------------------------------------------------------
V5_RANK = "/hpc/projects/icd.fast.ops/models/diffex/viewer_assets_v5/_rankings"
V5_PARQUET = {"geneKO": f"{V5_RANK}/pma_shap_phase_geneKO.parquet",     # NEW shap_screen phase rankings (same cells as traversals)
              "complex": f"{V5_RANK}/pma_shap_phase_complex.parquet"}
V5_CLASS_COL = {"geneKO": "gene", "complex": "predicted_class"}
OUT_V5 = f"{C.OUT}/viewer_assets_v5/top_cells"
V5_RECORDS = OUT_V5 + "/_v5_records_{grain}.json"


def _v5_records(grain, top_n, names=None):
    """{class: [top_n cell records]} straight from the slim v5 parquet (already per-class-bag ranked).
    Complexes collapse to the base complex name (v4 convention) then dedup-by-position + cap to top_n."""
    import pandas as pd
    ccol = V5_CLASS_COL[grain]
    df = pd.read_parquet(V5_PARQUET[grain],
                         columns=[ccol, "experiment", "well", "x_pheno", "y_pheno", "segmentation", "pma_attention", "rank"])
    df = df[df["rank"] <= top_n * 3 if grain == "complex" else df["rank"] <= top_n]   # complex: extra headroom for dedup
    if grain == "complex":
        df[ccol] = df[ccol].str.split(",").str[0].str.strip()
    if names:
        df = df[df[ccol].isin(set(names))]
    recs = {}
    for cls, g in df.groupby(ccol):
        cells = [{"exp": r.experiment, "well": r.well, "x": float(r.x_pheno), "y": float(r.y_pheno),
                  "seg": str(r.segmentation), "rank": int(r.rank), "score": round(float(r.pma_attention), 5)}
                 for r in g.itertuples()]
        recs[cls] = _cap(cells, top_n)      # dedup by position, keep top_n by score, re-rank 1..N
    return recs


def prepare_v5_records(grain, out=OUT_V5, top_n=TOP_N, names=None):
    recs = _v5_records(grain, top_n, names)
    os.makedirs(out, exist_ok=True)
    with open(V5_RECORDS.format(grain=grain), "w") as f:
        json.dump(recs, f)
    print(f"[topcells-v5] {grain}: prepared {len(recs)} classes (top {top_n}) -> {V5_RECORDS.format(grain=grain)}")
    return list(recs)


def crop_v5_shard(grain, classes, out=OUT_V5):
    """SLURM job: crop this shard's classes' cells into the shared crops/ dir (additive, unique per position)."""
    recs = json.load(open(V5_RECORDS.format(grain=grain)))
    acc = {c: recs[c] for c in classes if c in recs}
    _crop_cells({}, acc, f"{out}/crops")
    return {"grain": grain, "classes": len(acc), "cells": sum(len(v) for v in acc.values())}


def finalize_v5_index(grain, out=OUT_V5, top_n=TOP_N):
    """Build index[genes|complexes] from prepared records + whichever crops exist; accuracy-only, attention empty."""
    recs = json.load(open(V5_RECORDS.format(grain=grain)))
    valid = {f[:-4] for f in os.listdir(f"{out}/crops") if f.endswith(".png")}
    acc = _recs(recs, "conf", valid)
    entries = {g: {"attention": [], "accuracy": acc.get(g, [])} for g in sorted(acc)}
    key = "genes" if grain == "geneKO" else "complexes"
    return _merge_index(out, top_n, key, entries)


def submit_v5(grain, out=OUT_V5, top_n=TOP_N, n_shards=24):
    """Prepare records, fan out crop shards on SLURM, then finalize the index for this grain."""
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    names = prepare_v5_records(grain, out, top_n)
    shards = [s for s in ([names[i::n_shards] for i in range(n_shards)]) if s]
    jobs = [{"name": f"topcells_v5_{grain}_{i}", "func": crop_v5_shard, "kwargs": {"grain": grain, "classes": s, "out": out}}
            for i, s in enumerate(shards)]
    print(f"[topcells-v5] submitting {len(jobs)} shards for {len(names)} {grain} classes (top {top_n})")
    submit_parallel_jobs(jobs, experiment=f"topcells_v5_{grain}",
                         slurm_params={"slurm_partition": "cpu", "cpus_per_task": 8, "mem_gb": 32, "timeout_min": 120},
                         log_dir=f"topcells_v5_{grain}", wait_for_completion=True)
    finalize_v5_index(grain, out, top_n)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("grain", choices=["geneKO", "complex"], help="top-accuracy cells for this grain (SLURM shards)")
    ap.add_argument("--finalize", action="store_true", help="rebuild the index for this grain from existing crops")
    ap.add_argument("--top-n", type=int, default=TOP_N)
    a = ap.parse_args()
    (finalize_v5_index if a.finalize else submit_v5)(a.grain, OUT_V5, a.top_n)
