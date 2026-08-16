"""f = α at peak centroid-recovery top-1 (nearest real-prototype among the grain's classes), for EVERY traversal.

PHASE: read the prebuilt centroid_bagsweep top-1 per class (no new compute).
FLUOR: per (marker, grain) SLURM-GPU shard — embed all classes' top-k real crops → prototypes, then each
perturbation's generated frames per α → per-α top-1 recovery → interpolated peak. webp-embedding approximation
(both real crops + gen frames are webp; CellDINO is ckpt-independent).
"""
import glob
import json
import os
import types

import numpy as np
from PIL import Image

V5 = "/hpc/projects/icd.fast.ops/models/diffex/viewer_assets_v5"
CB = "/hpc/projects/icd.fast.ops/analysis/figure4_traversals/centroid_bagsweep_v5new"
OUT = "/hpc/projects/icd.fast.ops/analysis/figure4_traversals/f_centroid_recovery"
TOPK = 100
BAG = "200"                                                              # phase bag for the top-1 curve


def _slug(s):
    from ops_model.models.interpretability.diffae.classifier.config import slugify
    return slugify(str(s))


def ipk(al, y):
    al = np.asarray(al, float); y = np.array([np.nan if v is None else v for v in y], float)
    pos = al > 0; a, v = al[pos], y[pos]; ok = ~np.isnan(v); a, v = a[ok], v[ok]
    if len(a) < 2:
        return float("nan")
    i = int(np.argmax(v))
    if i == 0 or i == len(a) - 1:
        return float(a[i])
    x3, y3 = a[i - 1:i + 2], v[i - 1:i + 2]
    c = np.polyfit(x3, y3, 2)
    return float(a[i]) if c[0] >= 0 else float(np.clip(-c[1] / (2 * c[0]), x3[0], x3[2]))


def phase_f():
    os.makedirs(OUT, exist_ok=True)
    res = {}
    for grain in ("geneKO", "complex"):
        by = json.load(open(f"{CB}/{grain}_bagsweep.json"))["by_bag"]
        b = by.get(BAG) or by[list(by)[0]]; al = sorted(b, key=float); alf = [float(a) for a in al]
        for cls in b[al[0]]["top1"]:
            f = ipk(alf, [b[a]["top1"].get(cls, np.nan) for a in al])
            res[f"phase/{grain}/{cls}"] = round(f, 3)
    json.dump(res, open(f"{OUT}/phase.json", "w"))
    return {"phase_traversals": len(res)}


def _load1(p):
    try:
        return np.asarray(Image.open(p).convert("L"), np.float32)
    except Exception:
        return None


EMBCACHE = f"{OUT}/embcache"


def _emb(paths, cfg, key=None):
    from concurrent.futures import ThreadPoolExecutor
    from ops_model.models.interpretability.diffae.classifier.celldino_features import embed_crops
    cp = f"{EMBCACHE}/{key}.npz" if key else None                        # persist CellDINO feats → reuse across metrics/reruns
    if cp and os.path.exists(cp):
        return np.load(cp)["features"]
    with ThreadPoolExecutor(max_workers=16) as ex:                       # parallel webp decode (I/O) overlaps GPU
        ims = [a for a in ex.map(_load1, paths) if a is not None]
    if not ims:
        return np.zeros((0, 1024), np.float32)
    feats = embed_crops(np.stack(ims)[:, None], cfg)
    if cp:
        os.makedirs(os.path.dirname(cp), exist_ok=True); np.savez(cp, features=feats.astype(np.float32))
    return feats


def run_marker_grain(mod, grain):
    if os.path.exists(f"{OUT}/fluor/{mod}__{grain}.json"):               # resume: skip completed shards
        return {"mod": mod, "grain": grain, "skip": "done"}
    cfg = types.SimpleNamespace(batch_size=128, celldino_z_score=True)
    block = "genes" if grain == "geneKO" else "complexes"
    idx = json.load(open(f"{V5}/top_cells/markers/{mod}/index.json"))
    cropdir = f"{V5}/top_cells/markers/{mod}/crops"
    names, cents, allreal = [], [], []
    for c, rec in idx.get(block, {}).items():
        keys = rec.get("accuracy") or rec.get("attention") or []
        e = _emb([f"{cropdir}/{r['img']}" for r in keys[:TOPK]], cfg, key=f"gal/{mod}/{grain}/{_slug(c)}")
        if len(e):
            names.append(c); cents.append(e.mean(0)); allreal.append(e)
    if len(names) < 3:
        return {"mod": mod, "grain": grain, "skip": "gallery<3"}
    R = np.concatenate(allreal); mu_r, sd_r = R.mean(0), R.std(0) + 1e-6
    cz = (np.stack(cents) - mu_r) / sd_r; cz = cz / (np.linalg.norm(cz, axis=1, keepdims=True) + 1e-9)
    ci = {c: i for i, c in enumerate(names)}
    res = {}
    for mp in glob.glob(f"{V5}/{mod}/{grain}/*/meta.json"):
        gd = os.path.dirname(mp); meta = json.load(open(mp)); cls = meta.get("target") or os.path.basename(gd)
        if cls not in ci:
            continue
        al = [float(a) for a in meta["alphas"]]; frames = sorted(os.path.basename(f) for f in glob.glob(f"{gd}/cell0/frame_*.webp"))
        if not frames:
            continue
        cells = sorted(glob.glob(f"{gd}/cell*/")); a0 = int(np.argmin(np.abs(np.array(al))))
        genA = [[f"{cd}{fr}" for cd in cells if os.path.exists(f"{cd}{fr}")] for fr in frames]
        gk = f"gen/{mod}/{grain}/{_slug(cls)}"
        g0 = _emb(genA[a0], cfg, key=f"{gk}/a{a0}")
        if not len(g0):
            continue
        mu_g, sd_g = g0.mean(0), g0.std(0) + 1e-6; ti = ci[cls]; top1 = []
        for ai in range(len(al)):
            e = _emb(genA[ai], cfg, key=f"{gk}/a{ai}")
            if not len(e):
                top1.append(None); continue
            gz = (e - mu_g) / sd_g; gz = gz / (np.linalg.norm(gz, axis=1, keepdims=True) + 1e-9)
            top1.append(float(np.mean(np.argmax(gz @ cz.T, axis=1) == ti)))
        res[f"{mod}/{grain}/{_slug(cls)}"] = {"f": round(ipk(al, top1), 3), "top1": top1, "alphas": al, "gallery": len(names)}
    os.makedirs(f"{OUT}/fluor", exist_ok=True)
    json.dump(res, open(f"{OUT}/fluor/{mod}__{grain}.json", "w"))
    return {"mod": mod, "grain": grain, "traversals": len(res)}


def _fluor_jobs():
    man = json.load(open(f"{V5}/manifest.json"))
    import re
    jobs = []
    for mk in man["markers"]:
        mc = mk.get("marker_channel")
        if not mc or re.match(r"(?i)phase", mc):
            continue
        mod = _slug(mc)
        for grain in ("geneKO", "complex"):
            if glob.glob(f"{V5}/{mod}/{grain}/*/meta.json") and os.path.exists(f"{V5}/top_cells/markers/{mod}/index.json"):
                jobs.append((mod, grain))
    return jobs


def submit():
    import os
    if os.environ.get("FCR_ENABLE") != "1":
        raise RuntimeError("fcr submit() disabled — set FCR_ENABLE=1 to re-enable (guard added to stop a stray resubmit loop)")
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    phase_f()
    jl = _fluor_jobs()
    jobs = [{"name": f"fcr_{m}_{g}"[:40], "func": run_marker_grain, "kwargs": {"mod": m, "grain": g}} for m, g in jl]
    print(f"[fcr] phase done; {len(jobs)} fluor (marker,grain) shards")
    submit_parallel_jobs(jobs, experiment="fcr",
                         slurm_params={"slurm_partition": "gpu", "slurm_gres": "gpu:1", "cpus_per_task": 8,
                                       "mem_gb": 48, "timeout_min": 360, "slurm_constraint": "[a100|l40s|a6000]",
                                       "slurm_exclude": "gpu-b-4"},   # broken CUDA driver; 360min (heavy embed) + reuse embcache
                         log_dir="fcr", wait_for_completion=False)


if __name__ == "__main__":
    submit()
