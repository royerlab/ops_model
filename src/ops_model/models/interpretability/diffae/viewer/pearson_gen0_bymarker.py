"""Diagnostic: per-perturbation Pearson r of the traversal α=0 frame vs its paired NTC anchor real cell.

A correct DDIM-inverted traversal reconstructs its anchor at α=0 (r≈0.99). A traversal built from a
STALE/old-ranking anchor set but now paired against the current NTC anchors mismatches → low r. This
sweep finds which markers/perturbations still carry old (single-rank) reference cells.

Pairing mirrors the viewer: gen = {asset_dir}/cell{c}/frame_{i0}.webp (i0 = index of α==0),
reference = {real_dir}/cell{c}/real.webp, same cell index c (default bag, offset 0). `__to__`
alt-anchor transitions are excluded (they intentionally don't start from the NTC anchor).

  python pearson_gen0_bymarker.py submit
  python pearson_gen0_bymarker.py merge
"""
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

ROOT = "/hpc/projects/icd.fast.ops/models/diffex/viewer_assets_v5"
SH = f"{ROOT}/_montage/_pearson_gen0_shards"
STEP = 4          # sample every 4th cell for speed (~10 cells/perturbation)
LOW = 0.8         # r below this = stale/mismatched anchor


def _load(p):
    try:
        return np.asarray(Image.open(p).convert("L"), dtype=np.float32).ravel()
    except Exception:
        return None


def _pear(a, b):
    if a is None or b is None or a.shape != b.shape:
        return np.nan
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan


def _score(asset_dir, real_dir, alphas, n_cells):
    if 0.0 not in alphas:
        return None
    i0 = f"{alphas.index(0.0):02d}"
    rs = []
    for c in range(0, n_cells, STEP):
        g = _load(f"{ROOT}/{asset_dir}/cell{c}/frame_{i0}.webp")
        r = _load(f"{ROOT}/{real_dir}/cell{c}/real.webp")
        v = _pear(g, r)
        if np.isfinite(v):
            rs.append(v)
    return (round(float(np.mean(rs)), 4), len(rs)) if rs else None


def shard(marker):
    os.makedirs(SH, exist_ok=True)
    res = {"geneKO": {}, "complex": {}}
    for grain in ("geneKO", "complex"):
        for f in glob.glob(f"{ROOT}/{marker}/{grain}/*/meta.json"):
            name = f.split("/")[-2]
            if "__to__" in name:
                continue
            m = json.load(open(f))
            if not m.get("has_real") or not m.get("n_cells"):
                continue
            s = _score(m["asset_dir"], m["real_dir"], m["alphas"], m["n_cells"])
            if s:
                res[grain][name] = {"r": s[0], "n": s[1]}
    json.dump(res, open(f"{SH}/{marker.replace('/', '_')}.json", "w"))
    lo = sum(1 for g in res.values() for v in g.values() if v["r"] < LOW)
    tot = sum(len(g) for g in res.values())
    return {"marker": marker, "n": tot, "low": lo}


def markers():
    return sorted(d for d in os.listdir(ROOT) if os.path.isdir(f"{ROOT}/{d}/geneKO") and not d.startswith("vs_"))


def submit():
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    jobs = [{"name": f"pear0_{m}"[:40], "func": shard, "kwargs": {"marker": m}} for m in markers()]
    print(f"[pearson-gen0] {len(jobs)} markers")
    submit_parallel_jobs(jobs, experiment="pearson_gen0",
                         slurm_params={"slurm_partition": "cpu", "cpus_per_task": 2, "mem_gb": 8, "timeout_min": 45},
                         log_dir="pearson_gen0", wait_for_completion=False)


def merge():
    rows = []
    for f in sorted(glob.glob(f"{SH}/*.json")):
        marker = os.path.basename(f)[:-5]; d = json.load(open(f))
        for grain in ("geneKO", "complex"):
            vals = d[grain]
            if not vals:
                continue
            rr = np.array([v["r"] for v in vals.values()])
            low = sorted((k, v["r"]) for k, v in vals.items() if v["r"] < LOW)
            rows.append({"marker": marker, "grain": grain, "n": len(rr),
                         "median_r": round(float(np.median(rr)), 3),
                         "n_low": int((rr < LOW).sum()),
                         "frac_low": round(float((rr < LOW).mean()), 3),
                         "low_examples": low[:8]})
    rows.sort(key=lambda x: -x["frac_low"])
    json.dump(rows, open(f"{ROOT}/_montage/pearson_gen0.json", "w"), indent=1)
    print(f"pearson_gen0.json: {len(rows)} marker/grain groups")
    print(f"\n{'marker':44s} {'grain':8s} {'n':>4s} {'medR':>6s} {'nLow':>5s} {'fracLow':>7s}")
    for r in rows:
        if r["n_low"]:
            print(f"{r['marker']:44s} {r['grain']:8s} {r['n']:4d} {r['median_r']:6.3f} {r['n_low']:5d} {r['frac_low']:7.3f}")
    aff = [r for r in rows if r["n_low"]]
    print(f"\n{len(aff)}/{len(rows)} groups have ≥1 low-r (stale-anchor) perturbation")


if __name__ == "__main__":
    cmd = sys.argv[1] if sys.argv[1:] else "submit"
    print({"submit": submit, "merge": merge}[cmd]())
