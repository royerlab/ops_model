"""Metric A (global-α0-std, bag-LEVEL pooled centroid recovery) on FLUOR — the SAME recipe as the phase
centroid_pooled_bagsweep, applied to the fcr embcache (no CellDINO recompute). Pure numpy → CPU, sharded
per (marker, grain). Output mirrors the phase pooled JSON so both are read/plotted identically.
"""
import glob
import json
import os

import numpy as np

FT = "/hpc/projects/icd.fast.ops/analysis/figure4_traversals"
EMB = f"{FT}/f_centroid_recovery/embcache"
V5 = "/hpc/projects/icd.fast.ops/models/diffex/viewer_assets_v5"
OUT = f"{FT}/metricA_fluor"
BAGS = [20, 50, 100, 200, 400]
NBOOT = 25


def _pooled(vecs, cz, ti):
    m = vecs.mean(0); m = m / (np.linalg.norm(m) + 1e-9)
    order = np.argsort(-(m @ cz.T)); rk = int(np.where(order == ti)[0][0]) + 1
    return float(order[0] == ti), float(1.0 / rk)


def _ipk(al, y):
    al = np.asarray(al, float); y = np.array([np.nan if v is None else v for v in y], float)
    pos = al > 0; a, v = al[pos], y[pos]; ok = ~np.isnan(v); a, v = a[ok], v[ok]
    if len(a) < 2:
        return float("nan")
    i = int(np.argmax(v))
    if i == 0 or i == len(a) - 1:
        return float(a[i])
    x3, y3 = a[i - 1:i + 2], v[i - 1:i + 2]; c = np.polyfit(x3, y3, 2)
    return float(a[i]) if c[0] >= 0 else float(np.clip(-c[1] / (2 * c[0]), x3[0], x3[2]))


def run_marker(mod, grain, device=None):
    if os.path.exists(f"{OUT}/{mod}__{grain}.json"):
        return {"mod": mod, "grain": grain, "skip": "done"}
    galdir = f"{EMB}/gal/{mod}/{grain}"; gendir = f"{EMB}/gen/{mod}/{grain}"
    metas = glob.glob(f"{V5}/{mod}/{grain}/*/meta.json")
    if not os.path.isdir(galdir) or not metas:
        return {"mod": mod, "grain": grain, "skip": "no cache"}
    al = [float(a) for a in json.load(open(metas[0]))["alphas"]]; a0 = int(np.argmin(np.abs(np.array(al))))
    # real class centroids + real-pop stats (from cached galleries)
    names, cents, allreal = [], [], []
    for gp in sorted(glob.glob(f"{galdir}/*.npz")):
        e = np.load(gp)["features"]
        if len(e):
            names.append(os.path.basename(gp)[:-4]); cents.append(e.mean(0)); allreal.append(e)
    if len(names) < 3:
        return {"mod": mod, "grain": grain, "skip": "gallery<3"}
    R = np.concatenate(allreal); mu_r, sd_r = R.mean(0), R.std(0) + 1e-6
    cz = (np.stack(cents) - mu_r) / sd_r; cz = cz / (np.linalg.norm(cz, axis=1, keepdims=True) + 1e-9)
    cidx = {n: i for i, n in enumerate(names)}
    # GLOBAL gen α=0 std (pool every class's α=0 gen)
    g0 = [np.load(f"{gendir}/{n}/a{a0}.npz")["features"] for n in names if os.path.exists(f"{gendir}/{n}/a{a0}.npz")]
    G0 = np.concatenate(g0); mu_g, sd_g = G0.mean(0), G0.std(0) + 1e-6
    rng = np.random.default_rng(0)
    gen = {B: {} for B in BAGS}; real = {B: {} for B in BAGS}
    for n in names:
        ti = cidx[n]
        for ai, a in enumerate(al):
            gp = f"{gendir}/{n}/a{ai}.npz"
            if not os.path.exists(gp):
                continue
            gv = np.load(gp)["features"]
            if not len(gv):
                continue
            for B in BAGS:
                zb = (gv[:B] - mu_g) / sd_g; zb = zb / (np.linalg.norm(zb, axis=1, keepdims=True) + 1e-9)
                t1, mp = _pooled(zb, cz, ti)
                r = gen[B].setdefault(str(a), {"top1": {}, "map": {}}); r["top1"][n] = t1; r["map"][n] = mp
        rz = (allreal[ti] - mu_r) / sd_r                                  # this class's gallery (real ceiling bootstrap)
        rz = rz / (np.linalg.norm(rz, axis=1, keepdims=True) + 1e-9)
        for B in BAGS:
            real[B][n] = float(np.mean([_pooled(rz[rng.integers(0, len(rz), B)], cz, ti)[0] for _ in range(NBOOT)]))
    # per-class peak-α (bag 200)
    b200 = gen[200]; per = {}
    for n in names:
        per[n] = round(_ipk(al, [b200.get(str(a), {"top1": {}})["top1"].get(n, np.nan) for a in al]), 3)
    os.makedirs(OUT, exist_ok=True)
    json.dump({"bags": BAGS, "alphas": al, "gen": {str(B): gen[B] for B in BAGS},
               "real": {str(B): real[B] for B in BAGS}, "f": per},
              open(f"{OUT}/{mod}__{grain}.json", "w"))
    return {"mod": mod, "grain": grain, "classes": len(names)}


def _jobs():
    import re
    man = json.load(open(f"{V5}/manifest.json")); jobs = []
    from ops_model.models.interpretability.diffae.classifier.config import slugify
    for mk in man["markers"]:
        mc = mk.get("marker_channel")
        if not mc or re.match(r"(?i)phase", mc):
            continue
        mod = slugify(mc)
        for grain in ("geneKO", "complex"):
            if os.path.isdir(f"{EMB}/gal/{mod}/{grain}") and glob.glob(f"{V5}/{mod}/{grain}/*/meta.json"):
                jobs.append((mod, grain))
    return jobs


def submit():
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    jl = _jobs()
    jobs = [{"name": f"Aflu_{m[:12]}_{g[:4]}", "func": run_marker, "kwargs": {"mod": m, "grain": g}} for m, g in jl]
    print(f"[metricA-fluor] {len(jobs)} (marker,grain) CPU shards")
    submit_parallel_jobs(jobs, experiment="Aflu",
                         slurm_params={"slurm_partition": "preempted", "cpus_per_task": 8, "mem_gb": 32, "timeout_min": 90},
                         log_dir="Aflu", wait_for_completion=False)


if __name__ == "__main__":
    submit()
