"""CORAL-degapped centroid-recovery f, SLURM-sharded (one job per channel + phase gene-chunks), threaded I/O.

Per channel we fit ONE gen->real CORAL map (mean+covariance only -> removes the shared real<->generated domain
warp, cannot encode per-class identity), align every generated cell into real space, then score nearest REAL
centroid per alpha. f = interpolated peak-alpha (peak top-1 < 0.05 -> f=1, no rescale). Output [f, peak].

  python coral_degap_f.py submit     # fan out all channel + phase shards
  python coral_degap_f.py merge      # combine shard json -> af_scores_coral.js
"""
import glob
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

FT = "/hpc/projects/icd.fast.ops/analysis/figure4_traversals"
EMB = f"{FT}/f_centroid_recovery/embcache"
CACHE_PH = f"{FT}/gen_real_map_cache_v5new"
CENT_PH = f"{FT}/gen_real_centroid_v5new"
OUT = f"{FT}/coral_degap_f"
THR = 0.05                                                   # peak top-1 recovery floor -> f=1
DOM_GENES = 100                                              # classes pooled for the gen-domain covariance
DOM_CAP = 15000                                              # subsample cap for the domain covariance
ALPHAS = [-5, -4, -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5]


def _slug(s):
    return re.sub(r"_+$", "", re.sub(r"^_+", "", re.sub(r"[^A-Za-z0-9]", "_", str(s))))


def _loadf(paths, key="features"):
    with ThreadPoolExecutor(max_workers=16) as ex:
        return list(ex.map(lambda p: np.load(p)[key].astype(np.float32), paths))


def ipk(al, y):
    al = np.asarray(al, float); y = np.array([np.nan if v is None else v for v in y], float)
    pos = al > 0; a, v = al[pos], y[pos]; ok = ~np.isnan(v); a, v = a[ok], v[ok]
    if len(a) < 2:
        return None
    i = int(np.argmax(v))
    if i == 0 or i == len(a) - 1:
        return float(a[i])
    x3, y3 = a[i - 1:i + 2], v[i - 1:i + 2]; c = np.polyfit(x3, y3, 2)
    return float(a[i]) if c[0] >= 0 else float(np.clip(-c[1] / (2 * c[0]), x3[0], x3[2]))


def _sympow(C, p, eps=1e-2):
    C = C + eps * np.trace(C) / C.shape[0] * np.eye(C.shape[0])
    w, V = np.linalg.eigh(C); w = np.clip(w, 1e-8, None)
    return (V * (w ** p)) @ V.T


def _norm(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def _curve(gens, al, ti, cz, standardize):
    """top-1 recovery per α under a given standardization; returns {f, peak, top1}."""
    top1 = []
    for gv in gens:
        if gv is None or not len(gv):
            top1.append(None); continue
        top1.append(round(float(np.mean(np.argmax(standardize(gv) @ cz.T, 1) == ti)), 3))
    pos = np.asarray(al) > 0
    peak = max([t for t, m in zip(top1, pos) if m and t is not None] or [0.0])
    f = 1.0 if peak < THR else (ipk(al, top1) or 1.0)
    return {"f": round(float(f), 3), "peak": round(float(peak), 3), "top1": top1}


# ---------------- one shard per (mod, grain): BOTH z-score and CORAL, per-α ----------------
def shard(mod, grain, genes=None, tag=None):
    """Score perturbations with z-score (per-KO gen-α0) AND CORAL (global de-gap). mod='phase' uses the
    top-1000 real centroids from CENT_PH; fluor uses the gallery-mean centroids. genes=None scores all
    (fluor); pass a subset + tag to chunk phase (its gen is huge)."""
    os.makedirs(OUT, exist_ok=True)
    gal = sorted(glob.glob(f"{EMB}/gal/{mod}/{grain}/*.npz"))
    names = [os.path.basename(f)[:-4] for f in gal]
    score_genes = genes if genes is not None else names
    E = _loadf(gal); R = np.concatenate(E)                       # real cells (fluor 40/class; phase 30/class)
    if mod == "phase":                                          # classify against the faithful top-1000 centroids
        cen = np.load(f"{CENT_PH}/{grain}_centroids.npz", allow_pickle=True)
        cnames = [str(x) for x in cen["names"]]; cents = cen["cents"].astype(np.float32)
        cidx = {n: i for i, n in enumerate(cnames)}
        cidx.update({_slug(n): i for i, n in enumerate(cnames)})   # gal stems are slugified (complex names have spaces)
    else:
        cents = np.stack([e.mean(0) for e in E]); cnames = names; cidx = {n: i for i, n in enumerate(names)}
    dom = []
    for g in names[:DOM_GENES]:
        dom += _loadf(sorted(glob.glob(f"{EMB}/gen/{mod}/{grain}/{g}/a*.npz")))
    Gd = np.concatenate(dom)
    if len(Gd) > DOM_CAP:
        Gd = Gd[np.random.default_rng(0).choice(len(Gd), DOM_CAP, replace=False)]
    mu_r, sd_r = R.mean(0), R.std(0) + 1e-6; mu_g = Gd.mean(0)
    W = _sympow(np.cov(Gd.T), -0.5) @ _sympow(np.cov(R.T), 0.5)
    cz = _norm((cents - mu_r) / sd_r)
    coral = lambda gv: _norm(((gv - mu_g) @ W) / sd_r)
    res = {}
    for g in score_genes:
        ti = cidx.get(g, cidx.get(_slug(g)))
        if ti is None:
            continue
        fs = sorted(glob.glob(f"{EMB}/gen/{mod}/{grain}/{g}/a*.npz"), key=lambda p: int(os.path.basename(p)[1:-4]))
        if not fs:
            continue
        ais = [int(os.path.basename(p)[1:-4]) for p in fs]; al = [ALPHAS[a] for a in ais]
        gens = _loadf(fs)
        a0 = int(np.argmin(np.abs(np.array(al)))); g0 = gens[a0]
        mu_p, sd_p = g0.mean(0), g0.std(0) + 1e-6
        zscore = lambda gv, mu_p=mu_p, sd_p=sd_p: _norm((gv - mu_p) / sd_p)
        res[f"{mod}/{grain}/{g}"] = {"alphas": al,
                                     "z": _curve(gens, al, ti, cz, zscore),
                                     "coral": _curve(gens, al, ti, cz, coral)}
    suffix = f"__{tag}" if tag else ""
    json.dump(res, open(f"{OUT}/{mod}__{grain}{suffix}.json", "w"))
    return {"mod": mod, "grain": grain, "n": len(res)}


def _jobs():
    """One shard per (mod, grain) for fluor; phase chunked by genes (its gen is large even compact)."""
    jobs = []
    for gal in sorted(glob.glob(f"{EMB}/gal/*/*")):
        if not os.path.isdir(gal):
            continue
        grain = os.path.basename(gal); mod = os.path.basename(os.path.dirname(gal))
        if grain not in ("geneKO", "complex") or not glob.glob(f"{gal}/*.npz"):
            continue
        if mod == "phase":                                     # chunk: phase gen is ~1.6MB/α/gene compact
            names = sorted(os.path.basename(f)[:-4] for f in glob.glob(f"{gal}/*.npz"))
            ch = 120
            for i in range(0, len(names), ch):
                jobs.append({"name": f"cor_phase_{grain}_{i}"[:40], "func": shard,
                             "kwargs": {"mod": "phase", "grain": grain, "genes": names[i:i + ch], "tag": str(i)}})
        else:
            jobs.append({"name": f"cor_{mod[:18]}_{grain}"[:40], "func": shard,
                         "kwargs": {"mod": mod, "grain": grain}})
    return jobs


def submit():
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    jobs = _jobs()
    print(f"[coral-degap-f] {len(jobs)} shards")
    submit_parallel_jobs(jobs, experiment="coral_degap_f",
                         slurm_params={"slurm_partition": "cpu", "cpus_per_task": 16, "mem_gb": 48, "timeout_min": 40},
                         log_dir="coral_degap_f", wait_for_completion=False)


def recache_phase(genes, grain):
    """Read each 60MB pickled gene cache ONCE (cold), write compact per-α .npz into the fluor embcache layout
    (gal/phase/{grain}/{g}.npz = real cells; gen/phase/{grain}/{g}/a{ai}.npz). Then phase sweeps are fluor-fast."""
    gdir = f"{EMB}/gal/phase/{grain}"; os.makedirs(gdir, exist_ok=True)
    done = 0
    for g in genes:
        p = f"{CACHE_PH}/{grain}/{g}.npz"
        if not os.path.exists(p):
            continue
        d = np.load(p, allow_pickle=True); al = [float(a) for a in d["alphas"]]
        np.savez(f"{gdir}/{g}.npz", features=np.asarray(d["real"], np.float32), alphas=np.array(al))
        cd = f"{EMB}/gen/phase/{grain}/{g}"; os.makedirs(cd, exist_ok=True)
        for ai in range(len(al)):
            z = d["gen"][ai]
            if z is not None and len(z):
                np.savez(f"{cd}/a{ai}.npz", features=np.asarray(z, np.float32))
        done += 1
    return {"grain": grain, "recached": done}


def submit_recache():
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    jobs = []
    for grain in ("geneKO", "complex"):
        genes = sorted(os.path.basename(f)[:-4] for f in glob.glob(f"{CACHE_PH}/{grain}/*.npz"))
        ch = 30
        for i in range(0, len(genes), ch):
            jobs.append({"name": f"rc_{grain}_{i}"[:40], "func": recache_phase,
                         "kwargs": {"grain": grain, "genes": genes[i:i + ch]}})
    print(f"[recache-phase] {len(jobs)} shards")
    submit_parallel_jobs(jobs, experiment="recache_phase",
                         slurm_params={"slurm_partition": "cpu", "cpus_per_task": 4, "mem_gb": 24, "timeout_min": 45},
                         log_dir="recache_phase", wait_for_completion=False)


def merge():
    """Combine shards -> unified.json {key: {alphas, z:{f,peak,top1}, coral:{f,peak,top1}}} and a comparison."""
    M = {}
    for p in glob.glob(f"{OUT}/*__*.json"):
        M.update(json.load(open(p)))
    json.dump(M, open(f"{OUT}/unified.json", "w"))
    # comparison: % gated to f=1 and median peak, per method, phase vs fluor
    import collections
    grp = collections.defaultdict(lambda: {"n": 0, "z1": 0, "c1": 0, "zpk": [], "cpk": []})
    for k, v in M.items():
        chan = k.split("/")[0]; dom = "phase" if chan == "phase" else "fluor"
        for key in (dom, "ALL"):
            g = grp[key]; g["n"] += 1
            g["z1"] += (v["z"]["f"] == 1.0); g["c1"] += (v["coral"]["f"] == 1.0)
            g["zpk"].append(v["z"]["peak"]); g["cpk"].append(v["coral"]["peak"])
    print(f"{'group':8} {'n':>6}  {'z-score f=1':>12}  {'CORAL f=1':>11}  {'z medPk':>8}  {'CORAL medPk':>11}")
    for key in ("phase", "fluor", "ALL"):
        g = grp[key]
        if not g["n"]:
            continue
        print(f"{key:8} {g['n']:>6}  {100*g['z1']/g['n']:>11.1f}%  {100*g['c1']/g['n']:>10.1f}%  "
              f"{float(np.median(g['zpk'])):>8.3f}  {float(np.median(g['cpk'])):>11.3f}")
    print(f"\nwrote {len(M)} keys -> {OUT}/unified.json")


if __name__ == "__main__":
    cmd = sys.argv[1] if sys.argv[1:] else "submit"
    print({"submit": submit, "submit_recache": submit_recache, "merge": merge}[cmd]())
