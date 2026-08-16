"""Metric C (per-domain std, per-CELL centroid recovery) on PHASE — the SAME recipe as centroid_recovery_fluor,
applied to gen_real_map_cache_v5new + the phase class centroids. Pure numpy → CPU, sharded per gene chunk.
gen standardized per-class on its own α=0 (per-domain); real centroids on real-pop; each gen cell → nearest
centroid among all classes → fraction correct (top-1). No CellDINO recompute.
"""
import glob
import json
import os

import numpy as np

FT = "/hpc/projects/icd.fast.ops/analysis/figure4_traversals"
CENTD = f"{FT}/gen_real_centroid_v5new"
OUT = f"{FT}/metricC_phase"


def _slug(s):
    from ops_model.models.interpretability.diffae.classifier.config import slugify
    return slugify(str(s))


def _cz(grain):
    d = np.load(f"{CENTD}/{grain}_centroids.npz", allow_pickle=True)
    cz = (d["cents"] - d["mu"]) / d["sd"]; cz = cz / (np.linalg.norm(cz, axis=1, keepdims=True) + 1e-9)
    cidx = {_slug(str(c)): i for i, c in enumerate(d["names"])}
    return cz, cidx


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


def run(genes, grain="geneKO", device=None):
    os.makedirs(OUT, exist_ok=True)
    if os.path.exists(f"{OUT}/{grain}_{_slug(genes[0])}.json"):
        return {"grain": grain, "skip": "done"}
    cz, cidx = _cz(grain); cache = f"{FT}/gen_real_map_cache_v5new/{grain}"; res = {}
    for g in genes:
        fp = f"{cache}/{g}.npz"; s = _slug(g)
        if not os.path.exists(fp) or s not in cidx:
            continue
        d = np.load(fp, allow_pickle=True); al = [float(a) for a in d["alphas"]]; ti = cidx[s]
        a0 = int(np.argmin(np.abs(np.array(al)))); g0 = np.asarray(d["gen"][a0], np.float32)
        mu_g, sd_g = g0.mean(0), g0.std(0) + 1e-6                          # per-class gen α=0 (per-domain)
        top1 = []
        for ai in range(len(al)):
            gv = d["gen"][ai]
            if gv is None or not len(gv):
                top1.append(None); continue
            gz = (np.asarray(gv, np.float32) - mu_g) / sd_g
            gz = gz / (np.linalg.norm(gz, axis=1, keepdims=True) + 1e-9)
            top1.append(float(np.mean(np.argmax(gz @ cz.T, axis=1) == ti)))    # per-cell nearest → fraction
        res[g] = {"f": round(_ipk(al, top1), 3), "top1": top1, "alphas": al}
    json.dump(res, open(f"{OUT}/{grain}_{_slug(genes[0])}.json", "w"))
    return {"grain": grain, "n": len(res)}


def submit():
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    jobs = []
    for grain in ("geneKO", "complex"):
        genes = sorted(os.path.basename(f)[:-4] for f in glob.glob(f"{FT}/gen_real_map_cache_v5new/{grain}/*.npz"))
        for i in range(0, len(genes), 40):
            jobs.append({"name": f"Cph_{grain[:4]}_{i}", "func": run, "kwargs": {"genes": genes[i:i + 40], "grain": grain}})
    print(f"[metricC-phase] {len(jobs)} CPU shards")
    submit_parallel_jobs(jobs, experiment="Cph",
                         slurm_params={"slurm_partition": "preempted", "cpus_per_task": 8, "mem_gb": 32, "timeout_min": 60},
                         log_dir="Cph", wait_for_completion=False)


if __name__ == "__main__":
    submit()
