"""Centroid-recovery mAP bag-sweep on the new multibag v5 cache, SHARDED for speed.
Stage 1 (mu): global α=0 standardization from a gene subset (α=0 is NTC-recon, tight → subset≈full).
Stage 2 (score): parallel shards, each scores its genes at bags {20,50,100,200,400} → partial json.
Stage 3 (merge): combine partials → {grain}_bagsweep.json (per-bag by_alpha of top1/top5/map).

  CBS_GRAIN=geneKO python centroid_bagsweep.py           # submit mu (inline) + score shards
  CBS_GRAIN=geneKO python centroid_bagsweep.py merge      # combine partials
"""
import os, glob, json, sys
import numpy as np

CV = "/hpc/projects/icd.fast.ops/analysis/figure4_traversals"
GRAIN = os.environ.get("CBS_GRAIN", "geneKO")
CACHE = f"{CV}/gen_real_map_cache_v5new/{GRAIN}"
CENTD = f"{CV}/gen_real_centroid_v5new"
STD = os.environ.get("CBS_STD", "global")                   # global (panel α=0 mu) | perbag (per-gene α=0, matches score_embs_v5)
OUT = f"{CV}/centroid_bagsweep_v5new" + ("_perbag" if STD == "perbag" else "")
PART = f"{OUT}/{GRAIN}_parts"
BAGS = [20, 50, 100, 200, 400]
MU_N = 150                                                  # genes for the global α=0 estimate


def _cz():
    from ops_model.models.interpretability.diffex.classifier.config import slugify
    d = np.load(f"{CENTD}/{GRAIN}_centroids.npz", allow_pickle=True)
    names = list(d["names"]); cidx = {slugify(str(c)): i for i, c in enumerate(names)}
    cz = (d["cents"] - d["mu"]) / d["sd"]; cz = cz / (np.linalg.norm(cz, axis=1, keepdims=True) + 1e-9)
    return cz, cidx


def compute_mu():
    os.makedirs(OUT, exist_ok=True)
    S = np.zeros(1024); SS = np.zeros(1024); n = 0
    for f in sorted(glob.glob(f"{CACHE}/*.npz"))[:MU_N]:
        d = np.load(f, allow_pickle=True); al = np.asarray(d["alphas"], float); a0 = int(np.argmin(np.abs(al)))
        z = d["gen"][a0]
        if z is not None and len(z):
            z = np.asarray(z, np.float32); S += z.sum(0); SS += (z.astype(np.float64) ** 2).sum(0); n += len(z)
    mu = S / n; sd = np.sqrt(np.clip(SS / n - mu ** 2, 1e-12, None)) + 1e-6
    np.savez(f"{OUT}/{GRAIN}_mu.npz", mu=mu.astype(np.float32), sd=sd.astype(np.float32), n=n)
    return {"grain": GRAIN, "mu_cells": n}


def score_shard(genes):
    from ops_model.models.interpretability.diffex.classifier.config import slugify
    os.makedirs(PART, exist_ok=True)
    cz, cidx = _cz()
    if STD == "global":
        m = np.load(f"{CV}/centroid_bagsweep_v5new/{GRAIN}_mu.npz"); mu, sd = m["mu"], m["sd"]
    by = {B: {} for B in BAGS}
    for g in genes:
        f = f"{CACHE}/{g}.npz"
        if not os.path.exists(f) or slugify(g) not in cidx:
            continue
        d = np.load(f, allow_pickle=True); al = [float(a) for a in d["alphas"]]; ti = cidx[slugify(g)]
        a0 = int(np.argmin(np.abs(np.asarray(al, float)))); gv0 = np.asarray(d["gen"][a0], np.float32)
        for ai, a in enumerate(al):
            gv = d["gen"][ai]
            if gv is None or not len(gv):
                continue
            gv = np.asarray(gv, np.float32)
            for B in BAGS:
                if STD == "perbag":                         # standardize on this gene's own α=0 frames (first-B), like score_embs_v5
                    mu = gv0[:B].mean(0); sd = gv0[:B].std(0) + 1e-6
                gz = (gv[:B] - mu) / sd; gz = gz / (np.linalg.norm(gz, axis=1, keepdims=True) + 1e-9)
                order = np.argsort(-(gz @ cz.T), axis=1); rk = np.where(order == ti)[1] + 1
                rec = by[B].setdefault(a, {"top1": {}, "top5": {}, "map": {}})
                rec["top1"][g] = float(np.mean(order[:, 0] == ti))
                rec["top5"][g] = float(np.mean([ti in r[:5] for r in order]))
                rec["map"][g] = float(np.mean(1.0 / rk))
    json.dump({"by_bag": by}, open(f"{PART}/{genes[0]}.json", "w"))
    return {"grain": GRAIN, "n": len(genes)}


def merge():
    cz, cidx = _cz()
    by = {str(B): {} for B in BAGS}
    for p in glob.glob(f"{PART}/*.json"):
        d = json.load(open(p))
        for B, ba in d["by_bag"].items():
            for a, rec in ba.items():
                dst = by[str(B)].setdefault(a, {"top1": {}, "top5": {}, "map": {}})
                for k in ("top1", "top5", "map"):
                    dst[k].update(rec[k])
    json.dump({"bags": BAGS, "by_bag": by, "n_classes": len(cidx)}, open(f"{OUT}/{GRAIN}_bagsweep.json", "w"))
    n = len(next(iter(by["400"].values()))["map"]) if by["400"] else 0
    return {"grain": GRAIN, "scored": n}


def main():
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    if STD == "global":
        print(compute_mu())                                 # inline: fast (α=0 of 150 genes); perbag needs no global mu
    os.makedirs(OUT, exist_ok=True)
    genes = sorted(os.path.basename(f)[:-4] for f in glob.glob(f"{CACHE}/*.npz"))
    ch = 40; shards = [genes[i:i + ch] for i in range(0, len(genes), ch)]
    jobs = [{"name": f"cbs_{GRAIN}_{i}", "func": score_shard, "kwargs": {"genes": s}} for i, s in enumerate(shards)]
    print(f"[centroid-bagsweep] {GRAIN}: {len(genes)} genes → {len(jobs)} score shards")
    submit_parallel_jobs(jobs, experiment=f"cbs_{GRAIN}",
                         slurm_params={"slurm_partition": "preempted", "cpus_per_task": 8, "mem_gb": 32, "timeout_min": 60},
                         log_dir=f"cbs_{GRAIN}", wait_for_completion=False)


if __name__ == "__main__":
    if sys.argv[1:2] == ["merge"]:
        print(merge())
    else:
        main()
