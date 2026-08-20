"""Bag-LEVEL centroid recovery on the new multibag v5 cache. For each class/α/bag B: pool the first-B generated
cells → their standardized mean (the generated class centroid) → is the nearest real centroid the true class?
Also a bootstrapped per-bag REAL ceiling: sample B real cells (with replacement from the ~30 cached) → mean →
nearest centroid, n_boot times. Both rise with bag (better centroid estimate), giving a proper per-bag reference.
Reuses gen_real_map_cache_v5new + the mu.npz from centroid_bagsweep. Sharded → merge.

  CBS_GRAIN=geneKO python centroid_pooled_bagsweep.py          # mu (reuse) + score shards
  CBS_GRAIN=geneKO python centroid_pooled_bagsweep.py merge
"""
import os, glob, json, sys
import numpy as np

CV = "/hpc/projects/icd.fast.ops/analysis/figure4_traversals"
GRAIN = os.environ.get("CBS_GRAIN", "geneKO")
CACHE = f"{CV}/gen_real_map_cache_v5new/{GRAIN}"
CENTD = f"{CV}/gen_real_centroid_v5new"
MU = f"{CV}/centroid_bagsweep_v5new/{GRAIN}_mu.npz"
STD = os.environ.get("CBS_STD", "global")                   # global (panel α=0 mu) | perbag (per-gene α=0, matches score_embs_v5)
OUT = f"{CV}/centroid_pooled_bagsweep_v5new" + ("_perbag" if STD == "perbag" else "")
PART = f"{OUT}/{GRAIN}_parts"
BAGS = [20, 50, 100, 200, 400]
NBOOT = 25


def _cz():
    from ops_model.models.interpretability.diffae.classifier.config import slugify
    d = np.load(f"{CENTD}/{GRAIN}_centroids.npz", allow_pickle=True)
    names = list(d["names"]); cidx = {slugify(str(c)): i for i, c in enumerate(names)}
    cz = (d["cents"] - d["mu"]) / d["sd"]; cz = cz / (np.linalg.norm(cz, axis=1, keepdims=True) + 1e-9)
    return cz, cidx, d["mu"], d["sd"]


def _pooled(vecs, cz, ti):
    """mean of standardized cells → normalize → nearest-centroid top1 + 1/rank (one class-centroid)."""
    m = vecs.mean(0); m = m / (np.linalg.norm(m) + 1e-9)
    order = np.argsort(-(m @ cz.T)); rk = int(np.where(order == ti)[0][0]) + 1
    return float(order[0] == ti), float(1.0 / rk)


def score_shard(genes):
    from ops_model.models.interpretability.diffae.classifier.config import slugify
    os.makedirs(PART, exist_ok=True)
    cz, cidx, mu_r, sd_r = _cz()
    if STD == "global":
        mg = np.load(MU); mu_g, sd_g = mg["mu"], mg["sd"]
    rng = np.random.default_rng(0)
    gen = {B: {} for B in BAGS}; real = {B: {} for B in BAGS}
    for g in genes:
        f = f"{CACHE}/{g}.npz"
        if not os.path.exists(f) or slugify(g) not in cidx:
            continue
        d = np.load(f, allow_pickle=True); al = [float(a) for a in d["alphas"]]; ti = cidx[slugify(g)]
        a0 = int(np.argmin(np.abs(np.asarray(al, float)))); gv0 = np.asarray(d["gen"][a0], np.float32)
        # generated: pooled bag centroid per α per bag
        for ai, a in enumerate(al):
            gv = d["gen"][ai]
            if gv is None or not len(gv):
                continue
            gv = np.asarray(gv, np.float32)
            for B in BAGS:
                if STD == "perbag":                         # per-gene α=0 (first-B), matches score_embs_v5
                    mu_g = gv0[:B].mean(0); sd_g = gv0[:B].std(0) + 1e-6
                zb = (gv[:B] - mu_g) / sd_g; zb = zb / (np.linalg.norm(zb, axis=1, keepdims=True) + 1e-9)
                t1, mp = _pooled(zb, cz, ti)
                r = gen[B].setdefault(a, {"top1": {}, "map": {}}); r["top1"][g] = t1; r["map"][g] = mp
        # real ceiling: bootstrap B real cells → mean → nearest (per bag)
        rr = d["real"]
        if rr is not None and len(rr):
            rz = (np.asarray(rr, np.float32) - mu_r) / sd_r; rz = rz / (np.linalg.norm(rz, axis=1, keepdims=True) + 1e-9)
            for B in BAGS:
                t1s = []
                for _ in range(NBOOT):
                    idx = rng.integers(0, len(rz), size=B)
                    t1s.append(_pooled(rz[idx], cz, ti)[0])
                real[B][g] = float(np.mean(t1s))
    json.dump({"gen": gen, "real": real}, open(f"{PART}/{genes[0]}.json", "w"))
    return {"grain": GRAIN, "n": len(genes)}


def merge():
    gen = {str(B): {} for B in BAGS}; real = {str(B): {} for B in BAGS}
    for p in glob.glob(f"{PART}/*.json"):
        d = json.load(open(p))
        for B in BAGS:
            for a, rec in d["gen"].get(str(B), {}).items():
                dst = gen[str(B)].setdefault(a, {"top1": {}, "map": {}})
                dst["top1"].update(rec["top1"]); dst["map"].update(rec["map"])
            real[str(B)].update(d["real"].get(str(B), {}))
    json.dump({"bags": BAGS, "gen": gen, "real": real}, open(f"{OUT}/{GRAIN}_pooled.json", "w"))
    return {"grain": GRAIN, "gen_classes": len(next(iter(gen['400'].values()))['map']) if gen['400'] else 0,
            "real_classes": len(real['400'])}


def main():
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    genes = sorted(os.path.basename(f)[:-4] for f in glob.glob(f"{CACHE}/*.npz"))
    ch = 40; shards = [genes[i:i + ch] for i in range(0, len(genes), ch)]
    jobs = [{"name": f"cpbs_{GRAIN}_{i}", "func": score_shard, "kwargs": {"genes": s}} for i, s in enumerate(shards)]
    print(f"[pooled-centroid] {GRAIN}: {len(genes)} genes → {len(jobs)} shards")
    submit_parallel_jobs(jobs, experiment=f"cpbs_{GRAIN}",
                         slurm_params={"slurm_partition": "preempted", "cpus_per_task": 8, "mem_gb": 32, "timeout_min": 60},
                         log_dir=f"cpbs_{GRAIN}", wait_for_completion=False)


if __name__ == "__main__":
    if sys.argv[1:2] == ["merge"]:
        print(merge())
    else:
        main()
