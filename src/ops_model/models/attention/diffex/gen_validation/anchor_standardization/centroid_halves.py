"""First-200 vs second-200 anchor comparison. Same directions/traversals; only the anchor cells differ between
the two 200-cell halves. Pooled bag-level centroid recovery of cells [0:200] vs [200:400] per class/α → does the
second-200 set recover worse (explaining the bag=400 dilution)? Reuses the mu.npz + centroids. Sharded → merge."""
import os, glob, json, sys
import numpy as np

CV = "/hpc/projects/icd.fast.ops/analysis/figure4_traversals"
GRAIN = os.environ.get("CBS_GRAIN", "geneKO")
CACHE = f"{CV}/gen_real_map_cache_v5new/{GRAIN}"
CENTD = f"{CV}/gen_real_centroid_v5new"
MU = f"{CV}/centroid_bagsweep_v5new/{GRAIN}_mu.npz"
OUT = f"{CV}/centroid_halves_v5new"
PART = f"{OUT}/{GRAIN}_parts"


def _cz():
    from ops_model.models.attention.diffex.classifier.config import slugify
    d = np.load(f"{CENTD}/{GRAIN}_centroids.npz", allow_pickle=True)
    names = list(d["names"]); cidx = {slugify(str(c)): i for i, c in enumerate(names)}
    cz = (d["cents"] - d["mu"]) / d["sd"]; cz = cz / (np.linalg.norm(cz, axis=1, keepdims=True) + 1e-9)
    return cz, cidx


def _top1(vecs, cz, ti):
    if not len(vecs):
        return None
    m = vecs.mean(0); m = m / (np.linalg.norm(m) + 1e-9)
    return float(np.argmax(m @ cz.T) == ti)


def score_shard(genes):
    from ops_model.models.attention.diffex.classifier.config import slugify
    os.makedirs(PART, exist_ok=True)
    cz, cidx = _cz(); mg = np.load(MU); mu_g, sd_g = mg["mu"], mg["sd"]
    first, second = {}, {}
    for g in genes:
        f = f"{CACHE}/{g}.npz"
        if not os.path.exists(f) or slugify(g) not in cidx:
            continue
        d = np.load(f, allow_pickle=True); al = [float(a) for a in d["alphas"]]; ti = cidx[slugify(g)]
        for ai, a in enumerate(al):
            gv = d["gen"][ai]
            if gv is None or len(gv) < 400:
                continue
            z = (np.asarray(gv, np.float32) - mu_g) / sd_g; z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)
            f1 = _top1(z[:200], cz, ti); f2 = _top1(z[200:400], cz, ti)
            if f1 is not None:
                first.setdefault(a, {})[g] = f1
            if f2 is not None:
                second.setdefault(a, {})[g] = f2
    json.dump({"first": first, "second": second}, open(f"{PART}/{genes[0]}.json", "w"))
    return {"grain": GRAIN, "n": len(genes)}


def merge():
    first, second = {}, {}
    for p in glob.glob(f"{PART}/*.json"):
        d = json.load(open(p))
        for a, r in d["first"].items():
            first.setdefault(a, {}).update(r)
        for a, r in d["second"].items():
            second.setdefault(a, {}).update(r)
    json.dump({"first": first, "second": second}, open(f"{OUT}/{GRAIN}_halves.json", "w"))
    return {"grain": GRAIN, "n": len(next(iter(first.values()))) if first else 0}


def main():
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    genes = sorted(os.path.basename(f)[:-4] for f in glob.glob(f"{CACHE}/*.npz"))
    ch = 40; shards = [genes[i:i + ch] for i in range(0, len(genes), ch)]
    jobs = [{"name": f"halves_{GRAIN}_{i}", "func": score_shard, "kwargs": {"genes": s}} for i, s in enumerate(shards)]
    print(f"[halves] {GRAIN}: {len(genes)} genes → {len(jobs)} shards")
    submit_parallel_jobs(jobs, experiment=f"halves_{GRAIN}",
                         slurm_params={"slurm_partition": "preempted", "cpus_per_task": 8, "mem_gb": 32, "timeout_min": 60},
                         log_dir=f"halves_{GRAIN}", wait_for_completion=False)


if __name__ == "__main__":
    if sys.argv[1:2] == ["merge"]:
        print(merge())
    else:
        main()
