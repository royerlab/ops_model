"""CONTROL: is the first-200 vs second-200 centroid-recovery gap (76% vs 13%) a per-bag standardization
artifact? The SetTransformer self-standardizes each bag on its OWN α=0 generated frames (cancels any
half-specific CellDINO offset) and sees NO gap; the pooled-centroid metric standardizes gen cells against a
GLOBAL/panel α=0 mean, so a half-specific offset survives. Here we recompute each half's centroid top-1 two
ways per gene: (global) shared {grain}_mu.npz vs (perbag) each half's own α=0 mean/std. If the gap collapses
under perbag, the 76/13 is definitively a standardization artifact of the centroid metric, not phenotype loss.
Real centroids stay real-standardized (unchanged) in both. Sharded → merge. CPU."""
import os, glob, json, sys
import numpy as np

CV = "/hpc/projects/icd.fast.ops/analysis/figure4_traversals"
GRAIN = os.environ.get("CBS_GRAIN", "geneKO")
CACHE = f"{CV}/gen_real_map_cache_v5new/{GRAIN}"
CENTD = f"{CV}/gen_real_centroid_v5new"
MU = f"{CV}/centroid_bagsweep_v5new/{GRAIN}_mu.npz"
OUT = f"{CV}/control_halves_v5new"
PART = f"{OUT}/{GRAIN}_parts"
EPS = 1e-6


def _cz():
    from ops_model.models.attention.diffex.classifier.config import slugify
    d = np.load(f"{CENTD}/{GRAIN}_centroids.npz", allow_pickle=True)
    names = list(d["names"]); cidx = {slugify(str(c)): i for i, c in enumerate(names)}
    cz = (d["cents"] - d["mu"]) / d["sd"]; cz = cz / (np.linalg.norm(cz, axis=1, keepdims=True) + 1e-9)
    return cz, cidx


def _t1(vecs, cz, ti):
    if not len(vecs):
        return None
    v = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
    m = v.mean(0); m = m / (np.linalg.norm(m) + 1e-9)
    return float(np.argmax(m @ cz.T) == ti)


def score_shard(genes):
    from ops_model.models.attention.diffex.classifier.config import slugify
    os.makedirs(PART, exist_ok=True)
    cz, cidx = _cz(); mg = np.load(MU); mu_g, sd_g = mg["mu"], mg["sd"]
    res = {s: {h: {} for h in ("first", "second")} for s in ("global", "perbag")}
    for g in genes:
        f = f"{CACHE}/{g}.npz"
        if not os.path.exists(f) or slugify(g) not in cidx:
            continue
        d = np.load(f, allow_pickle=True); al = [float(a) for a in d["alphas"]]; ti = cidx[slugify(g)]
        a0 = int(np.argmin(np.abs(al))); a0v = d["gen"][a0]
        if a0v is None or len(a0v) < 400:
            continue
        a0v = np.asarray(a0v, np.float32)
        mu1 = a0v[:200].mean(0); sd1 = a0v[:200].std(0) + EPS
        mu2 = a0v[200:400].mean(0); sd2 = a0v[200:400].std(0) + EPS
        for ai, a in enumerate(al):
            gv = d["gen"][ai]
            if gv is None or len(gv) < 400:
                continue
            gv = np.asarray(gv, np.float32); h1, h2 = gv[:200], gv[200:400]
            for s, z1, z2 in [("global", (h1 - mu_g) / sd_g, (h2 - mu_g) / sd_g),
                              ("perbag", (h1 - mu1) / sd1, (h2 - mu2) / sd2)]:
                t1 = _t1(z1, cz, ti); t2 = _t1(z2, cz, ti)
                if t1 is not None:
                    res[s]["first"].setdefault(a, {})[g] = t1
                if t2 is not None:
                    res[s]["second"].setdefault(a, {})[g] = t2
    json.dump(res, open(f"{PART}/{genes[0]}.json", "w"))
    return {"grain": GRAIN, "n": len(genes)}


def merge():
    res = {s: {h: {} for h in ("first", "second")} for s in ("global", "perbag")}
    for p in glob.glob(f"{PART}/*.json"):
        d = json.load(open(p))
        for s in ("global", "perbag"):
            for h in ("first", "second"):
                for a, r in d[s][h].items():
                    res[s][h].setdefault(a, {}).update(r)
    json.dump(res, open(f"{OUT}/{GRAIN}_control.json", "w"))
    # peak-α summary
    out = {}
    for s in ("global", "perbag"):
        agg = {}
        for h in ("first", "second"):
            byA = {float(a): np.mean(list(r.values())) for a, r in res[s][h].items() if r}
            k = max(byA, key=byA.get) if byA else None
            agg[h] = {"peak_alpha": k, "top1": (byA[k] if k is not None else None), "n": len(next(iter(res[s][h].values()))) if res[s][h] else 0}
        out[s] = agg
    print(json.dumps(out, indent=2))
    return out


def main():
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    genes = sorted(os.path.basename(f)[:-4] for f in glob.glob(f"{CACHE}/*.npz"))
    ch = 40; shards = [genes[i:i + ch] for i in range(0, len(genes), ch)]
    jobs = [{"name": f"ctrl_{GRAIN}_{i}", "func": score_shard, "kwargs": {"genes": s}} for i, s in enumerate(shards)]
    print(f"[control] {GRAIN}: {len(genes)} genes → {len(jobs)} shards")
    submit_parallel_jobs(jobs, experiment=f"ctrl_{GRAIN}",
                         slurm_params={"slurm_partition": "preempted", "cpus_per_task": 8, "mem_gb": 32, "timeout_min": 60},
                         log_dir=f"ctrl_{GRAIN}", wait_for_completion=False)


if __name__ == "__main__":
    if sys.argv[1:2] == ["merge"]:
        print(merge())
    else:
        main()
