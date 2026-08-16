"""Closest-approach α: mean cosine similarity of the generated cells to the class's TOP-predictive real centroid,
at BOTH top-100 and top-1000 cells — in the SAME per-domain-standardized cosine space as gen_real_centroid.score()
(real centroids z-scored by real μ/σ, gen z-scored by pooled gen-α0 μ/σ, both L2-normalized). Per class we record
the per-α mean similarity to its own top-k centroid; "closest approach" = the α that maximizes it. SLURM-sharded.

  precompute(grain)  -> {grain}_ref.npz  (gen-α0 μ/σ + top100/top1k real centroids, real-standardized + L2)
  score_shard(grain, genes) -> per-class {gene}.json with sim_top100 / sim_top1k vs α
  submit(grain)      -> shard score_shard across SLURM (CPU, no GPU)
"""
import glob
import json
import os

import numpy as np

CV = "/hpc/projects/icd.fast.ops/analysis/figure4_traversals"
CACHE = f"{CV}/gen_real_map_cache_v5new"
CENTDIR = f"{CV}/gen_real_centroid_v5new"
OUT = f"{CV}/gen_dist_closest"


def _slug(s):
    from ops_model.models.interpretability.diffae.classifier.config import slugify
    return slugify(str(s))


def precompute(grain):
    """gen-α0 pooled μ/σ (sampled) + top-100 & top-1000 real centroids (real-standardized, L2) → {grain}_ref.npz."""
    os.makedirs(OUT, exist_ok=True)
    C = np.load(f"{CENTDIR}/{grain}_centroids.npz", allow_pickle=True)
    mu_r, sd_r = C["mu"], C["sd"]
    E = np.load(f"{CENTDIR}/{grain}_embs.npz", allow_pickle=True)
    feats, labels = E["feats"], E["labels"]
    bounds, start = {}, 0                                              # embs are contiguous + rank-ordered per class
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            bounds[str(labels[start])] = (start, i); start = i

    def zc(c):
        z = (c - mu_r) / sd_r
        return (z / (np.linalg.norm(z) + 1e-9)).astype(np.float32)

    nm, cz100, cz1k, bs, be = [], [], [], [], []
    for c, (a, b) in bounds.items():
        blk = feats[a:b].astype(np.float32)
        nm.append(c); cz100.append(zc(blk[:100].mean(0))); cz1k.append(zc(blk[:1000].mean(0)))
        bs.append(a); be.append(b)                                    # embs row range per class (for the energy-distance variant)

    caches = sorted(glob.glob(f"{CACHE}/{grain}/*.npz"))
    step = max(1, len(caches) // 60)                                  # ~60 caches → robust pooled gen-α0 stats
    g0 = []
    for f in caches[::step]:
        dd = np.load(f, allow_pickle=True)
        a0 = int(np.argmin(np.abs(np.asarray(dd["alphas"], float))))
        z0 = dd["gen"][a0]
        if z0 is not None and len(z0):
            g0.append(np.asarray(z0, np.float32))
    g0 = np.concatenate(g0); mu_g, sd_g = g0.mean(0), g0.std(0) + 1e-6
    np.savez(f"{OUT}/{grain}_ref.npz", names=np.array(nm), cz100=np.stack(cz100), cz1k=np.stack(cz1k),
             bs=np.array(bs), be=np.array(be), mu_r=mu_r.astype(np.float32), sd_r=sd_r.astype(np.float32),
             mu_g=mu_g.astype(np.float32), sd_g=sd_g.astype(np.float32))
    return {"grain": grain, "classes": len(nm), "gen0_cells": len(g0)}


def _zl2(x, mu, sd):
    z = (x - mu) / sd
    return (z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)).astype(np.float32)


def _pdist_mean(A):                                                  # mean pairwise Euclidean on L2-normed rows (‖a-b‖=√(2-2cos))
    G = A @ A.T; D = np.sqrt(np.maximum(2 - 2 * G, 0)); n = len(A)
    return D.sum() / max(n * (n - 1), 1)


def _cdist_mean(A, B):
    return np.sqrt(np.maximum(2 - 2 * (A @ B.T), 0)).mean()


def score_shard_energy(grain, genes):
    """Energy distance between gen(α) and the real top-K KO cells (K=100/200/1k), per class → per-α curves.
    Minimum energy = α where the generated distribution best matches the real distribution."""
    R = np.load(f"{OUT}/{grain}_ref.npz", allow_pickle=True)
    names = list(R["names"]); bs = R["bs"]; be = R["be"]; mu_r = R["mu_r"]; sd_r = R["sd_r"]; mu_g = R["mu_g"]; sd_g = R["sd_g"]
    ci = {_slug(c): i for i, c in enumerate(names)}
    E = np.load(f"{CENTDIR}/{grain}_embs.npz", allow_pickle=True, mmap_mode="r"); feats = E["feats"]
    os.makedirs(f"{OUT}/{grain}_energy", exist_ok=True); done = 0
    for g in genes:
        f = f"{CACHE}/{grain}/{g}.npz"; outp = f"{OUT}/{grain}_energy/{g}.json"
        if not os.path.exists(f) or os.path.exists(outp):
            continue
        dd = np.load(f, allow_pickle=True); gene = str(dd["gene"]); al = [float(a) for a in dd["alphas"]]
        s = _slug(gene)
        if s not in ci:
            continue
        i = ci[s]; real = np.asarray(feats[bs[i]:be[i]], np.float32)
        Ys = {k: _zl2(real[:k], mu_r, sd_r) for k in (100, 200, 1000) if len(real) >= min(k, 100)}
        dYY = {k: _pdist_mean(Y) for k, Y in Ys.items()}
        out = {k: [] for k in Ys}
        for ai in range(len(al)):
            gv = dd["gen"][ai]
            if gv is None or not len(gv):
                for k in Ys: out[k].append(None)
                continue
            X = _zl2(np.asarray(gv, np.float32), mu_g, sd_g); dXX = _pdist_mean(X)
            for k, Y in Ys.items():
                out[k].append(float(2 * _cdist_mean(X, Y) - dXX - dYY[k]))
        json.dump({"gene": gene, "alphas": al, "energy_top100": out.get(100), "energy_top200": out.get(200), "energy_top1k": out.get(1000)},
                  open(outp, "w"))
        done += 1
    return {"grain": grain, "done": done}


def submit_energy(grain):
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    genes = sorted(os.path.basename(f)[:-4] for f in glob.glob(f"{CACHE}/{grain}/*.npz"))
    ch = 60
    shards = [genes[i:i + ch] for i in range(0, len(genes), ch)]
    jobs = [{"name": f"gde_{grain}_{i}", "func": score_shard_energy, "kwargs": {"grain": grain, "genes": s}} for i, s in enumerate(shards)]
    print(f"[gde] {grain}: {len(genes)} genes → {len(jobs)} shards")
    submit_parallel_jobs(jobs, experiment=f"gde_{grain}",
                         slurm_params={"slurm_partition": "preempted", "cpus_per_task": 8, "mem_gb": 64, "timeout_min": 120},
                         log_dir=f"gde_{grain}", wait_for_completion=False)


def run_energy():
    for g in ("geneKO", "complex"):
        print(precompute(g)); submit_energy(g)


def score_shard(grain, genes):
    R = np.load(f"{OUT}/{grain}_ref.npz", allow_pickle=True)
    names = list(R["names"]); cz100 = R["cz100"]; cz1k = R["cz1k"]; mu_g = R["mu_g"]; sd_g = R["sd_g"]
    ci = {_slug(c): i for i, c in enumerate(names)}
    os.makedirs(f"{OUT}/{grain}", exist_ok=True); done = 0
    for g in genes:
        f = f"{CACHE}/{grain}/{g}.npz"; outp = f"{OUT}/{grain}/{g}.json"
        if not os.path.exists(f) or os.path.exists(outp):
            continue
        dd = np.load(f, allow_pickle=True); gene = str(dd["gene"]); al = [float(a) for a in dd["alphas"]]
        s = _slug(gene)
        if s not in ci:
            continue
        i = ci[s]; c100 = cz100[i]; c1k = cz1k[i]; s100 = []; s1k = []
        for ai in range(len(al)):
            gv = dd["gen"][ai]
            if gv is None or not len(gv):
                s100.append(None); s1k.append(None); continue
            gz = (np.asarray(gv, np.float32) - mu_g) / sd_g
            gz = gz / (np.linalg.norm(gz, axis=1, keepdims=True) + 1e-9)
            s100.append(float((gz @ c100).mean())); s1k.append(float((gz @ c1k).mean()))
        json.dump({"gene": gene, "alphas": al, "sim_top100": s100, "sim_top1k": s1k}, open(outp, "w"))
        done += 1
    return {"grain": grain, "done": done}


def submit(grain):
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    genes = sorted(os.path.basename(f)[:-4] for f in glob.glob(f"{CACHE}/{grain}/*.npz"))
    ch = 80
    shards = [genes[i:i + ch] for i in range(0, len(genes), ch)]
    jobs = [{"name": f"gdc_{grain}_{i}", "func": score_shard, "kwargs": {"grain": grain, "genes": s}} for i, s in enumerate(shards)]
    print(f"[gdc] {grain}: {len(genes)} genes → {len(jobs)} shards")
    submit_parallel_jobs(jobs, experiment=f"gdc_{grain}",
                         slurm_params={"slurm_partition": "preempted", "cpus_per_task": 8, "mem_gb": 48, "timeout_min": 90},
                         log_dir=f"gdc_{grain}", wait_for_completion=False)


def run_all():
    for g in ("geneKO", "complex"):
        print(precompute(g)); submit(g)


if __name__ == "__main__":
    run_all()
