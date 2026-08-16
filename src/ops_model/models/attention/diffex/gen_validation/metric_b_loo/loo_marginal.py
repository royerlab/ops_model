"""f via 'removal-based ranking' (Alex Lin's LOO marginal, applied to GENERATED cells):
drop one generated cell (at α) into a bag of REAL cells of its class and measure its marginal contribution to
the set classifier's P(true class): P(true | real bag + gen cell) − P(true | real bag). Averaged over random real
bags / gen cells / bag sizes. f = α maximizing the generated cell's marginal (most 'real-phenotype-like').
Uses our v5 SetTransformer (set_classifier); gen cells z-standardized on gen-α0 (real-cell space).
"""
import glob
import json
import os

import numpy as np

CACHE = "/hpc/projects/icd.fast.ops/analysis/figure4_traversals/gen_real_map_cache_v5new/geneKO"
REAL_STORE = "/hpc/projects/icd.fast.ops/models/alex_lin_attention/v5/paper_v2_phase/val"   # ~8k real cells/gene, classifier space (no recompute)
OUT = "/hpc/projects/icd.fast.ops/analysis/figure4_traversals/loo_marginal"
SIZES = (20, 50, 100, 200)                                                     # match the paper's multibag sizes
REPS = 30


def _ipk(al, y):
    al = np.asarray(al, float); y = np.asarray(y, float); pos = al > 0; a, v = al[pos], y[pos]
    i = int(np.argmax(v))
    if i == 0 or i == len(a) - 1:
        return float(a[i])
    x3, y3 = a[i - 1:i + 2], v[i - 1:i + 2]; c = np.polyfit(x3, y3, 2)
    return float(a[i]) if c[0] >= 0 else float(np.clip(-c[1] / (2 * c[0]), x3[0], x3[2]))


def run(genes=("POLR1B", "TOMM20", "MICOS13", "AARS", "PSMB7"), device="cuda"):
    import torch
    from ops_model.models.attention.diffex.viewer.set_classifier import load_set_classifier, score_bags, V5_CKPT_ROOT, V5_RUNS
    run_ = V5_RUNS[("phase", "geneKO")]
    model, cmap, c2i = load_set_classifier(run=run_, device=device, root=V5_CKPT_ROOT)
    ci = c2i.get("Phase2D", 0); rng = np.random.default_rng(0)
    os.makedirs(OUT, exist_ok=True); res = {}
    for g in genes:
        fp = f"{CACHE}/{g}.npz"
        if g not in cmap or not os.path.exists(fp):
            print(f"skip {g}"); continue
        rp = f"{REAL_STORE}/{g}.pt"
        if not os.path.exists(rp):
            print(f"skip {g}: no real store"); continue
        gi = cmap[g]; d = np.load(fp, allow_pickle=True); al = [float(a) for a in d["alphas"]]
        real = torch.load(rp, map_location="cpu")["embeddings"].numpy().astype(np.float32); nR = len(real)   # ~8k real cells (classifier space)
        z0 = int(np.argmin(np.abs(np.array(al)))); g0 = np.asarray(d["gen"][z0], np.float32)
        mu, sd = g0.mean(0), g0.std(0) + 1e-6
        ncell = min(len(g0), 40)                                              # per-cell scores kept for the viewer

        def _sample_bags():
            sb = {}
            for sz in SIZES:
                if sz <= nR:
                    rbs = np.stack([real[rng.choice(nR, sz, replace=False)] for _ in range(REPS)])
                    sb[sz] = (rbs, np.asarray(score_bags(model, rbs, channel_idx=ci, device=device)[:, gi]))
            return sb

        def _cell_marg(query, sb):                                             # per-query mean LOO marginal over the bags
            out = np.full(len(query), np.nan)
            for k in range(len(query)):
                ms = []
                for rbs, p_loo in sb.values():
                    full = np.concatenate([np.repeat(query[k][None, None], len(rbs), 0), rbs], axis=1)
                    ms.extend(np.asarray(score_bags(model, full, channel_idx=ci, device=device)[:, gi]) - p_loo)
                out[k] = np.mean(ms) if ms else np.nan
            return out

        refb = _sample_bags()
        if not refb:
            print(f"skip {g}: nR<{min(SIZES)}"); continue
        real_ref = _cell_marg(real[rng.choice(nR, min(150, nR), replace=False)], refb)   # real-cell marginal distribution (the ranking reference)
        real_ref = real_ref[~np.isnan(real_ref)]
        percell = [[None] * len(al) for _ in range(ncell)]; rankcurve = []
        for ai in range(len(al)):
            gv = d["gen"][ai]
            if gv is None or not len(gv):
                rankcurve.append(np.nan); continue
            G = (np.asarray(gv, np.float32)[:ncell] - mu) / sd
            gm = _cell_marg(G, _sample_bags())                                  # gen-cell marginals (fresh bags)
            pct = np.array([float((real_ref < m).mean()) for m in gm])          # percentile-rank amongst real cells (0-1)
            for cidx in range(ncell):
                percell[cidx][ai] = None if np.isnan(gm[cidx]) else round(float(pct[cidx]), 4)
            rankcurve.append(float(np.nanmean(pct)))
        f = _ipk(al, rankcurve)
        res[g] = {"f": round(f, 2), "alphas": al, "rank_vs_real": [None if np.isnan(m) else round(m, 4) for m in rankcurve],
                  "real_ref_median": round(float(np.median(real_ref)), 5), "per_cell_rank": percell}   # per_cell_rank[ncell][nα] = percentile among real → viewer
        print(f"{g:10s} f(peak rank-vs-real)={f:.2f}")
        pos = [i for i in range(len(al)) if al[i] >= 0]
        print("   α    :", " ".join(f"{al[i]:5.1f}" for i in pos))
        print("   rank :", " ".join(f"{rankcurve[i]:5.2f}" if not np.isnan(rankcurve[i]) else '  -- ' for i in pos), "(percentile vs real)")
    json.dump(res, open(f"{OUT}/phase_test.json", "w"))
    return res


def submit():
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    submit_parallel_jobs([{"name": "loo_phase", "func": run, "kwargs": {}}], experiment="loo",
                         slurm_params={"slurm_partition": "preempted", "slurm_gres": "gpu:1", "cpus_per_task": 8,
                                       "mem_gb": 48, "timeout_min": 120, "slurm_constraint": "[a40|a6000|l40s]"},
                         log_dir="loo", wait_for_completion=False)


if __name__ == "__main__":
    submit()
