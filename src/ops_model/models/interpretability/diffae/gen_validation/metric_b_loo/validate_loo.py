"""VALIDATION: reproduce Alex Lin's per-cell LOO score/rank (fullrank_phase_cells_trainval) with OUR set classifier
+ score_bags, to prove the ranking system is replicated. For one gene at one bag size, compute each sampled cell's
LOO marginal P(true|bag)-P(true|bag-cell) over random bags of REAL cells, and correlate against Alex's `score`/`rank`.
"""
import numpy as np

STORE = "/hpc/projects/icd.fast.ops/models/alex_lin_attention/v5/paper_v2_phase/val"
FULLRANK = "/hpc/projects/icd.fast.ops/models/alex_lin_attention/v5/phase/fullrank_phase_cells_trainval.parquet"
OUT = "/hpc/projects/icd.fast.ops/analysis/figure4_traversals/validate_loo.json"


def run(gene="POLR1B", bag=100, reps=400, nsample=250, device="cuda"):
    import json, torch, pandas as pd
    from scipy.stats import pearsonr, spearmanr
    from ops_model.models.interpretability.diffae.viewer.set_classifier import load_set_classifier, score_bags, V5_CKPT_ROOT, V5_RUNS
    model, cmap, c2i = load_set_classifier(run=V5_RUNS[("phase", "geneKO")], device=device, root=V5_CKPT_ROOT)
    gi = cmap[gene]; ci = c2i.get("Phase2D", 0)
    d = torch.load(f"{STORE}/{gene}.pt", map_location="cpu")
    emb = d["embeddings"].numpy().astype(np.float32)
    segflat = [int(s) for lst in d["cell_metadata"]["segmentation_id"] for s in lst]        # aligned to embedding rows
    assert len(segflat) == len(emb), (len(segflat), len(emb))
    seg2row = {s: i for i, s in enumerate(segflat)}
    ax = pd.read_parquet(FULLRANK, columns=["gene", "bag_size", "split", "rank", "score", "segmentation_id"])
    ax = ax[(ax["gene"] == gene) & (ax["bag_size"] == bag) & (ax["split"] == "val")]
    ax = ax[ax["segmentation_id"].isin(seg2row)]
    print(f"{gene}: {len(emb)} rows, Alex val@bag{bag} = {len(ax)} cells matched")
    ax = ax.sample(min(nsample, len(ax)), random_state=0)
    rows = [seg2row[s] for s in ax["segmentation_id"]]
    rng = np.random.default_rng(0); N = len(emb)
    mine = []
    for r in rows:
        ms = []
        for _ in range(reps):
            idx = rng.choice(N, bag, replace=False)
            rb = emb[idx]                                                                    # real bag (Alex ranks within-gene bags)
            p_full = float(score_bags(model, np.concatenate([emb[r][None], rb[:bag - 1]])[None], channel_idx=ci, device=device)[0][gi])
            p_loo = float(score_bags(model, rb[:bag - 1][None], channel_idx=ci, device=device)[0][gi])
            ms.append(p_full - p_loo)
        mine.append(float(np.mean(ms)))
    a = ax["score"].to_numpy(); m = np.array(mine)
    pr = pearsonr(a, m); sr = spearmanr(a, m)
    print(f"\n=== VALIDATION {gene} bag={bag} (n={len(m)}) ===")
    print(f"  Alex score : mean={a.mean():+.4f}  range=[{a.min():+.4f},{a.max():+.4f}]")
    print(f"  My  score  : mean={m.mean():+.4f}  range=[{m.min():+.4f},{m.max():+.4f}]")
    print(f"  Pearson r  = {pr[0]:.3f}   Spearman ρ = {sr[0]:.3f}   (1.0 = replicated)")
    ex = sorted(zip(ax["rank"].tolist(), a.tolist(), m.tolist()))[:8]
    print("  Alex-rank  Alex-score  My-score")
    for rk, av, mv in ex:
        print(f"    {rk:5d}    {av:+.4f}    {mv:+.4f}")
    json.dump({"gene": gene, "bag": bag, "n": len(m), "pearson": pr[0], "spearman": sr[0],
               "alex": a.tolist(), "mine": m.tolist(), "alex_rank": ax["rank"].tolist()}, open(OUT, "w"))


def submit():
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    submit_parallel_jobs([{"name": "valloo", "func": run, "kwargs": {}}], experiment="valloo",
                         slurm_params={"slurm_partition": "preempted", "slurm_gres": "gpu:1", "cpus_per_task": 8,
                                       "mem_gb": 48, "timeout_min": 120, "slurm_constraint": "[a40|a6000|l40s]"},
                         log_dir="valloo", wait_for_completion=False)


if __name__ == "__main__":
    submit()


def exact(device="cuda"):
    """Exact deterministic check: single-cell P(POLR1B|cell) for the top cell (train row 22709) vs Alex conf=0.26616."""
    import torch, numpy as np
    from ops_model.models.interpretability.diffae.viewer.set_classifier import load_set_classifier, score_bags, V5_CKPT_ROOT, V5_RUNS
    m, cmap, c2i = load_set_classifier(run=V5_RUNS[("phase", "geneKO")], device=device, root=V5_CKPT_ROOT)
    gi = cmap["POLR1B"]; ci = c2i.get("Phase2D", 0)
    d = torch.load("/hpc/projects/icd.fast.ops/models/alex_lin_attention/v5/paper_v2_phase/train/POLR1B.pt", map_location="cpu")
    cell = d["embeddings"].numpy().astype(np.float32)[22709]
    prob = score_bags(m, cell[None][None], channel_idx=ci, device=device)[0]
    p1 = float(prob[gi]); top = int(np.argmax(prob))
    inv = {v: k for k, v in cmap.items()}
    print(f"single-cell P(POLR1B|cell) = {p1:.5f}   vs Alex conf = 0.26616   diff={abs(p1-0.26616):.5f}")
    print(f"argmax class = {inv.get(top)} ({prob[top]:.5f})   POLR1B rank = {int((np.argsort(-prob)==gi).argmax())+1}")
