"""Re-score the new multibag v5 traversals with the v5 SetTransformer at MULTIPLE bag sizes {20,50,100,200,400}.
Reads the CellDINO cache (gen = list[A] of (n_cells,1024) per gene) and runs score_embs_v5 at each bag → per-gene
{bag: scores_v5-dict}. GPU (loads the classifier once per shard). The rank/P(target)/top-k plots then draw one
line per bag size. bag=B scores the FIRST B cells (deterministic), standardized on the α=0 frames — same method
as the stored bag=45, just swept.
"""
import os, glob, json
import numpy as np

CV = "/hpc/projects/icd.fast.ops/analysis/figure4_traversals"
GRAIN = os.environ.get("BSS_GRAIN", "geneKO")
CACHE = f"{CV}/gen_real_map_cache_v5new/{GRAIN}"
OUT = f"{CV}/bag_sweep_v5new/{GRAIN}"
BAGS = [20, 50, 100, 200, 400]


def score_shard(genes):
    import torch
    from ops_model.models.attention.diffex.viewer.score_generated import score_embs_v5
    from ops_model.models.attention.diffex.viewer.set_classifier import load_set_classifier, V5_CKPT_ROOT, V5_RUNS
    os.makedirs(OUT, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from ops_model.models.attention.diffex.classifier.config import slugify
    run = V5_RUNS[("phase", "geneKO" if GRAIN == "geneKO" else "complex_ebionly")]
    m, g2i, c2i = load_set_classifier(run=run, device=dev, root=V5_CKPT_ROOT)
    ci = c2i.get("Phase2D", 0)
    slug2orig = {slugify(k): k for k in g2i}          # cache genes are slugified (KRTAP2_3); g2i uses originals (KRTAP2-3)
    done = 0
    for g in genes:
        f = f"{CACHE}/{g}.npz"; outp = f"{OUT}/{g}.json"
        if not os.path.exists(f) or os.path.exists(outp):
            continue
        d = np.load(f, allow_pickle=True); gene = str(d["gene"]); al = [float(a) for a in d["alphas"]]
        tgt = gene if gene in g2i else slug2orig.get(slugify(gene), gene)   # normalize slug→classifier name
        embs = [None if d["gen"][ai] is None else np.asarray(d["gen"][ai], np.float32) for ai in range(len(al))]
        rec = {}
        for B in BAGS:
            rec[B] = score_embs_v5(embs, al, tgt, m, g2i, ci, run, device=dev, bag=B)
        json.dump({"gene": gene, "alphas": al, "by_bag": rec}, open(outp, "w"))
        done += 1
    return {"grain": GRAIN, "done": done}


def main():
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    genes = sorted(os.path.basename(f)[:-4] for f in glob.glob(f"{CACHE}/*.npz"))
    ch = 60
    shards = [genes[i:i + ch] for i in range(0, len(genes), ch)]
    jobs = [{"name": f"bss_{GRAIN}_{i}", "func": score_shard, "kwargs": {"genes": s}} for i, s in enumerate(shards)]
    print(f"[bag-sweep] {GRAIN}: {len(genes)} genes → {len(jobs)} shards, bags={BAGS}")
    submit_parallel_jobs(jobs, experiment=f"bagsweep_{GRAIN}",
                         slurm_params={"slurm_partition": "preempted", "slurm_gres": "gpu:1", "cpus_per_task": 8,
                                       "mem_gb": 48, "timeout_min": 120, "slurm_constraint": "[a40|a6000|l40s]"},
                         log_dir=f"bagsweep_{GRAIN}", wait_for_completion=False)


if __name__ == "__main__":
    main()
