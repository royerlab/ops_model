"""SetTransformer scored on the two anchor halves: cells [0:200] (hand-picked) vs [200:400] (strict multibag NTC).
Per gene → {first, second} scores_v5-dicts (P(target), rank, top1, top5 per α). Same score_embs_v5 as the bag
sweep, just fed each 200-cell half. GPU, sharded → per-gene json."""
import os, glob, json
import numpy as np

CV = "/hpc/projects/icd.fast.ops/analysis/figure4_traversals"
GRAIN = os.environ.get("BSS_GRAIN", "geneKO")
CACHE = f"{CV}/gen_real_map_cache_v5new/{GRAIN}"
OUT = f"{CV}/st_halves_v5new/{GRAIN}"


def score_shard(genes):
    import torch
    from ops_model.models.interpretability._internal.viewer.score_generated import score_embs_v5
    from ops_model.models.interpretability._internal.viewer.set_classifier import load_set_classifier, V5_CKPT_ROOT, V5_RUNS
    from ops_model.models.interpretability.diffae.classifier.config import slugify
    os.makedirs(OUT, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    run = V5_RUNS[("phase", "geneKO" if GRAIN == "geneKO" else "complex_ebionly")]
    m, g2i, c2i = load_set_classifier(run=run, device=dev, root=V5_CKPT_ROOT)
    ci = c2i.get("Phase2D", 0); slug2orig = {slugify(k): k for k in g2i}
    done = 0
    for g in genes:
        f = f"{CACHE}/{g}.npz"; outp = f"{OUT}/{g}.json"
        if not os.path.exists(f) or os.path.exists(outp):
            continue
        d = np.load(f, allow_pickle=True); gene = str(d["gene"]); al = [float(a) for a in d["alphas"]]
        tgt = gene if gene in g2i else slug2orig.get(slugify(gene), gene)
        embs = [None if d["gen"][ai] is None else np.asarray(d["gen"][ai], np.float32) for ai in range(len(al))]
        if any(e is not None and len(e) < 400 for e in embs):
            continue
        first = [None if e is None else e[:200] for e in embs]
        second = [None if e is None else e[200:400] for e in embs]
        rec = {"first": score_embs_v5(first, al, tgt, m, g2i, ci, run, device=dev, bag=200),
               "second": score_embs_v5(second, al, tgt, m, g2i, ci, run, device=dev, bag=200)}
        json.dump({"gene": gene, "alphas": al, **rec}, open(outp, "w")); done += 1
    return {"grain": GRAIN, "done": done}


def main():
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    genes = sorted(os.path.basename(f)[:-4] for f in glob.glob(f"{CACHE}/*.npz"))
    ch = 60; shards = [genes[i:i + ch] for i in range(0, len(genes), ch)]
    jobs = [{"name": f"sth_{GRAIN}_{i}", "func": score_shard, "kwargs": {"genes": s}} for i, s in enumerate(shards)]
    print(f"[st-halves] {GRAIN}: {len(genes)} genes → {len(jobs)} shards")
    submit_parallel_jobs(jobs, experiment=f"sth_{GRAIN}",
                         slurm_params={"slurm_partition": "preempted", "slurm_gres": "gpu:1", "cpus_per_task": 8,
                                       "mem_gb": 48, "timeout_min": 120, "slurm_constraint": "[a40|a6000|l40s]"},
                         log_dir=f"sth_{GRAIN}", wait_for_completion=False)


if __name__ == "__main__":
    main()
