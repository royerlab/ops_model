"""Re-score all v5 traversals to add `rank_target` (1-indexed rank of the target class) to scores_v5.json.

score_embs_v5 now emits rank_target alongside p_target/top1/top5; this pass re-runs the SetTransformer scorer
over both anchor pools so the viewer's "target rank" overlay has data. One pass fully subsumes the old top5 backfill.
"""
import os, glob, json

BASE = "/hpc/projects/icd.fast.ops/models/diffex"
POOLS = ["viewer_assets_v5", "viewer_assets_v5_accpool"]


def _retarget(assets):
    """Point the score module at `assets` and return the module (V5_BASE is read at call time)."""
    import ops_model.models.interpretability.diffex.viewer.score_generated as SG
    SG.V5_BASE = f"{BASE}/{assets}/phase"
    return SG


def rescore_shard(assets, grain, targets, bag=20):
    SG = _retarget(assets)
    SG.score_targets(grain, targets, bag=bag)


def anchor_shard(assets, grain):
    SG = _retarget(assets)
    SG.score_anchor_traversals(grain)


def _targets(assets, grain):
    """NTC-anchor target names (from each traversal's meta.json) for a pool/grain, excluding A→B dirs."""
    sub = "geneKO" if grain == "geneKO" else "complex"
    out = []
    for d in sorted(glob.glob(f"{BASE}/{assets}/phase/{sub}/*")):
        if not os.path.isdir(d) or "__to__" in os.path.basename(d):
            continue
        mp = f"{d}/meta.json"
        if os.path.exists(mp):
            out.append(json.load(open(mp))["target"])
    return out


def main(n_shards=24):
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    jobs = []
    for assets in POOLS:
        for grain in ["geneKO", "complex"]:
            names = _targets(assets, grain)
            shards = [s for s in (names[i::n_shards] for i in range(n_shards)) if s]
            for i, s in enumerate(shards):
                jobs.append({"name": f"rank_{assets[-4:]}_{grain}_{i}", "func": rescore_shard,
                             "kwargs": {"assets": assets, "grain": grain, "targets": s, "bag": 20}})
        # A→B alt-anchor dirs live only in the attention pool
        if glob.glob(f"{BASE}/{assets}/phase/geneKO/*__to__*"):
            for grain in ["geneKO", "complex"]:
                jobs.append({"name": f"rank_{assets[-4:]}_alt_{grain}", "func": anchor_shard,
                             "kwargs": {"assets": assets, "grain": grain}})
    print(f"[rank-rescore] submitting {len(jobs)} jobs across {POOLS}")
    submit_parallel_jobs(
        jobs, experiment="diffex_rank_rescore",
        slurm_params={"slurm_partition": "gpu", "slurm_gres": "gpu:1",
                      "cpus_per_task": 8, "mem_gb": 64, "timeout_min": 180},
        log_dir="diffex_rank_rescore", wait_for_completion=False)


if __name__ == "__main__":
    main()
