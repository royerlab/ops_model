"""Verify the valid200 cache embeddings equal the original embed_crops path on the SAME saved frames.
If cosine≈1 the cache build is faithful (frames genuinely differ); if not, the cache build is the bug.
"""
import numpy as np

CV = "/hpc/projects/icd.fast.ops/analysis/figure4_traversals"
B = "/hpc/projects/icd.fast.ops/models/diffex"


def check(gene="AACS", ai=6):
    from ops_model.models.interpretability.diffae.viewer.score_generated import _emb_frames
    from ops_model.models.interpretability.diffae.classifier.celldino_features import embed_crops
    from ops_model.models.interpretability.diffae.directions.config import DirConfig
    trav = f"{B}/viewer_assets_valid200/phase/geneKO/{gene}"
    cfg = DirConfig(grain="geneKO", target=gene, device="cuda")
    embB = np.asarray(_emb_frames(cfg, trav, ai, embed_crops), np.float32)     # original path
    d = np.load(f"{CV}/gen_real_map_cache_valid200/geneKO/{gene}.npz", allow_pickle=True)
    embA = np.asarray(d["gen"][ai], np.float32)                                # my cache
    n = min(len(embA), len(embB)); A, Bm = embA[:n], embB[:n]
    cos = (A * Bm).sum(1) / ((np.linalg.norm(A, 1 if False else None, axis=1) + 1e-9) * (np.linalg.norm(Bm, axis=1) + 1e-9))
    return {"gene": gene, "ai": ai, "nA": len(embA), "nB": len(embB),
            "cos_mean": float(cos.mean()), "cos_min": float(cos.min()),
            "max_abs_diff": float(np.abs(A - Bm).max())}


def main():
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    submit_parallel_jobs(jobs_to_submit=[{"name": "embcheck", "func": check, "kwargs": {}}],
                         experiment="embcheck",
                         slurm_params={"slurm_partition": "preempted", "gpus_per_node": 1, "cpus_per_task": 8,
                                       "mem_gb": 48, "timeout_min": 30, "slurm_constraint": "[a40|a6000|l40s]"},
                         log_dir="embcheck", wait_for_completion=False)


if __name__ == "__main__":
    main()
