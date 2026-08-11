"""Populate the `real` field for the 42 recovered geneKO in the new-v5 cache (they were built gen-only).
Gather each gene's top-30 accuracy real cells (dashed ranking name for KRTAP) → embed_crops → write into
gen_real_map_cache_v5new/geneKO/<slug>.npz so the pooled-centroid REAL ceiling covers the full 1000."""
import os, glob
import numpy as np

CV = "/hpc/projects/icd.fast.ops/analysis/figure4_traversals"
CACHE = f"{CV}/gen_real_map_cache_v5new/geneKO"
RANKP = "/hpc/projects/icd.fast.ops/models/diffex/viewer_assets_v5/_rankings/pma_v5_phase_geneKO.parquet"
KREAL = 30


def _drop42():
    import glob as g
    single = {os.path.basename(x) for x in g.glob("/hpc/projects/icd.fast.ops/models/diffex/viewer_assets_v5/phase/geneKO/*")
              if os.path.isdir(x) and "__to__" not in x and not os.path.basename(x).startswith("_")}
    old = {os.path.basename(f)[:-4] for f in g.glob(f"{CV}/gen_real_map_cache/geneKO/*.npz")}
    return sorted(single - old)


def run():
    import pandas as pd
    from ops_model.models.interpretability.diffex.viewer.precompute import _gather_class
    from ops_model.models.interpretability.diffex.directions.config import DirConfig
    from ops_model.models.interpretability.diffex.classifier.config import slugify
    genes = _drop42()
    orig = {slugify(str(x)): str(x) for x in pd.read_parquet(RANKP, columns=["gene"])["gene"].unique()}  # slug→ranking name
    cfg = DirConfig(grain="geneKO", target=genes[0], device="cuda"); cfg.num_workers = 12
    done = 0
    for g in genes:
        cp = f"{CACHE}/{g}.npz"
        if not os.path.exists(cp):
            continue
        d = np.load(cp, allow_pickle=True)
        if len(np.asarray(d["real"])):                       # already has real
            continue
        name = orig.get(slugify(g), g)                       # dashed name for KRTAP, plain otherwise
        _, embs = _gather_class(cfg, name, KREAL, parquet=RANKP)
        if not len(embs):
            print(f"no real for {g} ({name})"); continue
        np.savez(cp, real=np.asarray(embs, np.float32)[:KREAL], gen=d["gen"], alphas=d["alphas"], gene=str(d["gene"]))
        done += 1
    return {"patched": done, "n_genes": len(genes)}


def main():
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    submit_parallel_jobs(jobs_to_submit=[{"name": "patchreal", "func": run, "kwargs": {}}],
                         experiment="patchreal",
                         slurm_params={"slurm_partition": "preempted", "slurm_gres": "gpu:1", "cpus_per_task": 12,
                                       "mem_gb": 64, "timeout_min": 120, "slurm_constraint": "[a40|a6000|l40s]"},
                         log_dir="patchreal", wait_for_completion=False)


if __name__ == "__main__":
    main()
