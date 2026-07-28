"""Driver: fan out 88 gather shards via submit_parallel_jobs, then submit a
single big-mem embed job. Both phases run on SLURM."""
from __future__ import annotations

import argparse
from pathlib import Path

from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs

from ops_model.models.attention.weighted_aggregation.run_single_cell_embedding import (
    DEFAULT_OUT, V5_PER_EXP, embed, gather_shard,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--top-k", type=int, default=1000)
    ap.add_argument("--n-landmark", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pca-var", type=float, default=0.8)
    ap.add_argument("--skip-phate", action="store_true")
    ap.add_argument("--skip-pca", action="store_true")
    ap.add_argument("--skip-raw", action="store_true")
    ap.add_argument("--cpu-embed", action="store_true",
                    help="Run embed on CPU (default: GPU via cuml)")
    ap.add_argument("--per-class-cap", type=int, default=None,
                    help="Downsample to top-K per class (e.g. 200 for a 5x faster run)")
    ap.add_argument("--ntc-normalize", action="store_true",
                    help="Subtract NTC centroid before UMAP/PHATE")
    ap.add_argument("--skip-gather", action="store_true",
                    help="Reuse existing shard files (skip phase 1)")
    ap.add_argument("--gather-only", action="store_true")
    ap.add_argument("--embed-local", action="store_true",
                    help="Run phase 2 in-process instead of submitting via SLURM")
    args = ap.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    shard_dir = out / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    # Derive a per-run slug for SLURM job names (from out-dir basename).
    slug = out.name.replace("single_cell_embedding_ebifb_", "sc_")

    # -------- Phase 1: parallel gather ------------------------------------
    if not args.skip_gather:
        h5ads = sorted(V5_PER_EXP.glob("*.h5ad"))
        print(f"phase 1: {len(h5ads)} shards → {shard_dir}")
        jobs = [{
            "name": f"gather_{p.stem}",
            "func": gather_shard,
            "kwargs": {"h5ad_path": str(p), "top_k": args.top_k,
                       "shard_dir": str(shard_dir)},
            "metadata": {"experiment": p.stem},
        } for p in h5ads]

        res = submit_parallel_jobs(
            jobs_to_submit=jobs,
            experiment=f"{slug}_gather",
            slurm_params={
                "timeout_min": 60,
                "cpus_per_task": 4,
                "mem_gb": 48,
                "slurm_partition": "cpu",
            },
            log_dir=f"{slug}_gather",
            manifest_prefix=f"{slug}_gather",
            wait_for_completion=True,
        )
        if not res.get("all_completed", False):
            raise RuntimeError(f"gather phase incomplete: {res.get('failed')}")

    if args.gather_only:
        print("gather-only mode: done")
        return

    # -------- Phase 2: single embed ---------------------------------------
    use_gpu = not args.cpu_embed
    if args.embed_local:
        print("phase 2: running embed in-process")
        embed(out, args.seed, args.n_landmark, args.pca_var,
              args.skip_phate, args.skip_pca, args.skip_raw, use_gpu,
              args.per_class_cap, args.ntc_normalize)
        return

    print(f"phase 2: submitting embed to SLURM (gpu={use_gpu})")
    slurm_params: dict = {
        "timeout_min": 6 * 60,
        "cpus_per_task": 8 if use_gpu else 32,
        "mem_gb": 128 if use_gpu else 256,
        "slurm_partition": "gpu" if use_gpu else "cpu",
    }
    if use_gpu:
        slurm_params["slurm_gres"] = "gpu:1"
    submit_parallel_jobs(
        jobs_to_submit=[{
            "name": f"{slug}_embed",
            "func": embed,
            "kwargs": {
                "out_dir": out,
                "seed": args.seed,
                "n_landmark": args.n_landmark,
                "pca_var": args.pca_var,
                "skip_phate": args.skip_phate,
                "skip_pca": args.skip_pca,
                "skip_raw": args.skip_raw,
                "use_gpu": use_gpu,
                "per_class_cap": args.per_class_cap,
                "ntc_normalize": args.ntc_normalize,
            },
        }],
        experiment=f"{slug}_embed",
        slurm_params=slurm_params,
        log_dir=f"{slug}_embed",
        manifest_prefix=f"{slug}_embed",
        wait_for_completion=False,
    )


if __name__ == "__main__":
    main()
