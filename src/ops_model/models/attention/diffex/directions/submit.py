"""Submit Stage 3 (directions + traversal) to SLURM (1 GPU).

    python -m ops_model.models.attention.diffex.directions.submit --target HSPA5
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs

from ..classifier.config import DEFAULT_OUT_ROOT, GRAINS, slugify
from .config import DirConfig
from .run import run_directions


def main():
    ap = argparse.ArgumentParser(description="Submit DiffEx Stage 3 to SLURM")
    ap.add_argument("--grain", choices=list(GRAINS), default="geneKO")
    ap.add_argument("--target", default="HSPA5")
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--dir-epochs", type=int, default=100)
    ap.add_argument("--diffae-ckpt", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--partition", default="gpu")
    ap.add_argument("--gres", default="gpu:1")
    ap.add_argument("--cpus", type=int, default=8)
    ap.add_argument("--mem-gb", type=int, default=64)
    ap.add_argument("--time-min", type=int, default=180)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = DirConfig(grain=args.grain, target=args.target, K=args.K,
                    dir_epochs=args.dir_epochs, device="cuda")
    if args.diffae_ckpt:
        cfg.diffae_ckpt = args.diffae_ckpt
    out = args.out_dir or f"{DEFAULT_OUT_ROOT}/directions/{args.grain}/{slugify(args.target)}"

    jobs = [{
        "name": f"diffex_dir_{slugify(args.target)}"[:64],
        "func": run_directions,
        "kwargs": {"cfg": cfg, "out_dir": str(Path(out).resolve())},
        "metadata": {"stage": "directions", "target": args.target},
    }]
    slurm_params = {
        "slurm_partition": args.partition, "slurm_gres": args.gres,
        "cpus_per_task": args.cpus, "mem_gb": args.mem_gb, "timeout_min": args.time_min,
    }
    submit_parallel_jobs(
        jobs_to_submit=jobs, experiment="diffex_directions",
        slurm_params=slurm_params, log_dir="diffex_directions",
        dry_run=args.dry_run, wait_for_completion=False,
    )


if __name__ == "__main__":
    main()
