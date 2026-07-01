"""Submit the DiffAE training to SLURM (1 GPU, longer wall clock).

    python -m ops_model.models.attention.diffex.diffae.submit
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs

from ..classifier.config import DEFAULT_OUT_ROOT
from .config import DiffAEConfig
from .run import run_diffae


def main():
    ap = argparse.ArgumentParser(description="Submit DiffAE training to SLURM")
    ap.add_argument("--n-crops", type=int, default=50_000)
    ap.add_argument("--crop-size", type=int, default=160)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out-dir", default=f"{DEFAULT_OUT_ROOT}/diffae/phase_v1")
    ap.add_argument("--partition", default="gpu")
    ap.add_argument("--gpus", type=int, default=1)
    # bracketed constraint on the gpu partition = fast ≥80GB GPUs, non-preemptible
    # (repo pattern in ops_process slurm_task_config.yaml). All fit batch 48.
    ap.add_argument("--constraint", default="[a100_80|h100|h200|6000_blackwell]")
    ap.add_argument("--cpus", type=int, default=8)
    ap.add_argument("--mem-gb", type=int, default=96)
    ap.add_argument("--time-min", type=int, default=720)
    ap.add_argument("--after", default=None,
                    help="SLURM job id: start afterany:<id> (resume=True continues training → auto-resubmit chains)")
    ap.add_argument("--name", default="diffae_phase_v1")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = DiffAEConfig(
        n_crops=args.n_crops, crop_size=args.crop_size, epochs=args.epochs,
        batch_size=args.batch_size, device="cuda",
    )
    jobs = [{
        "name": args.name,
        "func": run_diffae,
        "kwargs": {"cfg": cfg, "out_dir": str(Path(args.out_dir).resolve())},
        "metadata": {"stage": "diffae"},
    }]
    slurm_params = {
        "slurm_partition": args.partition, "gpus_per_node": args.gpus,
        "cpus_per_task": args.cpus, "mem_gb": args.mem_gb, "timeout_min": args.time_min,
        "slurm_setup": ["export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"],
    }
    if args.constraint:
        slurm_params["slurm_constraint"] = args.constraint
    if args.after:  # resume-chain: wait for the prior job to end (any reason), then continue
        slurm_params["slurm_additional_parameters"] = {"dependency": f"afterany:{args.after}"}
    submit_parallel_jobs(
        jobs_to_submit=jobs, experiment="diffae",
        slurm_params=slurm_params, log_dir="diffae",
        dry_run=args.dry_run, wait_for_completion=False,
    )


if __name__ == "__main__":
    main()
