"""Submit the DiffAE training to SLURM (1 GPU, longer wall clock).

    python -m ops_model.models.attention.diffex.diffae.submit

=============================== RUNBOOK ===============================
Checkpoints (root /hpc/projects/icd.fast.ops/models/diffex/diffae/) and their
best cond_ratio (emb/noise conditioning strength; higher = stronger edits):
  phase_v1/            50k crops,  ep120, 0.468   <- PRODUCTION (all traversals use this)
  phase_v1_500k/       500k scratch, ep12, 0.416  (undertrained; parked)
  phase_v1_500k_warm/  500k warm-from-v1, 0.542   (more-data test; being resumed)

RESUME an existing run (continue where it stopped): resume=True is the config
default and train state (model+ema+opt+epoch) is saved EVERY epoch to
<out>/diffae_train_state.pt. Just re-submit the SAME --out-dir/--n-crops and it
picks up automatically. Do NOT pass --init-ckpt on a resume (train_state wins).

MEMORY GOTCHA (500k): the 500k crop cache is 51 GB float32 and load_diffae_crops
normalizes it -> ~102 GB transient peak. Use mem_gb>=200 for 500k runs; 96 GB
OOM-kills (esp. on a shared node). 50k runs are fine at 96 GB.

CHAIN across the 720-min walltime: submit N jobs, each with
slurm_additional_parameters={"dependency": f"afterany:<prev_id>"} (see --after).
afterany fires even on failure, so verify link 0 clears the cache-load before
trusting the chain.

WATCH cond_ratio trend:
  python -c "import torch;h=torch.load('<out>/diffae_train_state.pt',map_location='cpu')['history'];print([round(e.get('cond_ratio',-1),3) for e in h if e.get('cond_ratio',-1)>0])"
======================================================================
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
    ap.add_argument("--augment-affine", action="store_true",
                    help="continuous rotation+scale+flip aug (else discrete dihedral)")
    ap.add_argument("--no-aug", action="store_true",
                    help="disable ALL augmentation (true v1-style, no dihedral/affine)")
    ap.add_argument("--init-ckpt", default=None,
                    help="warm-start: load these weights into the fresh model before training")
    ap.add_argument("--marker-channel", default=None,
                    help="fluor mode: fluor-CSV `channel` value (e.g. 'nucleolus-GC_NPM3')")
    ap.add_argument("--anndata-paths", default=None,
                    help="no-PMA markers: comma-separated per-exp CellDINO anndata h5ad paths")
    ap.add_argument("--channel", default="Phase2D",
                    help="raw pheno-zarr channel to read (Phase2D | GFP | mCherry | Cy5)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    affine = args.augment_affine and not args.no_aug
    dihedral = (not args.augment_affine) and not args.no_aug
    cfg = DiffAEConfig(
        n_crops=args.n_crops, crop_size=args.crop_size, epochs=args.epochs,
        batch_size=args.batch_size, device="cuda",
        augment_affine=affine, augment_dihedral=dihedral, init_ckpt=args.init_ckpt,
        marker_channel=args.marker_channel, channel=args.channel,
        anndata_paths=tuple(args.anndata_paths.split(",")) if args.anndata_paths else (),
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
