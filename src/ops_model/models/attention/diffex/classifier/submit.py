"""Submit the classifier sweep to SLURM (GPU) via submit_parallel_jobs.

    # one gene, both models
    python -m ops_model.models.attention.diffex.classifier.submit --gene HSPA5 --models B C

    # all 98 EBI complexes, model C
    python -m ops_model.models.attention.diffex.classifier.submit --grain complex --all-classes --models C

    # specific classes
    python -m ops_model.models.attention.diffex.classifier.submit --grain complex \
        --classes "19S proteasome regulatory complex" "Commander complex" --models C

One GPU job per (class, model). Outputs under <out-dir>/<grain>/<class-slug>/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs

from .config import DEFAULT_OUT_ROOT, GRAINS, Config, slugify
from .run import run_poc


def _all_classes(parquet: str, class_col: str) -> list[str]:
    """Distinct class values present in the parquet (rank-1 top rows = cheap).
    Includes NTC — a useful negative-control bin (its classifier should be ~chance)."""
    df = pd.read_parquet(
        parquet, filters=[("rank", "==", 1), ("rank_type", "==", "top")],
        columns=[class_col],
    )
    return sorted(df[class_col].astype(str).unique())


def main():
    ap = argparse.ArgumentParser(description="Submit DiffEx classifier sweep to SLURM")
    ap.add_argument("--grain", choices=list(GRAINS), default="geneKO")
    ap.add_argument("--gene", default="HSPA5", help="single class value (ignored if --classes/--all-classes)")
    ap.add_argument("--classes", nargs="+", default=None, help="explicit class values")
    ap.add_argument("--all-classes", action="store_true", help="every class in the grain's parquet")
    ap.add_argument("--models", nargs="+", choices=["B", "C"], default=["C"])
    ap.add_argument("--n-per-class", type=int, default=1000)
    ap.add_argument("--crop-size", type=int, default=160)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--split-mode", choices=["experiment", "random"], default="experiment")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_ROOT)
    # SLURM
    ap.add_argument("--partition", default="gpu")
    ap.add_argument("--gres", default="gpu:1")
    ap.add_argument("--cpus", type=int, default=8)
    ap.add_argument("--mem-gb", type=int, default=64)
    ap.add_argument("--time-min", type=int, default=240)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    grain = GRAINS[args.grain]
    parquet, class_col = grain["parquet"], grain["class_col"]
    if args.all_classes:
        classes = _all_classes(parquet, class_col)
    elif args.classes:
        classes = args.classes
    else:
        classes = [args.gene]
    print(f"grain={args.grain} class_col={class_col} -> {len(classes)} classes; models={args.models}")

    base = Path(args.out_dir).resolve() / args.grain
    jobs = []
    for cls in classes:
        slug = slugify(cls)
        for m in args.models:
            cfg = Config(
                gene=cls, class_col=class_col, pma_parquet=parquet,
                n_per_class=args.n_per_class, crop_size=args.crop_size,
                epochs=args.epochs, split_mode=args.split_mode, device="cuda",
            )
            jobs.append({
                "name": f"diffex_{args.grain}_{slug}_{m}"[:64],
                "func": run_poc,
                "kwargs": {"cfg": cfg, "model": m, "out_dir": str(base / slug)},
                "metadata": {"grain": args.grain, "class": cls, "model": m},
            })

    slurm_params = {
        "slurm_partition": args.partition, "slurm_gres": args.gres,
        "cpus_per_task": args.cpus, "mem_gb": args.mem_gb, "timeout_min": args.time_min,
    }
    submit_parallel_jobs(
        jobs_to_submit=jobs, experiment=f"diffex_clf_{args.grain}",
        slurm_params=slurm_params, log_dir=f"diffex_clf_{args.grain}",
        dry_run=args.dry_run, wait_for_completion=False,
    )


if __name__ == "__main__":
    main()
