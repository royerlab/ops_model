"""Thin runner: invoke pca_optimization with a per-cell weight parquet + column.

Replaces the fragile monkey-patch flow in
``run_weighted_pca_monkey_patch.py``. Uses the upstream
``--weight-parquet`` + ``--weight-column`` flags added to pca_optimization
(see phase1.py's per-cell weighting block). Weighting is applied inside
phase1 itself; the tripwire in phase1 raises RuntimeError if the flag is set
but no experiment matches.

Usage
-----
    # Simple single-column strategy
    python -m ops_model.models.attention.weighted_aggregation.run_weighted_pca \\
        --weight-parquet /path/to/sidecar.parquet \\
        --weight-column v5_gko \\
        --paper-v2 \\
        --signal-set phase_only \\
        --run-tag attention/v5_gko \\
        --slurm

Any additional args after ``--`` are forwarded to pca_optimization verbatim
(e.g. ``-- --fixed-threshold 0.8 --chad-annotation ...``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weight-parquet", required=True,
                   help="Path to per-cell weight parquet")
    p.add_argument("--weight-column", required=True,
                   help="Column in --weight-parquet to use as per-cell weight")
    p.add_argument("--paper-v2", action="store_true",
                   help="Use paper_v2 features/experiments")
    p.add_argument("--paper-v1", action="store_true",
                   help="Use paper_v1 features/experiments")
    p.add_argument("--signal-set", default="phase_only",
                   choices=["phase_only", "no_phase", "all_livecell"])
    p.add_argument("--run-tag", required=True,
                   help="Output subdir under <root>/cell_dino/zscore_per_exp/paper_vX/")
    p.add_argument("--output-dir",
                   default="/hpc/projects/icd.fast.ops/organelle_attribution/"
                           "pca_optimized_v0.3")
    p.add_argument("--chad-annotation",
                   default="/hpc/projects/icd.fast.ops/configs/gene_clusters/"
                           "chad_positive_controls_v4.yml")
    p.add_argument("--fixed-threshold", type=float, default=0.80)
    p.add_argument("--slurm", action="store_true")
    p.add_argument("--slurm-partition", default="gpu")
    p.add_argument("--aggregate-only", action="store_true")
    args = p.parse_args()

    from ops_model.post_process.combination.pca_optimization import main as pca_main

    pca_argv = [
        "--output-dir", str(args.output_dir),
        "--cell-dino",
        "--zscore-per-experiment",
        "--run-tag", args.run_tag,
        "--chad-annotation", str(args.chad_annotation),
        "--fixed-threshold", str(args.fixed_threshold),
        "--slurm-partition", args.slurm_partition,
        "--weight-parquet", str(args.weight_parquet),
        "--weight-column", args.weight_column,
    ]
    if args.paper_v2:
        pca_argv.append("--paper-v2")
    elif args.paper_v1:
        pca_argv.append("--paper-v1")

    if args.signal_set == "phase_only":
        pca_argv.append("--phase-only")
        pca_argv.append("--no-second-pca")  # single channel → 2nd-pass is no-op
    elif args.signal_set == "no_phase":
        pca_argv.append("--no-phase")
    if args.slurm:
        pca_argv.append("--slurm")
    if args.aggregate_only:
        pca_argv.append("--aggregate-only")

    print(f"Invoking pca_optimization with:\n  {' '.join(pca_argv)}\n")
    sys.argv = ["pca_optimization"] + pca_argv
    return pca_main()


if __name__ == "__main__":
    sys.exit(main())
