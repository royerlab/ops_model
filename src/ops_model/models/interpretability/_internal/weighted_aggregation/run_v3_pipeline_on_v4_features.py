"""Run the canonical v3 ``pca_optimization`` pipeline on Alex Lin's v4
cell-dino features as a sanity check on the v4 expansion sweep's mAP values.

Approach: monkey-patch ``ops_utils.data.feature_discovery.find_cell_h5ad_path``
to return our v4 per-experiment h5ad (built by ``build_v4_per_exp_h5ads.py``)
whenever pca_optimization asks for the Phase cell-dino features. Everything
downstream (per-exp z-score, PCA at 80%, NTC z-score, 4 mAP metrics) runs
unmodified. The output lands in a new ``alex_lin_features/`` subdir so it
doesn't clobber the existing v3 results.

Why: we suspect our v4 expansion sweep's all-cells baseline (EBI=0.7959 vs
v3's 0.5094) may be inflated by a bug somewhere in the bespoke v4 selection
+ aggregation code. Running v3's well-tested pipeline on the same v4 input
features should reproduce the all-cells number if our v4 sweep is correct.

Usage
-----
    # Submit pca_optimization to SLURM with output under alex_lin_features/
    uv run python run_v3_pipeline_on_v4_features.py \\
        --output-dir /hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized_v0.3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

V4_PER_EXP = Path(
    "/hpc/projects/icd.fast.ops/models/alex_lin_attention/v4/expansion_v1/"
    "per_experiment_v4"
)


def _install_monkey_patch() -> None:
    """Redirect find_cell_h5ad_path → v4 per-experiment h5ads for Phase channel."""
    from ops_utils.data import feature_discovery as fd

    _original = fd.find_cell_h5ad_path

    def _patched(experiment, channel, storage_roots, feature_dir, metadata_path):
        # Only redirect when pca_optimization asks for Phase cell-dino features.
        if "phase" in str(channel).lower():
            v4_path = V4_PER_EXP / f"{experiment}.h5ad"
            if v4_path.exists():
                return v4_path
        # Fall through to v3's discovery for anything else (we expect nothing
        # else in --phase-only mode, but be safe).
        return _original(experiment, channel, storage_roots, feature_dir, metadata_path)

    fd.find_cell_h5ad_path = _patched
    # Also patch the symbol re-exported into pca_optimization's phase1.py
    try:
        from ops_model.post_process.combination.pca_optimization import phase1 as p1
        p1.find_cell_h5ad_path = _patched
    except Exception:
        pass


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir",
                   default="/hpc/projects/icd.fast.ops/organelle_attribution/"
                           "pca_optimized_v0.3",
                   help="Root output dir; results land under "
                        "<root>/cell_dino/zscore_per_exp/alex_lin_features/paper_v1/phase_only/...")
    p.add_argument("--chad-annotation",
                   default="/hpc/projects/icd.fast.ops/configs/gene_clusters/"
                           "chad_positive_controls_v4.yml")
    p.add_argument("--slurm", action="store_true",
                   help="Submit via SLURM (recommended)")
    p.add_argument("--aggregate-only", action="store_true",
                   help="Skip Phase 1 (per_signal/ already exists), only run aggregation")
    p.add_argument("--fixed-threshold", type=float, default=0.80,
                   help="PCA variance cutoff — set =0 to enable the full sweep "
                        "(default 0.80, fixed = no sweep)")
    args = p.parse_args()

    _install_monkey_patch()
    print(f"✓ monkey-patched find_cell_h5ad_path → {V4_PER_EXP}/<exp>.h5ad")

    from ops_model.post_process.combination.pca_optimization import main as pca_main

    pca_argv = [
        "--output-dir", str(args.output_dir),
        "--cell-dino",
        "--zscore-per-experiment",
        "--phase-only",
        "--paper-v1",
        "--run-tag", "alex_lin_features",
        "--chad-annotation", str(args.chad_annotation),
        "--fixed-threshold", str(args.fixed_threshold),
        "--no-second-pca",
    ]
    if args.slurm:
        pca_argv.append("--slurm")
    if args.aggregate_only:
        pca_argv.append("--aggregate-only")

    print(f"Invoking pca_optimization with:\n  {' '.join(pca_argv)}")
    sys.argv = ["pca_optimization"] + pca_argv
    return pca_main()


if __name__ == "__main__":
    sys.exit(main())
