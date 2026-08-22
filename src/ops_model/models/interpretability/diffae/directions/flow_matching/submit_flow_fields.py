"""Submit `flow_field.build_flow_field` across every target of a grain (SLURM, 1 GPU/job).

Mirrors the existing full-library buildout pattern (mean_diff + traversal already ran at this
~1000-gene scale — see interpretability/diffae/PLAN.md, "Full NTC drain"). Resume is automatic:
a target whose `metrics.json` already exists under --out-root/<grain>/<slug>/ is skipped, so a
re-run after a partial failure only rebuilds what's missing.

    python -m ops_model.models.interpretability.diffae.directions.submit_flow_fields --grain complex
    python -m ops_model.models.interpretability.diffae.directions.submit_flow_fields --grain geneKO
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs

from ...classifier.config import DEFAULT_OUT_ROOT, GRAINS, slugify
from ..config import DirConfig
from .flow_field import build_flow_field


def _list_targets(grain: str) -> list:
    g = GRAINS[grain]
    df = pd.read_parquet(g["parquet"], columns=[g["class_col"], "rank_type"])
    vals = sorted(df[df.rank_type == "top"][g["class_col"]].unique())
    return [v for v in vals if v != "NTC"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Build OT-coupled flow fields for every target of a grain")
    ap.add_argument("--grain", choices=["geneKO", "complex"], required=True)
    ap.add_argument("--out-root", default=f"{DEFAULT_OUT_ROOT}/flow_fields")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--w", type=float, default=1.0)
    ap.add_argument("--flow-steps", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=None, help="cap target count (smoke-test a subset)")
    ap.add_argument("--partition", default="gpu")
    ap.add_argument("--gres", default="gpu:1")
    ap.add_argument("--cpus", type=int, default=8)
    ap.add_argument("--mem-gb", type=int, default=64)
    ap.add_argument("--time-min", type=int, default=90)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    targets = _list_targets(args.grain)
    if args.limit:
        targets = targets[: args.limit]
    out_root = Path(args.out_root) / args.grain

    jobs, skipped = [], 0
    for target in targets:
        out_dir = out_root / slugify(target)
        if (out_dir / "metrics.json").exists():
            skipped += 1
            continue
        cfg = DirConfig(grain=args.grain, target=target, device="cuda")
        jobs.append({
            "name": f"flowfield_{args.grain}_{slugify(target)}"[:64],
            "func": build_flow_field,
            "kwargs": {"cfg": cfg, "out_dir": str(out_dir), "seed": args.seed, "w": args.w,
                      "flow_steps": args.flow_steps},
            "metadata": {"stage": "flow_field", "grain": args.grain, "target": target},
        })
    print(f"{args.grain}: {len(targets)} targets total, {skipped} already built, {len(jobs)} to submit")
    if not jobs:
        return

    slurm_params = {
        "slurm_partition": args.partition, "slurm_gres": args.gres,
        "cpus_per_task": args.cpus, "mem_gb": args.mem_gb, "timeout_min": args.time_min,
    }
    submit_parallel_jobs(
        jobs_to_submit=jobs, experiment=f"flow_fields_{args.grain}",
        slurm_params=slurm_params, log_dir=f"flow_fields_{args.grain}",
        dry_run=args.dry_run, wait_for_completion=True,
    )


if __name__ == "__main__":
    main()
