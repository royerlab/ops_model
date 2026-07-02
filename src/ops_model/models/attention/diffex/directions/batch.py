"""Batch DiffEx: strips + NTC→KO GIF for the top-ranked geneKOs and EBI complexes.

Reads the k10-ranked CSVs from the attention-selection page (top-N by rank_by_K10_mAP),
submits one GPU job per target. Each job: run_directions at w=5 only → per-cell strips +
scores, then a GIF for the auto-picked best-Δscore cell.

    python -m ops_model.models.attention.diffex.directions.batch \
        --genes-csv <k10_ranked_all_geneKOs.csv> --complex-csv <k10_ranked_all_complexes.csv> \
        --n-genes 50 --n-complex 20
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs

from ..classifier.config import DEFAULT_OUT_ROOT, slugify
from .config import DirConfig
from .make_gifs import make_all_gifs, make_gif
from .run import run_directions


def _short(name: str, n: int = 18) -> str:
    """Short GIF header label for a complex name (genes pass through unchanged)."""
    s = name
    for suf in (" complex", " subunit", " variant"):
        s = s.replace(suf, "")
    s = s.strip().strip(",")
    return s if len(s) <= n else s[: n - 1] + "…"


def run_target(grain: str, target: str, label: str, w: float = 5.0) -> dict:
    cfg = DirConfig(grain=grain, target=target, device="cuda")
    cfg.guidance_scales = (w,)                      # w=5 only (batch)
    out = f"{DEFAULT_OUT_ROOT}/directions/{grain}/{slugify(target)}"
    run_directions(cfg, out)
    sc = np.load(f"{out}/scores_w{w:g}.npy")        # (n_cells, n_alphas)
    delta = sc[:, -1] - sc[:, 0]
    best = int(np.argmax(delta))                    # cell that moves most toward the phenotype
    make_gif(grain, target, best, w, label)
    return {"target": target, "best_cell": best, "delta": float(delta[best])}


def all_gifs_target(grain: str, target: str, label: str, w: float = 5.0) -> list:
    """Render GIFs for every traversed cell of one target (strips must already exist)."""
    return make_all_gifs(grain, target, label, w=w)


def build_targets(genes_csv: str, complex_csv: str, n_genes: int, n_complex: int):
    g = pd.read_csv(genes_csv).sort_values("rank_by_K10_mAP").head(n_genes)
    c = pd.read_csv(complex_csv).sort_values("rank_by_K10_mAP").head(n_complex)
    jobs = [("geneKO", t, t) for t in g["geneKO"].tolist()]
    jobs += [("complex", t, _short(t)) for t in c["complex_name"].tolist()]
    return jobs


def main():
    ap = argparse.ArgumentParser(description="Batch DiffEx strips+GIFs for top-ranked targets")
    ap.add_argument("--genes-csv", required=True)
    ap.add_argument("--complex-csv", required=True)
    ap.add_argument("--n-genes", type=int, default=50)
    ap.add_argument("--n-complex", type=int, default=20)
    ap.add_argument("--w", type=float, default=5.0)
    ap.add_argument("--gifs-only", action="store_true",
                    help="strips already exist: only render GIFs for all traversed cells")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    targets = build_targets(args.genes_csv, args.complex_csv, args.n_genes, args.n_complex)
    print(f"{len(targets)} targets: {args.n_genes} geneKO + {args.n_complex} complex"
          f"{'  [gifs-only, all cells]' if args.gifs_only else ''}")
    func = all_gifs_target if args.gifs_only else run_target
    jobs = [{
        "name": f"dx_{grain}_{slugify(target)[:24]}",
        "func": func,
        "kwargs": {"grain": grain, "target": target, "label": label, "w": args.w},
        "metadata": {"stage": "batch_gifs" if args.gifs_only else "batch_directions",
                     "grain": grain, "target": target},
    } for grain, target, label in targets]

    submit_parallel_jobs(
        jobs_to_submit=jobs, experiment="diffex_batch",
        slurm_params={"slurm_partition": "gpu", "gpus_per_node": 1, "cpus_per_task": 8,
                      "mem_gb": 64, "timeout_min": 90,
                      "slurm_constraint": "[a100_80|h100|h200|6000_blackwell]",
                      "slurm_setup": ["export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"]},
        log_dir="diffex_batch", dry_run=args.dry_run, wait_for_completion=False,
    )


if __name__ == "__main__":
    main()
