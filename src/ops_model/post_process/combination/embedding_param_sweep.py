"""Standalone UMAP / PHATE parameter sweep — shape-only canvases.

Points at a single run directory containing
``guide_pca_optimized.h5ad`` and/or ``gene_embedding_pca_optimized.h5ad``,
re-fits UMAP and PHATE across a 2D primary parameter grid plus several 1D
auxiliary strips, and writes shape-only canvas PNGs (no overlays, no labels)
under ``<run_dir>/embedding_sweep/`` so the user can pick a parameter regime
by eye.

Each `(level, embedder, param-combo)` fit is independent, so the script fans
out across CPU cores via ``joblib.Parallel`` (process pool) sized by
``ops_utils.hpc.resource_manager.get_optimal_workers``. By default the whole
sweep is submitted to SLURM as a single multi-CPU job via
``ops_utils.hpc.slurm_batch_utils.submit_parallel_jobs``; pass ``--local`` to
run on the current node instead.

Run with::

    # SLURM (default)
    python -m ops_model.post_process.combination.embedding_param_sweep

    # local
    python -m ops_model.post_process.combination.embedding_param_sweep --local
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import anndata as ad
import numpy as np

from ops_utils.analysis.embedding_plots import clean_X_for_embedding


# ---------------------------------------------------------------------------
# Defaults — wide search; CLI flags can shrink/extend any of these
# ---------------------------------------------------------------------------

DEFAULT_UMAP_N_NEIGHBORS = (2, 5, 10, 15, 25, 50, 100, 200, 500)
DEFAULT_UMAP_MIN_DIST = (0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.99)
DEFAULT_UMAP_METRIC = ("euclidean", "cosine", "correlation", "manhattan", "chebyshev")
DEFAULT_UMAP_SPREAD = (0.5, 1.0, 1.5, 2.0, 3.0)

DEFAULT_PHATE_KNN = (3, 5, 10, 15, 25, 50, 100)
DEFAULT_PHATE_DECAY = (2, 5, 10, 20, 40, 80)
DEFAULT_PHATE_T = ("auto", 5, 10, 20, 40, 80)
DEFAULT_PHATE_GAMMA = (0.0, 0.5, 1.0)

CANONICAL_UMAP = {"n_neighbors": 15, "min_dist": 0.1, "metric": "euclidean", "spread": 1.0}
CANONICAL_PHATE = {"knn": 15, "decay": 15, "t": "auto", "gamma": 1.0}

DEFAULT_RUN_DIR = (
    "/hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized_v0.3/"
    "cell_dino/zscore_per_exp/paper_v1/all/fixed_80%/cosine/second_pca_consensus"
)


# ---------------------------------------------------------------------------
# Sweep workers
# ---------------------------------------------------------------------------


def _fit_umap(
    X: np.ndarray, seed: int, *, n_neighbors: int, min_dist: float,
    metric: str = "euclidean", spread: float = 1.0,
) -> Optional[np.ndarray]:
    from umap import UMAP

    nn = max(2, min(n_neighbors, X.shape[0] - 1))
    model = UMAP(
        n_components=2, n_neighbors=nn, min_dist=float(min_dist),
        metric=metric, spread=float(spread), random_state=seed,
    )
    return model.fit_transform(X)


def _fit_phate(
    X: np.ndarray, seed: int, *, knn: int, decay: float,
    t="auto", gamma: float = 1.0, n_jobs: int = 1,
) -> Optional[np.ndarray]:
    import phate

    knn_eff = max(2, min(knn, X.shape[0] - 1))
    return phate.PHATE(
        n_components=2, knn=knn_eff, decay=float(decay), t=t,
        gamma=float(gamma), n_jobs=n_jobs, random_state=seed, verbose=0,
    ).fit_transform(X)


def _fit_one(X: np.ndarray, seed: int, embedder: str, kwargs: Dict) -> Optional[np.ndarray]:
    """Worker entrypoint: fit one (embedder, kwargs) on X. Module-level so it
    is picklable by the joblib loky backend."""
    try:
        if embedder == "umap":
            return _fit_umap(X, seed, **kwargs)
        return _fit_phate(X, seed, n_jobs=1, **kwargs)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Canvas drawing
# ---------------------------------------------------------------------------


def _draw_grid(
    coords_grid: Dict[Tuple, Optional[np.ndarray]],
    row_vals: Sequence, col_vals: Sequence,
    row_label: str, col_label: str,
    title: str, out_path: Path, plt,
    default_key: Optional[Tuple] = None,
) -> None:
    nr, nc = len(row_vals), len(col_vals)
    fig, axes = plt.subplots(
        nr, nc, figsize=(3.2 * nc + 0.6, 3.2 * nr + 0.8),
        squeeze=False,
    )
    for i, rv in enumerate(row_vals):
        for j, cv in enumerate(col_vals):
            ax = axes[i, j]
            ax.set_xticks([])
            ax.set_yticks([])
            coords = coords_grid.get((rv, cv))
            if coords is None:
                ax.set_facecolor("#f5f5f5")
                ax.text(0.5, 0.5, "—", transform=ax.transAxes,
                        ha="center", va="center", fontsize=28, color="#999")
            else:
                ax.scatter(
                    coords[:, 0], coords[:, 1],
                    s=2, c="0.25", alpha=0.4, linewidths=0,
                )
                ax.set_aspect("equal", adjustable="datalim")
            is_default = default_key is not None and (rv, cv) == default_key
            tile_title = f"{row_label}={rv}\n{col_label}={cv}"
            if is_default:
                tile_title += "\n(pipeline default)"
                for spine in ax.spines.values():
                    spine.set_edgecolor("#d62728")
                    spine.set_linewidth(4.0)
            ax.set_title(
                tile_title,
                fontsize=18, fontweight="bold", pad=4,
                color="#d62728" if is_default else "black",
            )
    fig.suptitle(title, fontsize=24, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Spec building + parallel fan-out
# ---------------------------------------------------------------------------


def _build_specs(cfg: Dict, embedders: Sequence[str]) -> List[Dict]:
    """Build a flat list of fit specs covering every (canvas, key) pair the
    user asked for. ``canvas`` is the bucket the result will land in; ``key``
    is its position inside that bucket."""
    specs: List[Dict] = []
    if "umap" in embedders:
        for nn in cfg["umap_n_neighbors"]:
            for md in cfg["umap_min_dist"]:
                specs.append({
                    "canvas": "umap_primary", "key": (nn, md), "embedder": "umap",
                    "kwargs": {"n_neighbors": nn, "min_dist": md,
                               "metric": CANONICAL_UMAP["metric"],
                               "spread": CANONICAL_UMAP["spread"]},
                })
        if cfg["secondary"]:
            for m in cfg["umap_metric_list"]:
                for s in cfg["umap_spread_list"]:
                    specs.append({
                        "canvas": "umap_secondary", "key": (m, s), "embedder": "umap",
                        "kwargs": {"n_neighbors": CANONICAL_UMAP["n_neighbors"],
                                   "min_dist": CANONICAL_UMAP["min_dist"],
                                   "metric": m, "spread": s},
                    })
    if "phate" in embedders:
        for k in cfg["phate_knn"]:
            for d in cfg["phate_decay"]:
                specs.append({
                    "canvas": "phate_primary", "key": (k, d), "embedder": "phate",
                    "kwargs": {"knn": k, "decay": d,
                               "t": CANONICAL_PHATE["t"],
                               "gamma": CANONICAL_PHATE["gamma"]},
                })
        if cfg["secondary"]:
            for tv in cfg["phate_t_list"]:
                for g in cfg["phate_gamma_list"]:
                    specs.append({
                        "canvas": "phate_secondary", "key": (tv, g), "embedder": "phate",
                        "kwargs": {"knn": CANONICAL_PHATE["knn"],
                                   "decay": CANONICAL_PHATE["decay"],
                                   "t": tv, "gamma": g},
                    })
    return specs


def _parallel_fit(
    X: np.ndarray, seed: int, specs: List[Dict], n_workers: int, _logger,
) -> List[Optional[np.ndarray]]:
    """Run every spec in parallel via joblib loky backend; return coords in spec order."""
    from joblib import Parallel, delayed

    total = len(specs)
    _logger.info("  Fanning out %d fits across %d workers", total, n_workers)
    t0 = time.time()
    results = Parallel(n_jobs=n_workers, backend="loky", verbose=5)(
        delayed(_fit_one)(X, seed, s["embedder"], s["kwargs"]) for s in specs
    )
    n_ok = sum(1 for r in results if r is not None)
    _logger.info(
        "  Fits done: %d/%d ok, %d failed — %.1f min wall, %.1fs/fit",
        n_ok, total, total - n_ok,
        (time.time() - t0) / 60, (time.time() - t0) / max(1, total),
    )
    return list(results)


# ---------------------------------------------------------------------------
# Per-level driver
# ---------------------------------------------------------------------------


def _process_level(
    level: str, adata, plots_dir: Path, embedders: Sequence[str],
    cfg: Dict, plt, _logger, n_workers: int,
) -> None:
    X = clean_X_for_embedding(adata)
    n_obs = X.shape[0]
    if cfg["max_points"] and n_obs > cfg["max_points"]:
        rng = np.random.default_rng(cfg["random_seed"])
        idx = rng.choice(n_obs, size=cfg["max_points"], replace=False)
        X = X[idx]
        _logger.info("  %s: subsampled %d -> %d obs", level, n_obs, X.shape[0])
    _logger.info("  %s: %d obs x %d features", level, X.shape[0], X.shape[1])

    specs = _build_specs(cfg, embedders)
    if not specs:
        _logger.warning("  %s: no embedders/params requested — skipping", level)
        return

    results = _parallel_fit(X, cfg["random_seed"], specs, n_workers, _logger)

    canvases: Dict[str, Dict] = {}
    for spec, coords in zip(specs, results):
        canvases.setdefault(spec["canvas"], {})[spec["key"]] = coords

    n_drawn = X.shape[0]

    if "umap_primary" in canvases:
        _draw_grid(
            canvases["umap_primary"], cfg["umap_n_neighbors"], cfg["umap_min_dist"],
            "n_neighbors", "min_dist",
            f"{level} UMAP primary sweep — {n_drawn} obs"
            f"\nheld: metric={CANONICAL_UMAP['metric']}, spread={CANONICAL_UMAP['spread']}",
            plots_dir / f"{level}_umap_primary_sweep.png", plt,
            default_key=(CANONICAL_UMAP["n_neighbors"], CANONICAL_UMAP["min_dist"]),
        )
        _logger.info("  Saved %s_umap_primary_sweep.png", level)
    if "umap_secondary" in canvases:
        _draw_grid(
            canvases["umap_secondary"], cfg["umap_metric_list"], cfg["umap_spread_list"],
            "metric", "spread",
            f"{level} UMAP secondary sweep — {n_drawn} obs"
            f"\nheld: n_neighbors={CANONICAL_UMAP['n_neighbors']}, "
            f"min_dist={CANONICAL_UMAP['min_dist']}",
            plots_dir / f"{level}_umap_secondary_sweep.png", plt,
            default_key=(CANONICAL_UMAP["metric"], CANONICAL_UMAP["spread"]),
        )
        _logger.info("  Saved %s_umap_secondary_sweep.png", level)
    if "phate_primary" in canvases:
        _draw_grid(
            canvases["phate_primary"], cfg["phate_knn"], cfg["phate_decay"],
            "knn", "decay",
            f"{level} PHATE primary sweep — {n_drawn} obs"
            f"\nheld: t={CANONICAL_PHATE['t']}, gamma={CANONICAL_PHATE['gamma']}",
            plots_dir / f"{level}_phate_primary_sweep.png", plt,
            default_key=(CANONICAL_PHATE["knn"], CANONICAL_PHATE["decay"]),
        )
        _logger.info("  Saved %s_phate_primary_sweep.png", level)
    if "phate_secondary" in canvases:
        _draw_grid(
            canvases["phate_secondary"], cfg["phate_t_list"], cfg["phate_gamma_list"],
            "t", "gamma",
            f"{level} PHATE secondary sweep — {n_drawn} obs"
            f"\nheld: knn={CANONICAL_PHATE['knn']}, decay={CANONICAL_PHATE['decay']}",
            plots_dir / f"{level}_phate_secondary_sweep.png", plt,
            default_key=(CANONICAL_PHATE["t"], CANONICAL_PHATE["gamma"]),
        )
        _logger.info("  Saved %s_phate_secondary_sweep.png", level)


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


def _resolve_n_workers(_logger) -> int:
    """Defer to ``ops_utils.hpc.resource_manager.get_optimal_workers`` — it
    already accounts for SLURM CPU/RAM allocation, cgroup limits, and
    sched_getaffinity."""
    from ops_utils.hpc.resource_manager import get_optimal_workers

    n = get_optimal_workers(
        use_gpu=False,
        model_ram_gb=1.5,
        data_ram_gb=0.5,
        cpu_safety_buffer=1,
        verbose=False,
    )
    return max(1, int(n))


def run_sweep(cfg: Dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _logger = logging.getLogger(__name__)

    run_dir = Path(cfg["run_dir"])
    if not run_dir.exists():
        raise SystemExit(f"ERROR: {run_dir} does not exist")

    plots_dir = run_dir / "embedding_sweep"
    plots_dir.mkdir(parents=True, exist_ok=True)

    n_workers = _resolve_n_workers(_logger)

    h5ad_paths = {
        "guide": run_dir / "guide_pca_optimized.h5ad",
        "gene": run_dir / "gene_embedding_pca_optimized.h5ad",
    }

    t_total = time.time()
    for level in cfg["levels"]:
        path = h5ad_paths[level]
        if not path.exists():
            _logger.warning("  %s: %s missing — skipping", level, path)
            continue
        _logger.info("Loading %s (%s)", level, path.name)
        adata = ad.read_h5ad(path)
        _process_level(
            level, adata, plots_dir, cfg["embedders"], cfg, plt, _logger, n_workers,
        )

    _logger.info("Done in %.1f min — wrote to %s", (time.time() - t_total) / 60, plots_dir)


# ---------------------------------------------------------------------------
# SLURM submission
# ---------------------------------------------------------------------------


def submit_slurm(cfg: Dict, slurm_params: Dict, dry_run: bool = False) -> Dict:
    """Submit ``run_sweep(cfg)`` as a single multi-CPU SLURM job and block on
    completion. Live progress, resource utilization, and pass/fail reporting
    are handled by ``submit_parallel_jobs`` — same machinery the rest of the
    pipeline scripts use."""
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs

    job_name = f"embedding_sweep_{Path(cfg['run_dir']).name}"
    return submit_parallel_jobs(
        jobs_to_submit=[{
            "name": job_name,
            "func": run_sweep,
            "kwargs": {"cfg": cfg},
            "metadata": {
                "run_dir": cfg["run_dir"],
                "levels": cfg["levels"],
                "embedders": cfg["embedders"],
            },
        }],
        experiment=job_name,
        slurm_params=slurm_params,
        log_dir="embedding_sweep",
        manifest_prefix="embedding_sweep",
        dry_run=dry_run,
        wait_for_completion=True,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_list(s: str, *, cast=str) -> List:
    out: List = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if cast is float:
            out.append(float(tok))
        elif cast is int:
            out.append(int(tok))
        else:
            out.append(tok)
    return out


def _parse_phate_t(s: str) -> List:
    out: List = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok == "auto":
            out.append("auto")
        else:
            out.append(int(tok))
    return out


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Sweep UMAP/PHATE parameters at guide and gene levels and save "
            "shape-only canvas PNGs."
        ),
    )
    p.add_argument("--run-dir", default=DEFAULT_RUN_DIR, type=str,
                   help=("Run directory containing guide_pca_optimized.h5ad and/or "
                         "gene_embedding_pca_optimized.h5ad. "
                         f"Default: {DEFAULT_RUN_DIR}"))
    p.add_argument("--levels", default="guide,gene", type=str,
                   help="Comma-separated subset of {guide,gene}")
    p.add_argument("--embedders", default="umap,phate", type=str,
                   help="Comma-separated subset of {umap,phate}")
    p.add_argument("--no-secondary", action="store_true",
                   help="Skip the secondary 2D grid (UMAP metric x spread, PHATE t x gamma)")
    p.add_argument("--random-seed", default=42, type=int)
    p.add_argument("--max-points", default=0, type=int,
                   help="0=use all; else random subsample to this many obs before fitting")

    # SLURM (submission is the default; pass --local to run on the current node)
    p.add_argument("--local", action="store_true",
                   help="Run the sweep on the current node instead of submitting to SLURM")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the SLURM submission plan without submitting (no effect with --local)")
    p.add_argument("--slurm-time", type=int, default=240,
                   help="SLURM time limit in minutes (default: 240)")
    p.add_argument("--slurm-memory", type=str, default="200GB",
                   help="SLURM memory request (default: 200GB)")
    p.add_argument("--slurm-cpus", type=int, default=32,
                   help="SLURM cpus_per_task — also caps the joblib worker pool (default: 32)")
    p.add_argument("--slurm-partition", type=str, default="cpu",
                   help="SLURM partition (default: cpu)")

    # UMAP
    p.add_argument("--umap-n-neighbors", default=",".join(str(v) for v in DEFAULT_UMAP_N_NEIGHBORS))
    p.add_argument("--umap-min-dist", default=",".join(str(v) for v in DEFAULT_UMAP_MIN_DIST))
    p.add_argument("--umap-metric-list", default=",".join(DEFAULT_UMAP_METRIC))
    p.add_argument("--umap-spread-list", default=",".join(str(v) for v in DEFAULT_UMAP_SPREAD))

    # PHATE
    p.add_argument("--phate-knn", default=",".join(str(v) for v in DEFAULT_PHATE_KNN))
    p.add_argument("--phate-decay", default=",".join(str(v) for v in DEFAULT_PHATE_DECAY))
    p.add_argument("--phate-t-list", default=",".join(str(v) for v in DEFAULT_PHATE_T))
    p.add_argument("--phate-gamma-list", default=",".join(str(v) for v in DEFAULT_PHATE_GAMMA))

    return p


def _cfg_from_args(args: argparse.Namespace) -> Dict:
    return {
        "run_dir": args.run_dir,
        "levels": _parse_list(args.levels),
        "embedders": _parse_list(args.embedders),
        "secondary": not args.no_secondary,
        "random_seed": args.random_seed,
        "max_points": args.max_points,
        "umap_n_neighbors": _parse_list(args.umap_n_neighbors, cast=int),
        "umap_min_dist": _parse_list(args.umap_min_dist, cast=float),
        "umap_metric_list": _parse_list(args.umap_metric_list),
        "umap_spread_list": _parse_list(args.umap_spread_list, cast=float),
        "phate_knn": _parse_list(args.phate_knn, cast=int),
        "phate_decay": _parse_list(args.phate_decay, cast=float),
        "phate_t_list": _parse_phate_t(args.phate_t_list),
        "phate_gamma_list": _parse_list(args.phate_gamma_list, cast=float),
    }


def _slurm_params_from_args(args: argparse.Namespace) -> Dict:
    return {
        "timeout_min": args.slurm_time,
        "mem": args.slurm_memory,
        "cpus_per_task": args.slurm_cpus,
        "slurm_partition": args.slurm_partition,
    }


def main() -> None:
    args = _build_parser().parse_args()
    cfg = _cfg_from_args(args)
    if args.local:
        run_sweep(cfg)
    else:
        submit_slurm(cfg, _slurm_params_from_args(args), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
