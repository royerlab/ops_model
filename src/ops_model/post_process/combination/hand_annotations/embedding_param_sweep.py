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

# Hand-annotated leiden clusters used to color the sweep panels and gauge
# how well a parameter regime separates real biology. Two flavours match the
# two embedding trees produced by pca_optimization.
DEFAULT_CLUSTER_ANNOTATIONS_LIVECELL = (
    Path(__file__).resolve().parent / "hand_annotated_cluster.txt"
)
DEFAULT_CLUSTER_ANNOTATIONS_WITHCP4I = (
    Path(__file__).resolve().parent / "hand_annotated_cluster_withcp4i.txt"
)


# ---------------------------------------------------------------------------
# Cluster-annotation parsing + coloring helpers
# (mirrors gene_best_marker_assignment so the two visualizations stay
# pixel-comparable across scripts).
# ---------------------------------------------------------------------------


def _parse_hand_annotated_clusters(path: Path) -> Dict[str, List[str]]:
    """Parse a hand_annotated_cluster.txt file → {cluster_name: [gene, ...]}."""
    out: Dict[str, List[str]] = {}
    name: Optional[str] = None
    genes: List[str] = []

    def _flush() -> None:
        nonlocal name, genes
        if name and genes:
            out[name] = list(genes)
        name = None
        genes = []

    for raw in Path(path).read_text().splitlines():
        s = raw.strip()
        if not s:
            _flush()
            continue
        if s.startswith("#"):
            continue
        if s.endswith(":"):
            _flush()
            name = s.rstrip(":").strip()
            continue
        if name is not None:
            genes.append(s)
    _flush()
    return out


def _build_cluster_color_map(
    embed_genes: np.ndarray,
    cluster_map: Dict[str, List[str]],
    *,
    ntc_prefix: str = "NTC",
    palette_name: str = "tab20",
):
    """Return (rgba, cluster_colors, ntc_mask) aligned to ``embed_genes``.

    rgba           : (N, 4) per-gene — cluster color at alpha 0.9, gray (0.30)
                     for un-annotated, alpha 0 for NTC (drawn separately as ✕).
    cluster_colors : {cluster_name → RGBA tuple} — exposed so the legend
                     mirrors the panel coloring exactly.
    ntc_mask       : bool array marking NTC genes (matched on ``ntc_prefix``).
    """
    import matplotlib

    embed_genes = np.asarray(embed_genes)
    cluster_names = list(cluster_map.keys())
    cmap = matplotlib.colormaps.get_cmap(palette_name)
    cluster_colors: Dict[str, tuple] = {
        name: tuple(cmap(i % 20)) for i, name in enumerate(cluster_names)
    }
    gene_to_cluster: Dict[str, str] = {}
    for nm, gs in cluster_map.items():
        for g in gs:
            gene_to_cluster.setdefault(str(g), nm)

    n = len(embed_genes)
    rgba = np.zeros((n, 4), dtype=np.float32)
    rgba[:, :3] = 0.78
    rgba[:, 3] = 0.30
    ntc_mask = np.zeros(n, dtype=bool)
    for i, g in enumerate(embed_genes):
        gs = str(g)
        if gs.startswith(ntc_prefix):
            ntc_mask[i] = True
            rgba[i, 3] = 0.0   # painted separately
            continue
        cl = gene_to_cluster.get(gs)
        if cl is not None:
            r, gc, b, _ = cluster_colors[cl]
            rgba[i, :3] = (r, gc, b)
            rgba[i, 3] = 0.90
    return rgba, cluster_colors, ntc_mask


def _save_cluster_color_legend(
    cluster_colors: Dict[str, tuple],
    out_path: Path, plt,
    *,
    ncol: int = 3,
) -> None:
    """One-off swatch reference: cluster name → color, in ``ncol`` columns."""
    import matplotlib.patches as mpatches

    items = list(cluster_colors.items())
    n = len(items)
    rows = (n + ncol - 1) // ncol
    fig, ax = plt.subplots(figsize=(2.8 * ncol, 0.18 * rows + 0.8))
    ax.axis("off")
    handles = [
        mpatches.Patch(facecolor=color, edgecolor="black",
                       linewidth=0.3, label=name)
        for name, color in items
    ]
    ax.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(0, 1),
        ncol=ncol, fontsize=7, frameon=False,
        handlelength=1.4, handleheight=1.2, labelspacing=0.4,
        columnspacing=1.2,
    )
    fig.suptitle(
        f"Hand-annotated cluster color legend ({n} clusters)",
        fontsize=12, fontweight="bold", y=0.99,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


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
    cluster_data: Optional[Dict] = None,
) -> None:
    """Draw the sweep canvas. When ``cluster_data`` is provided every panel
    paints points colored by their hand-annotated cluster (gray for un-
    annotated genes; red ✕ for NTCs) so the eye can compare cluster
    separation across param combinations at a glance.

    ``cluster_data`` keys (all required when set):
        ``rgba``     — (n_pts, 4) per-point RGBA aligned to coords order
        ``ntc_mask`` — bool (n_pts,) marking NTC genes
    """
    nr, nc = len(row_vals), len(col_vals)
    fig, axes = plt.subplots(
        nr, nc, figsize=(3.2 * nc + 0.6, 3.2 * nr + 0.8),
        squeeze=False,
    )
    rgba = cluster_data["rgba"] if cluster_data else None
    ntc_mask = cluster_data["ntc_mask"] if cluster_data else None
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
                if rgba is not None and rgba.shape[0] == coords.shape[0]:
                    ax.scatter(
                        coords[:, 0], coords[:, 1],
                        s=4, c=rgba, linewidths=0,
                    )
                    if ntc_mask is not None and ntc_mask.any():
                        ax.scatter(
                            coords[ntc_mask, 0], coords[ntc_mask, 1],
                            marker="x", s=14, c="red", linewidths=0.8,
                            alpha=0.55, zorder=4,
                        )
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
    sub_idx: Optional[np.ndarray] = None
    if cfg["max_points"] and n_obs > cfg["max_points"]:
        rng = np.random.default_rng(cfg["random_seed"])
        sub_idx = rng.choice(n_obs, size=cfg["max_points"], replace=False)
        X = X[sub_idx]
        _logger.info("  %s: subsampled %d -> %d obs", level, n_obs, X.shape[0])
    _logger.info("  %s: %d obs x %d features", level, X.shape[0], X.shape[1])

    # Build cluster-color overlay if requested. Indices stay aligned to the
    # post-subsample X order so every panel's scatter colors line up exactly.
    cluster_data: Optional[Dict] = None
    if cfg.get("cluster_annotations"):
        cluster_path = Path(cfg["cluster_annotations"])
        if not cluster_path.is_file():
            _logger.warning(
                "  %s: cluster annotations file missing — falling back to gray panels: %s",
                level, cluster_path,
            )
        else:
            gene_col = ("geneKO_name" if "geneKO_name" in adata.obs.columns
                        else "perturbation")
            embed_genes = adata.obs[gene_col].astype(str).values
            if sub_idx is not None:
                embed_genes = embed_genes[sub_idx]
            cluster_map = _parse_hand_annotated_clusters(cluster_path)
            rgba, cluster_colors, ntc_mask = _build_cluster_color_map(
                embed_genes, cluster_map,
            )
            n_assigned = int(((rgba[:, 3] >= 0.85) & ~ntc_mask).sum())
            _logger.info(
                "  %s: cluster overlay enabled — %d/%d genes mapped to %d clusters "
                "(plus %d NTC), source=%s",
                level, n_assigned, len(embed_genes), len(cluster_colors),
                int(ntc_mask.sum()), cluster_path.name,
            )
            cluster_data = {
                "rgba": rgba,
                "cluster_colors": cluster_colors,
                "ntc_mask": ntc_mask,
                "embed_genes": embed_genes,
            }
            # One-off legend reference dropped at the level it applies to
            _save_cluster_color_legend(
                cluster_colors,
                plots_dir / f"{level}_cluster_color_legend.png", plt,
            )

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
            cluster_data=cluster_data,
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
            cluster_data=cluster_data,
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
            cluster_data=cluster_data,
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
            cluster_data=cluster_data,
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
    p.add_argument(
        "--cluster-annotations", default=None, type=str,
        help="Path to a hand_annotated_cluster.txt file. When provided, every "
             "sweep panel paints points colored by their cluster + a red ✕ "
             "overlay for NTC genes, plus a one-off cluster_color_legend.png "
             "is saved per level. Default: auto-resolves to "
             "hand_annotated_cluster_withcp4i.txt if --run-dir contains "
             "with_cp/with_4i/, else hand_annotated_cluster.txt.",
    )
    p.add_argument(
        "--no-cluster-overlay", action="store_true",
        help="Skip the cluster color overlay even when the default annotation "
             "file resolves (uniform-gray panels like the original sweep).",
    )

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
    # Held-constant ("canonical") values used in the secondary sweep when
    # the corresponding axis is NOT being swept. Default to CANONICAL_PHATE.
    p.add_argument("--phate-canonical-knn", type=int, default=CANONICAL_PHATE["knn"],
                   help="Override the held knn used in the PHATE secondary sweep "
                        "AND highlighted as the 'pipeline default' in the primary "
                        f"sweep (default: {CANONICAL_PHATE['knn']}).")
    p.add_argument("--phate-canonical-decay", type=float, default=CANONICAL_PHATE["decay"],
                   help="Override the held decay used in the PHATE secondary sweep "
                        "AND highlighted in the primary sweep "
                        f"(default: {CANONICAL_PHATE['decay']}).")

    return p


def _cfg_from_args(args: argparse.Namespace) -> Dict:
    # Allow callers to override the "canonical / pipeline default" PHATE
    # knn + decay so the secondary sweep (t × gamma) and the highlighted
    # default tile in the primary sweep both track the user-chosen values.
    if hasattr(args, "phate_canonical_knn"):
        CANONICAL_PHATE["knn"] = int(args.phate_canonical_knn)
    if hasattr(args, "phate_canonical_decay"):
        CANONICAL_PHATE["decay"] = float(args.phate_canonical_decay)
    # Auto-resolve cluster annotations: if the run-dir path includes
    # ``with_cp/with_4i`` use the joint flavour, else the live-cell-only one.
    if args.no_cluster_overlay:
        cluster_annotations: Optional[str] = None
    elif args.cluster_annotations is not None:
        cluster_annotations = args.cluster_annotations
    else:
        cluster_annotations = str(
            DEFAULT_CLUSTER_ANNOTATIONS_WITHCP4I
            if "with_cp/with_4i" in str(args.run_dir)
            else DEFAULT_CLUSTER_ANNOTATIONS_LIVECELL
        )
    return {
        "run_dir": args.run_dir,
        "levels": _parse_list(args.levels),
        "embedders": _parse_list(args.embedders),
        "secondary": not args.no_secondary,
        "random_seed": args.random_seed,
        "max_points": args.max_points,
        "cluster_annotations": cluster_annotations,
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
