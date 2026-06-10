"""How many geneKOs hit a distinctiveness-mAP threshold at each cell-per-gene budget?

The v4 expansion sweep cached per-bin guide-level h5ads at
``bin_guide_means/sweep_a_random_<K>.h5ad`` (head-agnostic random sampling
= the vanilla geneKO baseline). This script:

  1. For each K bin, runs ``phenotypic_distinctivness`` on the NTC-normalized
     guide-level features to get the PER-GENE distinctiveness mAP array.
  2. For each of multiple thresholds (default 0.3 / 0.5 / 0.7), counts how
     many genes cross the threshold.
  3. Plots count vs cells-per-gene (log x), one curve per threshold.

Answers: "if we used just N cells per geneKO, how many genes have their
phenotype actually resolvable in those N cells?" — a direct counterpart to
the aggregate mAP curves in the rest of the expansion sweep.

Usage::

    # Submit one SLURM task per K (9 tasks; ~5 min wall once they land)
    uv run python -m ops_model.models.attention.titration.expansion.count_genes_above_threshold --slurm

    # Replot from cached per-gene CSVs (no SLURM)
    uv run python -m ops_model.models.attention.titration.expansion.count_genes_above_threshold --replot
"""
from __future__ import annotations

import argparse
import logging
import re
import time
from pathlib import Path
from typing import List, Optional

import anndata as ad
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


DEFAULT_GUIDE_DIR = Path(
    "/hpc/projects/icd.fast.ops/models/alex_lin_attention/v4/expansion_v3_trainval/bin_guide_means"
)
DEFAULT_OUTPUT_DIR = Path(
    "/hpc/projects/icd.fast.ops/models/alex_lin_attention/v4/expansion_v3_trainval/genes_above_threshold"
)
DEFAULT_BIN_PATTERN = "sweep_a_geneko_top"
DEFAULT_THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


# ---------------------------------------------------------------------------
# Worker (top-level, picklable for SLURM)
# ---------------------------------------------------------------------------

def score_one_bin(
    guide_h5ad_path: str,
    K: int,
    output_dir: str,
    null_size: int = 10_000,
    metric: str = "distinctiveness",
) -> str:
    """Score per-entity mAP for ONE K bin; save as CSV.

    ``metric`` selects the score function:
      - ``"distinctiveness"`` → per-gene gene-vs-NTC mAP (default)
      - ``"ebi"`` → per-EBI-complex consistency mAP (requires guide → gene
        aggregation first)
    """
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    _logger = logging.getLogger(f"bin[K={K},{metric}]")
    t0 = time.time()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"per_gene_map_t{int(K)}.csv"
    if out_csv.exists():
        return f"SKIP K={K}: shard already exists"

    from ops_model.features.anndata_utils import (
        aggregate_to_level, normalize_guide_adata,
    )

    guide_a = ad.read_h5ad(guide_h5ad_path)
    n_ntc = int((guide_a.obs["perturbation"].astype(str) == "NTC").sum())
    if n_ntc < 2:
        raise RuntimeError(f"[K={K}] need >=2 NTC guides, got {n_ntc}")
    guide_a = normalize_guide_adata(guide_a, norm_method="ntc")

    if metric == "distinctiveness":
        from ops_utils.analysis.map_scores import phenotypic_distinctivness
        map_df, ratio = phenotypic_distinctivness(
            guide_a, plot_results=False, null_size=null_size,
        )
    elif metric in ("ebi", "chad"):
        gene_a = aggregate_to_level(guide_a, "gene", method="mean",
                                    preserve_batch_info=False)
        # Drop NaN rows that aggregate_to_level can leave behind
        X = gene_a.X
        nan_mask = np.isnan(X).any(axis=1)
        if hasattr(nan_mask, "A1"):
            nan_mask = nan_mask.A1
        if nan_mask.any():
            gene_a = gene_a[~nan_mask].copy()
        if metric == "ebi":
            from ops_utils.analysis.map_scores import phenotypic_consistency_ebi
            map_df, ratio = phenotypic_consistency_ebi(
                gene_a, plot_results=False, null_size=null_size,
                cache_similarity=False,
            )
        else:  # "chad"
            from ops_utils.analysis.map_scores import (
                phenotypic_consistency_manual_annotation,
            )
            import tempfile, yaml as _yaml
            # The atlas labels CHAD pages using v5_hierarchy (121 flat
            # named clusters + 13 supercategory entries that aggregate
            # sub-clusters via a 'components' field). phenotypic_consistency_
            # manual_annotation only handles flat entries with 'genes' —
            # so pre-flatten v5 to a tmp YAML containing only entries
            # with 'genes' (drops the 13 supercategories, keeps the 121
            # flat clusters that match the atlas's page names).
            CHAD_V5 = (
                "/hpc/projects/icd.fast.ops/configs/gene_clusters/"
                "chad_positive_controls_v5_hierarchy.yml"
            )
            with open(CHAD_V5) as _fh:
                _v5 = _yaml.safe_load(_fh) or {}
            _flat = {k: v for k, v in _v5.items()
                     if isinstance(v, dict) and "genes" in v}
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yml", delete=False,
            ) as _tmp:
                _yaml.safe_dump(_flat, _tmp)
                _tmp_path = _tmp.name
            try:
                map_df, ratio = phenotypic_consistency_manual_annotation(
                    gene_a, plot_results=False, null_size=null_size,
                    cache_similarity=False, annotation_path=_tmp_path,
                )
            finally:
                Path(_tmp_path).unlink(missing_ok=True)
    else:
        raise ValueError(f"unknown metric: {metric!r}")

    out = pd.DataFrame(map_df).copy()
    out["cells_per_gene"] = int(K)
    out[f"{metric}_ratio"] = float(ratio)
    out.to_csv(out_csv, index=False)
    return (
        f"OK K={K} [{metric}]: n_rows={len(out)} "
        f"map_mean={out['mean_average_precision'].mean():.3f} "
        f"({int(time.time() - t0)}s)"
    )


# ---------------------------------------------------------------------------
# Aggregator + plot
# ---------------------------------------------------------------------------

def _discover_bins(guide_dir: Path, pattern: str) -> List[tuple]:
    """Return sorted (K, h5ad_path) tuples for <pattern>_<K>.h5ad."""
    out = []
    for p in sorted(guide_dir.glob(f"{pattern}_*.h5ad")):
        m = re.match(rf"{re.escape(pattern)}_(\d+)\.h5ad", p.name)
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out, key=lambda x: x[0])


def aggregate(output_dir: Path, thresholds: List[float]) -> pd.DataFrame:
    """Count genes above each threshold per K, return long-form DataFrame."""
    rows = []
    for csv in sorted(output_dir.glob("per_gene_map_t*.csv")):
        df = pd.read_csv(csv)
        K = int(df["cells_per_gene"].iloc[0])
        # Drop NTC if present in the map_df
        if "perturbation" in df.columns:
            df = df[df["perturbation"].astype(str) != "NTC"]
        for t in thresholds:
            n = int((df["mean_average_precision"] > t).sum())
            rows.append({
                "cells_per_gene": K,
                "threshold": t,
                "n_genes_above": n,
                "n_genes_total": len(df),
            })
    return pd.DataFrame(rows).sort_values(["threshold", "cells_per_gene"])


def plot_curves(counts: pd.DataFrame, out_dir: Path, head_label: str = "geneKO",
                metric: str = "distinctiveness") -> None:
    import matplotlib
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["svg.fonttype"] = "none"
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    def fmt(n):
        if n >= 1e6: return f"{n / 1e6:.1f}M"
        if n >= 1e3: return f"{n / 1e3:.0f}K"
        return f"{int(n)}"

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    n_total = int(counts["n_genes_total"].iloc[0]) if len(counts) else 0
    thresholds = sorted(counts["threshold"].unique())
    cmap = plt.get_cmap("viridis")
    for i, t in enumerate(thresholds):
        g = counts[counts["threshold"] == t].sort_values("cells_per_gene")
        c = cmap(i / max(len(thresholds) - 1, 1))
        ax.plot(g["cells_per_gene"], g["n_genes_above"], marker="o",
                linewidth=2.0, markersize=6, color=c,
                label=f"mAP > {t:.1f}")
    if metric == "ebi":
        entity, metric_label = "EBI complexes", "EBI consistency mAP"
    elif metric == "chad":
        entity, metric_label = "CHAD complexes", "CHAD consistency mAP"
    else:
        entity, metric_label = "geneKOs", "distinctiveness mAP"

    ax.set_xscale("log")
    ax.set_xlim(left=g["cells_per_gene"].min() if len(g) else 100)
    ax.set_xlabel("cells per geneKO (log10)", fontsize=11)
    ax.set_ylabel(f"# {entity} above threshold (of {n_total} total)",
                  fontsize=11)
    # Round the y-axis ceiling up to the next round 100 (so a panel of 98 EBI
    # complexes gets a top tick at 100). Pick a step that yields ~10 ticks.
    if n_total:
        y_max = ((n_total + 99) // 100) * 100  # ceil to nearest 100
        ax.set_ylim(0, y_max)
        if y_max <= 100:
            step = 10
        elif y_max <= 400:
            step = 50
        elif y_max <= 1100:
            step = 100
        else:
            step = 200
        ax.set_yticks(list(range(0, y_max + 1, step)))
        ax.axhline(n_total, color="#cccccc", linestyle="--", linewidth=0.8, zorder=0)
    ax.set_title(f"Top-attention cell selection ({head_label} head) — "
                 f"how many {entity} are resolved\nat each cells-per-geneKO budget? "
                 f"(top-K cells/gene; {metric_label} threshold)", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt(v)))
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.legend(loc="upper left", fontsize=10, frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out_dir / f"genes_above_threshold.{ext}", dpi=150,
                    bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s/genes_above_threshold.[png/pdf/svg]", out_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--guide-dir", type=Path, default=DEFAULT_GUIDE_DIR,
                   help="Dir with sweep_a_random_<K>.h5ad guide-level features")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--bin-pattern", default=DEFAULT_BIN_PATTERN,
                   help="Glob prefix for bin h5ads (default: sweep_a_geneko_top "
                        "= geneKO attention head's top-K cells; alternatives: "
                        "sweep_a_random, sweep_a_ebi_top, sweep_a_chad_top)")
    p.add_argument("--thresholds", nargs="*", type=float,
                   default=list(DEFAULT_THRESHOLDS))
    p.add_argument("--null-size", type=int, default=10_000,
                   help="Null bootstrap sample size for the mAP score function")
    p.add_argument("--metric", default="distinctiveness",
                   choices=("distinctiveness", "ebi", "chad"),
                   help="Score function: 'distinctiveness' (per-gene vs NTC), "
                        "'ebi' (per-EBI-complex consistency, gene-level "
                        "aggregation), or 'chad' (per-CHAD-cluster consistency, "
                        "gene-level aggregation, uses chad_positive_controls_v5_"
                        "hierarchy.yml). Pair distinctiveness with geneKO-head "
                        "bins, ebi with EBI-head bins, chad with CHAD-head bins.")
    p.add_argument("--slurm", action="store_true")
    p.add_argument("--slurm-memory", default="16GB")
    p.add_argument("--slurm-time", type=int, default=60)
    p.add_argument("--slurm-cpus", type=int, default=8,
                   help="cpus_per_task. phenotypic_distinctivness uses "
                        "joblib.Parallel(n_jobs=min(get_optimal_workers, 8)) "
                        "internally, so 8 saturates its thread cap.")
    p.add_argument("--slurm-partition", default="cpu,gpu")
    p.add_argument("--replot", action="store_true",
                   help="Skip scoring; aggregate existing per-bin CSVs + replot")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    args = _build_parser().parse_args(argv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    bins = _discover_bins(args.guide_dir.resolve(), args.bin_pattern)
    logger.info("found %d %s bins: K = %s", len(bins), args.bin_pattern,
                [k for k, _ in bins])
    if not bins:
        logger.error("no %s_*.h5ad bins found in %s",
                     args.bin_pattern, args.guide_dir)
        return 1

    if not args.replot:
        if args.slurm:
            from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
            jobs = [{
                "name": f"genes_above_t_K={K}",
                "func": score_one_bin,
                "kwargs": {
                    "guide_h5ad_path": str(path),
                    "K": int(K),
                    "output_dir": str(output_dir),
                    "null_size": int(args.null_size),
                    "metric": args.metric,
                },
                "metadata": {"K": int(K)},
            } for K, path in bins]
            slurm_params = {
                "timeout_min": args.slurm_time,
                "mem": args.slurm_memory,
                "cpus_per_task": args.slurm_cpus,
                "slurm_partition": args.slurm_partition,
            }
            submit_parallel_jobs(
                jobs_to_submit=jobs,
                experiment="genes_above_threshold",
                slurm_params=slurm_params,
                log_dir="slurm_step_logs/genes_above_threshold",
                manifest_prefix="genes_above_threshold",
                wait_for_completion=True,
            )
        else:
            for K, path in bins:
                logger.info(score_one_bin(
                    str(path), int(K), str(output_dir),
                    null_size=args.null_size, metric=args.metric,
                ))

    counts = aggregate(output_dir, args.thresholds)
    if counts.empty:
        logger.warning("no per-bin CSVs in %s — nothing to plot", output_dir)
        return 1
    counts_csv = output_dir / "genes_above_threshold.csv"
    counts.to_csv(counts_csv, index=False)
    logger.info("wrote %s (%d rows)", counts_csv, len(counts))
    # Derive head label from the bin pattern, e.g. sweep_a_ebi_top → "EBI",
    # sweep_a_geneko_top → "geneKO", sweep_a_chad_top → "CHAD".
    head_label = "geneKO"
    parts = args.bin_pattern.split("_")
    for tok in parts:
        if tok.lower() in {"ebi", "chad", "geneko"}:
            head_label = {"ebi": "EBI", "chad": "CHAD", "geneko": "geneKO"}[tok.lower()]
            break
    plot_curves(counts, output_dir, head_label=head_label, metric=args.metric)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
