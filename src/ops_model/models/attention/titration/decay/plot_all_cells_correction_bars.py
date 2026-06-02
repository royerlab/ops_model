"""Bar plot: 4 mAP metrics — all-cells baselines, BEFORE vs AFTER ISS-sidecar
correction.

Compares the all-cells PCA baseline numbers from:

  OLD (stale labels):
    .../paper_v1/phase_only/fixed_80%/cosine/{gene,guide}_pca_optimized.h5ad
  NEW (sidecar-corrected labels):
    .../paper_v1/corrected/phase_only/fixed_80%/cosine/{gene,guide}_pca_optimized.h5ad

Computes all 4 metrics directly via the canonical functions in
``ops_utils.analysis.map_scores`` (no reinvention):

  - EBI consistency        — phenotypic_consistency_ebi              (gene-level)
  - CHAD consistency       — phenotypic_consistency_manual_annotation (gene-level)
  - Distinctiveness        — phenotypic_distinctivness                (guide-level)
  - Phenotypic activity    — phenotypic_activity_assesment            (guide-level)

Cached OLD JSONs at ``models/alex_lin_attention/v3/attention_v3/cdino/`` are
reused where they exist (EBI/CHAD/distinct); activity isn't cached, so we
compute it for OLD. All 4 NEW values are computed from the corrected h5ads.

Writes the plot + a side-by-side CSV alongside the corrected outputs.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional

import anndata as ad
import numpy as np
import pandas as pd

from ops_utils.analysis.map_scores import (
    phenotypic_activity_assesment,
    phenotypic_consistency_ebi,
    phenotypic_consistency_manual_annotation,
    phenotypic_distinctivness,
)

logger = logging.getLogger(__name__)

ROOT = Path(
    "/hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized_v0.3/"
    "cell_dino/zscore_per_exp/paper_v1"
)
OLD_DIR = ROOT / "phase_only/fixed_80%/cosine"
NEW_DIR = ROOT / "corrected/phase_only/fixed_80%/cosine"
CACHE_DIR = Path(
    "/hpc/projects/icd.fast.ops/models/alex_lin_attention/v3/attention_v3/cdino"
)
CHAD_YAML = Path(
    "/hpc/projects/icd.ops/configs/gene_clusters/chad_positive_controls_v4.yml"
)
NULL_SIZE = 10_000


def _load_cached_baseline(p: Path, val_key: str) -> Optional[float]:
    if not p.exists():
        return None
    try:
        return float(json.loads(p.read_text())[val_key])
    except Exception:
        return None


def _compute_all_four(label: str, gene_h5ad: Path, guide_h5ad: Path) -> Dict[str, float]:
    """Compute all 4 mAPs on (gene_h5ad, guide_h5ad). Returns dict keyed by metric."""
    logger.info(f"[{label}] EBI consistency on {gene_h5ad.name}…")
    gene = ad.read_h5ad(gene_h5ad)
    ebi_df, _ = phenotypic_consistency_ebi(
        gene, plot_results=False, null_size=NULL_SIZE, cache_similarity=True,
    )
    ebi_mAP = float(ebi_df["mean_average_precision"].mean())

    logger.info(f"[{label}] CHAD consistency on {gene_h5ad.name}…")
    chad_df, _ = phenotypic_consistency_manual_annotation(
        gene, plot_results=False, null_size=NULL_SIZE,
        cache_similarity=True, annotation_path=str(CHAD_YAML),
    )
    chad_mAP = float(chad_df["mean_average_precision"].mean())

    logger.info(f"[{label}] Phenotypic activity on {guide_h5ad.name}…")
    guide = ad.read_h5ad(guide_h5ad)
    act_df, _ = phenotypic_activity_assesment(
        guide, plot_results=False, null_size=NULL_SIZE,
    )
    act_mAP = float(act_df["mean_average_precision"].mean())

    logger.info(f"[{label}] Distinctiveness on {guide_h5ad.name}…")
    dist_df, _ = phenotypic_distinctivness(
        guide, activity_map=act_df, plot_results=False, null_size=NULL_SIZE,
        active_only=False,
    )
    dist_mAP = float(dist_df["mean_average_precision"].mean())

    return {"ebi": ebi_mAP, "chad": chad_mAP,
             "activity": act_mAP, "distinctiveness": dist_mAP}


def main():
    logging.basicConfig(level=logging.INFO,
                         format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=NEW_DIR,
                    help="Where to write the comparison plot + CSV "
                         f"(default: {NEW_DIR})")
    args = ap.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- OLD: reuse cached JSONs where they exist (EBI/CHAD/distinct) ---
    old_vals: Dict[str, float] = {}
    cached_paths = {
        "ebi":             (CACHE_DIR / "all_cells_pca_baseline.json",            "mean_map"),
        "chad":            (CACHE_DIR / "all_cells_pca_chad_baseline.json",       "mean_map"),
        "distinctiveness": (CACHE_DIR / "all_cells_pca_distinctiveness_baseline.json",
                            "mean_map_dist"),
    }
    for metric, (p, key) in cached_paths.items():
        v = _load_cached_baseline(p, key)
        if v is not None:
            logger.info(f"[OLD] {metric}: {v:.4f} (cached at {p.name})")
            old_vals[metric] = v

    # Activity has no cached OLD baseline → compute from the old guide h5ad.
    if "activity" not in old_vals:
        guide_p = OLD_DIR / "guide_pca_optimized.h5ad"
        logger.info(f"[OLD] activity: computing from {guide_p}…")
        guide = ad.read_h5ad(guide_p)
        act_df, _ = phenotypic_activity_assesment(
            guide, plot_results=False, null_size=NULL_SIZE,
        )
        old_vals["activity"] = float(act_df["mean_average_precision"].mean())
        logger.info(f"[OLD] activity: {old_vals['activity']:.4f}")

    # Any cached metric missing? Compute from old h5ads.
    if any(m not in old_vals for m in ("ebi", "chad", "distinctiveness")):
        full = _compute_all_four(
            "OLD",
            OLD_DIR / "gene_pca_optimized.h5ad",
            OLD_DIR / "guide_pca_optimized.h5ad",
        )
        for m, v in full.items():
            old_vals.setdefault(m, v)

    # --- NEW (corrected): compute all 4 from corrected h5ads ---
    new_vals = _compute_all_four(
        "NEW",
        NEW_DIR / "gene_pca_optimized.h5ad",
        NEW_DIR / "guide_pca_optimized.h5ad",
    )

    # --- Side-by-side CSV ---
    rows = []
    for m in ("ebi", "chad", "distinctiveness", "activity"):
        old, new = old_vals[m], new_vals[m]
        rows.append({"metric": m, "old": old, "new": new,
                      "delta": new - old,
                      "rel_change_pct": 100 * (new - old) / max(old, 1e-9)})
    df = pd.DataFrame(rows)
    csv_path = out_dir / "all_cells_4mAP_correction_deltas.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Wrote {csv_path}")
    logger.info("\n" + df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # --- Total cell-count (per gene_pca_optimized.h5ad's n_cells sum) ---
    def _n_cells_aggregated(gene_h5ad: Path) -> int:
        g = ad.read_h5ad(gene_h5ad, backed="r")
        return int(g.obs["n_cells"].sum()) if "n_cells" in g.obs.columns else 0
    old_n = _n_cells_aggregated(OLD_DIR / "gene_pca_optimized.h5ad")
    new_n = _n_cells_aggregated(NEW_DIR / "gene_pca_optimized.h5ad")
    logger.info(f"Total cells aggregated:  OLD={old_n:,}  NEW={new_n:,}  delta={new_n - old_n:+,}")

    # --- Bar plot ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = ["ebi", "chad", "distinctiveness", "activity"]
    labels = {"ebi": "EBI\nconsistency",
                "chad": "CHAD\nconsistency",
                "distinctiveness": "Distinctiveness",
                "activity": "Phenotypic\nactivity"}
    metric_old = [old_vals[m] for m in metrics]
    metric_new = [new_vals[m] for m in metrics]

    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    x = np.arange(len(metrics))
    w = 0.36
    bars_old = ax.bar(x - w / 2, metric_old, width=w,
                       color="#7A7A7A", edgecolor="black", linewidth=0.7,
                       label=f"Before correction (stale ISS labels)\n"
                             f"n = {old_n:,} cells aggregated")
    bars_new = ax.bar(x + w / 2, metric_new, width=w,
                       color="#1F9B4A", edgecolor="black", linewidth=0.7,
                       label=f"After correction (current ISS labels)\n"
                             f"n = {new_n:,} cells aggregated"
                             f"  ({new_n - old_n:+,})")

    # value annotations on top of each bar
    for b in bars_old:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                f"{b.get_height():.3f}", ha="center", va="bottom",
                fontsize=9.5)
    for b in bars_new:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                f"{b.get_height():.3f}", ha="center", va="bottom",
                fontsize=9.5)
    # delta arrow + Δ above each metric group
    for i, m in enumerate(metrics):
        delta = new_vals[m] - old_vals[m]
        top = max(metric_old[i], metric_new[i]) + 0.045
        color = "#1F9B4A" if delta >= 0 else "#C0392B"
        ax.text(x[i], top, f"Δ {delta:+.3f}", ha="center", va="bottom",
                 fontsize=10, fontweight="bold", color=color)

    ax.set_xticks(x)
    ax.set_xticklabels([labels[m] for m in metrics], fontsize=12)
    ax.set_ylabel("mean mAP (PCA 97-d, NTC-normed, all cells)", fontsize=12)
    ax.set_title("All-cells mAP per metric — before vs after ISS-sidecar correction\n"
                  "paper-v1 / cell_dino / phase_only / fixed_80% / cosine",
                  fontsize=12.5, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
               borderaxespad=0.0, framealpha=0.9, fontsize=10)
    ymax = max(max(metric_old), max(metric_new)) + 0.10
    ax.set_ylim(0, ymax)

    fig.tight_layout()
    for ext in ("pdf", "png", "svg"):
        p = out_dir / f"all_cells_4mAP_correction_bars.{ext}"
        fig.savefig(p, bbox_inches="tight",
                     dpi=200 if ext == "png" else None)
        logger.info(f"Wrote {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
