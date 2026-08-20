"""Gene-level UMAP + mAP scoring for top-attention consolidated h5ads.

Thin wrapper that mirrors `pca_optimization.aggregate_channels`'s scoring flow:
    cells -> guide (mean) -> NTC z-score -> gene (mean) -> activity / dist /
    CORUM / CHAD scoring + UMAP at gene level.

Requires the input h5ad to carry `sgRNA` in obs and NTC rows (i.e. produced by
consolidate_top_attention_cells.py with `--add-ntc`). Skips NTC normalization
when no NTC rows are present and falls back to gene-level CORUM/CHAD only.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np

from ops_model.features.anndata_utils import (
    aggregate_to_level,
    normalize_guide_adata,
)
from ops_utils.analysis.embedding_plots import clean_X_for_embedding
from ops_utils.analysis.map_scores import (
    compute_auc_score,
    phenotypic_activity_assesment,
    phenotypic_consistency_corum,
    phenotypic_consistency_manual_annotation,
    phenotypic_distinctivness,
)

CHAD_ANNOTATION_PATH = Path(
    "/hpc/projects/icd.fast.ops/configs/gene_clusters/chad_positive_controls_v5_hierarchy.yml"
)


def run_embed_and_score(
    input_h5ad: Path,
    output_dir: Path,
    random_seed: int,
    null_size: int,
    distance: str,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = output_dir / "metrics"
    plots_dir = output_dir / "plots"
    metrics_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)

    print(f"\n{'='*72}\nTop-attention UMAP + mAP\n{'='*72}")
    print(f"Input  : {input_h5ad}")
    print(f"Output : {output_dir}\n")

    t0 = time.time()
    adata = ad.read_h5ad(input_h5ad)
    print(f"Loaded: {adata.n_obs:,} cells x {adata.n_vars:,} features "
          f"({time.time()-t0:.1f}s)")

    # aggregate_to_level expects 'perturbation' (gene-level) and 'sgRNA' (guide).
    if "perturbation" not in adata.obs.columns:
        adata.obs["perturbation"] = adata.obs["gene"].astype(str)
    has_sgrna = "sgRNA" in adata.obs.columns and (adata.obs["sgRNA"].astype(str) != "").any()
    has_ntc = (adata.obs["perturbation"].astype(str) == "NTC").any()
    print(f"  sgRNA present: {has_sgrna}, NTC rows present: {has_ntc}")

    adata.X = clean_X_for_embedding(adata)

    # --- Cells -> guide (if sgRNA available) ---
    adata_guide = None
    if has_sgrna:
        # Drop cells with empty sgRNA so groupby doesn't lump them all together.
        keep = adata.obs["sgRNA"].astype(str) != ""
        if (~keep).any():
            print(f"  dropping {(~keep).sum():,} cells with no sgRNA")
            adata = adata[keep].copy()
        adata_guide = aggregate_to_level(
            adata, level="guide", method="mean",
            preserve_batch_info=False, subsample_controls=False,
        )
        adata_guide.X = clean_X_for_embedding(adata_guide)
        print(f"Guide-level: {adata_guide.n_obs:,} guides x {adata_guide.n_vars:,} features")

        # NTC z-score (skips silently if no NTC perturbation in obs).
        if has_ntc:
            print("NTC z-scoring guide-level features...")
            adata_guide = normalize_guide_adata(adata_guide, norm_method="ntc")
            adata_guide.X = clean_X_for_embedding(adata_guide)

    # --- Gene-level (from guide if available, else from cells) ---
    src = adata_guide if adata_guide is not None else adata
    adata_gene = aggregate_to_level(
        src, level="gene", method="mean",
        preserve_batch_info=False, subsample_controls=False,
    )
    adata_gene.obs_names = adata_gene.obs["perturbation"].astype(str).values
    adata_gene.X = clean_X_for_embedding(adata_gene)
    print(f"Gene-level: {adata_gene.n_obs:,} genes x {adata_gene.n_vars:,} features")

    # --- UMAP at gene level (matches pca_optimization params) ---
    print("\nRunning UMAP...")
    t0 = time.time()
    from umap import UMAP
    nn = max(2, min(10, adata_gene.n_obs - 1))
    coords = UMAP(
        n_components=2, n_neighbors=nn, min_dist=0.25, random_state=random_seed,
    ).fit_transform(np.asarray(adata_gene.X, dtype=np.float32))
    adata_gene.obsm["X_umap"] = coords.astype(np.float32)
    print(f"  done in {time.time()-t0:.1f}s")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    is_ntc = (adata_gene.obs["perturbation"].astype(str).str.upper().str.startswith("NTC"))
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.scatter(coords[~is_ntc, 0], coords[~is_ntc, 1], s=14, alpha=0.6,
               c="steelblue", edgecolors="none", label="KO genes")
    if is_ntc.any():
        ax.scatter(coords[is_ntc, 0], coords[is_ntc, 1], s=80, alpha=0.95,
                   c="#E03030", marker="D", edgecolors="black", linewidths=0.5,
                   label=f"NTC ({int(is_ntc.sum())})", zorder=5)
    ax.legend(loc="best")
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_title(f"Top-attention {input_h5ad.stem} — gene UMAP "
                 f"(n={adata_gene.n_obs:,}, {adata_gene.n_vars:,} feats)")
    fig.tight_layout()
    fig.savefig(plots_dir / "umap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    if adata_guide is not None:
        adata_guide.write_h5ad(output_dir / "guide_aggregated.h5ad")
    adata_gene.write_h5ad(output_dir / "gene_aggregated.h5ad")
    print(f"Saved gene_aggregated.h5ad" + (" + guide_aggregated.h5ad" if adata_guide is not None else ""))

    # --- Scoring ---
    summary = [
        f"# Top-attention mAP scoring summary",
        f"",
        f"Input:  `{input_h5ad}`",
        f"Output: `{output_dir}`",
        f"",
        f"- Cells: **{adata.n_obs:,}**, features: **{adata.n_vars:,}**",
        f"- Guides: **{adata_guide.n_obs if adata_guide is not None else 0:,}** "
        f"({'NTC z-scored' if has_ntc else 'no NTC; raw'})",
        f"- Genes: **{adata_gene.n_obs:,}**",
        f"- distance: {distance}, null_size: {null_size:,}",
        f"",
    ]

    activity_map = None
    if adata_guide is not None and has_ntc:
        print("\nActivity scoring (vs NTC null)...")
        t0 = time.time()
        try:
            activity_map, active_ratio = phenotypic_activity_assesment(
                adata_guide, plot_results=False, null_size=null_size, distance=distance,
            )
            activity_map.to_csv(metrics_dir / "phenotypic_activity.csv", index=False)
            auc = compute_auc_score(activity_map)
            print(f"  Activity: {active_ratio:.1%} active, AUC={auc:.4f} ({time.time()-t0:.1f}s)")
            summary.append(f"## Activity: **{active_ratio:.1%}** active "
                           f"(AUC={auc:.4f})")
        except Exception as exc:
            print(f"  Activity failed: {exc!r}")
            summary.append(f"## Activity: FAILED — {exc!r}")
        summary.append("")

    if adata_guide is not None:
        print("\nDistinctiveness (guide-level)...")
        t0 = time.time()
        try:
            dist_df, dist_ratio = phenotypic_distinctivness(
                adata_guide, activity_map=activity_map,
                plot_results=False, null_size=null_size, distance=distance,
                active_only=False,
            )
            dist_df.to_csv(metrics_dir / "phenotypic_distinctiveness.csv", index=False)
            print(f"  Distinctiveness: {dist_ratio:.1%} ({time.time()-t0:.1f}s)")
            summary.append(f"## Distinctiveness: **{dist_ratio:.1%}** "
                           f"({len(dist_df):,} perturbations)")
        except Exception as exc:
            print(f"  Distinctiveness failed: {exc!r}")
            summary.append(f"## Distinctiveness: FAILED — {exc!r}")
        summary.append("")

    print("\nCORUM consistency (gene-level)...")
    t0 = time.time()
    try:
        corum_df, corum_ratio = phenotypic_consistency_corum(
            adata_gene, plot_results=False, null_size=null_size,
            cache_similarity=True, distance=distance,
        )
        corum_df.to_csv(metrics_dir / "phenotypic_consistency_corum.csv", index=False)
        print(f"  CORUM: {corum_ratio:.1%} ({time.time()-t0:.1f}s)")
        summary.append(f"## CORUM consistency: **{corum_ratio:.1%}** "
                       f"({len(corum_df):,} complexes)")
    except Exception as exc:
        print(f"  CORUM failed: {exc!r}")
        summary.append(f"## CORUM consistency: FAILED — {exc!r}")
    summary.append("")

    print("\nCHAD consistency (gene-level)...")
    t0 = time.time()
    try:
        chad_df, chad_ratio = phenotypic_consistency_manual_annotation(
            adata_gene, plot_results=False, null_size=null_size,
            cache_similarity=True, distance=distance,
            annotation_path=str(CHAD_ANNOTATION_PATH),
        )
        chad_df.to_csv(metrics_dir / "phenotypic_consistency_chad.csv", index=False)
        print(f"  CHAD: {chad_ratio:.1%} ({time.time()-t0:.1f}s)")
        summary.append(f"## CHAD consistency: **{chad_ratio:.1%}** "
                       f"({len(chad_df):,} clusters)")
    except Exception as exc:
        print(f"  CHAD failed: {exc!r}")
        summary.append(f"## CHAD consistency: FAILED — {exc!r}")
    summary.append("")

    (output_dir / "summary.md").write_text("\n".join(summary))
    print(f"\nSummary -> {output_dir / 'summary.md'}\nDone.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--null-size", type=int, default=100_000)
    p.add_argument("--distance", default="cosine", choices=["cosine", "euclidean"])

    p.add_argument("--local", action="store_true")
    p.add_argument("--mem", default="200G")
    p.add_argument("--cpus", type=int, default=8)
    p.add_argument("--timeout-min", type=int, default=240)
    p.add_argument("--partition", default="cpu")
    p.add_argument("--no-wait", action="store_true")
    args = p.parse_args()

    if args.local:
        return run_embed_and_score(
            input_h5ad=args.input,
            output_dir=args.output_dir,
            random_seed=args.random_seed,
            null_size=args.null_size,
            distance=args.distance,
        )

    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    name = f"embed_score_{args.input.stem}"
    jobs = [{
        "name": name,
        "func": run_embed_and_score,
        "kwargs": {
            "input_h5ad": args.input,
            "output_dir": args.output_dir,
            "random_seed": args.random_seed,
            "null_size": args.null_size,
            "distance": args.distance,
        },
        "metadata": {"experiment": name},
    }]
    slurm_params = {
        "timeout_min": args.timeout_min,
        "mem": args.mem,
        "cpus_per_task": args.cpus,
        "slurm_partition": args.partition,
    }
    result = submit_parallel_jobs(
        jobs_to_submit=jobs,
        experiment=name,
        slurm_params=slurm_params,
        log_dir=name,
        manifest_prefix=name,
        wait_for_completion=not args.no_wait,
    )
    if args.no_wait:
        return 0 if result.get("success") else 1
    return 0 if result.get("all_completed") else 1


if __name__ == "__main__":
    sys.exit(main())
