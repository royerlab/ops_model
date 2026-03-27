"""CLI entry point for OPS embedding evaluation.

Run from the monorepo root with ``uv run``.

Examples
--------
Guide-level only::

    uv run run_eval \\
        --guide_embedding /path/to/guide_embeddings.h5ad

Gene-level only::

    uv run run_eval \\
        --gene_embedding /path/to/gene_embeddings.h5ad

Both levels, merged into one output row::

    uv run run_eval \\
        --guide_embedding /path/to/guide_embeddings.h5ad \\
        --gene_embedding /path/to/gene_embeddings.h5ad

Save CSVs and plots alongside the metrics CSV::

    uv run run_eval \\
        --guide_embedding /path/to/guide_embeddings.h5ad \\
        --gene_embedding /path/to/gene_embeddings.h5ad \\
        --save-outputs

.. note::
    When both embeddings are provided, the ``activity_map`` from the guide-level
    evaluation is passed to the gene-level evaluator to filter by active genes.
    When only ``--gene_embedding`` is provided, a warning is printed and all
    genes are assumed active.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import anndata as ad
import pandas as pd

from ops_model.eval.evaluate_guide import evaluate_guide_level
from ops_model.eval.evaluate_gene import evaluate_gene_level
from ops_model.eval.eval_io import save_guide_eval, save_gene_eval


def _default_output_path(embedding_path: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parent = Path(embedding_path).parent
    return str(parent / f"{timestamp}_eval.csv")


def main() -> None:
    """Run evaluation on one or both embedding levels and write a summary CSV."""
    parser = argparse.ArgumentParser(
        description="Evaluate OPS embedding quality at guide and/or gene level."
    )
    parser.add_argument(
        "--guide_embedding",
        type=str,
        default=None,
        help="Path to guide-level h5ad file.",
    )
    parser.add_argument(
        "--gene_embedding",
        type=str,
        default=None,
        help="Path to gene-level h5ad file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path. Defaults to <embedding_dir>/<timestamp>_eval.csv.",
    )
    parser.add_argument(
        "--save-outputs",
        action="store_true",
        help="Save per-metric CSVs and plots alongside the summary CSV.",
    )
    parser.add_argument(
        "--distinctiveness-active-only",
        action="store_true",
        help="Compute distinctiveness over active genes only (default: all genes).",
    )
    args = parser.parse_args()

    if not args.guide_embedding and not args.gene_embedding:
        parser.error(
            "At least one of --guide_embedding or --gene_embedding must be provided."
        )

    output_path = args.output or _default_output_path(
        args.guide_embedding or args.gene_embedding  # type: ignore[arg-type]
    )
    output_dir = Path(output_path).parent

    results: dict = {}
    activity_map = None

    if args.guide_embedding:
        adata_guide = ad.read_h5ad(args.guide_embedding)
        guide_metrics, activity_map, distinctiveness_map = evaluate_guide_level(
            adata_guide,
            distinctiveness_active_only=args.distinctiveness_active_only,
        )
        del adata_guide
        results.update(guide_metrics)
        results["guide_embedding_path"] = args.guide_embedding
        if args.save_outputs:
            save_guide_eval(activity_map, distinctiveness_map, guide_metrics, output_dir)

    if args.gene_embedding:
        adata_gene = ad.read_h5ad(args.gene_embedding)
        gene_metrics, consistency_corum_map, consistency_manual_map = evaluate_gene_level(
            adata_gene, activity_map=activity_map
        )
        results.update(gene_metrics)
        results["gene_embedding_path"] = args.gene_embedding
        if args.save_outputs:
            save_gene_eval(consistency_corum_map, consistency_manual_map, gene_metrics, output_dir)

    pd.DataFrame([results]).to_csv(output_path, index=False)
    print(f"Evaluation results written to {output_path}")


if __name__ == "__main__":
    main()
