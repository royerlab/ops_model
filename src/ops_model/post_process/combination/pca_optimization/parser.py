"""Argparse parser for the pca_optimization CLI.

Extracted from ``pca_optimization.py`` to keep the (large) argparse
configuration out of the main module. The single entry point is
``_build_parser()`` — call it, then ``main()`` reads ``args`` and
dispatches.

The only module-level dependency on ``pca_optimization`` is
lazily inside ``_build_parser`` so the two modules can re-import each
other at module load time without a circular dependency.
"""

from __future__ import annotations

import argparse


def _build_parser():
    """Build argparse parser for the PCA optimization CLI."""
    # Lazy import to avoid a circular dependency: pca_optimization.py
    # re-imports _build_parser from this module at module load time.

    parser = argparse.ArgumentParser(
        description="Per-signal pooled PCA optimization for organelle attribution"
    )
    parser.add_argument(
        "--signal",
        dest="signal_specs",
        action="append",
        metavar="NAME=PATHS",
        required=True,
        help="A named signal group to build, as NAME=path1[,path2,...] pointing "
             "at per-signal h5ads (obs carrying sgRNA / perturbation / "
             "experiment, X = the embedding). Repeat the flag for several "
             "groups; multiple paths under one name are pooled. NAME becomes an "
             "output directory component, so it must be unique and contain no "
             "path separator.",
    )
    parser.add_argument(
        "--gene-panel",
        type=str,
        required=True,
        help="Path to the annotated gene-panel CSV used to add pathway/family "
             "columns to the aggregated gene-level output.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML config whose keys are the CLI argument names as "
        "snake_case dest names (e.g. cell_dino, phase_only, output_dir, "
        "experiments, fixed_threshold, slurm). Config values populate the "
        "defaults; any flag passed explicitly on the command line overrides the "
        "config. See pca_optimization/example_config.yml.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        required=True,
        help="Root output directory (feature-type and variant subdirs are added "
             "automatically beneath it).",
    )
    parser.add_argument(
        "--norm-method",
        type=str,
        default="ntc",
        choices=["ntc", "global"],
        help="Normalization method (default: ntc)",
    )
    parser.add_argument(
        "--distance",
        type=str,
        default="cosine",
        choices=["cosine", "euclidean"],
        help="Distance metric for mAP scoring (default: cosine)",
    )
    parser.add_argument(
        "--fixed-threshold",
        type=float,
        default=0.80,
        help="Skip the variance sweep and use a single fixed PCA threshold (default: 0.80). "
        "Pass --fixed-threshold 0 to disable and run the full consensus sweep instead.",
    )
    parser.add_argument(
        "--slurm",
        action="store_true",
        help="Submit Phase 1 signal-group SLURM jobs + Phase 2 aggregation job",
    )
    parser.add_argument(
        "--slurm-memory",
        type=str,
        default="200GB",
        help="SLURM memory per signal-group job (default: 200GB)",
    )
    parser.add_argument(
        "--slurm-time",
        type=int,
        default=10,
        help="SLURM time limit per signal-group job in minutes (default: 10)",
    )
    parser.add_argument(
        "--slurm-cpus",
        type=int,
        default=16,
        help="SLURM CPUs per signal-group job (default: 16)",
    )
    parser.add_argument(
        "--slurm-partition",
        type=str,
        default="cpu,gpu",
        help="SLURM partition (default: cpu,gpu)",
    )
    parser.add_argument(
        "-y", "--yes", action="store_true", help="Skip confirmation prompt"
    )
    parser.add_argument(
        "--slurm-agg-memory",
        type=str,
        default="600GB",
        help="SLURM memory for aggregation job (default: 600GB)",
    )
    parser.add_argument(
        "--slurm-agg-time",
        type=int,
        default=180,
        help="SLURM time limit for aggregation job in minutes (default: 180). "
             "Phase 2 = concat + score + 2nd-pass PCA + Leiden + GO enrichment "
             "across ~12 resolutions; the GO enrichment loop is the long pole "
             "(~5-10 min per resolution at OP/CP scale).",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing per_signal/ directory before Phase 1 to ensure a fresh run.",
    )
    parser.add_argument(
        "--umap-n-neighbors",
        type=int,
        default=None,
        help="Override gene-level UMAP n_neighbors when used with "
        "the --sweep-seed cache (forces a refit at this value; the cache is "
        "--sweep-seed cache, which is keyed on default n_neighbors=15).",
    )
    parser.add_argument(
        "--umap-min-dist",
        type=float,
        default=None,
        help="Override gene-level UMAP min_dist "
        "(forces a refit; default UMAP value is 0.1).",
    )
    parser.add_argument(
        "--umap-type",
        type=str,
        default="max",
        choices=["max", "gav"],
        help="UMAP recipe to use for all UMAP fits in this pipeline. "
        "'max' (default): scanpy sc.pp.neighbors(n_neighbors=8, use_rep='X_pca') + "
        "sc.tl.umap(min_dist=0.25, alpha=1.0, gamma=1.5, maxiter=2000, "
        "init_pos=X_pca[:, :2]) — PCA-anchored, biology-aware layout. "
        "'gav' (legacy): umap-learn UMAP(n_neighbors=min(10, n-1), min_dist=0.25) "
        "fit directly on the feature matrix with default spectral init. "
        "The chosen recipe is recorded in adata.uns['umap']['params']['umap_type'].",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for UMAP / PHATE / NTC-split, threaded through "
             "aggregate_channels, apply_second_pass_pca, and the SLURM helpers. "
             "Same seed → bit-identical embeddings. When unset, resolves "
             "per --umap-type: 1 for 'max' (Max's recipe), 42 for 'gav' (legacy).",
    )
    parser.add_argument(
        "--second-pca",
        dest="second_pca",
        action="store_true",
        default=True,
        help="Also run a second-pass PCA after the main pipeline (full run or --aggregate-only) "
        "finishes. The 2nd-pass reads <output_dir>/guide_pca_optimized.h5ad, fits PCA on the "
        "horizontally concatenated NTC-normalized guide features, retains top --second-pca-threshold "
        "of variance, re-aggregates to gene level, re-scores all metrics, and writes results to "
        "<output_dir>/<--second-pca-subdir>/. In SLURM mode this is bundled into the same SLURM job "
        "as the aggregation step. Default: True.",
    )
    parser.add_argument(
        "--no-second-pca",
        dest="second_pca",
        action="store_false",
        help="Disable the chained 2nd-pass PCA.",
    )
    parser.add_argument(
        "--second-pca-only",
        action="store_true",
        help="Only run the 2nd-pass PCA on an existing aggregate output (skips Phase 1 + Phase 2). "
        "Use this when you've already run the main pipeline and just want to (re-)compute the "
        "2nd-pass.",
    )
    parser.add_argument(
        "--second-pca-threshold",
        type=float,
        default=0.0,
        help="Cumulative variance threshold for the second-pass PCA. "
        "Default: 0 → run the sweep and pick the consensus peak (max of "
        "normalized activity+distinctiveness+CHAD across thresholds), "
        "writing to second_pca_consensus/. "
        "Pass a positive value (e.g. 0.80) to use a fixed threshold, "
        "writing to second_pca_<pct>/.",
    )
    parser.add_argument(
        "--second-pca-subdir",
        type=str,
        default=None,
        help="Subdir name under <output_dir> for second-pass PCA outputs "
        "(default: second_pca_<threshold>).",
    )
    parser.add_argument(
        "--second-pca-no-sweep",
        action="store_true",
        help="Skip the variance-threshold sweep in --second-pca mode (faster, but no "
        "sweep CSV/plot to compare against the chosen threshold).",
    )
    parser.add_argument(
        "--second-pca-sweep-thresholds",
        type=str,
        default=None,
        help="Comma-separated variance thresholds for the second-pass sweep "
        "(default: same as DEFAULT_SWEEP_THRESHOLDS).",
    )
    parser.add_argument(
        "--second-pca-consensus-metrics",
        type=str,
        default=None,
        help="Comma-separated subset of {activity, distinctiveness, ebi, chad} "
             "to use for the 2nd-pass PCA threshold consensus pick. Default: "
             "activity,distinctiveness,ebi (writes to canonical "
             "second_pca_consensus/). Any non-default subset writes to "
             "second_pca_consensus_<TAG>/ where TAG is a "
             "deterministic ABBREV_ABBREV concat (e.g. activity,distinctiveness,"
             "chad → _ACT_DIST_CHAD ; activity alone → _ACT ; "
             "distinctiveness,ebi → _DIST_EBI).",
    )
    parser.add_argument(
        "--sweep-metric",
        type=str,
        default="mean_map",
        choices=["ratio", "mean_map"],
        help="Per-threshold scoring mode for the 2nd-pass PCA sweep. "
             "'mean_map' (default): continuous mean of per-item mAP — more "
             "stable near close threshold ties, lands in a sibling "
             "second_pca_consensus_MEANMAP/ subdir so it doesn't clobber any "
             "existing ratio-based output. 'ratio': fraction-significant "
             "counts (coarser), lands in the canonical second_pca_consensus/.",
    )
    parser.add_argument(
        "--target-cells",
        dest="target_cells",
        type=int,
        default=None,
        help="Force every signal group to this exact cell count under --downsampled, "
             "overriding the auto-computed `max(min_signal_count, 750k)` target. "
             "Useful for cross-run matching (e.g. CP and live-cell at the same N).",
    )
    parser.add_argument(
        "--downsample-per-guide",
        action="store_true",
        help="Cap cells per sgRNA at --cells-per-guide instead of downsampling "
             "to a total budget. Sampling only — it no longer changes the "
             "output path.",
    )
    parser.add_argument(
        "--cells-per-guide",
        type=int,
        default=250,
        help="Per-sgRNA cell cap used with --downsample-per-guide (default: 250).",
    )
    parser.add_argument(
        "--phase-memory",
        type=str,
        default="600GB",
        help="SLURM memory for Phase signal job (default: 600GB). Phase ~50M cells needs more.",
    )
    # Feature-mode flags — exactly one must be passed (no implicit default).
    parser.add_argument(
        "--dino",
        action="store_true",
        help="Use legacy DINO embeddings (feature_dir=dino_features). "
             "Output → dino/ subdir.",
    )
    parser.add_argument(
        "--cell-profiler",
        action="store_true",
        help="Use CellProfiler morphological features. Output → cellprofiler/ subdir.",
    )
    parser.add_argument(
        "--cell-dino",
        action="store_true",
        help="Use cell-level DINO features (feature_dir=cell_dino_features). "
             "Output → cell_dino/ subdir.",
    )
    parser.add_argument(
        "--dynaclr",
        action="store_true",
        help="Use DynaCLR features (feature_dir=dynaclr_features). "
             "Output → dynaclr/ subdir.",
    )
    parser.add_argument(
        "--subcell",
        action="store_true",
        help="Use SubCell features (feature_dir=subcell_features). "
             "Output → subcell/ subdir.",
    )
    parser.add_argument(
        "--exclude-dud-guides", dest="exclude_dud_guides",
        action="store_true", default=True,
        help="Filter out known dud sgRNAs (default: True). See DUD_GUIDES constant.",
    )
    parser.add_argument(
        "--no-exclude-dud-guides", dest="exclude_dud_guides",
        action="store_false",
        help="Keep dud sgRNAs in the cell pool.",
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help="Optional cohort/run subfolder inserted into the output path after "
             "the feature/zscore/paper_v1 subdirs and before the channel-set / "
             "threshold subdirs. Accepts multi-segment paths "
             "(e.g. 'paper_v1/validation_4exp_phase_only'). Pure organization — "
             "does not filter experiments.",
    )
    parser.add_argument(
        "--agg-method",
        type=str,
        default="mean",
        choices=["mean", "median"],
        help="Aggregation method for cells→guides and guides→geneKOs. Default: "
             "mean. ``median`` swaps both reductions; output is written to a "
             "separate ``agg_median/`` subdir so existing mean outputs are not "
             "overwritten. The PCA threshold sweep itself stays on mean so "
             "threshold selection is not biased by the agg method.",
    )
    parser.add_argument(
        "--chad-annotation",
        type=str,
        required=True,
        help="Path to the CHAD annotation YAML used for consistency scoring "
             "(mapping cluster -> {name, genes}).",
    )
    parser.add_argument(
        "--ebi-annotation",
        type=str,
        required=True,
        help="Path to EBI Complex Portal YAML for the 5th consistency score. "
             "Each entry is {name, genes:[...]} (same schema as CHAD). The "
             "score lands in metrics/phenotypic_consistency_ebi.csv with a "
             "dedicated mAP-vs-p-value volcano at plots/map_ebi_volcano.png.",
    )
    parser.add_argument(
        "--zscore-per-experiment", dest="zscore_per_experiment",
        action="store_true", default=True,
        help="Apply per-experiment z-score scaling to features before PCA. "
             "Output → zscore_per_exp/ subdir. Default: True.",
    )
    parser.add_argument(
        "--no-zscore-per-experiment", dest="zscore_per_experiment",
        action="store_false",
        help="Disable per-experiment z-score scaling.",
    )
    parser.add_argument(
        "--preserve-batch",
        action="store_true",
        help="Preserve experiment identity in guide/gene aggregation (for batch effect inspection). "
        "Skips the variance sweep; uses pca.variance_cutoff from the attribution config. "
        "Phase 2 aggregation is skipped. Output → batch/ subdir.",
    )
    parser.add_argument(
        "--no-pca",
        action="store_true",
        help="Skip PCA reduction entirely; export the full feature matrix. "
        "Phase 2 aggregation is skipped. Output → no_pca/ subdir.",
    )
    parser.add_argument(
        "--apply-iss-sidecar",
        action="store_true",
        help="When loading each per-experiment cell h5ad, apply the "
        "`<h5ad>_obs_corrected.parquet` sidecar produced by "
        "`ops_model.data.iss_drift_fix` so `obs[\"perturbation\"]` / "
        "`obs[\"sgRNA\"]` reflect the current ISS calls instead of the "
        "stale frozen snapshot. Cells flagged as `orphan_in_h5ad` are "
        "dropped (their seg_id is gone from the current ISS calls). "
        "Recommended for new analyses; pair with a dedicated output_path "
        "(e.g. paper_v1/phase_only_corrected/) to keep stale baselines intact.",
    )
    return parser
