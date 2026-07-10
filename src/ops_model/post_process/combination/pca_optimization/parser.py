"""Argparse parser for the pca_optimization CLI.

Extracted from ``pca_optimization.py`` to keep the (large) argparse
configuration out of the main module. The single entry point is
``_build_parser()`` — call it, then ``main()`` reads ``args`` and
dispatches.

The only module-level dependency on ``pca_optimization`` is
``DEFAULT_OP_ROOT`` (used as the ``--op-root`` default). It is imported
lazily inside ``_build_parser`` so the two modules can re-import each
other at module load time without a circular dependency.
"""

from __future__ import annotations

import argparse


def _build_parser():
    """Build argparse parser for the PCA optimization CLI."""
    # Lazy import to avoid a circular dependency: pca_optimization.py
    # re-imports _build_parser from this module at module load time.
    from ops_model.post_process.combination.pca_optimization import DEFAULT_OP_ROOT

    parser = argparse.ArgumentParser(
        description="Per-signal pooled PCA optimization for organelle attribution"
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
        "--signal-paths",
        dest="signal_paths",
        default=None,
        help="(Config-only) Combine embeddings that live OUTSIDE the standard "
        "experiment layout. Set in the --config YAML under `signal_paths:` as a "
        "mapping of signal-group name -> h5ad path (or list of paths to pool). "
        "Each h5ad must have the same schema as the discovery "
        "features_processed_*.h5ad (obs: sgRNA / perturbation / experiment; "
        "X = embedding). When set, experiment discovery is skipped. Output → "
        "<output-dir>/external/.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="/hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized_v0.3",
        help="Root output directory (feature-type and channel-subset subdirs are added automatically)",
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
        "--aggregate-only",
        action="store_true",
        help="Only run Phase 2 aggregation (skips PCA sweeps, reads existing per_signal/ h5ads).",
    )
    parser.add_argument(
        "--umap-only",
        action="store_true",
        help="Only generate embedding plots from existing optimized h5ads.",
    )
    parser.add_argument(
        "--overlays-only",
        action="store_true",
        help="Re-generate the interactive HTML overlay pages from existing "
        "guide_pca_optimized.h5ad + gene_embedding_pca_optimized.h5ad. "
        "Skips the full pipeline, PCA sweep, and scoring. UMAP/PHATE are "
        "reused from the h5ads unless --seed differs from the stored "
        "random_state, in which case those embeddings are refit and the "
        "h5ads are rewritten before overlays are regenerated.",
    )
    parser.add_argument(
        "--umap-n-neighbors",
        type=int,
        default=None,
        help="Override gene-level UMAP n_neighbors when used with "
        "--overlays-only (forces a refit at this value; bypasses the "
        "--sweep-seed cache, which is keyed on default n_neighbors=15).",
    )
    parser.add_argument(
        "--umap-min-dist",
        type=float,
        default=None,
        help="Override gene-level UMAP min_dist when used with --overlays-only "
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
        "--sweep-seed",
        action="store_true",
        help="Fit gene-level UMAP at --sweep-seed-n consecutive seeds and "
        "save a single PNG canvas (sqrt(n)×sqrt(n) panels) so different "
        "seed-driven layouts can be compared at a glance. Reads the existing "
        "gene_embedding_pca_optimized.h5ad — no other outputs are touched.",
    )
    parser.add_argument(
        "--sweep-seed-n",
        type=int,
        default=36,
        help="Number of consecutive seeds to fit in --sweep-seed mode "
        "(default: 36 → 6×6 grid).",
    )
    parser.add_argument(
        "--sweep-seed-base",
        type=int,
        default=0,
        help="Starting seed value for --sweep-seed mode. Seeds tried are "
        "[base, base+1, ..., base+n-1] (default: 0).",
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
        "--chrom-arm-correct",
        action="store_true",
        help="Before running --second-pca-only, regress out chromosome-arm "
        "effects from <output_dir>/guide_pca_optimized.h5ad and write the "
        "corrected guide-level h5ad to "
        "guide_pca_optimized_chrom_arm_corr.h5ad next to the original. The "
        "2nd-pass PCA then runs on the corrected h5ad and lands under "
        "second_pca_consensus_chrom_arm_corr/ (or second_pca_<pct>_chrom_arm_corr/) "
        "— never clobbers the existing untouched outputs. See "
        "ops_model/post_process/combination/guide_chrom_arm_correction.py for "
        "the kNN-cohesion test + per-arm median regression details.",
    )
    parser.add_argument(
        "--chrom-arm-knn",
        type=int,
        default=25,
        help="kNN value for the per-guide arm-cohesion test (default 25). "
        "Larger k = more statistical power per guide. Notebook uses 15; we "
        "default to 25 to catch deeper sig populations per arm.",
    )
    parser.add_argument(
        "--chrom-arm-qval",
        type=float,
        default=0.05,
        help="FDR threshold for flagging a guide as significantly "
        "arm-cohesive (default 0.05). Notebook uses 0.01; we default to "
        "0.05 for a stronger correction across more guides.",
    )
    parser.add_argument(
        "--chrom-arm-min-genes",
        type=int,
        default=10,
        help="Minimum number of significant guides on an arm before that "
        "arm's median is regressed out; smaller arms are left untouched "
        "(default 10).",
    )
    parser.add_argument(
        "--chrom-arm-skip-second-pca",
        action="store_true",
        help="With --chrom-arm-correct, skip the 2nd-pass PCA step after the "
        "correction. Output lands in <output_dir>/chrom_arm_corr<method>/ "
        "(no second_pca_consensus prefix) with full activity/distinctiveness/"
        "CORUM/CHAD scoring + plots run directly on the corrected guide "
        "h5ad.",
    )
    parser.add_argument(
        "--chrom-arm-after-second-pca",
        action="store_true",
        help="Reverse the order: run 2nd-pass PCA FIRST (re-uses existing "
        "second_pca_consensus/ when present), then apply chrom-arm "
        "correction in the ~29-sPC space, then score. Output lands in "
        "<output_dir>/second_pca_consensus_then_chrom_arm_corr<method>/. "
        "Mutually exclusive with --chrom-arm-skip-second-pca.",
    )
    parser.add_argument(
        "--chrom-arm-method",
        type=str,
        default="cohesion",
        choices=["cohesion", "centering", "scanpy_regress"],
        help="Chromosome-arm correction strategy. 'cohesion' (default, "
        "notebook): kNN cohesion test flags sig guides, per-arm median is "
        "subtracted only from sig members (~5%% of guides modified). "
        "'centering': every arm with ≥--chrom-arm-min-genes members has its "
        "arm-mean offset replaced with the overall mean (~all annotated "
        "guides modified). 'scanpy_regress': call sc.pp.regress_out with "
        "chrom_arm as the categorical covariate (residuals = X minus "
        "per-arm mean, no global-mean add-back; small/unmapped arms folded "
        "into an 'unmapped' sentinel that's not part of the regression). "
        "Each method's outputs land in their own subdir suffix "
        "(_chrom_arm_corr, _chrom_arm_corr_centering, "
        "_chrom_arm_corr_scanpy_regress) so methods never clobber each other.",
    )
    parser.add_argument(
        "--chrom-arm-map-csv",
        type=str,
        default=None,
        help="Path to a cached symbol→arm CSV (columns: symbol, chrom_arm). "
        "When this file exists, the chrom-arm helper skips the mygene "
        "network call and reads the mapping straight from disk — useful on "
        "SLURM nodes with no outbound internet. Default: shared cache at "
        "/hpc/projects/icd.fast.ops/configs/library/chrom_arm_mapping.csv "
        "(falls back to <input_dir>/chrom_arm_mapping.csv off-cluster).",
    )
    parser.add_argument(
        "--downsampled",
        action="store_true",
        help="Equalise cells across signal groups by downsampling to the smallest group "
        "(floor 750k). Default mode uses all cells per group. Output → downsampled/.",
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
        dest="downsample_per_guide",
        action="store_true",
        help="Cap each sgRNA at --cells-per-guide cells (pooled across experiments). "
             "Replaces proportional-per-experiment downsampling with a global per-sgRNA cap. "
             "Implies --downsampled. Output → downsampled_per_guide/.",
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
        "--organelle-profiler",
        "--op",
        dest="organelle_profiler",
        action="store_true",
        help="Use OrganelleProfiler consolidated per-marker h5ads (the output "
             "of organelle_profiler.feature_extraction.consolidate_all_cells). "
             "Each all_cells_*.h5ad is treated as one signal group (cells "
             "already pooled across experiments). Output → organelle_profiler/ "
             "subdir.",
    )
    parser.add_argument(
        "--op-root",
        type=str,
        default=DEFAULT_OP_ROOT,
        help=f"Directory containing OrganelleProfiler all_cells_*.h5ad files "
             f"(default: {DEFAULT_OP_ROOT}). Used with --organelle-profiler.",
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
        "--direct",
        action="store_true",
        help="Use -o/--output-dir as the exact output path (skip automatic dino/all/… nesting).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover experiments and print the signal-group manifest, then exit without processing.",
    )
    parser.add_argument(
        "--include-cellpainting",
        action="store_true",
        help="Include legacy CP1_*/CP2_* channels stored inside the standard feature_dir "
             "(older Cell Painting layout). For the newer per-experiment _cp/ sibling layout, "
             "use --with-cp instead. Output → with_cellpainting/ subdir.",
    )
    parser.add_argument(
        "--with-4i",
        dest="include_4i",
        action="store_true",
        help="Also scan parallel <feature_dir>_4i/ sibling directories (e.g. dino_features_4i/) "
             "and pool those channels alongside standard ones. 4i channels are tagged so they "
             "form their own signal groups (label suffix ' (4i)'). Composes with --with-cp and "
             "--include-cellpainting. Output → with_4i/ subdir.",
    )
    parser.add_argument(
        "--with-cp",
        dest="include_cp",
        action="store_true",
        help="Also scan parallel <feature_dir>_cp/ sibling directories (e.g. cell_dino_features_cp/) "
             "for the new Cell Painting layout (e.g. ops0094 ConA/Hoechst/NPM1/Phalloidin/TOMM20/Tubulin/WGA). "
             "Channels are tagged so they form their own signal groups (label suffix ' (cp)'). "
             "Composes with --with-4i. Output → with_cp/ subdir.",
    )
    parser.add_argument(
        "--only-4i",
        action="store_true",
        help="Run only the 4i sibling-dir channels (skip the standard <feature_dir>/). "
             "Implies --with-4i and turns off the standard scan. Output → only_4i/ subdir.",
    )
    parser.add_argument(
        "--only-cp",
        action="store_true",
        help="Run only the cp sibling-dir channels (skip the standard <feature_dir>/). "
             "Implies --with-cp and turns off the standard scan. Output → only_cp/ subdir. "
             "Composes with --only-4i (then both 4i and cp siblings, no standard).",
    )
    parser.add_argument(
        "--paper-v1",
        type=str,
        nargs="?",
        const="/hpc/projects/icd.fast.ops/configs/good_experiment_list_v1.yml",
        default=None,
        help="Restrict discovery to the exact experiment list in the paper-v1 YAML "
             "(default path: /hpc/projects/icd.fast.ops/configs/good_experiment_list_v1.yml). "
             "Errors out if any expected experiment is missing from discovery. "
             "Output → paper_v1/ subdir.",
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
        "--chromosome-csv",
        type=str,
        default=None,
        help="Path to a CSV with columns (perturbation, chromosome, chromosome_arm) "
             "used to color the gene-level UMAP + PHATE by chromosomal location "
             "(<chromosome><arm>, e.g. '12q'). Coords come from the current run's "
             "freshly computed embeddings, not from any UMAP1/UMAP2 columns in "
             "this CSV.",
    )
    parser.add_argument(
        "--chromosome-only",
        action="store_true",
        help="Combine with --overlays-only to regenerate ONLY the chromosome "
             "UMAP + PHATE plots (no seed refit, no other overlays). Requires "
             "--chromosome-csv. Fast path for restyling the chromosome plot "
             "after edits to its renderer.",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        default=None,
        help="Comma-separated experiment short names (e.g. ops0031,ops0035) to restrict to. "
             "Only these experiments will be included in signal groups. Useful for A/B comparisons.",
    )
    parser.add_argument(
        "--signals",
        type=str,
        default=None,
        help="Comma-separated signal-group names (e.g. \"Phase,5xUPRE,ER_SEC61B\") to "
             "restrict Phase 1 to. Useful for retrying just the failed shards from a "
             "prior submission. Successful per_signal/ outputs from the prior run are "
             "preserved on disk. Names must match the canonical signal labels (the same "
             "ones used as per_signal/<name>_guide.h5ad filenames).",
    )
    parser.add_argument(
        "--match-v02",
        action="store_true",
        help="Restrict to the same experiments used in pca_optimized_v0.2 (reads v0.2 manifest). "
             "Useful for controlled A/B comparison of features with identical experiment sets.",
    )
    parser.add_argument(
        "--chad-annotation",
        type=str,
        default=None,
        help="Path to custom CHAD annotation YAML for consistency scoring. "
             "Defaults to chad_positive_controls_v5_hierarchy.yml.",
    )
    parser.add_argument(
        "--ebi-annotation",
        type=str,
        default="/hpc/projects/icd.fast.ops/configs/gene_clusters/EBI_complexes_v1_old_gene_names.yaml",
        help="Path to EBI Complex Portal YAML for the 5th consistency score. "
             "Each entry is {name, genes:[...]} (same schema as CHAD). The "
             "score lands in metrics/phenotypic_consistency_ebi.csv with a "
             "dedicated mAP-vs-p-value volcano at plots/map_ebi_volcano.png.",
    )
    parser.add_argument(
        "--chad-umap-only", action="store_true",
        help="Only regenerate the CHAD-colored UMAP from existing gene_embedding_pca_optimized.h5ad.",
    )
    parser.add_argument(
        "--chad-umap-output", type=str, default=None,
        help="Output filename for CHAD UMAP (saved under plots/).",
    )
    parser.add_argument(
        "--chad-cluster-range", type=str, default=None,
        help="Filter CHAD clusters to a range by integer key (e.g. '1-76' or '100-162').",
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
    phase_group = parser.add_mutually_exclusive_group()
    phase_group.add_argument(
        "--phase-only",
        action="store_true",
        help="Include only Phase (label-free brightfield) channels. Output → phase_only/.",
    )
    phase_group.add_argument(
        "--no-phase",
        action="store_true",
        help="Exclude Phase channels, fluorescent only. Output → no_phase/.",
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
