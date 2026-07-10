"""Per-signal pooled PCA optimization & pre-reduction.

Pools cells across experiments sharing the same biological signal, fits PCA, sweeps
variance thresholds to find the optimal number of PCs, then aggregates all signals into
combined guide/gene h5ads and scores 4 phenotypic metrics (activity, distinctiveness,
CORUM consistency, CHAD consistency).

Two-phase SLURM architecture
-----------------------------
Phase 1  One SLURM job per biological signal group -- pool & downsample cells, PCA sweep,
         save per-signal h5ad.  Output → <root>/per_signal/
Phase 2  One aggregation job -- load per-signal h5ads, hconcat, NTC-normalize, score all
         4 metrics (also per-reporter), compute embeddings, save plots.

ROOT = /hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized_all

12 variants — feature type × channel subset
-------------------------------------------
Each variant produces an independent output subtree and can be compared via
analysis/compare_map_scores.py.  Replace --slurm with --aggregate-only --slurm to re-run
Phase 2 only (e.g. after code changes) without redoing the PCA sweeps.

  Variant                           Flags                                              Output subdir
  ───────────────────────────────── ────────────────────────────────────────────────── ─────────────────────────────────
  DINO        all                   --slurm                                            dino/all/
  DINO        phase-only            --phase-only --slurm                               dino/phase_only/
  DINO        no-phase              --no-phase --slurm                                 dino/no_phase/
  DINO        downsampled           --downsampled --slurm                              dino/downsampled/
  DINO        phase-only-ds         --phase-only --downsampled --slurm                 dino/phase_only_downsampled/
  DINO        no-phase-ds           --no-phase --downsampled --slurm                   dino/no_phase_downsampled/
  CellProfiler all                  --cell-profiler --slurm                            cellprofiler/all/
  CellProfiler phase-only           --phase-only --cell-profiler --slurm               cellprofiler/phase_only/
  CellProfiler no-phase             --no-phase --cell-profiler --slurm                 cellprofiler/no_phase/
  CellProfiler downsampled          --downsampled --cell-profiler --slurm              cellprofiler/downsampled/
  CellProfiler phase-only-ds        --phase-only --downsampled --cell-profiler --slurm cellprofiler/phase_only_downsampled/
  CellProfiler no-phase-ds          --no-phase --downsampled --cell-profiler --slurm   cellprofiler/no_phase_downsampled/

  Channel subsets:
    (default)    all fluorescent + phase channels, all cells pooled per signal group
    --phase-only label-free brightfield (Phase) only
    --no-phase   fluorescent channels only (excludes Phase)
    --downsampled cells equalised across signal groups (floor 750k/group, top 3 exps per signal)
    Combine --phase-only/--no-phase with --downsampled for filtered + downsampled variants.

  Append --aggregate-only to re-run Phase 2 only (e.g. after code changes).
  Use run_aggregate_all.sh to submit all 12 --aggregate-only jobs in parallel.

Validation cohort run (4 experiments, Phase only, validation500 library)
------------------------------------------------------------------------
The validation experiments (ops0146/0147/0150/0151) use the validation500
library, which has its own CHAD cluster file. ``--run-tag`` tucks the cohort
under a ``paper_v1/<leaf>`` parent for organization (no actual --paper-v1 flag,
since these experiments aren't in the paper_v1 YAML)::

    python -m ops_model.post_process.combination.pca_optimization \\
        --output-dir /hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized_v0.3 \\
        --cell-dino \\
        --zscore-per-experiment \\
        --run-tag paper_v1/validation_4exp_phase_only \\
        --experiments ops0146,ops0147,ops0150,ops0151 \\
        --phase-only \\
        --chad-annotation /hpc/projects/icd.fast.ops/configs/gene_clusters/val_library_chad_positive_controls_v1.yml \\
        --slurm

  → cell_dino/zscore_per_exp/paper_v1/validation_4exp_phase_only/phase_only/consensus_sweep/cosine/

Output structure
----------------
  <root>/
    dino/
      all/          (default)
      phase_only/   (--phase-only)
      no_phase/     (--no-phase)
      downsampled/  (--downsampled)
    cellprofiler/
      all/
      phase_only/
      no_phase/
      downsampled/
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd

from ops_model.features.anndata_utils import (
    aggregate_to_level,
    hconcat_by_perturbation,
    normalize_guide_adata,
    split_ntc_for_embedding,
)
from ops_utils.analysis.embedding_plots import (
    build_metric_lookup,
    clean_X_for_embedding,
    get_perts_col,
    plot_embedding_overlay,
)
from ops_utils.analysis.map_scores import (
    compute_auc_score,
    phenotypic_activity_assesment,
    phenotypic_consistency_manual_annotation,
    phenotypic_distinctivness,
    plot_map_scatter,
)
from ops_utils.analysis.pca import fit_pca, n_pcs_for_threshold
from ops_utils.analysis.pca_sweep_plots import (
    plot_channel_peaks_bar,
    plot_metric_map_bar,
    plot_pca_sweep,
    plot_sweep_curves_summary,
)
from ops_utils.data.feature_discovery import (
    build_signal_groups,
    count_cells_per_signal_group,
    discover_cellprofiler_experiments,
    discover_dino_experiments,
    find_cell_h5ad_path,
    get_channel_maps_path,
    get_storage_roots,
    load_attribution_config,
    load_cell_h5ad,
    resolve_channel_label,
    sanitize_signal_filename,
)
from ops_utils.data.positive_controls import (
    plot_positive_controls_grid,
)

from ops_model.post_process.combination.pipeline_add_ons.chromosome import (
    _load_chromosome_map,
    _plot_chromosome_overlay,
    _plot_chromosome_overlay_html,
)
from ops_model.post_process.combination.pipeline_add_ons.op_signal import (
    _discover_op_files,
    pca_sweep_op_signal,
)
from ops_model.post_process.combination.pca_optimization.parser import (
    _build_parser,
)
from ops_model.post_process.combination.pca_optimization.slurm import (
    _aggregate_then_second_pca,
    _build_second_pca_kwargs,
    _make_agg_slurm_params,
    _make_slurm_params,
    _submit_aggregation_slurm,
    _submit_phase1_slurm,
)
from ops_model.post_process.combination.pca_optimization.sweep_core import (
    _init_sweep_logger,
    _prepare_for_copairs,
    _run_guide_threshold_sweep,
    _run_threshold_sweep,
    _save_raw_outputs,
    _save_sweep_outputs,
    _score_activity_per_threshold,
)
from ops_model.post_process.combination.pca_optimization.aggregation import (
    ANNOTATED_GENE_PANEL_PATH,
    _annotate_genes_from_panel,
    _atomic_write_h5ad,
    _concat_and_normalize,
    _load_per_unit_blocks,
    _plot_chad_umap,
    _save_aggregated_h5ads,
    _save_per_reporter_metric_matrices,
    _score_activity_aggregated,
    _score_single_reporter_metrics,
)
from ops_model.post_process.combination.pca_optimization.embeddings import (
    _compute_and_plot_embeddings,
    _score_consistency,
    _score_distinctiveness,
    _score_ebi_plus,
)
from ops_model.post_process.combination.pca_optimization.phase1 import (
    pca_sweep_pooled_signal,
)
from ops_model.post_process.combination.pca_optimization.phase2 import (
    _save_pc_marker_contributions,
    aggregate_channels,
    apply_second_pass_pca,
)
from ops_model.post_process.combination.pca_optimization.handlers import (
    _discover_experiment_pairs,
    _fit_umap_one_seed,
    _handle_aggregate_only,
    _handle_chad_umap_only,
    _handle_downsampled,
    _handle_external,
    _handle_op,
    _handle_overlays_only,
    _handle_second_pca,
    _handle_sweep_seed,
    _handle_umap_only,
    _recompute_embeddings_for_seed,
    _run_overlays_only,
    _run_seed_sweep,
    _stored_embedding_seed,
    _try_load_swept_umap,
    run_chrom_arm_then_second_pca,
    run_second_pca_then_chrom_arm,
)

logger = logging.getLogger(__name__)

DEFAULT_SWEEP_THRESHOLDS = [
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
    0.99,
]
# CellProfiler features are hand-crafted and independent (not redundant like DINO embeddings),
# so PCA is destructive at high thresholds. Optimal region is ~50% variance explained.
DEFAULT_SWEEP_THRESHOLDS_CP = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
MIN_PCS = 10  # Minimum PCs for peak selection (avoids degenerate 1-PC artifact)
PCA_FIT_CAP = 5_000_000  # Cells used to fit PCA axes; larger datasets use passthrough (fit subsample, transform all)

# Consistency-score annotation paths — module-level globals so submitit can
# pickle helper functions that reference them. Default values are picked up
# from CLI flags in main() (``--chad-annotation`` / ``--ebi-annotation``).
CHAD_ANNOTATION_PATH: Optional[str] = None
EBI_ANNOTATION_PATH: Optional[str] = (
    "/hpc/projects/icd.fast.ops/configs/gene_clusters/"
    "EBI_complexes_v1_old_gene_names.yaml"
)

# Default location of the OrganelleProfiler consolidated per-marker h5ads.
# Mirrors organelle_profiler.feature_extraction.consolidate_all_cells.DEFAULT_OUTPUT_DIR.
DEFAULT_OP_ROOT = (
    "/hpc/projects/intracellular_dashboard/fast_ops/models/"
    "alex_lin_attention/all_cells_v2"
)

# Dud sgRNAs known to produce off-target/toxic phenotypes — filtered out by default.
# Source: cell_dino_final.yml cell_filters.
DUD_GUIDES = frozenset({
    "TCCCATGACTTGTTGTCATG",
    "GCAGGCAAATTCTGAACTTG",
    "GGGTGGTATCATAGCCACCC",
    "CACATCCCCAATGGGGAGTT",
    "TATTCAAAGTTGATGTTGGA",
})


# =============================================================================
# CLI
# =============================================================================



def _load_and_validate_config(config_path: str) -> dict:
    """Load a YAML config and validate its keys against the CLI argument set.

    Keys must be argparse ``dest`` names (snake_case), so a config is just the
    CLI args expressed as YAML (``--cell-dino`` → ``cell_dino``). Returns the
    parsed dict; the caller feeds it to ``parser.set_defaults(**cfg)``.
    """
    import yaml

    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(
            f"Config {config_path} must be a YAML mapping of arg→value, "
            f"got {type(cfg).__name__}."
        )
    valid_dests = {
        a.dest for a in _build_parser()._actions if a.dest not in ("help", "config")
    }
    unknown = sorted(set(cfg) - valid_dests)
    if unknown:
        raise ValueError(
            f"Unknown config key(s): {unknown}. Keys must match CLI argument names "
            f"as snake_case dest names (e.g. cell_dino, phase_only, output_dir, "
            f"fixed_threshold). Run the module with --help for the full list."
        )
    # set_defaults bypasses argparse's mutually-exclusive-group check, so guard
    # the one pair a config can realistically set together. (The "exactly one
    # feature-mode flag" rule is still enforced in run() below.)
    if cfg.get("phase_only") and cfg.get("no_phase"):
        raise ValueError(
            "Config sets both phase_only and no_phase (mutually exclusive)."
        )
    return cfg


def run_from_config(config_path: str):
    """Programmatic entry point: run the pipeline from a YAML config (no CLI).

    Equivalent to ``--config <path>`` on the command line. See
    ``pca_optimization/example_config.yml`` for the key set.
    """
    cfg = _load_and_validate_config(config_path)
    parser = _build_parser()
    parser.set_defaults(**cfg)
    run(parser.parse_args([]))


def main():
    # Force line-buffered stdout so progress prints appear in real time when
    # launched under `uv run`, `nohup`, or any other wrapper that pipes
    # stdout. Otherwise multi-minute discovery + submission steps look like
    # a silent hang.
    import sys as _sys
    try:
        _sys.stdout.reconfigure(line_buffering=True)  # Python 3.7+
    except (AttributeError, ValueError):
        pass

    args = _build_parser().parse_args()
    if getattr(args, "config", None):
        # Config file populates argparse defaults; any explicit CLI flag still
        # overrides it (re-parse the same argv against the config-seeded parser).
        cfg = _load_and_validate_config(args.config)
        parser = _build_parser()
        parser.set_defaults(**cfg)
        args = parser.parse_args()
    run(args)


def run(args):
    global CHAD_ANNOTATION_PATH, EBI_ANNOTATION_PATH
    CHAD_ANNOTATION_PATH = args.chad_annotation
    EBI_ANNOTATION_PATH = args.ebi_annotation
    # --seed default depends on --umap-type: max → 1 (Max's recipe), gav → 42 (legacy).
    if args.seed is None:
        args.seed = 1 if getattr(args, "umap_type", "max") == "max" else 42
        print(f"--seed unset, resolved to {args.seed} (umap_type={args.umap_type})")
    output_dir = Path(args.output_dir)

    # External mode: combine explicit per-signal h5ads given in the config's
    # `signal_paths` mapping (embeddings outside the experiment structure).
    # Bypasses the feature-mode requirement and experiment discovery.
    if getattr(args, "signal_paths", None):
        args.phase_filter = None
        args.all_cells = not getattr(args, "downsampled", False)
        out = output_dir if args.direct else output_dir / "external"
        print(f"External mode (signal_paths): output → {out}")
        _handle_external(args, out)
        return

    # --only-4i / --only-cp imply the corresponding --with-* and turn off the
    # standard scan. Apply once here so both --direct and standard paths see it.
    only_4i = getattr(args, "only_4i", False)
    only_cp = getattr(args, "only_cp", False)
    if only_4i:
        args.include_4i = True
    if only_cp:
        args.include_cp = True
    args.include_standard = not (only_4i or only_cp)

    # --direct: use the given path as-is, skip all automatic nesting
    if args.direct:
        args.phase_filter = None
        args.all_cells = True
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Direct mode: output → {output_dir}")
        if args.sweep_seed:
            _handle_sweep_seed(args, output_dir)
        elif args.chad_umap_only:
            _handle_chad_umap_only(args, output_dir)
        elif args.umap_only:
            _handle_umap_only(args, output_dir)
        elif args.overlays_only:
            _handle_overlays_only(args, output_dir)
        elif args.second_pca_only:
            _handle_second_pca(args, output_dir)
        elif args.aggregate_only:
            _handle_aggregate_only(args, output_dir)
        else:
            _handle_downsampled(args, output_dir, None)
        return

    # Nest output under feature-type subdir. Exactly one feature-mode flag is
    # required — there is no implicit DINO default, callers must opt in.
    cp_override = None
    feature_flags = [
        ("dino", getattr(args, "dino", False)),
        ("cell_profiler", getattr(args, "cell_profiler", False)),
        ("cell_dino", getattr(args, "cell_dino", False)),
        ("dynaclr", getattr(args, "dynaclr", False)),
        ("subcell", getattr(args, "subcell", False)),
        ("organelle_profiler", getattr(args, "organelle_profiler", False)),
    ]
    active = [name for name, on in feature_flags if on]
    if len(active) == 0:
        raise ValueError(
            "Pass exactly one feature-mode flag: --dino, --cell-dino, "
            "--cell-profiler, --dynaclr, --subcell, or --organelle-profiler."
        )
    if len(active) > 1:
        raise ValueError(
            f"Feature-mode flags are mutually exclusive; got: "
            f"{', '.join('--' + n.replace('_', '-') for n in active)}"
        )
    if args.cell_profiler:
        cp_override = "cell-profiler"
        output_dir = output_dir / "cellprofiler"
        print(
            f"CellProfiler mode: features from 3-assembly/cell-profiler/anndata_objects/"
        )
        print(
            f"PCA sweep thresholds: {DEFAULT_SWEEP_THRESHOLDS_CP} (lower range — CP features are independent)"
        )
        print(f"Output: {output_dir}")
    elif args.cell_dino:
        cp_override = "cell_dino_features"
        output_dir = output_dir / "cell_dino"
        print(f"Cell-DINO mode: features from 3-assembly/cell_dino_features/")
        print(f"Output: {output_dir}")
    elif getattr(args, "dynaclr", False):
        cp_override = "dynaclr_features"
        output_dir = output_dir / "dynaclr"
        print(f"DynaCLR mode: features from 3-assembly/dynaclr_features/")
        print(f"Output: {output_dir}")
    elif getattr(args, "subcell", False):
        cp_override = "subcell_features"
        output_dir = output_dir / "subcell"
        print(f"SubCell mode: features from 3-assembly/subcell_features/")
        print(f"Output: {output_dir}")
    elif getattr(args, "organelle_profiler", False):
        cp_override = "organelle_profiler"
        output_dir = output_dir / "organelle_profiler"
        print(f"OrganelleProfiler mode: features from {args.op_root}")
        print(f"Output: {output_dir}")
    else:  # args.dino
        # cp_override stays None — _discover_experiment_pairs falls back to
        # attr_config["feature_dir"] (typically "dino_features").
        output_dir = output_dir / "dino"
        print(f"DINO mode: features from 3-assembly/dino_features/")
        print(f"Output: {output_dir}")

    # Nest under zscore subdir if requested
    if args.zscore_per_experiment:
        output_dir = output_dir / "zscore_per_exp"
        print(f"Per-experiment z-score scaling enabled: output → {output_dir}")

    # paper_v1 sits at the top of the channel-set hierarchy so the v1 cohort
    # is the primary partition; with_cp / with_4i / cellpainting nest under it.
    if getattr(args, "paper_v1", None):
        output_dir = output_dir / "paper_v1"
        print(f"Paper-v1 experiment list enforced: output → {output_dir}")

    # --run-tag accepts a multi-segment relative path (e.g.
    # "paper_v1/validation_4exp_phase_only") so callers can recreate cohort
    # folders that don't have a dedicated flag.
    if getattr(args, "run_tag", None):
        tag = args.run_tag.strip().strip("/")
        if tag:
            output_dir = output_dir / tag
            print(f"Run tag: output → {output_dir}")

    # Nest under cell-painting subdir if requested
    if args.include_cellpainting:
        output_dir = output_dir / "with_cellpainting"
        print(f"Cell Painting channels included: output → {output_dir}")

    # Nest under cp-sibling subdir if requested (new layout). Goes ABOVE
    # with_4i so combined --with-cp --with-4i nests as with_cp/with_4i/.
    if getattr(args, "include_cp", False):
        sub = "only_cp" if getattr(args, "only_cp", False) and not args.include_standard else "with_cp"
        output_dir = output_dir / sub
        print(f"CP sibling-dir channels included: output → {output_dir}")

    # Nest under 4i subdir if requested (composes with --include-cellpainting / --with-cp)
    if getattr(args, "include_4i", False):
        sub = "only_4i" if getattr(args, "only_4i", False) and not args.include_standard else "with_4i"
        output_dir = output_dir / sub
        print(f"4i sibling-dir channels included: output → {output_dir}")

    # --downsample-per-guide implies --downsampled
    if args.downsample_per_guide:
        args.downsampled = True
    _ds_suffix = "_per_guide" if args.downsample_per_guide else ""

    # Nest under channel-subset subdir
    if args.phase_only and args.downsampled:
        output_dir = output_dir / f"phase_only_downsampled{_ds_suffix}"
        args.phase_filter = "phase_only"
        print(f"Phase-only downsampled mode: output → {output_dir}")
    elif args.no_phase and args.downsampled:
        output_dir = output_dir / f"no_phase_downsampled{_ds_suffix}"
        args.phase_filter = "no_phase"
        print(f"No-phase downsampled mode: output → {output_dir}")
    elif args.phase_only:
        output_dir = output_dir / "phase_only"
        args.phase_filter = "phase_only"
        print(f"Phase-only mode: output → {output_dir}")
    elif args.no_phase:
        output_dir = output_dir / "no_phase"
        args.phase_filter = "no_phase"
        print(f"No-phase mode: output → {output_dir}")
    elif args.downsampled:
        output_dir = output_dir / f"downsampled{_ds_suffix}"
        args.phase_filter = None
        print(f"Downsampled mode: output → {output_dir}")
    else:
        output_dir = output_dir / "all_livecell"
        args.phase_filter = None
        print(f"All live-cell mode (default): output → {output_dir}")

    # all_cells=True is now always the default (non-downsampled path)
    args.all_cells = not args.downsampled

    # Nest under mode-specific subdir
    if args.no_pca:
        mode_tag = "no_pca_batch" if args.preserve_batch else "no_pca"
        output_dir = output_dir / mode_tag
        print(f"No-PCA mode — output → {output_dir}")
    elif args.preserve_batch:
        output_dir = output_dir / "batch"
        print(f"Preserve-batch mode — output → {output_dir}")
    elif args.fixed_threshold is not None and args.fixed_threshold > 0:
        thresh_tag = f"fixed_{args.fixed_threshold:.0%}"
        output_dir = output_dir / thresh_tag
        print(f"Fixed threshold: {args.fixed_threshold:.0%} — output → {output_dir}")
    else:
        output_dir = output_dir / "consensus_sweep"
        print(f"Consensus sweep mode — output → {output_dir}")

    # Nest under distance metric subdir
    output_dir = output_dir / args.distance
    print(f"Distance metric: {args.distance} — output → {output_dir}")

    # Aggregation-method subdir: mean is the default and stays at the canonical
    # path; non-mean (currently only median) gets its own subtree so existing
    # outputs are never overwritten.
    if getattr(args, "agg_method", "mean") != "mean":
        output_dir = output_dir / f"agg_{args.agg_method}"
        print(f"Aggregation method: {args.agg_method} — output → {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Dispatch to mode handler. ``--sweep-seed`` is checked first so that
    # combining it with ``--second-pca-only`` (which is also a subdir
    # navigator) doesn't accidentally trigger the full second-pass pipeline.
    if args.sweep_seed:
        _handle_sweep_seed(args, output_dir)
    elif args.chad_umap_only:
        _handle_chad_umap_only(args, output_dir)
    elif args.umap_only:
        _handle_umap_only(args, output_dir)
    elif args.overlays_only:
        _handle_overlays_only(args, output_dir)
    elif args.second_pca_only:
        _handle_second_pca(args, output_dir)
    elif args.aggregate_only:
        _handle_aggregate_only(args, output_dir)
    elif getattr(args, "organelle_profiler", False):
        _handle_op(args, output_dir)
    else:
        _handle_downsampled(args, output_dir, cp_override)

