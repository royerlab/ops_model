"""Per-signal pooled PCA optimization & pre-reduction.

Pools cells across the h5ads given for each signal group, fits PCA, sweeps
variance thresholds to find the optimal number of PCs, then aggregates all
signals into combined guide/gene h5ads and scores the phenotypic metrics
(activity, distinctiveness, CORUM / CHAD / EBI consistency).

Two-phase SLURM architecture
-----------------------------
Phase 1  One job per signal group -- pool cells, PCA sweep, save per-signal
         h5ad.  Output -> <root>/per_signal/
Phase 2  One aggregation job -- load per-signal h5ads, hconcat, NTC-normalize,
         score the metrics (also per-reporter), compute embeddings, save plots.

Inputs are always explicit
--------------------------
Signal groups are named on the command line; there is no experiment discovery
and no channel/experiment filtering. Each ``--signal`` names one group and lists
the per-signal h5ads to pool into it::

    python -m ops_model.post_process.combination.pca_optimization \
        -o <output_root> \
        --cell-dino --zscore-per-experiment \
        --chad-annotation <chad.yml> \
        --ebi-annotation <ebi_complexes.yaml> \
        --gene-panel <annotated_gene_panel.csv> \
        --signal Phase=<...>/ops0146_Phase.h5ad,<...>/ops0147_Phase.h5ad \
        --signal ER_SEC61B=<...>/ops0146_ER.h5ad \
        --run-tag my_run \
        --slurm

Output structure
----------------
The feature-mode flag, ``--zscore-per-experiment``, ``--run-tag``, the
threshold mode, the distance metric and a non-mean ``--agg-method`` each add a
path segment beneath ``-o``, e.g.::

    <root>/cell_dino/zscore_per_exp/my_run/fixed_80%/cosine/
      per_signal/          Phase 1 output, one h5ad per signal group
      guide_pca_optimized.h5ad
      gene_pca_optimized.h5ad
      metrics/             one CSV per phenotypic metric
      plots/
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
from ops_utils.data.positive_controls import (
    plot_positive_controls_grid,
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
    _handle_external,
    _handle_second_pca,
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
EBI_ANNOTATION_PATH: Optional[str] = None
GENE_PANEL_PATH: Optional[str] = None

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


def _parse_signal_specs(specs: list) -> dict:
    """Parse repeated ``--signal NAME=path1,path2`` into {name: [paths]}.

    Each path must be an existing h5ad with the same schema discovery used to
    produce (``obs`` carrying sgRNA / perturbation / experiment, ``X`` = the
    embedding). Multiple paths under one name are pooled into that signal group.
    Names become output directory components, so they must be unique and free of
    path separators.
    """
    groups: dict = {}
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(
                f"--signal expects NAME=path1[,path2,...], got {spec!r}"
            )
        name, _, path_str = spec.partition("=")
        name = name.strip()
        if not name:
            raise SystemExit(f"--signal has an empty name: {spec!r}")
        if "/" in name or "\\" in name:
            raise SystemExit(
                f"--signal name {name!r} can't contain a path separator "
                f"(it becomes an output directory name)"
            )
        if name in groups:
            raise SystemExit(f"--signal {name!r} given more than once")
        paths = [Path(p.strip()) for p in path_str.split(",") if p.strip()]
        if not paths:
            raise SystemExit(f"--signal {name!r} lists no paths")
        missing = [str(x) for x in paths if not x.is_file()]
        if missing:
            raise SystemExit(
                f"--signal {name!r}: missing h5ad(s):\n  " + "\n  ".join(missing)
            )
        groups[name] = [str(x) for x in sorted(paths)]
    return groups


def run(args):
    global CHAD_ANNOTATION_PATH, EBI_ANNOTATION_PATH, GENE_PANEL_PATH
    CHAD_ANNOTATION_PATH = args.chad_annotation
    EBI_ANNOTATION_PATH = args.ebi_annotation
    GENE_PANEL_PATH = args.gene_panel
    # --seed default depends on --umap-type: max → 1 (Max's recipe), gav → 42 (legacy).
    if args.seed is None:
        args.seed = 1 if getattr(args, "umap_type", "max") == "max" else 42
        print(f"--seed unset, resolved to {args.seed} (umap_type={args.umap_type})")
    output_dir = Path(args.output_dir)

    # Signals are always given explicitly as --signal NAME=path1[,path2,...].
    signal_paths = _parse_signal_specs(args.signal_specs)
    args.signal_paths = signal_paths
    n_files = sum(len(v) for v in signal_paths.values())
    print(f"{len(signal_paths)} signal group(s), {n_files} h5ad(s)")

    feature_flags = [
        ("dino", getattr(args, "dino", False)),
        ("cell_profiler", getattr(args, "cell_profiler", False)),
        ("cell_dino", getattr(args, "cell_dino", False)),
        ("dynaclr", getattr(args, "dynaclr", False)),
        ("subcell", getattr(args, "subcell", False)),
    ]
    active = [name for name, on in feature_flags if on]
    if len(active) == 0:
        raise ValueError(
            "Pass exactly one feature-mode flag: --dino, --cell-dino, "
            "--cell-profiler, --dynaclr, or --subcell."
        )
    if len(active) > 1:
        raise ValueError(
            f"Feature-mode flags are mutually exclusive; got: "
            f"{', '.join('--' + n.replace('_', '-') for n in active)}"
        )
    if args.cell_profiler:
        output_dir = output_dir / "cellprofiler"
        print(
            f"CellProfiler mode: features from 3-assembly/cell-profiler/anndata_objects/"
        )
        print(
            f"PCA sweep thresholds: {DEFAULT_SWEEP_THRESHOLDS_CP} (lower range — CP features are independent)"
        )
        print(f"Output: {output_dir}")
    elif args.cell_dino:
        output_dir = output_dir / "cell_dino"
        print(f"Cell-DINO mode: features from 3-assembly/cell_dino_features/")
        print(f"Output: {output_dir}")
    elif getattr(args, "dynaclr", False):
        output_dir = output_dir / "dynaclr"
        print(f"DynaCLR mode: features from 3-assembly/dynaclr_features/")
        print(f"Output: {output_dir}")
    elif getattr(args, "subcell", False):
        output_dir = output_dir / "subcell"
        print(f"SubCell mode: features from 3-assembly/subcell_features/")
        print(f"Output: {output_dir}")
    else:  # args.dino
        output_dir = output_dir / "dino"
        print(f"DINO mode: features from 3-assembly/dino_features/")
        print(f"Output: {output_dir}")

    # Nest under zscore subdir if requested
    if args.zscore_per_experiment:
        output_dir = output_dir / "zscore_per_exp"
        print(f"Per-experiment z-score scaling enabled: output → {output_dir}")

    # --run-tag accepts a multi-segment relative path so callers can group runs.
    if getattr(args, "run_tag", None):
        tag = args.run_tag.strip().strip("/")
        if tag:
            output_dir = output_dir / tag
            print(f"Run tag: {tag} — output → {output_dir}")

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

    # Only two modes remain: re-run the 2nd-pass PCA over an existing output
    # tree, or run the pipeline over the explicitly-listed signals.
    if args.second_pca_only:
        _handle_second_pca(args, output_dir)
    else:
        _handle_external(args, output_dir)

