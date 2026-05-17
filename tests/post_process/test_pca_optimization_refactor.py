"""Safety-net tests for the pca_optimization.py refactor.

These tests pin down structural surfaces of the module so that each refactor
commit (extracting helpers into submodules, moving CLI handlers, etc.) can be
validated cheaply. They intentionally do NOT exercise heavy data paths — the
file is dominated by SLURM-driven AnnData I/O that needs real per_signal/
h5ads from the HPC filesystem. Instead, the tests cover:

* the module imports cleanly,
* module-level constants are preserved,
* ``_build_parser()`` accepts representative CLI invocations and resolves
  flags to the expected ``args`` namespace,
* every "subcommand" flag combination still produces a ``--help`` that
  exits zero,
* small pure helpers (``_make_slurm_params``, ``_make_agg_slurm_params``,
  ``_build_second_pca_kwargs``, ``_stored_embedding_seed``,
  ``_load_chromosome_map``, ``_discover_op_files``) keep their contracts.

If any of these break during the refactor, the commit that broke them is the
last passing commit's child.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# import & top-level surface
# ---------------------------------------------------------------------------


def test_module_imports():
    """The module must import without side effects or missing symbols."""
    from ops_model.post_process.combination import pca_optimization  # noqa: F401


def test_public_surface_present():
    """Pin the public entry points the rest of the codebase imports."""
    from ops_model.post_process.combination import pca_optimization as mod

    expected = [
        "main",
        "_build_parser",
        "pca_sweep_pooled_signal",
        "pca_sweep_op_signal",
        "aggregate_channels",
        "apply_second_pass_pca",
        "run_second_pca_then_chrom_arm",
        "run_chrom_arm_then_second_pca",
        "_discover_op_files",
        "_load_chromosome_map",
        "_make_slurm_params",
        "_make_agg_slurm_params",
        "_build_second_pca_kwargs",
        "_stored_embedding_seed",
    ]
    missing = [name for name in expected if not hasattr(mod, name)]
    assert not missing, f"Missing public/helper names after refactor: {missing}"


def test_handler_dispatch_targets_present():
    """Every CLI mode flag must resolve to a handler function in main()."""
    from ops_model.post_process.combination import pca_optimization as mod

    handlers = [
        "_handle_chad_umap_only",
        "_handle_umap_only",
        "_handle_overlays_only",
        "_handle_aggregate_only",
        "_handle_second_pca",
        "_handle_sweep_seed",
        "_handle_downsampled",
        "_handle_op",
    ]
    missing = [name for name in handlers if not callable(getattr(mod, name, None))]
    assert not missing, f"Missing handler functions after refactor: {missing}"


# ---------------------------------------------------------------------------
# module-level constants
# ---------------------------------------------------------------------------


def test_constants_pinned():
    from ops_model.post_process.combination import pca_optimization as mod

    assert mod.MIN_PCS == 10
    assert mod.PCA_FIT_CAP == 5_000_000
    assert isinstance(mod.DEFAULT_SWEEP_THRESHOLDS, list)
    assert mod.DEFAULT_SWEEP_THRESHOLDS[0] == 0.20
    assert mod.DEFAULT_SWEEP_THRESHOLDS[-1] == 0.99
    # CP thresholds run a tighter, lower-variance range.
    assert isinstance(mod.DEFAULT_SWEEP_THRESHOLDS_CP, list)
    assert mod.DEFAULT_SWEEP_THRESHOLDS_CP[0] == 0.30
    assert mod.DEFAULT_SWEEP_THRESHOLDS_CP[-1] == 0.70


def test_annotation_path_defaults():
    from ops_model.post_process.combination import pca_optimization as mod

    # CHAD path is populated from --chad-annotation in main(); the module
    # default must stay None so submitit pickling is reproducible.
    assert mod.CHAD_ANNOTATION_PATH is None
    assert mod.EBI_ANNOTATION_PATH is not None
    assert mod.EBI_ANNOTATION_PATH.endswith(".yaml")
    assert "EBI_complexes" in mod.EBI_ANNOTATION_PATH
    assert mod.DEFAULT_OP_ROOT.endswith("all_cells_v2")


def test_dud_guides_frozenset():
    from ops_model.post_process.combination import pca_optimization as mod

    assert isinstance(mod.DUD_GUIDES, frozenset)
    # If the file format ever changes accidentally the set will collapse.
    assert len(mod.DUD_GUIDES) > 0


# ---------------------------------------------------------------------------
# _build_parser smoke tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def parser():
    from ops_model.post_process.combination import pca_optimization as mod

    return mod._build_parser()


def test_parser_default(parser):
    args = parser.parse_args([])
    # Defaults must stay stable so the docs in the module docstring stay valid.
    assert args.norm_method == "ntc"
    assert args.distance == "cosine"
    assert args.fixed_threshold == 0.80
    assert args.slurm is False
    assert args.aggregate_only is False
    assert args.umap_only is False
    assert args.overlays_only is False
    assert args.organelle_profiler is False


def test_parser_aggregate_only(parser):
    args = parser.parse_args(["--aggregate-only"])
    assert args.aggregate_only is True


def test_parser_umap_only(parser):
    args = parser.parse_args(["--umap-only"])
    assert args.umap_only is True


def test_parser_overlays_only(parser):
    args = parser.parse_args(["--overlays-only"])
    assert args.overlays_only is True


def test_parser_chad_umap_only(parser):
    args = parser.parse_args(["--chad-umap-only"])
    assert args.chad_umap_only is True


def test_parser_op_signal(parser):
    args = parser.parse_args(["--op", "--op-root", "/tmp/op_root"])
    assert args.organelle_profiler is True
    assert args.op_root == "/tmp/op_root"


def test_parser_phase_only(parser):
    args = parser.parse_args(["--phase-only"])
    assert args.phase_only is True
    assert args.no_phase is False


def test_parser_no_phase(parser):
    args = parser.parse_args(["--no-phase"])
    assert args.no_phase is True
    assert args.phase_only is False


def test_parser_downsampled(parser):
    args = parser.parse_args(["--downsampled"])
    assert args.downsampled is True


def test_parser_cell_profiler(parser):
    args = parser.parse_args(["--cell-profiler"])
    assert args.cell_profiler is True


def test_parser_second_pca_full(parser):
    args = parser.parse_args(
        [
            "--second-pca",
            "--second-pca-threshold",
            "0.85",
            "--second-pca-sweep-thresholds",
            "0.5,0.6,0.7,0.8,0.9",
        ]
    )
    assert args.second_pca is True
    assert args.second_pca_threshold == 0.85
    assert args.second_pca_sweep_thresholds == "0.5,0.6,0.7,0.8,0.9"


def test_parser_no_second_pca(parser):
    args = parser.parse_args(["--no-second-pca"])
    assert args.second_pca is False


def test_parser_agg_method(parser):
    args = parser.parse_args(["--agg-method", "median"])
    assert args.agg_method == "median"


def test_parser_chromosome_csv(parser):
    args = parser.parse_args(["--chromosome-csv", "/tmp/chrom.csv"])
    assert args.chromosome_csv == "/tmp/chrom.csv"


def test_parser_chrom_arm_correct(parser):
    args = parser.parse_args(["--chrom-arm-correct"])
    assert args.chrom_arm_correct is True


def test_parser_sweep_seed(parser):
    args = parser.parse_args(["--sweep-seed", "--sweep-seed-n", "5"])
    assert args.sweep_seed is True
    assert args.sweep_seed_n == 5


def test_parser_slurm_block(parser):
    args = parser.parse_args(
        [
            "--slurm",
            "--slurm-memory", "300GB",
            "--slurm-time", "60",
            "--slurm-cpus", "8",
            "--slurm-partition", "gpu",
            "--slurm-agg-memory", "800GB",
            "--slurm-agg-time", "240",
        ]
    )
    assert args.slurm is True
    assert args.slurm_memory == "300GB"
    assert args.slurm_time == 60
    assert args.slurm_cpus == 8
    assert args.slurm_partition == "gpu"
    assert args.slurm_agg_memory == "800GB"
    assert args.slurm_agg_time == 240


def test_parser_dry_run(parser):
    """--dry-run is required for safe end-to-end smoke testing of the CLI."""
    args = parser.parse_args(["--dry-run"])
    assert args.dry_run is True


def test_help_exits_zero(capsys):
    """``--help`` short-circuits with SystemExit(0) and prints usage."""
    from ops_model.post_process.combination import pca_optimization as mod

    with pytest.raises(SystemExit) as exc_info:
        mod._build_parser().parse_args(["--help"])
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "PCA" in out or "pca" in out


def test_cli_help_subprocess():
    """End-to-end smoke: ``python -m … --help`` exits 0 with non-empty output.

    Catches problems that an in-process parser test misses: import errors at
    ``__main__`` entry, accidental side effects on module load, broken
    argparse defaults that reference globals not yet defined, etc.
    """
    import subprocess

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ops_model.post_process.combination.pca_optimization",
            "--help",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"--help failed: {result.stderr!r}"
    assert "--output-dir" in result.stdout
    assert "--slurm" in result.stdout
    assert "--aggregate-only" in result.stdout


# ---------------------------------------------------------------------------
# pure helpers — SLURM param builders
# ---------------------------------------------------------------------------


def test_make_slurm_params():
    from ops_model.post_process.combination import pca_optimization as mod

    args = SimpleNamespace(
        slurm_time=30,
        slurm_memory="200GB",
        slurm_cpus=16,
        slurm_partition="cpu,gpu",
    )
    params = mod._make_slurm_params(args)
    assert params == {
        "timeout_min": 30,
        "mem": "200GB",
        "cpus_per_task": 16,
        "slurm_partition": "cpu,gpu",
    }


def test_make_agg_slurm_params():
    from ops_model.post_process.combination import pca_optimization as mod

    args = SimpleNamespace(
        slurm_agg_time=180,
        slurm_agg_memory="600GB",
        slurm_cpus=16,
        slurm_partition="cpu,gpu",
    )
    params = mod._make_agg_slurm_params(args)
    assert params == {
        "timeout_min": 180,
        "mem": "600GB",
        "cpus_per_task": 16,
        "slurm_partition": "cpu,gpu",
    }


# ---------------------------------------------------------------------------
# pure helpers — _build_second_pca_kwargs
# ---------------------------------------------------------------------------


def test_build_second_pca_kwargs_disabled():
    from ops_model.post_process.combination import pca_optimization as mod

    args = SimpleNamespace(second_pca=False)
    assert mod._build_second_pca_kwargs(args) is None


def test_build_second_pca_kwargs_enabled_no_sweep():
    from ops_model.post_process.combination import pca_optimization as mod

    args = SimpleNamespace(
        second_pca=True,
        second_pca_threshold=0.80,
        second_pca_subdir="second_pca_consensus",
        second_pca_no_sweep=True,
        second_pca_sweep_thresholds=None,
    )
    out = mod._build_second_pca_kwargs(args)
    assert out == {
        "second_pca_threshold": 0.80,
        "second_pca_subdir": "second_pca_consensus",
        "second_pca_run_sweep": False,
        "second_pca_sweep_thresholds": None,
    }


def test_build_second_pca_kwargs_parses_sweep_thresholds():
    from ops_model.post_process.combination import pca_optimization as mod

    args = SimpleNamespace(
        second_pca=True,
        second_pca_threshold=0.80,
        second_pca_subdir=None,
        second_pca_no_sweep=False,
        second_pca_sweep_thresholds="0.5, 0.7, 0.9",
    )
    out = mod._build_second_pca_kwargs(args)
    assert out["second_pca_run_sweep"] is True
    assert out["second_pca_sweep_thresholds"] == [0.5, 0.7, 0.9]


# ---------------------------------------------------------------------------
# pure helpers — _stored_embedding_seed
# ---------------------------------------------------------------------------


def test_stored_embedding_seed_missing_uns():
    from ops_model.post_process.combination import pca_optimization as mod

    adata = SimpleNamespace(uns={})
    assert mod._stored_embedding_seed(adata, "umap") is None


def test_stored_embedding_seed_no_params():
    from ops_model.post_process.combination import pca_optimization as mod

    adata = SimpleNamespace(uns={"umap": {}})
    assert mod._stored_embedding_seed(adata, "umap") is None


def test_stored_embedding_seed_present():
    from ops_model.post_process.combination import pca_optimization as mod

    adata = SimpleNamespace(uns={"umap": {"params": {"random_state": "7"}}})
    assert mod._stored_embedding_seed(adata, "umap") == 7


def test_stored_embedding_seed_bad_value():
    from ops_model.post_process.combination import pca_optimization as mod

    adata = SimpleNamespace(uns={"umap": {"params": {"random_state": "not-an-int"}}})
    assert mod._stored_embedding_seed(adata, "umap") is None


# ---------------------------------------------------------------------------
# pure helpers — _discover_op_files
# ---------------------------------------------------------------------------


def test_discover_op_files_missing_root(tmp_path):
    from ops_model.post_process.combination import pca_optimization as mod

    missing = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        mod._discover_op_files(str(missing))


def test_discover_op_files_empty_root(tmp_path):
    from ops_model.post_process.combination import pca_optimization as mod

    with pytest.raises(FileNotFoundError):
        mod._discover_op_files(str(tmp_path))


# ---------------------------------------------------------------------------
# pure helpers — _load_chromosome_map
# ---------------------------------------------------------------------------


def _quiet_logger():
    return logging.getLogger("test_pca_opt_refactor")


def test_load_chromosome_map_unreadable_returns_none(tmp_path):
    from ops_model.post_process.combination import pca_optimization as mod

    bogus = tmp_path / "missing.csv"
    out = mod._load_chromosome_map(str(bogus), _quiet_logger())
    assert out is None


def test_load_chromosome_map_legacy_panel_schema(tmp_path):
    from ops_model.post_process.combination import pca_optimization as mod

    csv = tmp_path / "panel.csv"
    pd.DataFrame(
        {
            "perturbation": ["TP53", "BRCA1", "MYC"],
            "chromosome": ["17", "17", "8"],
            "chromosome_arm": ["p", "q", "q"],
        }
    ).to_csv(csv, index=False)
    out = mod._load_chromosome_map(str(csv), _quiet_logger())
    assert out is not None
    assert set(out.index) == {"TP53", "BRCA1", "MYC"}
    assert out.loc["TP53", "chr_arm"] == "17p"
    assert out.loc["BRCA1", "chr_arm"] == "17q"


def test_load_chromosome_map_symbol_schema(tmp_path):
    from ops_model.post_process.combination import pca_optimization as mod

    csv = tmp_path / "symbol.csv"
    pd.DataFrame(
        {
            "symbol": ["TP53", "BRCA1"],
            "chrom_arm": ["chr17p", "chr17q"],
        }
    ).to_csv(csv, index=False)
    out = mod._load_chromosome_map(str(csv), _quiet_logger())
    assert out is not None
    assert out.loc["TP53", "chr_arm"] == "17p"
    assert out.loc["BRCA1", "chr_arm"] == "17q"


def test_load_chromosome_map_unknown_schema_returns_none(tmp_path):
    from ops_model.post_process.combination import pca_optimization as mod

    csv = tmp_path / "weird.csv"
    pd.DataFrame({"foo": [1, 2], "bar": ["a", "b"]}).to_csv(csv, index=False)
    out = mod._load_chromosome_map(str(csv), _quiet_logger())
    assert out is None
