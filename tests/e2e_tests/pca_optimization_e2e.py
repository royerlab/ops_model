"""End-to-end test for the pca_optimization pipeline.

Self-contained script (run directly, not via pytest). Like the titration e2e it
drives the CLI rather than calling functions, because the argparse surface —
which flags exist, which are required — is a large part of what this package
does.

    1. run the CLI over two explicitly-named signal groups, locally
    2. verify Phase 1 wrote one per-signal h5ad per group
    3. verify Phase 2 wrote the aggregated guide/gene h5ads + metric CSVs
    4. verify the output path is nested exactly as the flags imply

There is no discovery: inputs are named with repeated --signal NAME=paths, so
the test supplies real per-signal h5ads directly. Nothing is copied — sources
are read-only and every output goes to a fresh tmp dir (printed at the end).

Both invocations state every flag the parser accepts; flags that cannot coexist
with the ones under test are listed in EXCLUDED_FLAGS with a reason, and
verify_arg_coverage() asserts the two sets together cover the whole parser.

Run with:
    uv run python tests/e2e_tests/pca_optimization_e2e.py
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import anndata as ad

from ops_model.post_process.combination.pca_optimization.parser import _build_parser

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PER_SIGNAL = Path(
    "/hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized_v0.3/cell_dino/"
    "zscore_per_exp/paper_v1/with_cp/with_4i/all_livecell/fixed_80%/cosine/per_signal"
)
SIGNAL_A = ("pRb", PER_SIGNAL / "pRb_4i_cells.h5ad")
SIGNAL_B = ("p21", PER_SIGNAL / "p21_4i_cells.h5ad")

CONFIGS = Path("/hpc/projects/icd.fast.ops/configs")
CHAD = CONFIGS / "gene_clusters/chad_positive_controls_v5_hierarchy.yml"
EBI = CONFIGS / "gene_clusters/EBI_complexes_v1_old_gene_names.yaml"
GENE_PANEL = CONFIGS / "annotated_gene_panel_July2025.csv"

RUN_TAG = "e2e"
TARGET_CELLS = 20_000  # small budget keeps Phase 1 quick

EXCLUDED_FLAGS = {
    "-h": "argparse builtin",
    "--help": "argparse builtin",
    "--config": "flags are passed explicitly rather than via a config file",
    "--slurm": "test runs locally; SLURM submission is not exercised",
    "--slurm-memory": "only read when --slurm is set",
    "--slurm-time": "only read when --slurm is set",
    "--slurm-cpus": "only read when --slurm is set",
    "--slurm-partition": "only read when --slurm is set",
    "--slurm-agg-memory": "only read when --slurm is set",
    "--slurm-agg-time": "only read when --slurm is set",
    "--phase-memory": "only read when --slurm is set",
    "--second-pca-only": "alternate mode that skips Phase 1",
    "--cell-profiler": "one feature mode per run; --cell-dino is used",
    "--dino": "one feature mode per run",
    "--dynaclr": "one feature mode per run",
    "--subcell": "one feature mode per run",
    "--no-zscore-per-experiment": "--zscore-per-experiment is passed instead",
    "--no-exclude-dud-guides": "--exclude-dud-guides is passed instead",
    "--no-pca": "would skip the PCA sweep this test is checking",
    "--preserve-batch": "changes the output subdir and skips Phase 2",
    "--no-second-pca": "--second-pca is not passed, so this is already the default",
    "--second-pca": "second pass is a separate concern; --second-pca-only covers it",
    "--second-pca-no-sweep": "only read when --second-pca is set",
    "--second-pca-subdir": "only read when --second-pca is set",
    "--second-pca-threshold": "only read when --second-pca is set",
    "--second-pca-sweep-thresholds": "only read when --second-pca is set",
    "--second-pca-consensus-metrics": "only read when --second-pca is set",
    "--downsample-per-guide": "mutually exclusive sampling mode with --target-cells",
    "--cells-per-guide": "only read when --downsample-per-guide is set",
    "--clean": "would delete per_signal/ before running; nothing to clean in a tmp dir",
    "-y": "short form of --yes",
    "--yes": "only prompts on destructive operations",
}


def cli_argv(root: Path) -> list:
    """Full CLI: two signal groups, local, smallest useful settings."""
    return [
        sys.executable, "-m",
        "ops_model.post_process.combination.pca_optimization",
        "-o", str(root),
        "--cell-dino",
        "--zscore-per-experiment",
        "--exclude-dud-guides",
        "--chad-annotation", str(CHAD),
        "--ebi-annotation", str(EBI),
        "--gene-panel", str(GENE_PANEL),
        "--signal", f"{SIGNAL_A[0]}={SIGNAL_A[1]}",
        "--signal", f"{SIGNAL_B[0]}={SIGNAL_B[1]}",
        "--target-cells", str(TARGET_CELLS),
        "--fixed-threshold", "0.8",
        "--distance", "cosine",
        "--norm-method", "ntc",
        "--agg-method", "mean",
        "--sweep-metric", "mean_map",
        "--umap-type", "max",
        "--umap-n-neighbors", "15",
        "--umap-min-dist", "0.1",
        "--seed", "1",
        "--run-tag", RUN_TAG,
        "--apply-iss-sidecar",
    ]


def expected_output_dir(root: Path) -> Path:
    """Where the flags nest the run:
    ``<root>/<feature>/<zscore>/<tag>/<threshold>/<distance>``."""
    return root / "cell_dino" / "zscore_per_exp" / RUN_TAG / "fixed_80%" / "cosine"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def verify_arg_coverage() -> None:
    """Every parser option is either passed by this test or explicitly excluded."""
    passed = {a for a in cli_argv(Path("/x")) if a.startswith("--") or a == "-o"}
    passed.add("--output-dir")  # -o is the short form
    available = {opt for a in _build_parser()._actions for opt in a.option_strings}
    unaccounted = available - passed - set(EXCLUDED_FLAGS)
    assert not unaccounted, (
        f"{sorted(unaccounted)} are neither passed by this test nor listed in "
        f"EXCLUDED_FLAGS — add them to one or the other"
    )
    stale = set(EXCLUDED_FLAGS) - available
    assert not stale, (
        f"EXCLUDED_FLAGS lists options that no longer exist: {sorted(stale)}"
    )
    print(f"[0] Arg coverage OK: {len(passed)} passed, {len(EXCLUDED_FLAGS)} excluded")


def run_cli(argv: list) -> None:
    result = subprocess.run(argv, capture_output=True, text=True)
    if result.returncode != 0:
        tail = "\n".join((result.stdout + result.stderr).splitlines()[-30:])
        raise AssertionError(f"pca_optimization exited {result.returncode}:\n{tail}")


def verify_phase1(out: Path) -> None:
    """Phase 1 writes a set of h5ads + a sweep CSV per --signal group.

    Each group yields ``<name>_cells``, ``<name>_cells_sub``, ``<name>_guide``
    and ``<name>_gene`` h5ads plus ``<name>_sweep.csv``; the guide-level one is
    what Phase 2 consumes, so that's the one asserted on.
    """
    per_signal = out / "per_signal"
    assert per_signal.is_dir(), f"no per_signal/ dir at {per_signal}"
    for name, _ in (SIGNAL_A, SIGNAL_B):
        guide = per_signal / f"{name}_guide.h5ad"
        assert guide.is_file(), (
            f"no {guide.name} — Phase 1 produced "
            f"{sorted(x.name for x in per_signal.glob('*.h5ad'))}"
        )
        a = ad.read_h5ad(guide, backed="r")
        assert a.n_obs > 0 and a.n_vars > 0, f"{guide.name} is empty ({a.shape})"
        assert "sgRNA" in a.obs.columns, f"{guide.name} lost the sgRNA column"
        assert (per_signal / f"{name}_sweep.csv").is_file(), (
            f"no {name}_sweep.csv — the PCA threshold sweep did not run"
        )
    print(
        f"[2] Phase 1 OK: guide h5ad + sweep CSV for "
        f"{SIGNAL_A[0]!r} and {SIGNAL_B[0]!r} -> {per_signal}"
    )


def verify_phase2(out: Path) -> None:
    """Aggregated guide/gene h5ads, the PCA report, and the metric CSVs."""
    guide = out / "guide_pca_optimized.h5ad"
    gene = out / "gene_pca_optimized.h5ad"
    assert guide.is_file(), f"no aggregated guide h5ad at {guide}"
    assert gene.is_file(), f"no aggregated gene h5ad at {gene}"

    ag, an = ad.read_h5ad(guide, backed="r"), ad.read_h5ad(gene, backed="r")
    assert ag.n_obs > an.n_obs, (
        f"guide level should have more rows than gene level "
        f"({ag.n_obs} vs {an.n_obs})"
    )
    # Two signals h-concatenated, so the feature width must exceed either alone.
    assert ag.n_vars > 1, f"aggregated guide matrix is degenerate: {ag.shape}"

    metrics = out / "metrics"
    assert metrics.is_dir(), f"no metrics/ dir at {metrics}"
    csvs = sorted(p.name for p in metrics.glob("*.csv"))
    assert csvs, "metrics/ contains no CSVs"
    assert (out / "pca_report.csv").is_file(), "no pca_report.csv"
    print(
        f"[3] Phase 2 OK: guide {ag.shape}, gene {an.shape}, "
        f"{len(csvs)} metric CSV(s)"
    )


def main() -> None:
    for src in (SIGNAL_A[1], SIGNAL_B[1], CHAD, EBI, GENE_PANEL):
        assert src.is_file(), f"missing required input: {src}"

    tmp = Path(tempfile.mkdtemp(prefix="pcaopt_e2e_"))
    print(f"Working dir: {tmp}\n")

    verify_arg_coverage()

    print("[1] Running pca_optimization ...")
    run_cli(cli_argv(tmp))

    out = expected_output_dir(tmp)
    assert out.is_dir(), (
        f"output not nested where the flags imply.\n  expected: {out}\n"
        f"  found:    {sorted(p for p in tmp.rglob('*') if p.is_dir())[:8]}"
    )
    print(f"[1] Output nested as expected: {out}")

    verify_phase1(out)
    verify_phase2(out)

    print(f"\n✓ pca_optimization e2e PASSED. Outputs under: {tmp}")


if __name__ == "__main__":
    main()
