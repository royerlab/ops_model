"""End-to-end test for the titration pipeline.

Self-contained script (run directly, not via pytest). Unlike the other e2e
tests, this one drives the scripts through their **command line**, because the
argparse surface — which flags exist, which are required, how they nest the
output path — is most of what these two scripts do.

    1. build a throwaway variant tree and symlink one real cells.h5ad into its
       per_signal/ dir, so titration.py sees exactly one reporter
    2. run titration.py's CLI -> one per-reporter curve + plots
    3. verify the CSV schema, values, and plot files
    4. run combined_titration.py's CLI with two markers in one group
    5. verify the combined CSV records both reporters

Both runs use the smallest schedule available (per-guide-min for titration,
--max-schedule-points 2 for combined) so the whole test takes ~2.5 minutes.
Nothing is copied: the source h5ads are symlinked read-only and every output
goes to a fresh tmp dir (printed at the end).

Both CLI invocations below state **every** flag the parser accepts, so a flag
that is renamed or removed makes this test fail loudly. Flags that cannot
coexist with the ones under test are listed in EXCLUDED_FLAGS with a reason,
and verify_arg_coverage() asserts the two sets together cover the whole parser.

Run with:
    uv run python tests/e2e_tests/titration_e2e.py
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

from ops_model.post_process.combination.titration import combined_titration as ct
from ops_model.post_process.combination.titration import titration as titr
from ops_model.post_process.combination.titration.titration import METRIC_COLUMNS

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PER_SIGNAL = Path(
    "/hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized_v0.3/cell_dino/"
    "zscore_per_exp/paper_v1/with_cp/with_4i/all_livecell/fixed_80%/cosine/per_signal"
)
REPORTER_A = PER_SIGNAL / "pRb_4i_cells.h5ad"
REPORTER_B = PER_SIGNAL / "p21_4i_cells.h5ad"

SIG_SAFE = "pRb_4i"          # sanitized signal name -> reporter subdir
COMBINED_LABEL = "pRb_p21"   # --group NAME
N_COMBINED_POINTS = 2        # --max-schedule-points

# The variant-selecting flags. Both runs share these, and the tmp tree is built
# at whatever path they resolve to — so this set can be changed freely.
VARIANT_FLAGS = [
    "--cell-dino",
    "--paper-v1",
    "--fixed-threshold", "0.8",
    "--distance", "cosine",
    "--zscore-per-experiment",
    "--norm-method", "ntc",
    "--run-tag", "e2e",
]

# Flags deliberately not exercised, and why. Together with the flags actually
# passed below these must account for every option in both parsers.
EXCLUDED_FLAGS = {
    "-h": "argparse builtin",
    "--help": "argparse builtin",
    "--slurm": "test runs locally; SLURM submission is not exercised",
    "--slurm-memory": "only read when --slurm is set",
    "--slurm-time": "only read when --slurm is set",
    "--phase-slurm-time": "only read when --slurm is set",
    "--slurm-cpus": "only read when --slurm is set",
    "--slurm-partition": "only read when --slurm is set",
    "--per-target-slurm": "requires --slurm; --no-per-target-slurm is passed instead",
    "--replot": "alternate mode that skips scoring; covered by its own path",
    "--compare-only": "alternate mode that skips scoring",
    "--cell-profiler": "mutually exclusive in effect with --cell-dino",
    "--downsampled": "selects a different variant tree than the one built here",
    "--downsample-per-guide": "selects a different variant tree",
    "--include-cellpainting": "selects a different variant tree",
    "--with-4i": "selects a different variant tree",
    "--with-cp": "selects a different variant tree",
    "--only-4i": "selects a different variant tree",
    "--only-cp": "selects a different variant tree",
    "--phase-only": "mutually exclusive with --no-phase; both change the path",
    "--no-phase": "mutually exclusive with --phase-only; both change the path",
    "--no-zscore-per-experiment": "--zscore-per-experiment is passed instead",
    "--min-exp-titration": "one titration mode per run; --per-guide-min-titration used",
    "--per-ko-min-titration": "one titration mode per run",
    "--per-gene-min-titration": "alias of --per-ko-min-titration",
    "--per-ko-max-titration": "one titration mode per run",
    "--per-gene-max-titration": "alias of --per-ko-max-titration",
    "--per-guide-max-titration": "one titration mode per run",
    # combined_titration only
    "--no-shared-start": "single group, so shared-start has no effect",
    "--no-compare": "single group, so the comparison step is skipped anyway",
}


def titration_argv(root: Path) -> list:
    """Full CLI for titration.py — one reporter, smallest schedule."""
    return [
        sys.executable, "-m",
        "ops_model.post_process.combination.titration.titration",
        "-o", str(root),
        *VARIANT_FLAGS,
        "--per-guide-min-titration",   # 3 points on this reporter: [3, 2, 1]
        "--no-per-target-slurm",
        "--no-cache",
        "--bootstrap", "1",
        "--bootstrap-replace",
    ]


def combined_argv(root: Path) -> list:
    """Full CLI for combined_titration.py — two markers in one group."""
    return [
        sys.executable, "-m",
        "ops_model.post_process.combination.titration.combined_titration",
        "-o", str(root),
        *VARIANT_FLAGS,
        "--per-guide-median-titration",
        "--no-per-target-slurm",
        "--no-cache",
        "--bootstrap", "1",
        "--bootstrap-replace",
        "--group", f"{COMBINED_LABEL}={REPORTER_A},{REPORTER_B}",
        "--max-schedule-points", str(N_COMBINED_POINTS),
        "--median-start-policy", "pool",
        "--n-workers", "2",
        "--second-pca-threshold", "0.0",
        "--seed", "42",
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def verify_arg_coverage() -> None:
    """Every option in both parsers is either passed or explicitly excluded."""
    passed = {a for a in titration_argv(Path("/x")) + combined_argv(Path("/x"))
              if a.startswith("--") or a == "-o"}
    passed.add("--output-dir")  # -o is the short form
    for name, parser in (("titration", titr._build_parser()),
                         ("combined_titration", ct._build_parser())):
        available = {opt for a in parser._actions for opt in a.option_strings}
        unaccounted = available - passed - set(EXCLUDED_FLAGS)
        assert not unaccounted, (
            f"{name}: {sorted(unaccounted)} are neither passed by this test nor "
            f"listed in EXCLUDED_FLAGS — add them to one or the other"
        )
    stale = set(EXCLUDED_FLAGS) - {
        opt for p in (titr._build_parser(), ct._build_parser())
        for a in p._actions for opt in a.option_strings
    }
    assert not stale, (
        f"EXCLUDED_FLAGS lists options that no longer exist: {sorted(stale)}"
    )
    print(f"[0] Arg coverage OK: {len(passed)} passed, {len(EXCLUDED_FLAGS)} excluded")


def build_variant_tree(root: Path) -> Path:
    """Symlink one reporter into the per_signal/ dir the flags resolve to."""
    args = titr._build_parser().parse_args(
        ["-o", str(root), *VARIANT_FLAGS, "--per-guide-min-titration"]
    )
    variant = titr._resolve_output_dir(args)
    per_signal = variant / "per_signal"
    per_signal.mkdir(parents=True, exist_ok=True)
    (per_signal / REPORTER_A.name).symlink_to(REPORTER_A)
    print(f"[1] Variant tree: {variant}\n    1 reporter symlinked into per_signal/")
    return variant


def run_cli(argv: list, step: str) -> None:
    """Run a CLI and fail with its tail output if it exits non-zero."""
    result = subprocess.run(argv, capture_output=True, text=True)
    if result.returncode != 0:
        tail = "\n".join((result.stdout + result.stderr).splitlines()[-25:])
        raise AssertionError(f"{step} exited {result.returncode}:\n{tail}")


def verify_metric_columns(df: pd.DataFrame, where: str) -> None:
    """Every metric column present, finite, and mAP strictly positive."""
    missing = [c for c in METRIC_COLUMNS if c not in df.columns]
    assert not missing, f"{where}: missing metric columns {missing}"
    nan_cols = [c for c in METRIC_COLUMNS if df[c].isna().any()]
    assert not nan_cols, f"{where}: NaN in {nan_cols} — scoring silently failed"
    # Ratios can legitimately be 0.0 at tiny cell budgets; mAP should not be.
    maps = [c for c in METRIC_COLUMNS if c.endswith("_map_mean")]
    nonpos = [c for c in maps if not (df[c] > 0).all()]
    assert not nonpos, f"{where}: non-positive mean mAP in {nonpos}"


def verify_titration_outputs(variant: Path) -> None:
    """One reporter curve, its plots, and the all-reporter combined CSV."""
    out = variant / "titration_guide_min"
    csv = out / SIG_SAFE / f"{SIG_SAFE}_titration.csv"
    assert csv.is_file(), f"no per-reporter CSV at {csv}"

    df = pd.read_csv(csv)
    assert len(df) == 3, f"expected 3 titration points ([3,2,1]), got {len(df)}"
    assert sorted(df["cells_per_guide"]) == [1, 2, 3], (
        f"unexpected schedule: {sorted(df['cells_per_guide'])}"
    )
    verify_metric_columns(df, "titration CSV")
    assert df["n_guides"].nunique() == 1, "guide count should not vary across points"
    print(f"[3] Curve OK: {len(df)} points x {df['n_guides'].iloc[0]} guides "
          f"-> {csv.name}")

    # 3 x-axis variants x 3 scales, PNG + SVG each
    plots = list((out / SIG_SAFE).glob(f"{SIG_SAFE}_titration_*.png"))
    svgs = list((out / SIG_SAFE).glob(f"{SIG_SAFE}_titration_*.svg"))
    assert len(plots) == 9, f"expected 9 PNGs (3 x-axes x 3 scales), got {len(plots)}"
    assert len(svgs) == 9, f"expected 9 SVGs, got {len(svgs)}"
    assert (out / "titration_combined.csv").is_file(), "no all-reporter combined CSV"
    print(f"[3] Plots OK: {len(plots)} PNG + {len(svgs)} SVG, "
          f"plus titration_combined.csv")


def verify_combined_outputs(variant: Path) -> None:
    """One combined curve built from two reporters."""
    out = (variant / "combined_titration" / "per_guide_median" / COMBINED_LABEL)
    csv = out / f"combined_titration_{COMBINED_LABEL}.csv"
    assert csv.is_file(), f"no combined CSV at {csv}"

    df = pd.read_csv(csv)
    assert len(df) == N_COMBINED_POINTS, (
        f"expected {N_COMBINED_POINTS} points, got {len(df)}"
    )
    verify_metric_columns(df, "combined CSV")
    assert (df["n_reporters"] == 2).all(), (
        f"combined run should record 2 reporters, got {df['n_reporters'].unique()}"
    )
    assert (df["group"] == COMBINED_LABEL).all(), "group column mislabelled"
    assert len(list(out.glob("*.png"))) == 3, "expected one PNG per scale"
    print(
        f"[5] Combined OK: {len(df)} points, n_reporters=2, "
        f"group={COMBINED_LABEL} -> {csv.name}"
    )


def main() -> None:
    for src in (REPORTER_A, REPORTER_B):
        assert src.is_file(), f"missing source h5ad: {src}"

    tmp = Path(tempfile.mkdtemp(prefix="titration_e2e_"))
    print(f"Working dir: {tmp}\n")

    verify_arg_coverage()
    variant = build_variant_tree(tmp)

    print("[2] Running titration.py ...")
    run_cli(titration_argv(tmp), "titration.py")
    verify_titration_outputs(variant)

    print("[4] Running combined_titration.py ...")
    run_cli(combined_argv(tmp), "combined_titration.py")
    verify_combined_outputs(variant)

    print(f"\n✓ Titration e2e PASSED. Outputs under: {tmp}")


if __name__ == "__main__":
    main()
