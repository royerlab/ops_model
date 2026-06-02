"""Run all 12 SHAP approaches in one command.

Wraps `run_shap_pipeline.py` and fires it for each combination of:
    variant   ∈ {top-attention, all-cells}
    contrast  ∈ {distinct, ntc, global}
    grain     ∈ {gene, complex}
= 12 runs total.

Each child invocation is its OWN SLURM-array submission (200 shards by
default), so the wall-clock cost of 12 sequential runs is roughly
12× a single-run wall-clock if --parallel is off. With --parallel,
all 12 child pipelines submit their SLURM arrays concurrently and the
cluster scheduler interleaves them — useful when QoS limits aren't
the bottleneck.

Usage:
  # All 12, sequential (default). Each waits for its SLURM array to finish
  # before the next is submitted.
  python organelle_profiler/scripts/ko_shap/run_all_shap.py

  # All 12, parallel — fires 12 background processes, each managing its
  # own SLURM array.
  python organelle_profiler/scripts/ko_shap/run_all_shap.py --parallel

  # Subset (filters compose):
  python ... --variants top-attention --grains complex      # 3 runs
  python ... --contrasts distinct --grains gene             # 2 runs

  # Dry-run prints the 12 commands without submitting anything.
  python ... --dry-run
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PIPELINE = REPO / "organelle_profiler" / "scripts" / "ko_shap" / "run_shap_pipeline.py"

VARIANTS  = ("top-attention", "all-cells")
CONTRASTS = ("distinct", "ntc", "global")
GRAINS    = ("gene", "complex")


def _build_cmd(variant: str, contrast: str, grain: str,
                 no_resume: bool, extra: list[str]) -> list[str]:
    cmd = [
        sys.executable, str(PIPELINE),
        "--variant", variant,
        "--contrast", contrast,
        "--aggregation-level", grain,
    ]
    if no_resume:
        cmd.append("--no-resume")
    cmd.extend(extra)
    return cmd


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--variants",  nargs="+", choices=VARIANTS,  default=list(VARIANTS),
                    help="Subset of variants to run. Default: both.")
    p.add_argument("--contrasts", nargs="+", choices=CONTRASTS, default=list(CONTRASTS),
                    help="Subset of contrasts to run. Default: all 3.")
    p.add_argument("--grains",    nargs="+", choices=GRAINS,    default=list(GRAINS),
                    help="Subset of aggregation levels. Default: both.")
    p.add_argument("--parallel",  action="store_true",
                    help="Fire all child pipelines as background subprocesses "
                         "(each manages its own SLURM array). Off → run sequentially.")
    p.add_argument("--no-resume", action="store_true", default=True,
                    help="Pass --no-resume to each child (default: True so bug "
                         "fixes don't get masked by stale shards).")
    p.add_argument("--resume", dest="no_resume", action="store_false",
                    help="Disable --no-resume; child pipelines may skip genes "
                         "already in existing shard CSVs.")
    p.add_argument("--dry-run", action="store_true",
                    help="Print the 12 commands without submitting any.")
    p.add_argument("--log-dir", type=Path, default=None,
                    help="If set, redirect each child's stdout/stderr to "
                         "<log-dir>/<variant>_<contrast>_<grain>.log. "
                         "Default: print to this process's stdout.")
    p.add_argument(
        "extra", nargs=argparse.REMAINDER,
        help="Any args after '--' are forwarded verbatim to each child "
             "(e.g. -- --n-shards 50 --mem 128GB).",
    )
    args = p.parse_args()

    # argparse keeps the '--' separator as first element of REMAINDER
    extra = list(args.extra)
    if extra and extra[0] == "--":
        extra = extra[1:]

    combos = [
        (v, c, g)
        for v in args.variants
        for c in args.contrasts
        for g in args.grains
    ]
    print(f"Planned: {len(combos)} runs "
          f"({len(args.variants)} variants × {len(args.contrasts)} contrasts × "
          f"{len(args.grains)} grains)")
    for v, c, g in combos:
        print(f"  • {v:<14s}  contrast={c:<8s}  grain={g}")
    if args.dry_run:
        print("\n[DRY RUN] Commands that would be submitted:\n")
        for v, c, g in combos:
            print("  $ " + " ".join(_build_cmd(v, c, g, args.no_resume, extra)))
        return

    if args.log_dir is not None:
        args.log_dir.mkdir(parents=True, exist_ok=True)

    # results: list of (tag, exit_code, elapsed_sec, log_path or None).
    # Tracked uniformly across sequential + parallel modes so the final
    # summary + exit code reflect every run.
    results: list[tuple[str, int, float, Path | None]] = []
    procs: list[tuple[str, subprocess.Popen, "object | None", Path | None, float]] = []
    t_start = time.time()

    for i, (v, c, g) in enumerate(combos, start=1):
        cmd = _build_cmd(v, c, g, args.no_resume, extra)
        tag = f"{v}_{c}_{g}"
        banner = f"\n{'='*72}\n[{i:>2d}/{len(combos)}] {tag}\n{'='*72}"
        log_path: Path | None = None
        if args.log_dir is not None:
            log_path = args.log_dir / f"{tag}.log"
            log_fh = open(log_path, "w")
            log_fh.write(banner + "\n" + " ".join(cmd) + "\n\n")
            log_fh.flush()
            stdout, stderr = log_fh, subprocess.STDOUT
            print(f"[{i:>2d}/{len(combos)}] {tag}  → tail -f {log_path}",
                  flush=True)
        else:
            print(banner, flush=True)
            print("$ " + " ".join(cmd), flush=True)
            log_fh = None
            stdout, stderr = None, None

        t_child = time.time()
        if args.parallel:
            proc = subprocess.Popen(cmd, stdout=stdout, stderr=stderr,
                                     env=os.environ.copy())
            procs.append((tag, proc, log_fh, log_path, t_child))
        else:
            ret = subprocess.run(cmd, stdout=stdout, stderr=stderr).returncode
            elapsed_child = time.time() - t_child
            if log_fh is not None:
                log_fh.close()
            results.append((tag, ret, elapsed_child, log_path))
            status = "OK" if ret == 0 else f"FAILED(exit={ret})"
            print(f"\n[{i:>2d}/{len(combos)}] {tag}  {status}  "
                  f"({elapsed_child/60:.1f} min)", flush=True)

    if args.parallel:
        print(f"\nWaiting on {len(procs)} parallel child pipelines...",
              flush=True)
        for tag, proc, log_fh, log_path, t_child in procs:
            ret = proc.wait()
            elapsed_child = time.time() - t_child
            if log_fh is not None:
                log_fh.close()
            results.append((tag, ret, elapsed_child, log_path))
            status = "OK" if ret == 0 else f"FAILED(exit={ret})"
            print(f"  [{tag}] {status}  ({elapsed_child/60:.1f} min)",
                  flush=True)

    elapsed = time.time() - t_start
    failed = [r for r in results if r[1] != 0]
    ok = [r for r in results if r[1] == 0]
    print(f"\n{'='*72}\nSUMMARY: {len(ok)} OK, {len(failed)} FAILED "
          f"of {len(results)} runs in {elapsed/60:.1f} min total\n{'='*72}")
    for tag, ret, elap, log_path in sorted(results, key=lambda r: r[1] != 0):
        status = "  OK   " if ret == 0 else f" FAIL({ret})"
        log_hint = f"  log: {log_path}" if log_path is not None else ""
        print(f"  {status}  {tag:<40s}  {elap/60:>5.1f} min{log_hint}")
    if failed:
        # Write a summary file next to the logs so the user can resume
        # only the failed ones (re-invoke with --variants/--contrasts/--grains
        # filters matching the FAILED list).
        if args.log_dir is not None:
            summary_path = args.log_dir / "FAILED.txt"
            summary_path.write_text(
                "# Failed (variant, contrast, grain) tuples — re-run with:\n"
                "# python run_all_shap.py --variants <v> --contrasts <c> --grains <g>\n"
                + "\n".join(t for t, _, _, _ in failed)
            )
            print(f"\nFAILED list: {summary_path}")
        # Non-zero exit so CI / monitoring catches the failure.
        sys.exit(1)
    print("\nAll runs OK.")


if __name__ == "__main__":
    main()
