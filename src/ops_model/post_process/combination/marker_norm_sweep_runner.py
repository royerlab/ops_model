"""Picklable worker for marker_normalization_sweep SLURM fanout.

The renderer lives in ``organelle_profiler/scripts/marker_normalization_sweep.py``
(a scripts/ file, not in an installed package), so submitit can't pickle it
directly. This thin wrapper subprocesses the script with the right CLI args
so it can be dispatched via ``submit_parallel_jobs``.
"""
from __future__ import annotations

import subprocess
import sys
from typing import List


SCRIPT = (
    "/hpc/mydata/gav.sturm/ops_mono/organelle_profiler/scripts/"
    "marker_normalization_sweep.py"
)


def run_marker_norm_sweep_subset(cli_args: List[str], label: str = "") -> str:
    """Run one (kind, sim_metric, strategy) shard of the marker norm sweep.

    Parameters
    ----------
    cli_args : list of str
        Extra CLI args appended to the script invocation. Should include
        ``--feature-kinds <kind>``, ``--sim-metrics <sim>``, ``--strategies
        <strategy>``, and ``--out-dir <path>``. Do NOT pass ``--slurm``.
    label : str
        Short tag for log lines.
    """
    cmd = [sys.executable, SCRIPT, *cli_args]
    print(f"[marker_norm_sweep] label={label}")
    print(f"[marker_norm_sweep] cmd: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    return "OK"
