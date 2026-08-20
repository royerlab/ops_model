"""Launch 5 expansion sweeps in parallel, one per percentile P ∈ {10,20,30,40,50},
each running the standard `random_low_removed_intersection` direction with a
matched P-tagged EBI parquet (second head) and `--intersection-percentile P`
(geneKO threshold). Each writes to its own `map_attention_expansion_strict_corrected_ebi_p{P}/`
subdir reusing the existing per-experiment cache.

After all 5 sweeps finish, build the percentile-comparison plots:
  - expansion_curves.png — 6 lines per metric (random + P=10/20/30/40/50), viridis colors
  - peak_bars.png        — bar at each curve's peak K, viridis, no on-bar text

Usage:
    python run_percentile_sweep.py             # launch + wait + plot
    python run_percentile_sweep.py --plot-only # skip launches, just regenerate plots
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

CDINO_DIR = Path(
    "/hpc/projects/icd.fast.ops/models/alex_lin_attention/v3/attention_v3/cdino"
)
PARQUET_DIR = Path(
    "/hpc/projects/icd.fast.ops/models/alex_lin_attention/v4"
)
RANDOM_CONTROL_CSV = (
    CDINO_DIR / "map_attention_expansion_strict_corrected/"
                "map_attention_expansion_strict_corrected.csv"
)
PERCENTILES = [10, 20, 30, 40, 50]
SCRIPT = Path(__file__).resolve().parent / "map_attention_decay.py"


def launch_one(P: int, log_dir: Path, dry_run: bool = False) -> int:
    """Launch the expansion sweep for one percentile as a detached background
    process whose stdout/stderr go to a log file. Returns the launch PID.
    Detaching (start_new_session=True) lets the orchestrator survive after
    this parent script exits.
    """
    parquet = PARQUET_DIR / f"pma_phase_cells_ebi_p{P:02d}.parquet"
    if not parquet.exists():
        raise FileNotFoundError(
            f"Missing parquet for P={P}: {parquet}. Generate it first via "
            f"convert_ebi_pma_to_parquet.py --percentiles {P}"
        )
    cmd = [
        sys.executable, str(SCRIPT),
        "--mode", "expansion",
        "--strict-cache",
        "--apply-iss-sidecar",
        "--pca-cache-only",
        "--pma-parquet-chad", str(parquet),
        "--intersection-percentile", str(P),
        "--run-suffix", f"ebi_p{P:02d}",
        "--no-chad",
    ]
    print(f"[P={P}%] {' '.join(cmd)}")
    if dry_run:
        return 0
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"orchestrator_p{P:02d}.log"
    log_fh = open(log_path, "w")
    proc = subprocess.Popen(
        cmd, stdout=log_fh, stderr=subprocess.STDOUT,
        start_new_session=True,  # detach so we survive parent exit
    )
    print(f"[P={P}%] launched PID={proc.pid}, log={log_path}")
    return proc.pid


def percentile_csv(P: int) -> Path:
    name = f"map_attention_expansion_strict_corrected_ebi_p{P:02d}"
    return CDINO_DIR / name / f"{name}.csv"


def _load_random_control() -> pd.DataFrame:
    if not RANDOM_CONTROL_CSV.exists():
        print(f"WARN: random control CSV not found at {RANDOM_CONTROL_CSV}")
        return pd.DataFrame()
    df = pd.read_csv(RANDOM_CONTROL_CSV)
    return df[(df["rank_type"] == "random") &
              (df["source"] == "geneKO_fast")].sort_values("rank_hi").copy()


def _load_intersection(P: int) -> pd.DataFrame:
    csv = percentile_csv(P)
    if not csv.exists():
        print(f"WARN: P={P} CSV missing at {csv}")
        return pd.DataFrame()
    df = pd.read_csv(csv)
    sub = df[(df["rank_type"] == "random_low_removed_intersection") &
             (df["source"] == "geneKO_fast")].sort_values("rank_hi").copy()
    sub["P"] = P
    return sub


def plot(out_dir: Path, baselines: dict) -> None:
    import matplotlib.pyplot as plt
    plt.rcParams["pdf.fonttype"] = 42
    from matplotlib import cm

    df_random = _load_random_control()
    frames = [f for f in (_load_intersection(P) for P in PERCENTILES)
              if not f.empty]
    if not frames:
        print("\nNo per-percentile CSVs yet — sweeps still running. "
              "Re-run with --plot-only after they finish.")
        return
    df_p = pd.concat(frames, ignore_index=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    df_p.to_csv(out_dir / "intersection_all_percentiles.csv", index=False)

    cmap = cm.get_cmap("viridis")
    Ps = sorted(df_p["P"].unique().tolist())
    p_colors = {P: cmap(0.15 + 0.7 * i / max(1, len(Ps) - 1))
                for i, P in enumerate(Ps)}
    rnd_color = "#666666"

    metrics = [
        ("EBI complex consistency mAP",  "mean_map_pca",       baselines["ebi"]),
        ("CHAD complex consistency mAP", "mean_map_chad_pca",  baselines["chad"]),
        ("Phenotypic distinctiveness mAP","mean_map_dist_pca", baselines["dist"]),
    ]

    # Drop count per P — diff vs random at max K.
    drop_counts = {}
    if not df_random.empty:
        kmax = float(df_random["rank_hi"].max())
        rmax_n = float(df_random[df_random["rank_hi"] == kmax]["n_cells"].iloc[0])
        for P in Ps:
            p_df = df_p[df_p["P"] == P]
            if p_df.empty:
                continue
            p_at_kmax = p_df[p_df["rank_hi"] == kmax]
            if p_at_kmax.empty:
                continue
            drop = int(round(rmax_n - float(p_at_kmax["n_cells"].iloc[0])))
            drop_counts[P] = max(0, drop)

    # ─────────── Expansion curves ───────────
    fig, axes = plt.subplots(1, 3, figsize=(22, 6.8), sharey=False)
    for ax, (title, col, base) in zip(axes, metrics):
        if not df_random.empty and col in df_random.columns:
            ax.plot(df_random["rank_hi"], df_random[col],
                    "o-", color=rnd_color, lw=2.4, ms=6,
                    markeredgecolor="black", markeredgewidth=0.4,
                    label="random (no filter)", zorder=4)
        for P in Ps:
            sub = df_p[df_p["P"] == P]
            if sub.empty or col not in sub.columns:
                continue
            drop = drop_counts.get(P, 0)
            drop_lbl = (f"−{drop/1e6:.1f}M" if drop >= 1e6
                        else f"−{drop/1e3:.0f}k" if drop >= 1e3
                        else f"−{drop:d}")
            ax.plot(sub["rank_hi"], sub[col], "s--",
                    color=p_colors[P], lw=2.0, ms=5,
                    markeredgecolor="black", markeredgewidth=0.4,
                    label=f"intersection P={P}% ({drop_lbl} cells)",
                    zorder=5)
        ax.axhline(base, ls=":", color="#d62728", lw=1.6, alpha=0.85,
                   label=f"all-cells baseline = {base:.3f}", zorder=3)
        ax.set_xscale("log")
        ax.set_xlabel("Cells per gene (cumulative K)", fontsize=13, labelpad=10)
        ax.set_ylabel(title, fontsize=13)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=9, loc="lower right", framealpha=0.92)
    fig.suptitle("Expansion curves — 2-of-2 (geneKO ∩ EBI) low-attention drop at varying percentile",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = out_dir / f"expansion_curves.{ext}"
        fig.savefig(out, dpi=150 if ext == "png" else None, bbox_inches="tight")
        print(f"Wrote {out}")
    plt.close(fig)

    # ─────────── Peak-K bars per group ───────────
    fig, axes = plt.subplots(1, 3, figsize=(20, 7.0), sharey=False)
    for ax, (title, col, base) in zip(axes, metrics):
        vals = []
        labels = []
        bar_colors = []
        if not df_random.empty and col in df_random.columns:
            r_peak_idx = df_random[col].idxmax()
            K_peak = int(df_random.loc[r_peak_idx, "rank_hi"])
            v = float(df_random.loc[r_peak_idx, col])
            vals.append(v); labels.append(f"random\n(no filter)\npeak K={K_peak//1000}k")
            bar_colors.append(rnd_color)
        for P in Ps:
            sub = df_p[df_p["P"] == P]
            if sub.empty or col not in sub.columns:
                continue
            i = sub[col].idxmax()
            K = int(sub.loc[i, "rank_hi"])
            v = float(sub.loc[i, col])
            drop = drop_counts.get(P, 0)
            drop_lbl = (f"−{drop/1e6:.1f}M" if drop >= 1e6
                        else f"−{drop/1e3:.0f}k" if drop >= 1e3
                        else f"−{drop:d}")
            vals.append(v)
            labels.append(f"P={P}%\npeak K={K//1000 if K>=1000 else K}{'k' if K>=1000 else ''}\n{drop_lbl} cells")
            bar_colors.append(p_colors[P])
        x = np.arange(len(vals))
        ax.bar(x, vals, color=bar_colors, edgecolor="black", linewidth=0.5)
        ax.axhline(base, ls=":", color="#d62728", lw=1.6, alpha=0.85,
                   label=f"all-cells = {base:.3f}", zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9.5)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylabel("peak mAP", fontsize=12)
        ax.set_xlabel("group", fontsize=12, labelpad=14)
        ax.tick_params(axis="x", pad=6)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=9, loc="lower right")
        ax.margins(x=0.04)
    fig.suptitle("Peak mAP per group — 2-of-2 (geneKO ∩ EBI) drop at varying percentile",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = out_dir / f"peak_bars.{ext}"
        fig.savefig(out, dpi=150 if ext == "png" else None, bbox_inches="tight")
        print(f"Wrote {out}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path,
                    default=CDINO_DIR / "intersection_percentile_sweep")
    ap.add_argument("--plot-only", action="store_true",
                    help="Skip launches, just regenerate plots from existing CSVs.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the launch commands without submitting.")
    ap.add_argument("--baseline-ebi",  type=float, default=0.5094)
    ap.add_argument("--baseline-chad", type=float, default=0.5145)
    ap.add_argument("--baseline-dist", type=float, default=0.4686)
    args = ap.parse_args()

    if not args.plot_only:
        # Launch all 5 sweeps; each is its own detached orchestrator that
        # submits its own SLURM array and waits on it. Detached so they
        # survive this parent exiting. Logs at args.out_dir/orchestrator_logs/.
        pids = []
        log_dir = args.out_dir / "orchestrator_logs"
        for P in PERCENTILES:
            pid = launch_one(P, log_dir, dry_run=args.dry_run)
            pids.append(pid)
        if args.dry_run:
            print(f"[dry-run] would launch {len(pids)} orchestrators")
            return
        print(f"\n[parallel] {len(pids)} orchestrators backgrounded: {pids}")
        print(f"[parallel] each is its own python process running its own "
              f"SLURM array. Wait at the shell, then re-run with --plot-only.")
        return  # don't try to plot — results aren't ready

    plot(args.out_dir,
         {"ebi": args.baseline_ebi,
          "chad": args.baseline_chad,
          "dist": args.baseline_dist})


if __name__ == "__main__":
    main()
