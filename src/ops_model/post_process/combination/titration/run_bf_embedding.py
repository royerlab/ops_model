"""Embedding step for the BF-slice titration pipeline.

Runs the *standard* pca_optimization embedding process — phase1 pooled PCA
(`pca_sweep_pooled_signal`) + phase2 aggregation (auto-chained by
`_submit_phase1_slurm`) — on the per-slice Cell-DINO features, treating each
channel (Focus3D, BF_z0..BF_z6) as its own signal. The pooling, per-experiment
z-score, and PCA are reused verbatim (no duplication).

Two bespoke pieces are unavoidable because the channel→biological-signal maps
don't know these channels:
  1. CSV → per-channel cell h5ad. The standard naming (`get_biological_signal`)
     collapses Focus3D/BF_z* to 'unknown'; we instead write
     `features_processed_<channel>.h5ad` (reusing `create_adata_object_embedding`
     for the CSV→AnnData conversion). `find_cell_h5ad_path` then resolves each
     channel via its raw-channel fallback (no `_unknown.h5ad` is written).
  2. signal_groups = {channel: [(exp, channel), ...]} — one signal per channel,
     bypassing `build_signal_groups`' biological collapse.

Phase2D is NOT processed here: the production Phase curve is reused as-is
(symlinked into the variant per_signal dir by the orchestrator).
"""

import argparse
from pathlib import Path
from types import SimpleNamespace

# Per-experiment feature dir holding the BF-titration Cell-DINO CSVs + h5ads
# (sibling to the production `cell_dino_features`, under <exp>/3-assembly/).
FEATURE_DIR = "cell_dino_features_bftitr"


def exp_feature_dir(experiment):
    from ops_model.data.paths import OpsPaths
    return Path(OpsPaths(experiment).base) / experiment / "3-assembly" / FEATURE_DIR


def convert_csvs_worker(experiment, channels):
    """CSV → features_processed_<channel>.h5ad per channel (no PCA; SLURM worker)."""
    import yaml
    from ops_model.features.evaluate_embeddings import create_adata_object_embedding
    from ops_utils.data.experiment import OpsDataset
    # cell_type (required by the validator) comes from the experiment's cell_line.
    cell_type = "A549"
    cfg_path = OpsDataset(experiment).config_paths["exp_config"]
    if cfg_path.exists():
        try:
            cell_type = yaml.safe_load(open(cfg_path)).get("cell_line") or "A549"
        except Exception:
            pass
    fdir = exp_feature_dir(experiment)
    out_dir = fdir / "anndata_objects"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for ch in channels:
        out = out_dir / f"features_processed_{ch}.h5ad"
        if out.exists():  # idempotent: skip already-converted channels on rerun
            print(f"  [skip] {out} exists")
            written.append(str(out))
            continue
        csv = fdir / f"cell_dino_features_{ch}.csv"
        if not csv.exists():
            print(f"  [WARN] missing {csv}; skipping {ch} for {experiment}")
            continue
        adata = create_adata_object_embedding(
            str(csv), channel=ch, experiment=experiment, cell_type=cell_type,
            embedding_type="cell_dino")
        adata.write_h5ad(out)
        written.append(str(out))
        print(f"  wrote {out}")
    return written


def _build_args(norm_method, distance, fixed_threshold,
                slurm_memory, slurm_time, slurm_cpus, slurm_partition):
    """Minimal argparse-like namespace consumed by _submit_phase1_slurm + job kwargs."""
    return SimpleNamespace(
        norm_method=norm_method, distance=distance, fixed_threshold=fixed_threshold,
        zscore_per_experiment=True, slurm=True,
        slurm_time=slurm_time, slurm_memory=slurm_memory, slurm_cpus=slurm_cpus,
        slurm_partition=slurm_partition,
        slurm_agg_time=slurm_time, slurm_agg_memory=slurm_memory,
        second_pca=False, seed=42, agg_method="mean", chromosome_csv=None,
        umap_type="max", second_pca_consensus_metrics=None, sweep_metric="mean_map",
        preserve_batch=False, no_pca=False, exclude_dud_guides=True,
        downsample_per_guide=False,
    )


def run_embedding(experiments, channels, variant_dir, norm_method="ntc",
                  distance="cosine", fixed_threshold=0.8,
                  # Every BF/Focus3D signal pools all cells across all experiments
                  # (~60M cells x 1024), so it lands in pca_optimization's >4M-cell
                  # high-memory tier — matching its phase_memory=600GB (the value
                  # that built prod's 59.7M-cell Phase). 200GB OOMs on the z-score copy.
                  slurm_memory="600GB", slurm_time=480, slurm_cpus=8,
                  slurm_partition="cpu,gpu", convert_only=False):
    """CSV→h5ad (SLURM), then phase1 pooled PCA + phase2 aggregation via _submit_phase1_slurm."""
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    from ops_model.post_process.combination.pca_optimization import (
        pca_sweep_pooled_signal, load_attribution_config, get_storage_roots,
        get_channel_maps_path, count_cells_per_signal_group)
    from ops_model.post_process.combination.pca_optimization.slurm import _submit_phase1_slurm

    variant_dir = Path(variant_dir)
    variant_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. CSV → cell h5ads, one SLURM job per (experiment, channel) ---
    # Each channel is independent (own features_processed_<ch>.h5ad), so fan out
    # per (exp, channel) for 8x parallelism. Skip already-converted pairs at
    # submission time so reruns only launch jobs for the missing/failed ones.
    total = len(experiments) * len(channels)
    conv_jobs = [{
        "name": f"bf_csv2h5ad_{exp}_{ch}",
        "func": convert_csvs_worker,
        "kwargs": {"experiment": exp, "channels": [ch]},
        "metadata": {"experiment": exp, "channel": ch},
    } for exp in experiments for ch in channels
        if not (exp_feature_dir(exp) / "anndata_objects" / f"features_processed_{ch}.h5ad").exists()]
    print(f"\n[embedding] CSV→h5ad: {total - len(conv_jobs)}/{total} already done, "
          f"{len(conv_jobs)} to convert")
    if conv_jobs:
        conv = submit_parallel_jobs(
            jobs_to_submit=conv_jobs, experiment="bf_titration",
            slurm_params={"timeout_min": 60, "mem": "48G", "cpus_per_task": 4,
                          "slurm_partition": "cpu,gpu"},
            log_dir="run_bf_titration/csv2h5ad", manifest_prefix="bf_csv2h5ad",
            wait_for_completion=True)
        if conv.get("failed"):
            raise RuntimeError(f"CSV→h5ad failed: {conv['failed']}")
    if convert_only:
        return

    # --- 2. signal_groups (one signal per channel) + cell-count pre-scan ---
    signal_groups = {ch: [(exp, ch) for exp in experiments] for ch in channels}
    storage_roots = get_storage_roots(load_attribution_config())
    maps_path = get_channel_maps_path()
    cell_counts = count_cells_per_signal_group(
        signal_groups, storage_roots, FEATURE_DIR, maps_path)

    # --- 3. phase1 jobs (reuse pca_sweep_pooled_signal) ---
    jobs = []
    for ch, pairs in signal_groups.items():
        kwargs = dict(
            signal=ch, exp_channel_pairs=pairs, output_dir=str(variant_dir),
            target_n_cells=cell_counts.get(ch, 0),  # all cells (no downsample)
            norm_method=norm_method, distance=distance,
            fixed_threshold=fixed_threshold, zscore_per_experiment=True,
            feature_dir_override=FEATURE_DIR,
        )
        jobs.append({"name": f"pca_bf_{ch}", "func": pca_sweep_pooled_signal, "kwargs": kwargs})

    # --- 4. submit phase1 ONLY (no auto-chained phase2) ---
    # Phase 2 aggregation is run separately, in parallel with titration, by the
    # orchestrator (submit_aggregation) once phase1 succeeds.
    args = _build_args(norm_method, distance, fixed_threshold,
                       slurm_memory, slurm_time, slurm_cpus, slurm_partition)
    result = _submit_phase1_slurm(
        jobs=jobs, args=args, agg_output=str(variant_dir),
        per_unit_subdir="per_signal", experiment_name="bf_titration",
        manifest_prefix="bf_titration", unit_label="signal",
        chain_aggregation=False)
    if result.get("failed"):
        raise RuntimeError(f"phase1 (embedding) failed: {result['failed']}")


def submit_aggregation(variant_dir, channels, norm_method="ntc", distance="cosine",
                       slurm_memory="200GB", slurm_time=240, slurm_cpus=8,
                       slurm_partition="cpu,gpu"):
    """Phase 2 aggregation, ONE per channel (separate folders) — not combined.

    `aggregate_channels` globs ``per_signal/*_guide.h5ad`` and concatenates, so to
    get a per-channel aggregate we isolate each channel's guide/gene/cells into its
    own ``<variant>/by_channel/<channel>/per_signal/`` and aggregate that alone.
    Each channel → ``by_channel/<channel>/`` with its own guide_pca_optimized.h5ad,
    gene_embedding_pca_optimized.h5ad, pca_report.csv + UMAP/PHATE. Submitted as one
    SLURM job per channel (parallel array); blocks until all finish.
    """
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    from ops_model.post_process.combination.pca_optimization import aggregate_channels
    variant_dir = Path(variant_dir)
    per_signal = variant_dir / "per_signal"

    jobs = []
    for ch in channels:
        if not (per_signal / f"{ch}_guide.h5ad").exists():
            print(f"  [WARN] {ch}_guide.h5ad missing in per_signal — skipping aggregation for {ch}")
            continue
        ch_dir = variant_dir / "by_channel" / ch
        ch_ps = ch_dir / "per_signal"
        ch_ps.mkdir(parents=True, exist_ok=True)
        for suf in ("guide", "gene", "cells"):
            src = per_signal / f"{ch}_{suf}.h5ad"
            dst = ch_ps / f"{ch}_{suf}.h5ad"
            if src.exists() and not (dst.exists() or dst.is_symlink()):
                dst.symlink_to(src)
        jobs.append({
            "name": f"bf_agg_{ch}",
            "func": aggregate_channels,
            "kwargs": dict(output_dir=str(ch_dir), norm_method=norm_method,
                           per_unit_subdir="per_signal", distance=distance),
            "metadata": {"channel": ch},
        })

    if not jobs:
        print("  no channels with per_signal guide h5ads — nothing to aggregate")
        return
    print(f"  Per-channel aggregation: {len(jobs)} channels → by_channel/<ch>/")
    result = submit_parallel_jobs(
        jobs_to_submit=jobs, experiment="bf_titration_agg",
        slurm_params={"timeout_min": slurm_time, "mem": slurm_memory,
                      "cpus_per_task": slurm_cpus, "slurm_partition": slurm_partition},
        log_dir="run_bf_titration/aggregate", manifest_prefix="bf_agg",
        wait_for_completion=True)
    if result.get("failed"):
        raise RuntimeError(f"per-channel aggregation failed: {result['failed']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiments", nargs="+", required=True)
    parser.add_argument("--channels", nargs="+", required=True)
    parser.add_argument("--variant-dir", required=True)
    parser.add_argument("--norm-method", default="ntc", choices=["ntc", "global"])
    parser.add_argument("--distance", default="cosine")
    parser.add_argument("--fixed-threshold", type=float, default=0.8)
    parser.add_argument("--convert-only", action="store_true")
    args = parser.parse_args()
    run_embedding(args.experiments, args.channels, args.variant_dir,
                  norm_method=args.norm_method, distance=args.distance,
                  fixed_threshold=args.fixed_threshold, convert_only=args.convert_only)


if __name__ == "__main__":
    main()
