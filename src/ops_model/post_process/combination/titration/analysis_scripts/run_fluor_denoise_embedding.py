"""Embedding step for the fluor-denoise titration pipeline.

Mirrors run_bf_embedding (same pca_optimization phase1 pooled PCA + phase2
aggregation, reused verbatim), with one difference: signals are marker-keyed
``<marker>_raw`` / ``<marker>_denoise`` rather than per-channel, and each signal
pools only ITS experiments (each experiment carries exactly one marker). The
Cell-DINO CSVs are written as ``cell_dino_features_<signal>.csv`` (the runner
renames the fluorophore-named CSV to the marker+variant signal), so the standard
CSV→h5ad→PCA path applies directly.
"""

import argparse
from pathlib import Path

# Per-experiment feature dir holding the fluor-denoise Cell-DINO CSVs + h5ads.
FEATURE_DIR = "cell_dino_features_fluordenoise"


def exp_feature_dir(experiment):
    from ops_model.data.paths import OpsPaths
    return Path(OpsPaths(experiment).base) / experiment / "3-assembly" / FEATURE_DIR


def convert_csvs_worker(experiment, channels):
    """CSV → features_processed_<signal>.h5ad per signal (no PCA; SLURM worker)."""
    import yaml
    from ops_model.features.evaluate_embeddings import create_adata_object_embedding
    from ops_utils.data.experiment import OpsDataset
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
        if out.exists():
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


def run_embedding(signal_groups, variant_dir, norm_method="ntc",
                  distance="cosine", fixed_threshold=0.8,
                  slurm_memory="400GB", slurm_time=480, slurm_cpus=8,
                  slurm_partition="cpu,gpu", convert_only=False):
    """CSV→h5ad (SLURM), then phase1 pooled PCA + phase2 aggregation.

    signal_groups: {signal_name: [(experiment, signal_name), ...]} — each signal
    pools only its own experiments (marker-keyed, unlike run_bf_embedding's
    every-channel-in-every-experiment grouping).
    """
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    from ops_model.post_process.combination.pca_optimization import (
        pca_sweep_pooled_signal, load_attribution_config, get_storage_roots,
        get_channel_maps_path, count_cells_per_signal_group)
    from ops_model.post_process.combination.pca_optimization.slurm import _submit_phase1_slurm
    from ops_model.post_process.combination.titration.run_bf_embedding import _build_args

    variant_dir = Path(variant_dir)
    variant_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. CSV → cell h5ads, one SLURM job per (experiment, signal) ---
    pairs_flat = sorted({(e, s) for grp in signal_groups.values() for e, s in grp})
    conv_jobs = [{
        "name": f"fluordn_csv2h5ad_{e}_{s}",
        "func": convert_csvs_worker,
        "kwargs": {"experiment": e, "channels": [s]},
        "metadata": {"experiment": e, "signal": s},
    } for e, s in pairs_flat
        if not (exp_feature_dir(e) / "anndata_objects" / f"features_processed_{s}.h5ad").exists()]
    print(f"\n[embedding] CSV→h5ad: {len(pairs_flat) - len(conv_jobs)}/{len(pairs_flat)} "
          f"already done, {len(conv_jobs)} to convert")
    if conv_jobs:
        conv = submit_parallel_jobs(
            jobs_to_submit=conv_jobs, experiment="fluor_denoise",
            slurm_params={"timeout_min": 60, "mem": "48G", "cpus_per_task": 4,
                          "slurm_partition": "cpu,gpu"},
            log_dir="fluor_denoise/csv2h5ad", manifest_prefix="fluordn_csv2h5ad",
            wait_for_completion=True)
        if conv.get("failed"):
            raise RuntimeError(f"CSV→h5ad failed: {conv['failed']}")
    if convert_only:
        return

    # --- 2. cell-count pre-scan ---
    storage_roots = get_storage_roots(load_attribution_config())
    maps_path = get_channel_maps_path()
    cell_counts = count_cells_per_signal_group(
        signal_groups, storage_roots, FEATURE_DIR, maps_path)

    # --- 3. phase1 jobs (reuse pca_sweep_pooled_signal), one per signal ---
    jobs = []
    for sig, pairs in signal_groups.items():
        kwargs = dict(
            signal=sig, exp_channel_pairs=pairs, output_dir=str(variant_dir),
            target_n_cells=cell_counts.get(sig, 0),
            norm_method=norm_method, distance=distance,
            fixed_threshold=fixed_threshold, zscore_per_experiment=True,
            feature_dir_override=FEATURE_DIR,
        )
        jobs.append({"name": f"pca_fluordn_{sig}", "func": pca_sweep_pooled_signal, "kwargs": kwargs})

    args = _build_args(norm_method, distance, fixed_threshold,
                       slurm_memory, slurm_time, slurm_cpus, slurm_partition)
    result = _submit_phase1_slurm(
        jobs=jobs, args=args, agg_output=str(variant_dir),
        per_unit_subdir="per_signal", experiment_name="fluor_denoise",
        manifest_prefix="fluor_denoise", unit_label="signal",
        chain_aggregation=False)
    if result.get("failed"):
        raise RuntimeError(f"phase1 (embedding) failed: {result['failed']}")


def submit_aggregation(variant_dir, signals, norm_method="ntc", distance="cosine",
                       slurm_memory="200GB", slurm_time=240, slurm_cpus=8,
                       slurm_partition="cpu,gpu"):
    """Phase 2 aggregation, ONE per signal (separate folders) — mirrors run_bf_embedding."""
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    from ops_model.post_process.combination.pca_optimization import aggregate_channels
    variant_dir = Path(variant_dir)
    per_signal = variant_dir / "per_signal"

    jobs = []
    for sig in signals:
        if not (per_signal / f"{sig}_guide.h5ad").exists():
            print(f"  [WARN] {sig}_guide.h5ad missing in per_signal — skipping aggregation for {sig}")
            continue
        ch_dir = variant_dir / "by_channel" / sig
        ch_ps = ch_dir / "per_signal"
        ch_ps.mkdir(parents=True, exist_ok=True)
        for suf in ("guide", "gene", "cells"):
            src = per_signal / f"{sig}_{suf}.h5ad"
            dst = ch_ps / f"{sig}_{suf}.h5ad"
            if src.exists() and not (dst.exists() or dst.is_symlink()):
                dst.symlink_to(src)
        jobs.append({
            "name": f"fluordn_agg_{sig}",
            "func": aggregate_channels,
            "kwargs": dict(output_dir=str(ch_dir), norm_method=norm_method,
                           per_unit_subdir="per_signal", distance=distance),
            "metadata": {"signal": sig},
        })

    if not jobs:
        print("  no signals with per_signal guide h5ads — nothing to aggregate")
        return
    print(f"  Per-signal aggregation: {len(jobs)} signals → by_channel/<signal>/")
    result = submit_parallel_jobs(
        jobs_to_submit=jobs, experiment="fluor_denoise_agg",
        slurm_params={"timeout_min": slurm_time, "mem": slurm_memory,
                      "cpus_per_task": slurm_cpus, "slurm_partition": slurm_partition},
        log_dir="fluor_denoise/aggregate", manifest_prefix="fluordn_agg",
        wait_for_completion=True)
    if result.get("failed"):
        raise RuntimeError(f"per-signal aggregation failed: {result['failed']}")
