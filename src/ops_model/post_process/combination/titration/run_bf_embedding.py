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
    from ops_model.features.evaluate_embeddings import create_adata_object_embedding
    fdir = exp_feature_dir(experiment)
    out_dir = fdir / "anndata_objects"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for ch in channels:
        csv = fdir / f"cell_dino_features_{ch}.csv"
        if not csv.exists():
            print(f"  [WARN] missing {csv}; skipping {ch} for {experiment}")
            continue
        adata = create_adata_object_embedding(
            str(csv), channel=ch, experiment=experiment, embedding_type="cell_dino")
        out = out_dir / f"features_processed_{ch}.h5ad"
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
                  slurm_memory="200GB", slurm_time=480, slurm_cpus=8,
                  slurm_partition="cpu,gpu", convert_only=False):
    """CSV→h5ad (SLURM), then phase1 pooled PCA + phase2 aggregation via _submit_phase1_slurm."""
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    from ops_model.post_process.combination.pca_optimization import (
        pca_sweep_pooled_signal, load_attribution_config, get_storage_roots,
        get_channel_maps_path, count_cells_per_signal_group)
    from ops_model.post_process.combination.pca_optimization.slurm import _submit_phase1_slurm

    variant_dir = Path(variant_dir)
    variant_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. CSV → per-channel cell h5ads (one SLURM job per experiment) ---
    print(f"\n[embedding] CSV→h5ad for {len(experiments)} experiments × {len(channels)} channels")
    conv_jobs = [{
        "name": f"bf_csv2h5ad_{exp}",
        "func": convert_csvs_worker,
        "kwargs": {"experiment": exp, "channels": list(channels)},
    } for exp in experiments]
    conv = submit_parallel_jobs(
        jobs_to_submit=conv_jobs, experiment="bf_titration",
        slurm_params={"timeout_min": 120, "mem": "64G", "cpus_per_task": 8,
                      "slurm_partition": "cpu"},
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

    # --- 4. submit phase1 + auto-chain phase2 aggregation ---
    args = _build_args(norm_method, distance, fixed_threshold,
                       slurm_memory, slurm_time, slurm_cpus, slurm_partition)
    _submit_phase1_slurm(
        jobs=jobs, args=args, agg_output=str(variant_dir),
        per_unit_subdir="per_signal", experiment_name="bf_titration",
        manifest_prefix="bf_titration", unit_label="signal")


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
