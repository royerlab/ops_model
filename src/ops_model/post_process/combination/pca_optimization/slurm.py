"""SLURM submission helpers for pca_optimization.

* ``_make_slurm_params`` / ``_make_agg_slurm_params`` — pure dict builders
  read off the parsed argparse Namespace.
* ``_build_second_pca_kwargs`` — pure helper that extracts the
  ``apply_second_pass_pca`` kwargs from argparse args.
* ``_aggregate_then_second_pca`` — picklable top-level worker that chains
  Phase 2 aggregation + the 2nd-pass PCA in a single SLURM slot.
* ``_submit_aggregation_slurm`` — fire one aggregation job (optionally
  chained with the 2nd pass).
* ``_submit_phase1_slurm`` — fire all Phase 1 signal-group jobs and
  auto-chain a Phase 2 job on completion.

Heavyweight callees (``aggregate_channels``, ``apply_second_pass_pca``)
live in ``pca_optimization`` and are imported lazily inside the worker /
submission functions so we don't pay the cycle at module load time.
"""

from __future__ import annotations

from typing import Dict, List, Optional


def _make_slurm_params(args):
    """Build standard SLURM params dict from parsed args."""
    return {
        "timeout_min": args.slurm_time,
        "mem": args.slurm_memory,
        "cpus_per_task": args.slurm_cpus,
        "slurm_partition": args.slurm_partition,
    }


def _make_agg_slurm_params(args):
    """Build aggregation SLURM params dict from parsed args."""
    return {
        "timeout_min": args.slurm_agg_time,
        "mem": args.slurm_agg_memory,
        "cpus_per_task": args.slurm_cpus,
        "slurm_partition": args.slurm_partition,
    }


def _aggregate_then_second_pca(
    output_dir: str,
    norm_method: str,
    per_unit_subdir: str,
    distance: str,
    second_pca_threshold: float,
    second_pca_subdir: Optional[str],
    second_pca_run_sweep: bool,
    second_pca_sweep_thresholds: Optional[List[float]],
    random_seed: int = 42,
    agg_method: str = "mean",
    chromosome_csv: Optional[str] = None,
    umap_type: str = "max",
    consensus_metrics=None,
    sweep_metric: str = "mean_map",
) -> str:
    """Run Phase 2 aggregation and the 2nd-pass PCA back-to-back.

    Top-level (picklable) helper for SLURM jobs that want both steps in one slot.
    ``consensus_metrics`` is a list (or comma-separated string) of metric names
    drawn from ``{activity, distinctiveness, ebi, chad}`` that drives the
    2nd-pass PCA consensus threshold pick. Default (``None``) =
    ``(activity, distinctiveness, ebi)``. Non-default sets land in
    ``second_pca_consensus_<TAG>/`` siblings so each subset keeps its own
    output without clobbering the canonical one.
    """
    from ops_model.post_process.combination.pca_optimization import (
        aggregate_channels,
        apply_second_pass_pca,
    )

    agg_result = aggregate_channels(
        output_dir=output_dir,
        norm_method=norm_method,
        per_unit_subdir=per_unit_subdir,
        distance=distance,
        random_seed=random_seed,
        agg_method=agg_method,
        chromosome_csv=chromosome_csv,
        umap_type=umap_type,
    )
    if str(agg_result).startswith("FAILED"):
        return agg_result
    second_result = apply_second_pass_pca(
        output_dir=output_dir,
        threshold=second_pca_threshold,
        distance=distance,
        norm_method=norm_method,
        subdir=second_pca_subdir,
        run_sweep=second_pca_run_sweep,
        sweep_thresholds=second_pca_sweep_thresholds,
        random_seed=random_seed,
        agg_method=agg_method,
        chromosome_csv=chromosome_csv,
        umap_type=umap_type,
        consensus_metrics=consensus_metrics,
        sweep_metric=sweep_metric,
    )
    return f"{agg_result} | 2nd-pca: {second_result}"


def _build_second_pca_kwargs(args) -> Optional[Dict]:
    """Return kwargs to pass to apply_second_pass_pca, or None if --second-pca was not set."""
    if not getattr(args, "second_pca", False):
        return None
    sweep_thresholds = None
    raw = getattr(args, "second_pca_sweep_thresholds", None)
    if raw:
        sweep_thresholds = [float(t) for t in raw.split(",") if t.strip()]
    return {
        "second_pca_threshold": args.second_pca_threshold,
        "second_pca_subdir": args.second_pca_subdir,
        "second_pca_run_sweep": not args.second_pca_no_sweep,
        "second_pca_sweep_thresholds": sweep_thresholds,
    }


def _submit_aggregation_slurm(
    agg_output,
    norm_method,
    per_unit_subdir,
    agg_slurm_params,
    experiment_name,
    manifest_prefix,
    distance="cosine",
    second_pca_kwargs: Optional[Dict] = None,
    random_seed: int = 42,
    agg_method: str = "mean",
    chromosome_csv: Optional[str] = None,
    umap_type: str = "max",
    consensus_metrics=None,
    sweep_metric: str = "mean_map",
):
    """Submit a single aggregation SLURM job (optionally chained with 2nd-pass PCA)."""
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs

    if second_pca_kwargs is not None:
        job_func = _aggregate_then_second_pca
        job_kwargs = {
            "output_dir": agg_output,
            "norm_method": norm_method,
            "per_unit_subdir": per_unit_subdir,
            "distance": distance,
            "random_seed": random_seed,
            "agg_method": agg_method,
            "chromosome_csv": chromosome_csv,
            "umap_type": umap_type,
            "consensus_metrics": consensus_metrics,
            "sweep_metric": sweep_metric,
            **second_pca_kwargs,
        }
        job_name = f"{manifest_prefix}_aggregate_2pca"
    else:
        from ops_model.post_process.combination.pca_optimization import aggregate_channels

        job_func = aggregate_channels
        job_kwargs = {
            "output_dir": agg_output,
            "norm_method": norm_method,
            "per_unit_subdir": per_unit_subdir,
            "distance": distance,
            "random_seed": random_seed,
            "agg_method": agg_method,
            "chromosome_csv": chromosome_csv,
            "umap_type": umap_type,
        }
        job_name = f"{manifest_prefix}_aggregate"

    agg_jobs = [
        {
            "name": job_name,
            "func": job_func,
            "kwargs": job_kwargs,
        }
    ]
    agg_result = submit_parallel_jobs(
        jobs_to_submit=agg_jobs,
        experiment=experiment_name,
        slurm_params=agg_slurm_params,
        log_dir="pca_optimization",
        manifest_prefix=manifest_prefix,
        wait_for_completion=True,
    )
    if agg_result.get("failed"):
        print("Aggregation FAILED")
    else:
        print("Aggregation complete")


def _submit_phase1_slurm(
    jobs,
    args,
    agg_output,
    per_unit_subdir,
    experiment_name,
    manifest_prefix,
    unit_label,
):
    """Submit Phase 1 SLURM jobs + auto-chain Phase 2 aggregation on completion."""
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs

    slurm_params = _make_slurm_params(args)
    agg_slurm_params = _make_agg_slurm_params(args)

    def _on_phase1_complete(submitted_jobs, experiment):
        print(f"\nAll {unit_label} jobs complete. Submitting aggregation SLURM job...")
        _submit_aggregation_slurm(
            agg_output,
            args.norm_method,
            per_unit_subdir,
            agg_slurm_params,
            f"{manifest_prefix}_aggregation",
            f"{manifest_prefix}_agg",
            distance=args.distance,
            second_pca_kwargs=_build_second_pca_kwargs(args),
            random_seed=getattr(args, "seed", 42),
            agg_method=getattr(args, "agg_method", "mean"),
            chromosome_csv=getattr(args, "chromosome_csv", None),
            umap_type=getattr(args, "umap_type", "max"),
            consensus_metrics=getattr(args, "second_pca_consensus_metrics", None),
            sweep_metric=getattr(args, "sweep_metric", "mean_map"),
        )

    print(f"\nSubmitting {len(jobs)} {unit_label} SLURM jobs...")
    result = submit_parallel_jobs(
        jobs_to_submit=jobs,
        experiment=experiment_name,
        slurm_params=slurm_params,
        log_dir="pca_optimization",
        manifest_prefix=f"{manifest_prefix}_opt",
        wait_for_completion=True,
        post_completion_callback=_on_phase1_complete,
    )
    if result.get("failed"):
        print(f"\nWarning: {len(result['failed'])} {unit_label} failed")
        for name in result["failed"]:
            print(f"  - {name}")
