"""CLI mode handlers for pca_optimization.

One function per ``main()`` dispatch branch — each implements a
mutually-exclusive CLI mode flag (``--chad-umap-only``, ``--umap-only``,
``--sweep-seed``, ``--aggregate-only``, ``--second-pca``-only paths, the
explicit-path ``signal_paths`` flow, and the default ``--downsampled`` flow).

These functions are the call-site surface area for orchestrating Phase 1
+ Phase 2 (delegating to ``phase1.pca_sweep_pooled_signal`` /
``phase2.aggregate_channels`` / ``phase2.apply_second_pass_pca`` via
either an in-process call or a SLURM submission helper).

Module globals from ``pca_optimization`` (CHAD_ANNOTATION_PATH,
DEFAULT_SWEEP_THRESHOLDS_CP, PCA_FIT_CAP) are imported lazily inside
the handlers that read them so the parent and this module can
re-import each other at load time without cycling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict


from ops_utils.data.feature_discovery import (
    sanitize_signal_filename,
)

from ops_model.post_process.combination.pca_optimization.phase1 import (
    pca_sweep_pooled_signal,
)
from ops_model.post_process.combination.pca_optimization.phase2 import (
    aggregate_channels,
    apply_second_pass_pca,
)
from ops_model.post_process.combination.pca_optimization.slurm import (
    _build_second_pca_kwargs,
    _make_agg_slurm_params,
    _make_slurm_params,
    _submit_aggregation_slurm,
)


def _handle_second_pca(args, output_dir):
    """Run a second-pass PCA on the existing aggregated guide h5ad.

    With ``--chrom-arm-correct``, first regresses out per-arm cohesion effects
    from the guide-level h5ad and runs the 2nd-pass on the corrected output;
    the corrected h5ad and its 2nd-pass subdir both get a ``_chrom_arm_corr``
    suffix so they never clobber an existing untouched run.
    """
    sweep_thresholds = None
    if args.second_pca_sweep_thresholds:
        sweep_thresholds = [
            float(t) for t in args.second_pca_sweep_thresholds.split(",") if t.strip()
        ]

    second_pca_kwargs = dict(
        output_dir=str(output_dir),
        threshold=args.second_pca_threshold,
        distance=args.distance,
        norm_method=args.norm_method,
        subdir=args.second_pca_subdir,
        run_sweep=not args.second_pca_no_sweep,
        sweep_thresholds=sweep_thresholds,
        random_seed=getattr(args, "seed", 42),
        agg_method=getattr(args, "agg_method", "mean"),
        umap_type=getattr(args, "umap_type", "max"),
        consensus_metrics=getattr(args, "second_pca_consensus_metrics", None),
        sweep_metric=getattr(args, "sweep_metric", "mean_map"),
    )

    job_name = "pca_second_pass"
    func = apply_second_pass_pca
    kwargs_for_run = second_pca_kwargs
    if args.slurm:
        from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs

        slurm_params = _make_agg_slurm_params(args)
        print(
            f"Submitting --second-pca as SLURM job "
            f"({slurm_params.get('mem')}, {slurm_params.get('timeout_min')}min)..."
        )
        agg_result = submit_parallel_jobs(
            jobs_to_submit=[
                {
                    "name": job_name,
                    "func": func,
                    "kwargs": kwargs_for_run,
                }
            ],
            experiment=job_name,
            slurm_params=slurm_params,
            log_dir="pca_optimization",
            manifest_prefix=job_name,
            wait_for_completion=True,
        )
        if agg_result.get("failed"):
            print("Second-pass PCA FAILED")
        else:
            print("Second-pass PCA complete")
    else:
        print(func(**kwargs_for_run))


def _handle_external(args, output_dir):
    """External mode: combine explicit per-signal h5ads from config ``signal_paths``.

    ``args.signal_paths`` maps a signal-group name -> one h5ad path or a list of
    paths. Each h5ad must have the same schema as the discovery
    ``features_processed_*.h5ad`` (obs with sgRNA / perturbation / experiment;
    ``X`` = embedding). Multiple paths under one signal are pooled. Experiment
    discovery is bypassed; the standard ``pca_sweep_pooled_signal`` worker (with
    an explicit-path override) + Phase 2 aggregation are reused unchanged.
    """
    ds_output_dir = output_dir
    ds_output_dir.mkdir(parents=True, exist_ok=True)

    if getattr(args, "clean", False):
        import shutil

        per_signal_dir = ds_output_dir / "per_signal"
        if per_signal_dir.exists():
            print(f"--clean: removing {per_signal_dir}")
            shutil.rmtree(per_signal_dir)

    spec = args.signal_paths
    if not isinstance(spec, dict) or not spec:
        print("ERROR: signal_paths must be a non-empty mapping of signal -> path(s).")
        return

    # Build signal groups + an explicit (exp_label, channel) -> path override.
    # Each file becomes its own synthetic "experiment" batch (channel == signal),
    # so per-experiment z-scoring treats each file independently.
    signal_groups: Dict[str, list] = {}
    cell_path_map: Dict[tuple, str] = {}
    missing: list = []
    for signal, paths in spec.items():
        if isinstance(paths, (str, Path)):
            paths = [paths]
        pairs = []
        for p in paths:
            p = Path(p)
            if not p.exists():
                missing.append(str(p))
                continue
            exp_label = p.stem
            pairs.append((exp_label, signal))
            cell_path_map[(exp_label, signal)] = str(p)
        if pairs:
            signal_groups[signal] = pairs

    if missing:
        print("ERROR: signal_paths references missing file(s):")
        for m in missing:
            print(f"  - {m}")
        return
    if not signal_groups:
        print("ERROR: no usable signal_paths entries.")
        return

    print(f"External mode: {len(signal_groups)} signal group(s) from explicit paths:")
    for sig, pairs in signal_groups.items():
        print(f"  {sig}: {len(pairs)} file(s)")

    # External files are user-provided; default to keeping all cells unless the
    # user opts into downsampling (--target-cells / --downsampled).
    target_n_cells = int(getattr(args, "target_cells", 0) or 0) or 10_000_000
    skip_phase2 = getattr(args, "preserve_batch", False) or getattr(args, "no_pca", False)

    def _job_kwargs(signal, pairs):
        kwargs = dict(
            signal=signal,
            exp_channel_pairs=pairs,
            output_dir=str(ds_output_dir),
            target_n_cells=target_n_cells,
            norm_method=args.norm_method,
            random_seed=getattr(args, "seed", 42),
            distance=args.distance,
            zscore_per_experiment=getattr(args, "zscore_per_experiment", False),
            exclude_dud_guides=getattr(args, "exclude_dud_guides", True),
            cell_paths=cell_path_map,
        )
        if args.fixed_threshold is not None and args.fixed_threshold > 0:
            kwargs["fixed_threshold"] = args.fixed_threshold
        if getattr(args, "preserve_batch", False):
            kwargs["preserve_batch"] = True
        if getattr(args, "no_pca", False):
            kwargs["no_pca"] = True
        if getattr(args, "agg_method", "mean") != "mean":
            kwargs["agg_method"] = args.agg_method
        if getattr(args, "downsample_per_guide", False):
            kwargs["downsample_per_guide"] = True
            kwargs["cells_per_guide"] = getattr(args, "cells_per_guide", 250)
        return kwargs

    if not args.slurm:
        print("\nRunning locally (sequential)...")
        for signal, pairs in signal_groups.items():
            print(f"  {pca_sweep_pooled_signal(**_job_kwargs(signal, pairs))}")
        if not skip_phase2:
            print(
                aggregate_channels(
                    output_dir=str(ds_output_dir),
                    norm_method=args.norm_method,
                    per_unit_subdir="per_signal",
                    distance=args.distance,
                    random_seed=getattr(args, "seed", 42),
                    agg_method=getattr(args, "agg_method", "mean"),
                                umap_type=getattr(args, "umap_type", "max"),
                )
            )
            if getattr(args, "second_pca", False):
                print("\nChaining 2nd-pass PCA on aggregate output...")
                _handle_second_pca(args, ds_output_dir)
        return

    # SLURM: one job per signal group, wait, then a chained aggregation job.
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs

    jobs = [
        {
            "name": f"pca_ext_{sanitize_signal_filename(sig)[:40]}",
            "func": pca_sweep_pooled_signal,
            "kwargs": _job_kwargs(sig, pairs),
            "metadata": {"signal": sig, "n_files": len(pairs)},
        }
        for sig, pairs in signal_groups.items()
    ]
    slurm_params = _make_slurm_params(args)
    print(
        f"\nSubmitting {len(jobs)} external signal job(s) "
        f"({slurm_params['mem']} each, {slurm_params['timeout_min']}min)..."
    )
    res = submit_parallel_jobs(
        jobs_to_submit=jobs,
        experiment="pca_ext",
        slurm_params=slurm_params,
        log_dir="pca_optimization",
        manifest_prefix="pca_ext",
        wait_for_completion=True,
    )
    if res.get("failed"):
        print(f"\n{len(res['failed'])} external signal job(s) failed. Check logs.")
        return
    print(f"\nAll {len(jobs)} external signal job(s) complete")

    if skip_phase2:
        print("  Phase 2 aggregation skipped (--preserve-batch or --no-pca mode)")
        return

    sp_kwargs = _build_second_pca_kwargs(args)
    print("\nSubmitting aggregation SLURM job...")
    _submit_aggregation_slurm(
        str(ds_output_dir),
        args.norm_method,
        "per_signal",
        _make_agg_slurm_params(args),
        "pca_ext_aggregation",
        "pca_ext_agg",
        distance=args.distance,
        second_pca_kwargs=sp_kwargs,
        random_seed=getattr(args, "seed", 42),
        agg_method=getattr(args, "agg_method", "mean"),
        umap_type=getattr(args, "umap_type", "max"),
        consensus_metrics=getattr(args, "second_pca_consensus_metrics", None),
    )


