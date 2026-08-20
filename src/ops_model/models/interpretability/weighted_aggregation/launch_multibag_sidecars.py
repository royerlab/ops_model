"""Fan out 3 SLURM jobs to build the multi-bag sidecars in parallel."""
from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
from ops_model.models.interpretability.weighted_aggregation.build_multibag_sidecars import (
    build_sidecar, SOURCES,
)

if __name__ == "__main__":
    jobs = [
        {"name": f"multibag_sidecar_{n}", "func": build_sidecar, "kwargs": {"name": n}}
        for n in SOURCES
    ]
    submit_parallel_jobs(
        jobs_to_submit=jobs,
        experiment="multibag_sidecars",
        slurm_params={
            "timeout_min": 90,
            "cpus_per_task": 8,
            "mem_gb": 128,
            "slurm_partition": "cpu",
        },
        log_dir="multibag_sidecars",
        manifest_prefix="multibag_sidecars",
        wait_for_completion=False,
    )
