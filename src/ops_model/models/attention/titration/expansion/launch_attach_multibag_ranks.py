"""Fan out multibag rank attachment across all per-experiment PCA h5ads via SLURM."""
from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
from ops_model.models.attention.titration.expansion.attach_multibag_ranks import (
    attach_one, SRC_PCA,
)

if __name__ == "__main__":
    exps = sorted(p.stem for p in SRC_PCA.glob("*.h5ad"))
    jobs = [
        {"name": f"attach_multibag_{e}", "func": attach_one, "kwargs": {"experiment": e}}
        for e in exps
    ]
    print(f"submitting {len(jobs)} attach jobs")
    submit_parallel_jobs(
        jobs_to_submit=jobs,
        experiment="attach_multibag_ranks",
        slurm_params={
            "timeout_min": 30,
            "cpus_per_task": 4,
            "mem_gb": 64,
            "slurm_partition": "cpu",
        },
        log_dir="attach_multibag_ranks",
        manifest_prefix="attach_multibag_ranks",
        wait_for_completion=False,
    )
