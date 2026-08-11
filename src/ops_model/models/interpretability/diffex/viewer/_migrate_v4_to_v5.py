"""Migrate the v4 viewer components that v5 models will NOT regenerate into viewer_assets_v5, so they work
under the v5 cache: the PC viewer (pcs/) and the minibinder + PC-axis traversals (phase/minibinder, phase/pc).

Left in v4 (still awaiting v5 model outputs): fluorescence marker traversals/montage and attention heads.
"""
import os, json, shutil, subprocess

ROOT = "/hpc/projects/icd.fast.ops/models/diffex"
V4, V5 = f"{ROOT}/viewer_assets", f"{ROOT}/viewer_assets_v5"
DIRS = ["pcs", "phase/minibinder", "phase/pc"]          # big trees → one SLURM rsync each
FILES = ["_minibinder_meta.json"]


def rsync_dir(rel):
    src, dst = f"{V4}/{rel}/", f"{V5}/{rel}/"
    os.makedirs(dst, exist_ok=True)
    subprocess.run(["rsync", "-a", src, dst], check=True)
    return f"{rel}: {subprocess.run(['du','-sh',dst],capture_output=True,text=True).stdout.split()[0]}"


def merge_manifest():
    """Append the v4 minibinder + pc targets (phase) into the v5 phase marker's target list (idempotent)."""
    v4 = json.load(open(f"{V4}/manifest.json"))
    v5 = json.load(open(f"{V5}/manifest.json"))
    # collect v4 minibinder+pc targets (they live under the phase marker)
    add = [t for mk in v4["markers"] for t in mk["targets"] if t["grain"] in ("minibinder", "pc")]
    m5 = v5["markers"][0]                                # v5 is phase-only, single marker
    have = {(t["grain"], t["asset_dir"]) for t in m5["targets"]}
    new = [t for t in add if (t["grain"], t["asset_dir"]) not in have]
    m5["targets"].extend(new)
    json.dump(v5, open(f"{V5}/manifest.json", "w"))
    print(f"[manifest] added {len(new)} targets (minibinder+pc); v5 marker now {len(m5['targets'])} targets")


def main():
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    jobs = [{"name": f"migrate_{r.replace('/','_')}", "func": rsync_dir, "kwargs": {"rel": r}} for r in DIRS]
    print(f"[migrate] copying {DIRS} v4 -> v5 via {len(jobs)} rsync jobs")
    submit_parallel_jobs(
        jobs, experiment="diffex_migrate_v5",
        slurm_params={"slurm_partition": "cpu", "cpus_per_task": 4, "mem_gb": 16, "timeout_min": 120},
        log_dir="diffex_migrate_v5", wait_for_completion=True)
    for f in FILES:
        shutil.copy(f"{V4}/{f}", f"{V5}/{f}"); print(f"[file] {f}")
    merge_manifest()


if __name__ == "__main__":
    main()
