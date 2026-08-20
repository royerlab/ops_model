"""Force-regenerate the stale (pre-DDIM-inversion) fluor traversals flagged by pearson_gen0_bymarker.

The stale subset (α=0 frame doesn't reconstruct its NTC anchor, Pearson r<0.5) was built by an old pass
and never force-rebuilt after the inverted-anchor fix. Their ranking cells all still exist, so we just
re-run the CURRENT gen (gen_marker_shard / gen_complex_shard) with force=True on exactly those targets.

  python regen_stale_traversals.py submit          # geneKO + complex, per-marker force=True on stale targets
  python regen_stale_traversals.py submit geneKO    # one grain only
"""
import json
import os
import sys

from . import catalog as C
from ..classifier.config import slugify
from . import _fluor_v5_build as GK
from . import _fluor_complex_build as CX

ROOT = "/hpc/projects/icd.fast.ops/models/diffex/viewer_assets_v5"
STALE = f"{ROOT}/_montage/pearson_gen0_stale.json"


def _stale():
    return json.load(open(STALE))


def jobs_geneKO(stale):
    rk = GK.build_rankings()                                   # {channel: (parq, [genes])}
    cm = {mc: (d, ch) for d, mc, ch in C.complete_markers()}
    slug2ch = {slugify(ch): ch for ch in rk}
    jobs = []
    for key, genes in stale.items():
        mk, grain = key.rsplit("/", 1)
        if grain != "geneKO" or mk not in slug2ch:
            continue
        ch_name = slug2ch[mk]
        if ch_name not in cm:
            continue
        d, rawch = cm[ch_name]; parq, avail = rk[ch_name]
        tgt = [g for g in genes if g in set(avail)]
        miss = [g for g in genes if g not in set(avail)]
        if miss:
            print(f"  [warn] {mk}: {len(miss)} stale genes not in current ranking (skip): {miss[:5]}")
        if tgt:
            jobs.append({"name": f"regenGK_{slugify(ch_name)[:14]}", "func": GK.gen_marker_shard,
                         "kwargs": {"mc": ch_name, "d": d, "ch": rawch, "targets": tgt, "parq": parq, "force": True}})
    return jobs


def jobs_complex(stale):
    rk = CX.build_rankings()                                   # {channel: (parq, [complexes])}
    cm = {mc: (d, ch) for d, mc, ch in C.complete_markers()}
    slug2ch = {slugify(ch): ch for ch in rk}
    jobs = []
    for key, cxs in stale.items():
        mk, grain = key.rsplit("/", 1)
        if grain != "complex" or mk not in slug2ch:
            continue
        ch_name = slug2ch[mk]
        if ch_name not in cm:
            continue
        d, rawch = cm[ch_name]; parq, avail = rk[ch_name]
        slug2cx = {slugify(c): c for c in avail}               # stale keys are dir slugs → complex names
        tgt = [slug2cx[c] for c in cxs if c in slug2cx]
        miss = [c for c in cxs if c not in slug2cx]
        if miss:
            print(f"  [warn] {mk}: {len(miss)} stale complexes not in current ranking (skip): {miss[:5]}")
        if tgt:
            jobs.append({"name": f"regenCX_{slugify(ch_name)[:14]}", "func": CX.gen_complex_shard,
                         "kwargs": {"mc": ch_name, "d": d, "ch": rawch, "targets": tgt, "parq": parq, "force": True}})
    return jobs


def submit(grain="both"):
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    os.environ["OPS_DIFFEX_ASSETS"] = "viewer_assets_v5"
    stale = _stale()
    jobs = []
    if grain in ("both", "geneKO"):
        jobs += jobs_geneKO(stale)
    if grain in ("both", "complex"):
        jobs += jobs_complex(stale)
    ntgt = sum(len(j["kwargs"]["targets"]) for j in jobs)
    print(f"[regen-stale] {len(jobs)} marker jobs, {ntgt} stale targets total")
    submit_parallel_jobs(jobs, experiment="diffex_regen_stale",
                         slurm_params={"slurm_partition": "gpu", "slurm_gres": "gpu:1",
                                       "cpus_per_task": 12, "mem_gb": 96, "timeout_min": 600},
                         log_dir="diffex_regen_stale", wait_for_completion=False)


def _done_since(asset_dir, cutoff):
    """A target is 'done' if its α=0 frame was rebuilt after `cutoff` (epoch s)."""
    p = f"{ROOT}/{asset_dir}/cell0/frame_08.webp"
    return os.path.exists(p) and os.path.getmtime(p) > cutoff


def resubmit(cutoff_str, chunk=30):
    """Re-shard the STILL-remaining stale targets into ~`chunk`-sized jobs (per marker, since each job loads
    one DiffAE checkpoint) so the heavy markers spread across many GPUs instead of one serial shard each.
    `cutoff_str` = launch time of the original batch (e.g. '2026-08-17 12:10'); targets rebuilt after it are skipped."""
    import time
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    os.environ["OPS_DIFFEX_ASSETS"] = "viewer_assets_v5"
    cutoff = time.mktime(time.strptime(cutoff_str, "%Y-%m-%d %H:%M"))
    stale = _stale()
    base = jobs_geneKO(stale) + jobs_complex(stale)                     # one job per marker w/ full stale target list
    jobs = []
    for j in base:
        mc, ch, grain_fn = j["kwargs"]["mc"], j["kwargs"]["ch"], j["func"]
        gr = "geneKO" if grain_fn is GK.gen_marker_shard else "complex"
        # slug for asset path: geneKO uses target name; complex uses slugify(target)
        rem = [t for t in j["kwargs"]["targets"]
               if not _done_since(f"{slugify(mc)}/{gr}/{t if gr == 'geneKO' else slugify(t)}", cutoff)]
        for i in range(0, len(rem), chunk):
            sub = rem[i:i + chunk]
            jobs.append({"name": f"rerun_{slugify(mc)[:12]}_{i // chunk}", "func": grain_fn,
                         "kwargs": {**j["kwargs"], "targets": sub}})
    ntgt = sum(len(j["kwargs"]["targets"]) for j in jobs)
    print(f"[regen-rerun] {len(jobs)} chunked jobs, {ntgt} remaining targets (chunk={chunk})")
    submit_parallel_jobs(jobs, experiment="diffex_regen_rerun",
                         slurm_params={"slurm_partition": "gpu", "slurm_gres": "gpu:1", "slurm_constraint": "h100|h200",
                                       "cpus_per_task": 12, "mem_gb": 96, "timeout_min": 300},
                         log_dir="diffex_regen_rerun", wait_for_completion=False)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "submit"
    if cmd == "resubmit":
        resubmit(sys.argv[2] if len(sys.argv) > 2 else "2026-08-17 12:10")
    else:
        submit(cmd if cmd in ("geneKO", "complex") else "both")
