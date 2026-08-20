"""Per-marker per-α LOO aggregate for the montage overlay.

Aggregates each traversal dir's scores_loo.json (mean over cells at α∈{1..5}) into
  _montage/loo_bymarker.json          = {markerSlug: {gene: {"1":m,..,"5":m}}}
  _montage/loo_bymarker_complex.json  = {markerSlug: {ebi_complex: {...}}}
markerSlug = the traversal dir name (== attnModality() slug; "phase" for phase). SLURM-sharded, one job per marker.

  python build_loo_bymarker.py submit
  python build_loo_bymarker.py merge
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = "/hpc/projects/icd.fast.ops/models/diffex/viewer_assets_v5"
SH = f"{ROOT}/_montage/_loo_bymarker_shards"
ALPHAS = [1, 2, 3, 4, 5]


def _agg(f):
    d = json.load(open(f)); al = d["alphas"]
    arr = np.array([[np.nan if v is None else v for v in c] for c in d["scores"]], float)
    r = {}
    for a in ALPHAS:
        if float(a) in al:
            m = np.nanmean(arr[:, al.index(float(a))])
            if np.isfinite(m): r[str(a)] = round(float(m), 4)
    return r


def shard(marker):
    os.makedirs(SH, exist_ok=True)
    res = {"geneKO": {}, "complex": {}}
    for grain in ("geneKO", "complex"):
        for f in glob.glob(f"{ROOT}/{marker}/{grain}/*/scores_loo.json"):
            name = f.split("/")[-2]
            if "__to__" in name: continue
            r = _agg(f)
            if r: res[grain][name] = r
    json.dump(res, open(f"{SH}/{marker.replace('/', '_')}.json", "w"))
    return {"marker": marker, "geneKO": len(res["geneKO"]), "complex": len(res["complex"])}


def markers():
    return sorted(d for d in os.listdir(ROOT) if os.path.isdir(f"{ROOT}/{d}/geneKO") and not d.startswith("vs_"))


def submit():
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    jobs = [{"name": f"loobm_{m}"[:40], "func": shard, "kwargs": {"marker": m}} for m in markers()]
    print(f"[loo-bymarker] {len(jobs)} markers")
    submit_parallel_jobs(jobs, experiment="loo_bymarker",
                         slurm_params={"slurm_partition": "cpu", "cpus_per_task": 2, "mem_gb": 8, "timeout_min": 30},
                         log_dir="loo_bymarker", wait_for_completion=False)


def merge():
    sc = list(json.load(open(f"{ROOT}/_montage/setacc_complex_rank.json")).keys())   # canonical ebi_complex labels
    norm = lambda s: "".join(c for c in s.lower() if c.isalnum()); kbn = {norm(k): k for k in sc}
    gk, cx = {}, {}
    for f in glob.glob(f"{SH}/*.json"):
        marker = os.path.basename(f)[:-5]; d = json.load(open(f))
        if d["geneKO"]: gk[marker] = d["geneKO"]
        if d["complex"]:
            cx[marker] = {kbn.get(norm(name), name.replace("_", " ")): v for name, v in d["complex"].items()}
    json.dump(gk, open(f"{ROOT}/_montage/loo_bymarker.json", "w"))
    json.dump(cx, open(f"{ROOT}/_montage/loo_bymarker_complex.json", "w"))
    print(f"loo_bymarker.json: {len(gk)} markers · geneKO median {int(np.median([len(v) for v in gk.values()]))} genes/marker")
    print(f"loo_bymarker_complex.json: {len(cx)} markers")


if __name__ == "__main__":
    cmd = sys.argv[1] if sys.argv[1:] else "submit"
    print({"submit": submit, "merge": merge}[cmd]())
