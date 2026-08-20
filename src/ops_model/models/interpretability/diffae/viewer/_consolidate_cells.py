"""Consolidate the top-accuracy anchor cells into the main v5 traversal dirs so all 45 NTC-anchor cells
live in one self-contained place (S3-deployable) instead of two pools + a viewer toggle.

Per traversal in viewer_assets_v5/phase/{sub}/{name} (that also exists in the accpool):
  - copy accpool cell0..24  ->  cell20..44   (frames + real.webp)
  - meta.json:  n_cells = 45,  cell_source = ['attention']*20 + ['accuracy']*25
  - scores.json (per-cell linear): first-20 attention + 25 accuracy  ->  45 entries
  - scores_v5.json (set-score): replaced with the accuracy pool's (bag-20 of the accuracy cells),
    tagged score_source='accuracy'; the prior attention set-score is preserved as scores_v5_attention.json
Idempotent: re-running skips already-copied cell dirs and rebuilds meta/scores from the source slices.
"""
import os, glob, json, shutil

ROOT = "/hpc/projects/icd.fast.ops/models/diffex"
V5, ACC = f"{ROOT}/viewer_assets_v5", f"{ROOT}/viewer_assets_v5_accpool"
ATTN_N = 20   # attention-anchored cells already present as cell0..19


def consolidate_one(sub, name):
    v5d, accd = f"{V5}/phase/{sub}/{name}", f"{ACC}/phase/{sub}/{name}"
    if not (os.path.isdir(v5d) and os.path.isdir(accd)):
        return f"skip {sub}/{name}: missing dir"
    acc_cells = sorted((c for c in os.listdir(accd) if c.startswith("cell")), key=lambda x: int(x[4:]))
    acc_n = len(acc_cells)
    for k in range(acc_n):                                   # copy accpool cell{k} -> v5 cell{ATTN_N+k}
        dst = f"{v5d}/cell{ATTN_N + k}"
        if not os.path.exists(dst):
            shutil.copytree(f"{accd}/cell{k}", dst)
    total = ATTN_N + acc_n
    # meta
    meta = json.load(open(f"{v5d}/meta.json"))
    meta["n_cells"] = total
    meta["cell_source"] = ["attention"] * ATTN_N + ["accuracy"] * acc_n
    json.dump(meta, open(f"{v5d}/meta.json", "w"))
    # per-cell linear scores.json: attention[:20] + accuracy[:acc_n]
    if os.path.exists(f"{v5d}/scores.json") and os.path.exists(f"{accd}/scores.json"):
        v5s, accs = json.load(open(f"{v5d}/scores.json")), json.load(open(f"{accd}/scores.json"))
        v5s["scores"] = v5s["scores"][:ATTN_N] + accs["scores"][:acc_n]
        json.dump(v5s, open(f"{v5d}/scores.json", "w"))
    # set-score scores_v5.json: use accuracy pool's; preserve attention's once
    if os.path.exists(f"{accd}/scores_v5.json"):
        if os.path.exists(f"{v5d}/scores_v5.json") and not os.path.exists(f"{v5d}/scores_v5_attention.json"):
            shutil.copy(f"{v5d}/scores_v5.json", f"{v5d}/scores_v5_attention.json")
        acc_sv = json.load(open(f"{accd}/scores_v5.json")); acc_sv["score_source"] = "accuracy"
        json.dump(acc_sv, open(f"{v5d}/scores_v5.json", "w"))
    return f"ok {sub}/{name}: {total} cells"


def shard(items):
    return [consolidate_one(sub, name) for sub, name in items]


def _targets():
    out = []
    for sub in ["geneKO", "complex"]:
        for d in sorted(glob.glob(f"{ACC}/phase/{sub}/*")):
            if os.path.isdir(d) and "__to__" not in os.path.basename(d):
                out.append((sub, os.path.basename(d)))
    return out


def consolidate_anchors():
    """Consolidate the shared NTC real-cell anchors (real_dir=phase/_anchors/NTC): copy accpool cell0..24
    -> cell20..44 so the 'show real cells' row has all 45. Other _anchors/* (alt-anchor A cells) stay as-is."""
    v5d, accd = f"{V5}/phase/_anchors/NTC", f"{ACC}/phase/_anchors/NTC"
    acc = sorted((c for c in os.listdir(accd) if c.startswith("cell")), key=lambda x: int(x[4:]))
    for k in range(len(acc)):
        dst = f"{v5d}/cell{ATTN_N + k}"
        if not os.path.exists(dst):
            shutil.copytree(f"{accd}/cell{k}", dst)
    print(f"[anchors] NTC real anchors: {ATTN_N + len(acc)} cells")


def finalize_manifest():
    """Set n_cells=45 in manifest.json for every merged target (matches the consolidated meta)."""
    mf = f"{V5}/manifest.json"
    m = json.load(open(mf))
    merged = 0
    for mk in m["markers"]:
        for t in mk.get("targets", []):
            v5d = f"{V5}/{t['asset_dir']}"
            mp = f"{v5d}/meta.json"
            if os.path.exists(mp):
                nc = json.load(open(mp)).get("n_cells")
                if nc and nc != t.get("n_cells"):
                    t["n_cells"] = nc; merged += 1
    json.dump(m, open(mf, "w"))
    print(f"[manifest] updated n_cells on {merged} targets")


def main(n_shards=32):
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    items = _targets()
    shards = [s for s in (items[i::n_shards] for i in range(n_shards)) if s]
    jobs = [{"name": f"consolidate_{i}", "func": shard, "kwargs": {"items": s}} for i, s in enumerate(shards)]
    print(f"[consolidate] {len(items)} traversals across {len(jobs)} shards")
    submit_parallel_jobs(
        jobs, experiment="diffex_consolidate_cells",
        slurm_params={"slurm_partition": "cpu", "cpus_per_task": 4, "mem_gb": 16, "timeout_min": 90},
        log_dir="diffex_consolidate_cells", wait_for_completion=True)
    consolidate_anchors()
    finalize_manifest()


if __name__ == "__main__":
    main()
