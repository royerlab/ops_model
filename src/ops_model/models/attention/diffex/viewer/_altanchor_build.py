"""Rebuild + expand the A->B alt-anchor traversals in viewer_assets_v5:
  - existing 40 pairs + 40 new curated pairs, BOTH directions (forward + reverse) → ~160 directed traversals
  - 45 anchor cells per traversal, ranked top-45 by v5 accuracy (pma_v5_phase_{geneKO,complex}.parquet)
  - then score each with the v5 SetTransformer → P(target B) + rank_target (both confidence scores)
Sharded by anchor class so no two jobs write the same _anchors/<a> real-cell dir (force rebuilds it).
"""
import os, json, glob
from . import catalog as C

ASSETS = "viewer_assets_v5"
ROOT = C.OUT
V5G = f"{ROOT}/{ASSETS}/_rankings/pma_v5_phase_geneKO.parquet"
V5C = f"{ROOT}/{ASSETS}/_rankings/pma_v5_phase_complex.parquet"
PHASE_CK = f"{C.DD}/phase_v1/diffae_best.pt"
NEW = "/tmp/claude-5957/-hpc-mydata-gav-sturm-ops-mono/33b63e8b-6d9f-4a90-8aa6-6681c0dc8408/scratchpad/new_pairs.json"
N_CELLS = 45


def _existing_pairs(grain):
    """(a,b) real names from each existing __to__ dir's meta (control=a, target=b)."""
    out = []
    for d in glob.glob(f"{ROOT}/{ASSETS}/phase/{grain}/*__to__*"):
        m = json.load(open(f"{d}/meta.json"))
        out.append((m["control"], m["target"]))
    return out


def _all_directed(grain):
    """Undirected {existing ∪ new} → both directions. Returns (directed_pairs, unique_classes)."""
    new = json.load(open(NEW))[grain]
    undirected = {frozenset((a, b)) for a, b in _existing_pairs(grain) + [tuple(p) for p in new] if a != b}
    directed = []
    for s in undirected:
        a, b = tuple(s) if len(s) == 2 else (next(iter(s)), next(iter(s)))
        directed += [(a, b), (b, a)]
    classes = sorted({c for p in directed for c in p})
    return directed, classes


def gen_shard(grain, classes, pairs):
    os.environ["OPS_DIFFEX_ASSETS"] = ASSETS
    from . import precompute as P
    P._ASSETS = ASSETS
    parq = V5G if grain == "geneKO" else V5C
    P.precompute_anchors_marker(grain=grain, classes=classes, ckpt=PHASE_CK, out_root=ROOT,
                                marker_channel=None, channel="Phase2D", n_cells=N_CELLS,
                                pairs=pairs, accuracy_parquet=parq, force=True)


def score_shard(grain):
    os.environ["OPS_DIFFEX_ASSETS"] = ASSETS
    from . import score_generated as SG
    SG.V5_BASE = f"{ROOT}/{ASSETS}/phase"
    SG.score_anchor_traversals(grain)


def main(n_shards=10):
    os.environ["OPS_DIFFEX_ASSETS"] = ASSETS
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    jobs = []
    for grain in ["geneKO", "complex"]:
        directed, _ = _all_directed(grain)
        by_anchor = {}
        for a, b in directed:
            by_anchor.setdefault(a, []).append((a, b))
        anchors = sorted(by_anchor)
        shards = [anchors[i::n_shards] for i in range(n_shards)]
        for i, sh in enumerate(shards):
            if not sh:
                continue
            pairs = [p for a in sh for p in by_anchor[a]]
            classes = sorted({c for p in pairs for c in p})
            jobs.append({"name": f"altanchor_{grain}_{i}", "func": gen_shard,
                         "kwargs": {"grain": grain, "classes": classes, "pairs": pairs}})
        print(f"[altanchor] {grain}: {len(directed)} directed pairs, {len(anchors)} anchors")
    print(f"[altanchor] submitting {len(jobs)} GPU gen shards")
    submit_parallel_jobs(jobs, experiment="diffex_altanchor",
                         slurm_params={"slurm_partition": "gpu", "slurm_gres": "gpu:1",
                                       "cpus_per_task": 12, "mem_gb": 96, "timeout_min": 240},
                         log_dir="diffex_altanchor", wait_for_completion=True)
    # score both grains (P(target)+rank) once generation is done
    sjobs = [{"name": f"altanchor_score_{g}", "func": score_shard, "kwargs": {"grain": g}} for g in ["geneKO", "complex"]]
    submit_parallel_jobs(sjobs, experiment="diffex_altanchor_score",
                         slurm_params={"slurm_partition": "gpu", "slurm_gres": "gpu:1",
                                       "cpus_per_task": 8, "mem_gb": 64, "timeout_min": 180},
                         log_dir="diffex_altanchor_score", wait_for_completion=False)


if __name__ == "__main__":
    main()
