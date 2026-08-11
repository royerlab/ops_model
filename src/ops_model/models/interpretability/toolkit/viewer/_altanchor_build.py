"""Rebuild the A->B alt-anchor traversals in viewer_assets_v5 with the MULTIRANK ranking:
  - the existing directed pairs (from each __to__ dir's meta), both directions
  - 45 anchor cells per traversal, ranked top-45 by the MULTIRANK (pma_shap_phase_{geneKO,complex}.parquet)
    for phase, and the per-marker fluor_shap ranking for fluorescence
  - then score each with the v5 SetTransformer → P(target B) + rank_target
Sharded by anchor class so no two jobs write the same _anchors/<a> real-cell dir (force rebuilds it).
"""
import os, json, glob
from . import catalog as C
from ops_model.models.interpretability.diffae.classifier.config import slugify

ASSETS = "viewer_assets_v5"
ROOT = C.OUT
V5G = f"{ROOT}/{ASSETS}/_rankings/pma_shap_phase_geneKO.parquet"    # MULTIRANK (was pma_v5)
V5C = f"{ROOT}/{ASSETS}/_rankings/pma_shap_phase_complex.parquet"
FRANK = f"{ROOT}/{ASSETS}/_rankings/fluor_shap"                     # per-marker: {grain}/{slug}.parquet
PHASE_CK = f"{C.DD}/phase_v1/diffae_best.pt"
NEW = ""            # curated-new-pairs json (optional; rebuild uses the existing __to__ pairs)
N_CELLS = 45


PAIRS_JSON = os.path.join(os.path.dirname(__file__), "altanchor_pairs.json")   # canonical committed pair list


def _all_directed(grain):
    """Canonical undirected pairs (altanchor_pairs.json) → both directions. Returns (directed_pairs, classes)."""
    undirected = {tuple(sorted((a, b))) for a, b in json.load(open(PAIRS_JSON))[grain] if a != b}
    directed = [d for a, b in undirected for d in ((a, b), (b, a))]
    return directed, sorted({c for p in directed for c in p})


def gen_shard(grain, classes, pairs):
    os.environ["OPS_DIFFEX_ASSETS"] = ASSETS
    from . import precompute as P
    P._ASSETS = ASSETS
    parq = V5G if grain == "geneKO" else V5C
    P.precompute_anchors_marker(grain=grain, classes=classes, ckpt=PHASE_CK, out_root=ROOT,
                                marker_channel=None, channel="Phase2D", n_cells=N_CELLS,
                                pairs=pairs, accuracy_parquet=parq, force=True, v5_score=True)


def gen_shard_fluor(d, marker_channel, channel, grain, classes, pairs):
    """Fluor A→B alt-anchors for ONE marker: anchor cells from the marker's fluor_shap MULTIRANK ranking,
    morphed in the marker channel with that marker's DiffAE checkpoint."""
    os.environ["OPS_DIFFEX_ASSETS"] = ASSETS
    from . import precompute as P
    P._ASSETS = ASSETS
    parq = f"{FRANK}/{grain}/{slugify(marker_channel)}.parquet"
    P.precompute_anchors_marker(grain=grain, classes=classes, ckpt=f"{C.DD}/{d}/diffae_best.pt", out_root=ROOT,
                                marker_channel=marker_channel, channel=channel, n_cells=N_CELLS,
                                pairs=pairs, accuracy_parquet=parq, force=True, v5_score=True)


def submit_fluor():
    """Same canonical A→B pairs for every fluor marker, filtered to pairs whose BOTH classes exist in that
    marker's fluor_shap ranking (a marker only covers the geneKOs it's distinctive for). One shard per marker×grain."""
    import pandas as pd
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    jobs = []
    for d, mc, ch in C.complete_markers():
        slug = slugify(mc)
        for grain in ["geneKO", "complex"]:
            parq = f"{FRANK}/{grain}/{slug}.parquet"
            if not os.path.exists(parq):
                continue
            cc = "gene" if grain == "geneKO" else "predicted_class"
            avail = set(pd.read_parquet(parq, columns=[cc])[cc].astype(str).unique())
            directed, _ = _all_directed(grain)
            pairs = [(a, b) for a, b in directed if a in avail and b in avail]
            if not pairs:
                continue
            classes = sorted({c for p in pairs for c in p})
            jobs.append({"name": f"falt_{grain[0]}_{slug[:12]}", "func": gen_shard_fluor,
                         "kwargs": {"d": d, "marker_channel": mc, "channel": ch, "grain": grain, "classes": classes, "pairs": pairs}})
    print(f"[altanchor-fluor] {len(jobs)} marker×grain shards (canonical pairs, marker-filtered)")
    submit_parallel_jobs(jobs, experiment="diffex_altanchor_fluor",
                         slurm_params={"slurm_partition": "preempted", "gpus_per_node": 1, "cpus_per_task": 12,
                                       "mem_gb": 96, "timeout_min": 300, "slurm_constraint": "[a100_80|h100|h200|6000_blackwell]",
                                       "slurm_additional_parameters": {"requeue": True},
                                       "slurm_setup": ["export OPS_DIFFEX_ASSETS=viewer_assets_v5"]},
                         log_dir="diffex_altanchor_fluor", wait_for_completion=False)


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
    print(f"[altanchor] submitting {len(jobs)} phase gen shards (MULTIRANK)")
    submit_parallel_jobs(jobs, experiment="diffex_altanchor",
                         slurm_params={"slurm_partition": "gpu", "gpus_per_node": 1, "cpus_per_task": 12,
                                       "mem_gb": 96, "timeout_min": 300,
                                       "slurm_setup": ["export OPS_DIFFEX_ASSETS=viewer_assets_v5"]},
                         log_dir="diffex_altanchor", wait_for_completion=False)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "fluor":
        submit_fluor()
    else:
        main()   # phase A→B alt-anchors (multirank)
