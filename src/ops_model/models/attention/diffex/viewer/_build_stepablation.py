"""DDIM-step ablation: does the inverted-anchor scheme need >50 steps to resolve? Regenerate a small fixed set
of high-recovery geneKO at ddim_steps in {50,100,200}, everything else identical to valid200 (inverted, w=1.5,
same anchors + v5 directions). Score centroid recovery + SetTransformer (inline v5) + a sharpness metric to see
whether 100/200 steps recovers the loss the inversion introduced at 50. Small set — 100/200 steps are 2x/4x slow.

  VAL_STEPS=100 python -m ...viewer._build_stepablation gen
"""
import os, sys, json
from pathlib import Path
import numpy as np

from . import catalog as C
from .precompute import precompute_marker

STEPS = int(os.environ.get("VAL_STEPS", "50"))
_TREE = f"viewer_assets_stepabl_s{STEPS}"
_SRC = "viewer_assets_valid200"                 # reuse its 200-NTC anchors
PHASE_CK = f"{C.DD}/phase_v1/diffae_best.pt"
V5G = f"{C.OUT}/viewer_assets_v5/_rankings/pma_v5_phase_geneKO.parquet"
ALPHAS = (0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0)
NCELLS = 45                                     # bag-size-invariant → 45 keeps 200-step tractable
FT = "/hpc/projects/icd.fast.ops/analysis/figure4_traversals"


def sel(n=15):
    d = json.load(open(f"{FT}/gen_real_centroid/geneKO_scored.json")); al = d["alphas"]; by = d["by_alpha"]
    best = {}
    for a in al:
        for g, v in by[str(a)]["top1"].items():
            best[g] = max(best.get(g, 0), v)
    return [g for g, _ in sorted(best.items(), key=lambda x: -x[1])[:n]]


def setup():
    root = Path(C.OUT) / _TREE; (root / "phase").mkdir(parents=True, exist_ok=True)
    for name, tgt in [("_directions", Path(C.OUT) / "viewer_assets_v5" / "_directions"),
                      ("phase/_anchors", Path(C.OUT) / _SRC / "phase" / "_anchors")]:
        ln = root / name
        if not ln.exists():
            ln.symlink_to(tgt); print(f"[stepabl s{STEPS}] symlinked {name}")


def build(targets):
    os.environ["OPS_DIFFEX_ASSETS"] = _TREE
    from . import precompute as P
    P._ASSETS = _TREE
    return precompute_marker(grain="geneKO", targets=list(targets), ckpt=PHASE_CK, out_root=C.OUT,
                             control="NTC", n_cells=NCELLS, alphas=ALPHAS, invert_anchors=True, w=1.5,
                             force=False, v5_score=True, accuracy_parquet=V5G, load_workers=12,
                             ddim_steps=STEPS)


def main():
    if sys.argv[1:2] == ["gen"]:
        from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
        setup()
        genes = sel(15); print(f"[stepabl s{STEPS}] genes: {genes}")
        tmo = {50: 180, 100: 300, 200: 600}[STEPS]
        submit_parallel_jobs(jobs_to_submit=[{"name": f"stepabl_s{STEPS}", "func": build, "kwargs": {"targets": genes}}],
                             experiment=f"stepabl_s{STEPS}",
                             slurm_params={"slurm_partition": "preempted", "gpus_per_node": 1, "cpus_per_task": 12,
                                           "mem_gb": 96, "timeout_min": tmo, "slurm_constraint": "[a40|a6000|l40s]",
                                           "slurm_setup": [f"export OPS_DIFFEX_ASSETS={_TREE}", f"export VAL_STEPS={STEPS}",
                                                           "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"]},
                             log_dir=f"stepabl_s{STEPS}", wait_for_completion=False)
    else:
        raise SystemExit("usage: _build_stepablation gen  (set VAL_STEPS)")


if __name__ == "__main__":
    main()
