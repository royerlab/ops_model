"""Top-100 PHASE cell montages for hand-picking representative cells, ranked by the multibag SHAP ranking
(pma_shap_phase_geneKO). One montage per gene, rank-ordered with rank + cell-key + conf, inverse blue seg
mask (reuses _materialize + render_montage). For the fig-4 new-phenotype review.

Run: OPS_DIFFEX_ASSETS=viewer_assets_v5 python phase_multibag_montages.py [GENE ...]
"""
import sys

import pandas as pd

from _setacc_common import _materialize
import debug_setacc_top100 as D

OUT_DIR = "/hpc/projects/icd.fast.ops/analysis/figure4_shap_montages"   # shared SHAP-montage review dir (phase + fluor)
RANK = "/hpc/projects/icd.fast.ops/models/diffex/viewer_assets_v5/_rankings/pma_shap_phase_geneKO.parquet"
PHASE_CH = "Phase2D"
N = 100

GENES = [  # gene -> literature-backed KO phenotype to look for
    ("KIF23", "multi-nucleation"),
    ("CAPZB", "stretched morphology"),
    ("SNRPD1", "dark vacuoles"),
    ("SAMM50", "globular mitochondria"),
    ("RAB7A", "enlarged & increased vesicles / lysosomes"),
    ("NTC", "control (phase geneKO + complex NTC pool)"),
]


def main(genes=None):
    D.OUT = OUT_DIR                                    # set here (runs in SLURM worker; module-level is skipped by cloudpickle)
    want = set(genes) if genes else None
    for g, ph in GENES:
        if want and g not in want:
            continue
        d = pd.read_parquet(RANK, filters=[("gene", "==", g)])
        if "rank_type" in d.columns:
            d = d[d["rank_type"] == "top"]
        d = d.sort_values("rank").head(N)
        raw, recs = _materialize(d, None, PHASE_CH, g)
        D.render_montage(raw, recs, f"KO — {g} ({ph}) · phase · multibag SHAP rank", f"phase_multibag_{g}")


def _job(gene):
    import os
    os.environ.setdefault("OPS_DIFFEX_ASSETS", "viewer_assets_v5")
    main([gene])


def submit(genes=None):
    """One SLURM cpu job per gene (crop materialization is memory-heavy — dies on the login node)."""
    import os
    import pathlib
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    figdir = str(pathlib.Path(__file__).resolve().parent)
    os.environ["PYTHONPATH"] = figdir + os.pathsep + os.environ.get("PYTHONPATH", "")
    os.environ.setdefault("OPS_DIFFEX_ASSETS", "viewer_assets_v5")
    gs = genes or [g for g, _ in GENES]
    jobs = [{"name": f"phmont_{g}", "func": _job, "kwargs": {"gene": g}} for g in gs]
    submit_parallel_jobs(jobs, experiment="diffex_phmont",
                         slurm_params={"slurm_partition": "cpu", "cpus_per_task": 8, "mem_gb": 64, "timeout_min": 90},
                         log_dir="diffex_phmont", wait_for_completion=False)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--submit":
        submit(sys.argv[2:] or None)
    else:
        main(sys.argv[1:] or None)
