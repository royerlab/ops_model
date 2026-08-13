"""Phase 'top-predictive cells' panel for the NEW-phenotype gene-KOs, MULTIBAG SHAP ranking
(pma_shap_phase_geneKO). NTC on top (a DISTINCT NTC cell per column, ranks 1-5), KO on bottom
(hand-picked from the phase_multibag montages). Real phase cells cropped from phenotyping_v3.zarr.

Run (SLURM): python figure4_setacc_panel_newpheno.py --submit
"""
import sys

import numpy as np
import pandas as pd

from ops_model.interpretability.diffae.figures._setacc_common import crop_pick_from_df, tile_at
from ops_model.interpretability.diffae.figures.setacc.figure4_setacc_panel import make_panel
from ops_model.paths import BASE_PATH

RANK = f"{BASE_PATH}/models/diffex/viewer_assets_v5/_rankings/pma_shap_phase_geneKO.parquet"
PHASE_CH = "Phase2D"

COLS = [  # KO rank = montage pick; NTC rank distinct per column (1-5)
    dict(slug="KIF23", key="KIF23", top_label="KIF23\n(multi-nucleation)", ko_rank=20, ntc_rank=1),
    dict(slug="CAPZB", key="CAPZB", top_label="CAPZB\n(stretched)", ko_rank=9, ntc_rank=2),
    dict(slug="SNRPD1", key="SNRPD1", top_label="SNRPD1\n(dark vacuoles)", ko_rank=84, ntc_rank=3),
    dict(slug="SAMM50", key="SAMM50", top_label="SAMM50\n(globular mito)", ko_rank=1, ntc_rank=4),
    dict(slug="RAB7A", key="RAB7A", top_label="RAB7A\n(enlarged vesicles)", ko_rank=4, ntc_rank=5),
]


def _df(cls):
    d = pd.read_parquet(RANK, filters=[("gene", "==", cls)])
    if "rank_type" in d.columns:
        d = d[d["rank_type"] == "top"]
    return d.sort_values("rank").reset_index(drop=True)


def column_tiles_shap(col):
    ko_raw, ko_recs, ko_pos = crop_pick_from_df(_df(col["key"]), col["ko_rank"], None, PHASE_CH, col["key"])
    ntc_raw, ntc_recs, ntc_pos = crop_pick_from_df(_df("NTC"), col["ntc_rank"], None, PHASE_CH, "NTC")
    lo, hi = np.percentile(np.concatenate([ko_raw.ravel(), ntc_raw.ravel()]), (1, 99))
    if hi - lo < 1e-6:
        hi = lo + 1
    ko_im, ko_r = tile_at(ko_raw, ko_recs, ko_pos, lo, hi)
    ntc_im, ntc_r = tile_at(ntc_raw, ntc_recs, ntc_pos, lo, hi)
    return ko_im, ntc_im, float(ko_r["score"]), float(ntc_r["score"])


def build():
    make_panel(COLS, "Top-predictive cells (phase)", "panelF_phase_newpheno",
               tiles_fn=column_tiles_shap, bottom_caption="Label-free 2D Phase",
               title_in=0.66)


def _job():
    import os
    os.environ.setdefault("OPS_DIFFEX_ASSETS", "viewer_assets_v5")
    build()


def submit():
    import os
    import pathlib
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    figdir = str(pathlib.Path(__file__).resolve().parent)
    os.environ["PYTHONPATH"] = figdir + os.pathsep + os.environ.get("PYTHONPATH", "")
    os.environ.setdefault("OPS_DIFFEX_ASSETS", "viewer_assets_v5")
    submit_parallel_jobs([{"name": "panelF_phase", "func": _job, "kwargs": {}}], experiment="diffex_panel",
                         slurm_params={"slurm_partition": "cpu", "cpus_per_task": 8, "mem_gb": 64, "timeout_min": 60},
                         log_dir="diffex_panel", wait_for_completion=False)


if __name__ == "__main__":
    submit() if (len(sys.argv) > 1 and sys.argv[1] == "--submit") else build()
