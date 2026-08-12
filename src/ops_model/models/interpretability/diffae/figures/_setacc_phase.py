"""Phase (label-free 2D Phase) set-accuracy groups for the panel-E-style figure. Phase uses single
global v5 rankings (not per-marker): geneKO keyed on `gene` (has NTC), complex keyed on
`predicted_class` (NO NTC → NTC pulled from the geneKO parquet). Channel = Phase2D. Reuses the
cropping/compositing from _setacc_common."""
import numpy as np
import pandas as pd

from ops_model.models.interpretability.diffae.figures._setacc_common import crop_pick_from_df, tile_at
from ops_model.paths import BASE_PATH

RANKS = f"{BASE_PATH}/models/diffex/viewer_assets_v5/_rankings"
PHASE_CH = "Phase2D"

# panel-E groups (published: TIMM23/Arp2-3 top, TIPARP/Core Mediator bottom); ko_rank/ntc_rank picked
# from the debug_phase montages.
GENE_COLS_PHASE = [
    dict(block="genes", key="TIMM23", top_label="TIMM23\n(gene-level)", ko_rank=1, ntc_rank=1),
    dict(block="genes", key="TIPARP", top_label="TIPARP\n(gene-level)", ko_rank=1, ntc_rank=1),
]
COMPLEX_COLS_PHASE = [
    dict(block="complexes", key="Actin-related protein 2/3 complex, ARPC1A-ACTR3B-ARPC5 variant",
         top_label="Arp2/3\ncomplex", ko_rank=8, ntc_rank=41),
    dict(block="complexes", key="Core mediator complex", top_label="Core Mediator\ncomplex",
         ko_rank=95, ntc_rank=46),
]
COLS_PHASE = GENE_COLS_PHASE + COMPLEX_COLS_PHASE


def phase_df(block, cls):
    if block == "genes":
        df = pd.read_parquet(f"{RANKS}/pma_v5_phase_geneKO.parquet")
        df = df[df["gene"] == cls]
    else:
        df = pd.read_parquet(f"{RANKS}/pma_v5_phase_complex.parquet")
        df = df[df["predicted_class"] == cls].copy()
        df["gene"] = cls                                    # so _materialize's gene->cls rename labels it
    df = df.sort_values("rank").reset_index(drop=True)
    if df.empty:
        raise ValueError(f"{cls!r} absent from phase {block}")
    return df


def phase_ntc():
    df = pd.read_parquet(f"{RANKS}/pma_v5_phase_geneKO.parquet")
    return df[df["gene"] == "NTC"].sort_values("rank").reset_index(drop=True)


def crop_pick_phase(block, cls, pick_rank, win=40):
    df = phase_ntc() if cls == "NTC" else phase_df(block, cls)
    return crop_pick_from_df(df, pick_rank, None, PHASE_CH, cls, win)


def column_tiles_phase(col):
    ko_raw, ko_recs, ko_pos = crop_pick_phase(col["block"], col["key"], col["ko_rank"])
    ntc_raw, ntc_recs, ntc_pos = crop_pick_phase(col["block"], "NTC", col["ntc_rank"])
    lo, hi = np.percentile(np.concatenate([ko_raw.ravel(), ntc_raw.ravel()]), (1, 99))
    if hi - lo < 1e-6:
        hi = lo + 1
    ko_im, ko_r = tile_at(ko_raw, ko_recs, ko_pos, lo, hi)
    ntc_im, ntc_r = tile_at(ntc_raw, ntc_recs, ntc_pos, lo, hi)
    return ko_im, ntc_im, float(ko_r["score"]), float(ntc_r["score"])
