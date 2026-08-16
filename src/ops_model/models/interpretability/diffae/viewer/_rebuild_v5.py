"""Rebuild v5 phase traversals with the CORRECTED anchor: v4 typical NTC anchor cells (pre-copied into
the shared _anchors/NTC cache, so the control gather is skipped) + v5 KD centroids (accuracy_parquet),
scoring the v5 SetTransformer inline (reuses gemb → no separate re-decode/re-embed pass). force=True
recomputes directions (v5_KD − v4_NTC) and frames. Removable after the rebuild."""
from . import catalog as C
from .precompute import precompute_marker

V5G = "/hpc/projects/icd.fast.ops/models/diffex/viewer_assets_v5/_rankings/pma_v5_phase_geneKO.parquet"
V5C = "/hpc/projects/icd.fast.ops/models/diffex/viewer_assets_v5/_rankings/pma_v5_phase_complex.parquet"
PHASE_CK = f"{C.DD}/phase_v1/diffae_best.pt"


def rebuild_shard(grain, targets):
    parq = V5G if grain == "geneKO" else V5C
    return precompute_marker(grain=grain, targets=list(targets), ckpt=PHASE_CK, out_root=C.OUT,
                             control="NTC", accuracy_parquet=parq, v5_score=True, force=True)


def accanchor_shard(grain, targets):
    """Reproduce the ORIGINAL v5 (accuracy-selected NTC anchor + v5 KD): run with OPS_DIFFEX_V5=1 so GRAINS
    parquet = v5 for BOTH control and KD (no accuracy_parquet override), isolated in viewer_assets_v5_accanchor."""
    return precompute_marker(grain=grain, targets=list(targets), ckpt=PHASE_CK, out_root=C.OUT,
                             control="NTC", v5_score=True, force=True)


SEL25 = "/hpc/projects/icd.fast.ops/analysis/figure4_traversals/ntc_accanchor_selected25.csv"


def build_accpool_anchor():
    """Accuracy-pool NTC anchor = the 25 hand-picked quality cells (z0 + control). Materialize+embed them and
    write ctrl.npz + cell0-24 real.webp into OPS_DIFFEX_ASSETS/phase/_anchors/NTC/ (viewer_assets_v5_accpool)."""
    import numpy as np, pandas as pd
    from pathlib import Path
    from concurrent.futures import ThreadPoolExecutor
    from .precompute import _gather_class, _ASSETS, _save_webp
    from ..generator.data import normalize
    from ..directions.config import DirConfig
    sel = pd.read_csv(SEL25)
    parq = pd.DataFrame({"gene": "NTC", "experiment": sel.experiment, "well": sel.well, "segmentation": sel.segmentation,
                         "x_pheno": sel.x_pheno, "y_pheno": sel.y_pheno, "pma_attention": sel.pma_attention,
                         "rank": range(1, len(sel) + 1), "rank_type": "top"})
    ptmp = f"{C.OUT}/{_ASSETS}/_ntc25.parquet"; Path(ptmp).parent.mkdir(parents=True, exist_ok=True); parq.to_parquet(ptmp)
    cfg = DirConfig(grain="geneKO", target="NTC", device="cuda")
    imgs, embs = _gather_class(cfg, "NTC", 25, parquet=ptmp)
    realdir = Path(C.OUT) / _ASSETS / "phase" / "_anchors" / "NTC"; realdir.mkdir(parents=True, exist_ok=True)
    real = normalize(imgs); tp = ThreadPoolExecutor(8)
    for c in range(len(real)):
        (realdir / f"cell{c}").mkdir(parents=True, exist_ok=True)
        tp.submit(_save_webp, realdir / f"cell{c}" / "real.webp", real[c, 0], 256)
    tp.shutdown(wait=True)
    np.savez(realdir / "ctrl.npz", ctrl_embs=embs, mu_ctrl=embs.mean(0))
    print(f"accpool anchor built: {embs.shape} -> {realdir}")


def accpool_shard(grain, targets):
    """Generate accuracy-pool traversals: pre-built 25 hand-picked anchor + v5 direction (OPS_DIFFEX_V5=1),
    the SAME ±5 α grid, n_cells=25, set-acc over a fixed 20-cell bag (v5_bag=20) for cross-approach comparison."""
    return precompute_marker(grain=grain, targets=list(targets), ckpt=PHASE_CK, out_root=C.OUT,
                             control="NTC", n_cells=25, v5_score=True, v5_bag=20, force=True)
