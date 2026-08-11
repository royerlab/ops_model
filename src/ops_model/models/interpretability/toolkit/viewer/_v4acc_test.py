"""DIAGNOSTIC (removable): generate the 40S→60S A→B traversal anchored on v4-ACCURACY-ranked complex
cells, to disentangle 'attention vs accuracy ranking' from 'new v5 cells'. Compared against the
existing v4-attention (viewer_assets) and v5-accuracy (viewer_assets_v5) 40S→60S. Monkeypatches the
complex parquet to a v4-accuracy ribosomal parquet; output isolated under viewer_assets_v4acc_test."""
from . import catalog as C

V4ACC = "/hpc/projects/icd.fast.ops/models/diffex/viewer_assets_v5/_rankings/pma_v4acc_phase_complex_ribo.parquet"
C40 = "40S cytosolic small ribosomal subunit"
C60 = "60S cytosolic large ribosomal subunit"


def run():
    from ops_model.models.interpretability.diffae.classifier.config import GRAINS
    GRAINS["complex"]["parquet"] = V4ACC                 # both anchor (A) and target (B) from v4-accuracy cells
    from ops_model.models.interpretability.diffae.traversal.precompute import precompute_anchors_marker
    return precompute_anchors_marker(grain="complex", classes=[C40, C60],
                                     ckpt=f"{C.DD}/phase_v1/diffae_best.pt", out_root=C.OUT)
