"""DIAGNOSTIC (removable): regenerate a couple of geneKO traversals with v4 NTC anchor cells
(attention-ranked → typical, in the DiffAE training distribution) but v5 KD centroids, to test
whether the v5 graininess / extreme-alpha blowout comes from selecting NTC anchors by set-accuracy
(which picks the most extreme/atypical control cells) instead of attention.

Launch with OPS_DIFFEX_ASSETS=viewer_assets_v5_anchortest and OPS_DIFFEX_V5 UNSET so the control
(NTC) gather reads the v4 parquet while the KD gather uses the v5 accuracy parquet override.
Output is isolated under viewer_assets_v5_anchortest/. Delete this module after the diagnosis."""
from . import catalog as C
from ops_model.models.interpretability.diffae.traversal.precompute import precompute_marker

V5_GENEKO = "/hpc/projects/icd.fast.ops/models/diffex/viewer_assets_v5/_rankings/pma_v5_phase_geneKO.parquet"
PHASE_CK = f"{C.DD}/phase_v1/diffae_best.pt"


def run_anchor_test(targets=("HSPA5", "SRSF3")):
    # control="NTC" → v4 parquet (OPS_DIFFEX_V5 unset); accuracy_parquet → v5 KD centroids.
    return precompute_marker(grain="geneKO", targets=list(targets), ckpt=PHASE_CK, out_root=C.OUT,
                             control="NTC", accuracy_parquet=V5_GENEKO, score=False, force=True)
