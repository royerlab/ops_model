"""Re-decode the 3 morpho traversal targets and save LOSSLESS native-160 float frames (frames_f32.npz)
alongside the existing webp — so the generated seg is measured at the model's true output resolution, with
NO bilinear 160→256 up-sample and NO webp q=90 compression (the real side is already lossless from the zarr,
so this makes the real-vs-generated comparison genuinely apples-to-apples).

Targets (measurement modality matches what the violin segments):
  mTOR    → lysosome_LysoTracker_live_cell_dye / geneKO / MTOR          (LysoTracker ckpt)
  TIM23   → mitochondria_ChromaLIVE_561_excitation / complex / TIM23…   (ChromaLIVE_mito ckpt)
  POLR1B  → phase / geneKO / POLR1B                                     (phase ckpt, phase nucleoli)

Run: OPS_DIFFEX_ASSETS=viewer_assets_v5 python -m ops_model.models.attention.diffex.viewer.build_morpho_float
"""
from . import catalog as C
from ._build_v5_inverted import CFRP_DIR, FRP_DIR, PHASE_CK
from ..classifier.config import slugify

_ASSETS = "viewer_assets_v5"
TIM23 = "TIM23 mitochondrial inner membrane pre-sequence translocase complex, TIM17A variant"

# (label, precompute_marker kwargs) — score/v5_score off (traversal direction is independent of the classifier);
# force=True guarantees the decode runs; save_float=True dumps gen[c,ai] (ncell×A×160×160 float32, [-1,1]).
JOBS = [
    ("MTOR_lyso", dict(
        grain="geneKO", targets=["MTOR"], ckpt=f"{C.DD}/fluor_LysoTracker/diffae_best.pt",
        marker_channel="lysosome_LysoTracker live-cell dye", channel="GFP",
        fluor_rank_parquet=f"{FRP_DIR}/{slugify('lysosome_LysoTracker live-cell dye')}.parquet", n_cells=40)),
    ("TIM23_chromalive561", dict(
        grain="complex", targets=[TIM23], ckpt=f"{C.DD}/fluor_ChromaLIVE_mito/diffae_best.pt",
        marker_channel="mitochondria_ChromaLIVE 561 excitation", channel="mCherry",
        fluor_rank_parquet=f"{CFRP_DIR}/{slugify('mitochondria_ChromaLIVE 561 excitation')}.parquet", n_cells=40)),
    ("POLR1B_phase", dict(
        grain="geneKO", targets=["POLR1B"], ckpt=PHASE_CK, marker_channel=None, channel="Phase2D",
        accuracy_parquet=f"{C.OUT}/viewer_assets_v5/_rankings/pma_shap_phase_geneKO.parquet", n_cells=100)),
]

_COMMON = dict(out_root=C.OUT, control="NTC", w=1.5, invert_anchors=True, ddim_steps=100,
               force=True, save_float=True, score=False, v5_score=False, save_gemb=False)


CHUNK = 5                                                 # cells per shard — direction+anchors are cached, so shards reuse them race-free


def _float_job(kwargs):
    import os
    os.environ["OPS_DIFFEX_ASSETS"] = _ASSETS
    from . import precompute as P
    P._ASSETS = _ASSETS                                   # frames_f32.npz → <OUT>/viewer_assets_v5/<modality>/<grain>/<slug>/cell{c}/
    return P.precompute_marker(**{**_COMMON, **kwargs})


def submit():
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    jobs = []                                             # shard each target by cell_range → one GPU per ~CHUNK cells (fluor 2, phase 5)
    for lbl, kw in JOBS:
        n = kw["n_cells"]
        for lo in range(0, n, CHUNK):
            hi = min(lo + CHUNK, n)
            jobs.append({"name": f"mf_{lbl[:11]}_{lo}", "func": _float_job,
                         "kwargs": {"kwargs": {**kw, "cell_range": (lo, hi), "skip_webp": True}}})
    sp = {"slurm_partition": "gpu", "gpus_per_node": 1, "cpus_per_task": 12, "mem_gb": 96,
          "timeout_min": 120, "slurm_constraint": "[a100_80|h100|h200|6000_blackwell]",
          "slurm_setup": ["export OPS_DIFFEX_ASSETS=viewer_assets_v5",
                          "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"]}
    print(f"[morpho-float] submitting {len(jobs)} cell-sharded lossless-float re-decodes → viewer_assets_v5")
    submit_parallel_jobs(jobs_to_submit=jobs, experiment="diffex_mfloat", slurm_params=sp,
                         log_dir="diffex_mfloat", wait_for_completion=False)


def _extend_anchors(modality, marker_channel, channel, ntc_frp, n, grain):
    """Extend a fluor marker's DDIM anchor cache 40→n by gathering n NTC (geneKO ranking, how the 40 were built),
    keeping cells 0..have identical. ctrl_embs/mu_ctrl untouched (direction cache stays valid). Returns old `have`."""
    import numpy as np, pandas as pd
    from pathlib import Path
    from concurrent.futures import ThreadPoolExecutor
    from .precompute import _gather_class, _save_webp
    from ..diffae.data import normalize
    from ..directions.config import DirConfig
    rd = Path(C.OUT) / _ASSETS / modality / "_anchors" / "NTC"
    z = dict(np.load(rd / "ctrl.npz"))
    have = len(z["anchor_imgs"])
    if have >= n:
        print(f"[extend] {modality} already {have} anchors"); return have
    cfg = DirConfig(grain=grain, target="NTC", control="NTC", device="cuda")
    cfg.marker_channel = marker_channel; cfg.channel = channel
    cfg._fluor_rows = pd.read_parquet(ntc_frp)
    imgs, _emb = _gather_class(cfg, "NTC", n)
    realN = normalize(imgs[:n])
    realN[:have] = z["anchor_imgs"][:have]                # keep existing anchors byte-identical
    tp = ThreadPoolExecutor(8)
    for c in range(have, n):
        (rd / f"cell{c}").mkdir(parents=True, exist_ok=True); tp.submit(_save_webp, rd / f"cell{c}" / "real.webp", realN[c, 0], 256)
    tp.shutdown(wait=True)
    np.savez(rd / "ctrl.npz", ctrl_embs=z["ctrl_embs"], mu_ctrl=z["mu_ctrl"], anchor_imgs=realN)
    print(f"[extend] {modality} {have}→{n} anchors -> {rd/'ctrl.npz'}")
    return have


def _extend_job(kwargs, ntc_frp, n):
    import os
    os.environ["OPS_DIFFEX_ASSETS"] = _ASSETS
    from . import precompute as P
    P._ASSETS = _ASSETS
    modality = slugify(kwargs["marker_channel"])
    have = _extend_anchors(modality, kwargs["marker_channel"], kwargs["channel"], ntc_frp, n, "geneKO")   # NTC anchors are geneKO (even for complex targets)
    P.precompute_marker(**{**_COMMON, **kwargs, "n_cells": n, "cell_range": (have, n), "skip_webp": True})   # decode only the NEW cells (0..have kept)


def submit_extend(n=100):
    """Extend the 2 fluor markers' anchors to n and decode float frames for the new cells (phase already 100)."""
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    lyso = dict(JOBS[0][1]); mito = dict(JOBS[1][1])
    jobs = [
        {"name": "mfext_lyso", "func": _extend_job,
         "kwargs": {"kwargs": lyso, "ntc_frp": f"{FRP_DIR}/{slugify(lyso['marker_channel'])}.parquet", "n": n}},
        {"name": "mfext_mito", "func": _extend_job,                                     # mito NTC anchors come from the geneKO ranking
         "kwargs": {"kwargs": mito, "ntc_frp": f"{FRP_DIR}/{slugify(mito['marker_channel'])}.parquet", "n": n}},
    ]
    sp = {"slurm_partition": "gpu", "gpus_per_node": 1, "cpus_per_task": 12, "mem_gb": 96,
          "timeout_min": 180, "slurm_constraint": "[a100_80|h100|h200|6000_blackwell]",
          "slurm_setup": ["export OPS_DIFFEX_ASSETS=viewer_assets_v5",
                          "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"]}
    print(f"[morpho-float] extending fluor anchors → {n} + decoding new cells")
    submit_parallel_jobs(jobs_to_submit=jobs, experiment="diffex_mfext", slurm_params=sp,
                         log_dir="diffex_mfext", wait_for_completion=False)


if __name__ == "__main__":
    import sys
    submit_extend() if "--extend" in sys.argv else submit()
