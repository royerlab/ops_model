"""Rebuild the 8 fig-4 morpho-traversal examples at n=100 cells (tight SEM). Per marker: back up + clear
the NTC anchor cache, re-gather 100 anchors + regenerate the picked target traversals (force), reusing the
EXACT v5 build config (phase = accuracy_parquet; fluor = fluor_rank_parquet). The NTC anchor cache is
per-modality (shared geneKO+complex), so the geneKO pass builds the 100-cell cache from the geneKO parquet
(thousands of NTC) and the complex pass reuses it (complex parquets only have ~30 NTC). _gather_class is
rank-deterministic → cells 0-39 preserved, other genes' existing traversals stay valid.

Then remeasure metrics:  OPS_DIFFEX_ASSETS=viewer_assets_v5 submit phase-morpho --n-cells 100
Usage: python rebuild_traversals_n100.py [modality ...]   (default = all 6)
"""
import os
import shutil
import sys
from pathlib import Path

from ops_model.models.attention.diffex.viewer import catalog as C
from ops_model.models.attention.diffex.classifier.config import slugify

ASSETS = "viewer_assets_v5"
RANK = f"{C.OUT}/{ASSETS}/_rankings/fluor_shap"
V5G = f"{C.OUT}/{ASSETS}/_rankings/pma_shap_phase_geneKO.parquet"
V5C = f"{C.OUT}/{ASSETS}/_rankings/pma_shap_phase_complex.parquet"
PHASE_CK = f"{C.DD}/phase_v1/diffae_best.pt"
NCELLS = 100

FLUOR = {  # modality -> (diffae_dir, marker_channel, channel)
    "nucleus_NucleoLIVE_Live_Cell_dye": ("fluor_NucleoLive", "nucleus_NucleoLIVE Live Cell dye", "mCherry"),
    "nucleolus_GC_NPM3": ("fluor_NPM3", "nucleolus-GC_NPM3", "GFP"),
    "mitochondria_ChromaLIVE_561_excitation": ("fluor_ChromaLIVE_mito", "mitochondria_ChromaLIVE 561 excitation", "mCherry"),
    "ER_Golgi_COP_II_SEC23A": ("fluor_ER_Golgi_COP_II_SEC23A", "ER/Golgi COP-II_SEC23A", "GFP"),
    "actin_filament_FastAct_SPY555_Live_Cell_Dye": ("fluor_FastAct", "actin filament_FastAct_SPY555 Live Cell Dye", "mCherry"),
    "lysosome_LysoTracker_live_cell_dye": ("fluor_LysoTracker", "lysosome_LysoTracker live-cell dye", "GFP"),
    "lipid_droplet_BODIPY_live_cell_dye": ("fluor_lipid_droplet_BODIPY_live_cell_dye", "lipid droplet_BODIPY live cell dye", "GFP"),
    "clathrin_vesicles_CLTA": ("fluor_clathrin_vesicles_CLTA", "clathrin vesicles_CLTA", "GFP"),
    "stress_granule_G3BP1": ("fluor_stress_granule_G3BP1", "stress granule_G3BP1", "GFP"),
    "chromatin_H2BC21": ("fluor_chromatin_H2BC21", "chromatin_H2BC21", "mCherry"),
    "nucleolus_DFC_FBL": ("fluor_nucleolus_DFC_FBL", "nucleolus-DFC_FBL", "GFP"),
    "autophagosome_MAP1LC3B": ("fluor_autophagosome_MAP1LC3B", "autophagosome_MAP1LC3B", "GFP"),
    "lysosome_LAMP1": ("fluor_lysosome_LAMP1", "lysosome_LAMP1", "GFP"),
    "proteasome_PSMB7": ("fluor_proteasome_PSMB7", "proteasome_PSMB7", "GFP"),
    "F_actin_Phalloidin": ("fluor_F_actin_Phalloidin", "F-actin_Phalloidin", "CP1_f_actin_Phalloidin"),
}
TARGETS = {  # modality -> {grain: [class,...]}  (the 8 picked fig-4 examples)
    "phase": {"geneKO": ["TOMM20", "MICOS13", "SAMM50"]},
    "nucleus_NucleoLIVE_Live_Cell_dye": {"geneKO": ["KIF23"]},
    "nucleolus_GC_NPM3": {"geneKO": ["POLR1B"], "complex": ["Chaperonin-containing T-complex"]},
    "mitochondria_ChromaLIVE_561_excitation": {"geneKO": ["TOMM20"],
        "complex": ["TIM23 mitochondrial inner membrane pre-sequence translocase complex, TIM17A variant"]},
    "ER_Golgi_COP_II_SEC23A": {"geneKO": ["GBF1"]},
    "actin_filament_FastAct_SPY555_Live_Cell_Dye": {"geneKO": ["CAPZB"]},
    "lysosome_LysoTracker_live_cell_dye": {"geneKO": ["LAMTOR2"]},
    "lipid_droplet_BODIPY_live_cell_dye": {"geneKO": ["RAB7A"]},
    "clathrin_vesicles_CLTA": {"geneKO": ["AP2M1"]},
    "stress_granule_G3BP1": {"geneKO": ["EIF2S2"]},
    "chromatin_H2BC21": {"geneKO": ["AURKB"]},
    "nucleolus_DFC_FBL": {"geneKO": ["NOP56"]},
    "autophagosome_MAP1LC3B": {"geneKO": ["ATG9A"]},
    "lysosome_LAMP1": {"geneKO": ["ATP6V1B2"]},
    "proteasome_PSMB7": {"geneKO": ["PSMB6"]},
    "F_actin_Phalloidin": {"geneKO": ["CAPZB"]},
}


def _clear_anchor(modality):
    """Back up the old (40-45 cell) ctrl.npz so the next gather rebuilds the cache fresh at NCELLS.
    Idempotent: if the cache is already >= NCELLS, keep it (so re-runs reuse the same 100 anchors)."""
    import numpy as np
    ad = Path(C.OUT) / ASSETS / modality / "_anchors" / "NTC"
    ck = ad / "ctrl.npz"
    if ck.exists():
        z = np.load(ck)
        have = z["anchor_imgs"].shape[0] if "anchor_imgs" in z.files else 0
        if have >= NCELLS:
            print(f"[keep] {modality}: anchor cache already {have} >= {NCELLS} — reuse"); return
        bak = ad / "ctrl.npz.n40bak"
        if bak.exists():
            bak.unlink()
        ck.rename(bak)
        print(f"[clear] {modality}: ctrl.npz ({have}) -> ctrl.npz.n40bak (rebuild @ {NCELLS})")


def rebuild_marker(modality):
    os.environ["OPS_DIFFEX_ASSETS"] = ASSETS
    from ops_model.models.attention.diffex.viewer import precompute as P
    P._ASSETS = ASSETS
    _clear_anchor(modality)
    tg = TARGETS[modality]
    if modality == "phase":
        P.precompute_marker(grain="geneKO", targets=tg["geneKO"], ckpt=PHASE_CK, out_root=C.OUT,
                            control="NTC", accuracy_parquet=V5G, v5_score=True, n_cells=NCELLS, force=True)
        return
    d, mc, ch = FLUOR[modality]
    if "geneKO" in tg:                                  # geneKO pass FIRST → builds the 100-cell NTC cache
        P.precompute_marker(grain="geneKO", targets=tg["geneKO"], ckpt=f"{C.DD}/{d}/diffae_best.pt", out_root=C.OUT,
                            marker_channel=mc, channel=ch, control="NTC", n_cells=NCELLS,
                            fluor_rank_parquet=f"{RANK}/geneKO/{slugify(mc)}.parquet", v5_score=True,
                            load_workers=12, force=True)
    if "complex" in tg:                                 # reuses the cache built above
        P.precompute_marker(grain="complex", targets=tg["complex"], ckpt=f"{C.DD}/{d}/diffae_best.pt", out_root=C.OUT,
                            marker_channel=mc, channel=ch, control="NTC", n_cells=NCELLS,
                            fluor_rank_parquet=f"{RANK}/complex/{slugify(mc)}.parquet", v5_score=True,
                            load_workers=12, force=True)


def gen_phase(targets):
    """Generate specific phase geneKO targets at n=100 (reuses the existing 100-cell phase anchor cache)."""
    os.environ["OPS_DIFFEX_ASSETS"] = ASSETS
    from ops_model.models.attention.diffex.viewer import precompute as P
    P._ASSETS = ASSETS
    _clear_anchor("phase")                                   # idempotent: keeps the 100-cell cache
    P.precompute_marker(grain="geneKO", targets=targets, ckpt=PHASE_CK, out_root=C.OUT,
                        control="NTC", accuracy_parquet=V5G, v5_score=True, n_cells=NCELLS, force=True)


def gen_phase_complex(targets):
    """Generate phase COMPLEX targets at n=100 (full complex names; reuses the 100-cell phase anchor cache)."""
    os.environ["OPS_DIFFEX_ASSETS"] = ASSETS
    from ops_model.models.attention.diffex.viewer import precompute as P
    P._ASSETS = ASSETS
    _clear_anchor("phase")
    P.precompute_marker(grain="complex", targets=targets, ckpt=PHASE_CK, out_root=C.OUT,
                        control="NTC", accuracy_parquet=V5C, v5_score=True, n_cells=NCELLS, force=True)


def submit_phase_targets(targets, tag, func=None):
    """One SLURM job PER target (parallel across GPUs) — the phase-100 anchor cache already exists and
    _clear_anchor is idempotent (>=100 → keep), so per-target jobs reuse it read-only (no re-gather race)."""
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    fn = func or gen_phase
    jobs = [{"name": f"trav100_{tag}_{slugify(str(t))[:14]}", "func": fn, "kwargs": {"targets": [t]}} for t in targets]
    submit_parallel_jobs(jobs, experiment="diffex_trav100",
                         slurm_params={"slurm_partition": "gpu", "slurm_gres": "gpu:1", "cpus_per_task": 12,
                                       "mem_gb": 96, "timeout_min": 300},
                         log_dir="diffex_trav100", wait_for_completion=False)


def gen_phase_chunk(targets, cell_range):
    """Generate a CELL-RANGE slice of phase geneKO targets. Parallel chunks share the cached 100-anchor + the
    cached direction (both ckpt-independent), each writing its own cell{c} dirs → parallelizes the per-cell
    DDIM inversion across GPUs instead of one long serial job. v5 scoring skipped (whole-target only)."""
    os.environ["OPS_DIFFEX_ASSETS"] = ASSETS
    from ops_model.models.attention.diffex.viewer import precompute as P
    P._ASSETS = ASSETS
    _clear_anchor("phase")                                   # idempotent: 100-anchor cache kept
    P.precompute_marker(grain="geneKO", targets=targets, ckpt=PHASE_CK, out_root=C.OUT,
                        control="NTC", accuracy_parquet=V5G, v5_score=False, n_cells=NCELLS, force=True,
                        cell_range=cell_range)


def submit_phase_chunked(targets, tag, n_chunks=5):
    """One SLURM job per (target, cell-chunk) — parallelize NCELLS across GPUs (not longer walltime)."""
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    step = (NCELLS + n_chunks - 1) // n_chunks
    jobs = []
    for t in targets:
        for i in range(n_chunks):
            s, e = i * step, min((i + 1) * step, NCELLS)
            if s >= e:
                continue
            jobs.append({"name": f"tv_{tag}_{slugify(str(t))[:8]}_{s}", "func": gen_phase_chunk,
                         "kwargs": {"targets": [t], "cell_range": (s, e)}})
    print(f"[chunked] {len(jobs)} jobs ({len(targets)} targets x {n_chunks} chunks of ~{step} cells)")
    submit_parallel_jobs(jobs, experiment="diffex_trav100",
                         slurm_params={"slurm_partition": "gpu", "slurm_gres": "gpu:1", "cpus_per_task": 12,
                                       "mem_gb": 96, "timeout_min": 120},
                         log_dir="diffex_trav100", wait_for_completion=False)


def main():
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    mods = sys.argv[1:] or list(TARGETS)
    jobs = [{"name": f"trav100_{slugify(m)[:16]}", "func": rebuild_marker, "kwargs": {"modality": m}} for m in mods]
    print(f"[trav-n100] {len(jobs)} marker job(s) @ n_cells={NCELLS}: {mods}")
    submit_parallel_jobs(jobs, experiment="diffex_trav100",
                         slurm_params={"slurm_partition": "gpu", "slurm_gres": "gpu:1", "cpus_per_task": 12,
                                       "mem_gb": 96, "timeout_min": 720},
                         log_dir="diffex_trav100", wait_for_completion=False)


if __name__ == "__main__":
    main()
