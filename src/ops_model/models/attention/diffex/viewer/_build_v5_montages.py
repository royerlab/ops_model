"""Per-marker embedding montages for completed v5 markers.

Same as the v4 montage build (alphas 1-5, cells 0-19, umap+phate) EXCEPT each marker's tiles are laid out
on ITS OWN gene embedding (marker_leaves.embedding_h5ad → paper_v2/markers/<leaf>/…) instead of the shared
phase embedding. Reads the merged inverted frames from viewer_assets_v5/<slug>/geneKO and writes tiles to
viewer_assets_v5/_montage/. Only builds markers whose geneKO is 100% present in viewer_assets_v5.

    python -m ops_model.models.attention.diffex.viewer._build_v5_montages
"""
import glob
import os

from . import catalog as C
from . import marker_leaves as ML
from .build_umap_montage import OUT
from ..classifier.config import slugify

V5 = "viewer_assets_v5"
CELLS = list(range(20))
ALPHAS = [1.0, 2.0, 3.0, 4.0, 5.0]
EMBS = ["umap", "phate"]


BACKUP = f"{OUT}/viewer_assets_v5_preinvert_backup"   # only markers we MERGED (inverted) have a backup here


def _completed_in_v5():
    """(marker_channel, slug) for markers whose INVERTED geneKO was merged into viewer_assets_v5 — i.e. the
    old build was moved to viewer_assets_v5_preinvert_backup/<slug>/ AND geneKO is 100% present in v5. This
    excludes markers still showing the OLD non-inverted build (no backup) so we only montage inverted frames."""
    from ._build_v5_inverted import _genes_for, FRP_DIR
    out = []
    for d, mc, ch in C.complete_markers():
        s = slugify(mc)
        frp = f"{FRP_DIR}/{s}.parquet"
        if not os.path.exists(frp):   # all v5 fluor is inverted now (_inv retired) → gate only on a ranking + complete geneKO
            continue
        exp = len(_genes_for(frp))
        dn = len([x for x in glob.glob(f"{OUT}/{V5}/{s}/geneKO/*")
                  if os.path.isdir(x) and "__to__" not in os.path.basename(x) and os.path.exists(f"{x}/meta.json")])
        if exp > 0 and dn >= exp:
            out.append((mc, s))
    return out


def mont_job(marker_channel, slug, cell, alpha, emb):
    """One (marker, cell, α, emb) montage → viewer_assets_v5/_montage, laid out on the marker's own embedding."""
    os.environ["OPS_DIFFEX_ASSETS"] = V5
    from . import build_umap_montage as BM
    BM._ASSETS = V5                                         # runtime override (import-time snapshot is unreliable)
    h5 = ML.embedding_h5ad(marker_channel) or f"{ML.PHASE_LEAF}/gene_embedding_pca_optimized.h5ad"
    oz = f"{OUT}/{V5}/_montage/{slug}_geneKO_{emb}_cell{cell}_a{alpha:g}.zarr"
    return BM.build_montage_web(h5ad=h5, out_zarr=oz, cell=cell, alpha=alpha, embedding=emb, modality=slug)


PHASE_CELLS = list(range(45))   # phase anchors = 45 cells (20 attention + 25 accuracy)


def phase_mont_job(cell, alpha, emb):
    """Phase montage on the phase gene embedding. Reads the inverted frames from viewer_assets_v5/phase
    (phase swapped into production 2026-07-23), writes tiles to viewer_assets_v5/_montage."""
    os.environ["OPS_DIFFEX_ASSETS"] = V5
    from . import build_umap_montage as BM
    BM._ASSETS = V5
    h5 = ML.embedding_h5ad(None)                        # phase_only leaf gene embedding
    oz = f"{OUT}/{V5}/_montage/phase_geneKO_{emb}_cell{cell}_a{alpha:g}.zarr"
    return BM.build_montage_web(h5ad=h5, out_zarr=oz, cell=cell, alpha=alpha, embedding=emb, modality="phase")


def submit_phase():
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    jobs = [{"name": f"mtg5_phase_{emb[:2]}_c{cell}_a{a:g}", "func": phase_mont_job,
             "kwargs": {"cell": cell, "alpha": a, "emb": emb}}
            for emb in EMBS for cell in PHASE_CELLS for a in ALPHAS]
    print(f"[v5mont] phase: {len(jobs)} montage jobs ({len(PHASE_CELLS)} cells × {len(ALPHAS)} α × {len(EMBS)} emb) → viewer_assets_v5/_montage")
    submit_parallel_jobs(
        jobs_to_submit=jobs, experiment="diffex_v5mont",
        slurm_params={"slurm_partition": "cpu", "cpus_per_task": 4, "mem_gb": 24, "timeout_min": 60,
                      "slurm_array_parallelism": 100,
                      "slurm_setup": ["export OPS_DIFFEX_ASSETS=viewer_assets_v5_inv"]},
        log_dir="diffex_v5mont", wait_for_completion=False)


def main():
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "phase":
        submit_phase(); return
    force = "force" in sys.argv[1:]      # mtime skip is unreliable (frames overwritten in place don't bump the dir mtime) → force a full rebuild
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    comp = _completed_in_v5()
    per_marker = sum(1 for mc, _ in comp if ML.embedding_h5ad(mc))
    print(f"[v5mont] {len(comp)} completed markers in v5 ({per_marker} with own embedding, "
          f"{len(comp) - per_marker} phase-fallback){' [FORCE]' if force else ''}")
    jobs = []
    for mc, s in comp:
        gk = f"{OUT}/{V5}/{s}/geneKO"
        cm = os.path.getmtime(gk) if os.path.isdir(gk) else 0     # content-aware skip: rebuild only if stale
        for emb in EMBS:
            for cell in CELLS:
                for a in ALPHAS:
                    tj = f"{OUT}/{V5}/_montage/{s}_geneKO_{emb}_cell{cell}_a{a:g}_tiles/tiles.json"
                    if not force and os.path.exists(tj) and os.path.getmtime(tj) >= cm:
                        continue                                  # montage already reflects current cache
                    jobs.append({"name": f"mtg5_{s[:10]}_{emb[:2]}_c{cell}_a{a:g}", "func": mont_job,
                                 "kwargs": {"marker_channel": mc, "slug": s, "cell": cell, "alpha": a, "emb": emb}})
    print(f"[v5mont] {len(jobs)} montage jobs ({len(comp)}×{len(EMBS)}×{len(CELLS)}×{len(ALPHAS)}) → viewer_assets_v5/_montage")
    submit_parallel_jobs(
        jobs_to_submit=jobs, experiment="diffex_v5mont",
        slurm_params={"slurm_partition": "cpu", "cpus_per_task": 4, "mem_gb": 24, "timeout_min": 60,
                      "slurm_array_parallelism": 100,
                      "slurm_setup": ["export OPS_DIFFEX_ASSETS=viewer_assets_v5"]},
        log_dir="diffex_v5mont", wait_for_completion=False)


if __name__ == "__main__":
    main()
