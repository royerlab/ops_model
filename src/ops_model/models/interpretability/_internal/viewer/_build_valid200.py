"""Scaled-up PHASE validation traversals: 200 NTC anchor cells × all ~1000 geneKO, forward alphas only
(0, 0.5, 1, 2, 3, 4, 5), DDIM-inverted anchors (faithful α=0, w=1.5) — same generator as the v5 production
build, just more cells and fewer alphas, written to a SEPARATE sibling tree so it stays out of the viewer.

Output: {C.OUT}/viewer_assets_valid200/phase/geneKO/<slug>/cell<c>/frame_<i>.webp  (+ scores_v5.json per target)
  (sibling of viewer_assets_v5 — can be symlinked into the viewer later if wanted.)

Reuses the v5 per-class directions (d_vec + gap are the PRODUCTION values we are validating) via a symlinked
_directions tree, so no direction re-fit — only the 200-anchor inversion + 200×7 decodes per target.

  python -m ops_model.models.interpretability._internal.viewer._build_valid200 anchor    # 1 GPU: build the 200-cell NTC anchor
  python -m ops_model.models.interpretability._internal.viewer._build_valid200 submit     # shard all 1000 geneKO (after anchor)
  python -m ops_model.models.interpretability._internal.viewer._build_valid200 all        # anchor job -> shards (afterok dep)
"""
import os
import sys
from pathlib import Path

from ops_model.models.interpretability.diffae.classifier.config import slugify
from . import catalog as C
from ops_model.models.interpretability.diffae.traversal.precompute import precompute_marker

W = float(os.environ.get("VALID200_W", "1.5"))                           # CFG guidance weight (baseline recovery used w=2.0)
_VALID = "viewer_assets_valid200" if W == 1.5 else f"viewer_assets_valid200_w{W:g}"
PHASE_CK = f"{C.DD}/phase_v1/diffae_best.pt"
V5G = f"{C.OUT}/viewer_assets_v5/_rankings/pma_v5_phase_geneKO.parquet"   # target-cell ranking (only used if a direction is uncached)
V5C = f"{C.OUT}/viewer_assets_v5/_rankings/pma_v5_phase_complex.parquet"  # complex target-cell ranking
NCELLS = 200
VALID_ALPHAS = (0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0)                       # forward-only morph strengths
CHUNK = 12                                                               # genes/shard (200×7 ≈ 1.8× v5 per-target → half the v5 chunk)


def _use_valid():
    """Point precompute at the sibling tree (env is unreliable across submitit workers → set _ASSETS too)."""
    os.environ["OPS_DIFFEX_ASSETS"] = _VALID
    from . import precompute as P
    P._ASSETS = _VALID
    return _VALID


def setup_dirs():
    """Symlink _directions from viewer_assets_v5 so the 1000 production phase-geneKO directions (d_vec+gap)
    are reused verbatim — validation measures the production traversal, not a refit."""
    root = Path(C.OUT) / _VALID
    root.mkdir(parents=True, exist_ok=True)
    ln = root / "_directions"
    v5dir = Path(C.OUT) / "viewer_assets_v5" / "_directions"
    if not ln.exists():
        ln.symlink_to(v5dir)
        print(f"[valid200] symlinked _directions -> {v5dir}")
    else:
        print(f"[valid200] _directions already present ({ln})")
    if _VALID != "viewer_assets_valid200":                               # w-variant: reuse the SAME 200 anchors (only w differs)
        (root / "phase").mkdir(parents=True, exist_ok=True)
        aln = root / "phase" / "_anchors"
        if not aln.exists():
            aln.symlink_to(Path(C.OUT) / "viewer_assets_valid200" / "phase" / "_anchors")
            print(f"[valid200] symlinked phase/_anchors -> viewer_assets_valid200 (shared 200 anchors)")


def build_anchor():
    """Build the 200-cell phase NTC anchor: top-200 ACCURACY-ranked NTC (from V5G, the same accuracy table the
    production build uses — NOT top-attention). Saves per-anchor CellDINO embeddings + LOSSLESS anchor_imgs
    (needed for faithful DDIM inversion) + real.webp, once."""
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor
    from ops_model.models.interpretability.diffae.traversal.precompute import _gather_class, _save_webp
    from ops_model.models.interpretability.diffae.generator.data import normalize
    from ops_model.models.interpretability.diffae.directions.config import DirConfig
    _use_valid()
    setup_dirs()
    cfg = DirConfig(grain="geneKO", target="NTC", control="NTC", device="cuda")
    imgs, embs = _gather_class(cfg, "NTC", NCELLS, parquet=V5G)           # top-200 ACCURACY-ranked NTC
    n = min(NCELLS, len(embs))
    real = normalize(imgs[:n])                                            # lossless anchors, cell0..n-1
    rd = Path(C.OUT) / _VALID / "phase" / "_anchors" / "NTC"; rd.mkdir(parents=True, exist_ok=True)
    tp = ThreadPoolExecutor(8)
    for c in range(n):
        (rd / f"cell{c}").mkdir(parents=True, exist_ok=True)
        tp.submit(_save_webp, rd / f"cell{c}" / "real.webp", real[c, 0], 256)
    tp.shutdown(wait=True)
    np.savez(rd / "ctrl.npz", ctrl_embs=embs[:n], mu_ctrl=embs[:n].mean(0), anchor_imgs=real)
    print(f"[valid200] built {n}-cell NTC anchor: {real.shape} -> {rd/'ctrl.npz'}")
    return {"n_anchor": n}


def build_valid(targets):
    """Generate 200-cell × forward-α phase traversals for a chunk of geneKO into the sibling tree, reusing the
    pre-built 200 anchors (inverted at startup) + symlinked v5 directions. force=False → resumable."""
    _use_valid()
    return precompute_marker(grain="geneKO", targets=list(targets), ckpt=PHASE_CK, out_root=C.OUT,
                             control="NTC", n_cells=NCELLS, alphas=VALID_ALPHAS, invert_anchors=True, w=W,
                             force=False, v5_score=True, accuracy_parquet=V5G, load_workers=12)


def build_valid_complex(targets):
    """Same recipe as build_valid but grain='complex': reuses the SAME 200-NTC anchor (control='NTC',
    grain-independent _anchors/NTC) + the production phase/complex directions. → 200-cell complex traversals."""
    _use_valid()
    return precompute_marker(grain="complex", targets=list(targets), ckpt=PHASE_CK, out_root=C.OUT,
                             control="NTC", n_cells=NCELLS, alphas=VALID_ALPHAS, invert_anchors=True, w=W,
                             force=False, v5_score=True, accuracy_parquet=V5C, load_workers=12)


def submit_complex(after=None):
    """Shard the 98 EBI complexes (200 cells × 7 α). Anchor already built by the geneKO run — no dep needed."""
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    cx = C.ebi_complexes()
    ch = lambda l, n: [l[i:i + n] for i in range(0, len(l), n)]
    jobs = [{"name": f"val200c_{i}", "func": build_valid_complex, "kwargs": {"targets": s}}
            for i, s in enumerate(ch(cx, CHUNK))]
    sp = _sp()
    if after:
        sp["slurm_additional_parameters"] = {"dependency": f"afterok:{after}"}
    print(f"[valid200] {len(cx)} complexes → {len(jobs)} shards (chunk {CHUNK}, 200 cells × {len(VALID_ALPHAS)} α, inverted w=1.5)")
    return submit_parallel_jobs(jobs_to_submit=jobs, experiment="diffex_valid200c",
                                slurm_params=sp, log_dir="diffex_valid200c", wait_for_completion=False)


def _sp(timeout=720):
    return {"slurm_partition": "gpu", "gpus_per_node": 1, "cpus_per_task": 12, "mem_gb": 96,
            "timeout_min": timeout, "slurm_constraint": "[a100_80|h100|h200|6000_blackwell]",
            "slurm_setup": [f"export OPS_DIFFEX_ASSETS={_VALID}", f"export VALID200_W={W:g}",
                            "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"]}


def submit_shards(after=None):
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    genes = C.all_genes()
    ch = lambda l, n: [l[i:i + n] for i in range(0, len(l), n)]
    jobs = [{"name": f"val200_g{i}", "func": build_valid, "kwargs": {"targets": s}}
            for i, s in enumerate(ch(genes, CHUNK))]
    sp = _sp()
    if after:
        sp["slurm_additional_parameters"] = {"dependency": f"afterok:{after}"}
    print(f"[valid200] {len(genes)} geneKO → {len(jobs)} shards (chunk {CHUNK}, 200 cells × {len(VALID_ALPHAS)} α, inverted w=1.5)"
          + (f" [after {after}]" if after else ""))
    return submit_parallel_jobs(jobs_to_submit=jobs, experiment="diffex_valid200",
                                slurm_params=sp, log_dir="diffex_valid200", wait_for_completion=False)


def submit_anchor():
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    setup_dirs()
    return submit_parallel_jobs(jobs_to_submit=[{"name": "val200_anchor", "func": build_anchor, "kwargs": {}}],
                                experiment="diffex_valid200", slurm_params=_sp(timeout=180),
                                log_dir="diffex_valid200", wait_for_completion=False)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "anchor":
        build_anchor()
    elif cmd == "setup":
        setup_dirs()
    elif cmd == "submit":
        submit_shards()
    elif cmd == "complex":
        submit_complex()
    elif cmd == "all":
        r = submit_anchor()
        aid = str(r.get("base_job_id") or r.get("job_id"))
        submit_shards(after=aid)
    else:
        raise SystemExit(f"unknown cmd {cmd!r}")


if __name__ == "__main__":
    main()
