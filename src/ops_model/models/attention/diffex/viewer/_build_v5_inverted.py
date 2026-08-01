"""FINAL v5 traversals with DDIM-inverted anchors (faithful α=0, w=1.5) into viewer_assets_v5.

MARKERS: anchor = top-40 accuracy NTC from the per-marker v5 ranking (_rankings/fluor/geneKO/<slug>.parquet
= fluor_rank_parquet). The existing v5 ctrl.npz (embeddings only) is dropped so the SAME accuracy cells are
re-materialized LOSSLESSLY (verified identical to v5: anchor-match 1.000) — enabling faithful inversion
(α=0-vs-real 0.988) instead of inverting the lossy real.webp. Run under OPS_DIFFEX_ASSETS=viewer_assets_v5.

COVERAGE: phase gets all 1000 geneKO + 98 complex; each fluor marker gets only the 35-423 geneKO (+~36-54
complex) whose per-cell rankings exist. That is Alex Lin's top1_acc>0.5@100-cell distinctiveness filter (see
_fluor_v5_build.py) — NOT missing data (cells exist for all 1000×55). Lower-acc combos need Alex to gen more.

    python -m ops_model.models.attention.diffex.viewer._build_v5_inverted markers
"""
import json
import os
import sys
from pathlib import Path

from . import catalog as C
from ..classifier.config import slugify
from .precompute import precompute_marker

FRP_DIR = f"{C.OUT}/viewer_assets_v5/_rankings/fluor_shap/geneKO"   # NEW shap_screen rankings (robust bin-size top-acc); old at _rankings/fluor/geneKO_OLD_qualifying backup
# OUTPUT tree: a FRESH dir so force=False gives skip-done resume (timeouts harmless) without clobbering the old
# non-inverted v5 traversals. _directions + each <mod>/_anchors are SYMLINKED back to v5 (setup_v5inv below), so
# the 11,219 ckpt/w-independent directions + prebuilt lossless anchors are reused with no re-gather.
_V5 = "viewer_assets_v5"                # single production assets tree (was viewer_assets_v5_inv; merged + retired)
_V5EMB = "viewer_assets_v5_emb"        # embeddings-only tree: CellDINO of float frames, no webp (parallel to _V5)


def _use_v5():
    """Force the target tree to _V5 at RUNTIME (inside the job) — the import-time snapshot of
    precompute._ASSETS is unreliable across submitit workers. Returns the assets dirname to use for paths."""
    os.environ["OPS_DIFFEX_ASSETS"] = _V5
    from . import precompute as P
    P._ASSETS = _V5
    return _V5


def _genes_for(frp):
    import pandas as pd
    g = pd.read_parquet(frp, columns=["gene"])["gene"].astype(str)
    return sorted(x for x in set(g) if not x.startswith("NTC"))


def build_marker(d, marker_channel, channel):
    """Drop the embeddings-only v5 ctrl.npz → precompute_marker re-materializes the SAME top-40 accuracy
    anchors LOSSLESS (fresh gather from fluor_rank_parquet) + inverts them; directions recompute from the
    same accuracy parquet (v5-equivalent)."""
    frp = f"{FRP_DIR}/{slugify(marker_channel)}.parquet"
    if not os.path.exists(frp):
        return f"skip {marker_channel}: no fluor_rank_parquet"
    ctrl = Path(C.OUT) / _V5 / slugify(marker_channel) / "_anchors" / "NTC" / "ctrl.npz"
    if ctrl.exists():
        os.remove(ctrl)                                  # force fresh lossless anchor gather (same v5 cells)
    return precompute_marker(grain="geneKO", targets=_genes_for(frp), marker_channel=marker_channel,
                             channel=channel, ckpt=f"{C.DD}/{d}/diffae_best.pt", out_root=C.OUT,
                             control="NTC", fluor_rank_parquet=frp, n_cells=40, invert_anchors=True,
                             w=1.5, force=True, v5_score=True)


CFRP_DIR = f"{C.OUT}/viewer_assets_v5/_rankings/fluor_shap/complex"   # NEW shap_screen EBI-pooled complex rankings
SEL25 = "/hpc/projects/icd.fast.ops/analysis/figure4_traversals/ntc_accanchor_selected25.csv"
PHASE_CK = f"{C.DD}/phase_v1/diffae_best.pt"
PMA_SHAP_G = f"{C.OUT}/viewer_assets_v5/_rankings/pma_shap_phase_geneKO.parquet"     # NEW shap_screen phase rankings
PMA_SHAP_C = f"{C.OUT}/viewer_assets_v5/_rankings/pma_shap_phase_complex.parquet"


def build_phase_shap(grain, targets, tree="viewer_assets_v5", save_gemb=False, ddim_steps=100, alphas=None,
                     n_cells=45, cell_range=None):
    """PHASE traversals from the NEW shap_screen rankings: force=True recomputes each target's direction from
    the new-shap cells (accuracy_parquet), keeping the cached NTC anchors. `tree` = output assets dir
    (production viewer_assets_v5 for the real rebuild, or a scratch tree + save_gemb for the 15-gene test).
    `alphas` overrides the α grid (test uses the 7 forward α to match the step-ablation arms; None → full 17).
    `cell_range` (lo,hi) + force → REUSE cached directions and only sample that cell slice (the 200-cell top-up)."""
    os.environ["OPS_DIFFEX_ASSETS"] = tree
    from . import precompute as P
    P._ASSETS = tree
    acc = PMA_SHAP_G if grain == "geneKO" else PMA_SHAP_C
    kw = {} if alphas is None else {"alphas": tuple(alphas)}
    if cell_range is not None:
        kw["cell_range"] = tuple(cell_range)
    return precompute_marker(grain=grain, targets=list(targets), ckpt=PHASE_CK, out_root=C.OUT,
                             control="NTC", n_cells=n_cells, invert_anchors=True, w=1.5, force=True,
                             v5_score=True, accuracy_parquet=acc, ddim_steps=ddim_steps,
                             save_gemb=save_gemb, load_workers=12, **kw)


def build_phase_anchor_200():
    """Extend the phase NTC anchor to 200 cells for the 200-cell top-up: cells 0..99 keep the existing 100
    (attention+accuracy) anchors; cells 100..199 append the next top-accuracy NTC (aligned real imgs + embs)
    so inversion stays faithful across all 200. Overwrites _anchors/NTC ctrl.npz + real.webp cell100..199."""
    import numpy as np, pandas as pd
    from pathlib import Path
    from concurrent.futures import ThreadPoolExecutor
    from .precompute import _gather_class, _save_webp
    from ..diffae.data import normalize
    from ..directions.config import DirConfig
    _use_v5()
    rd = Path(C.OUT) / _V5 / "phase" / "_anchors" / "NTC"
    z = dict(np.load(rd / "ctrl.npz"))
    have = len(z["anchor_imgs"])
    if have >= 200:
        return {"note": f"anchor already {have} cells"}
    cfg = DirConfig(grain="geneKO", target="NTC", control="NTC", device="cuda")
    imgs, emb = _gather_class(cfg, "NTC", 200)                      # 200 top-accuracy NTC (aligned imgs↔embs)
    real200 = normalize(imgs[:200]); emb200 = emb[:200]
    real200[:have] = z["anchor_imgs"][:have]                        # keep the existing 0..have anchors identical
    emb200[:have] = z["ctrl_embs"][:have]
    tp = ThreadPoolExecutor(8)
    for c in range(have, 200):
        (rd / f"cell{c}").mkdir(parents=True, exist_ok=True); tp.submit(_save_webp, rd / f"cell{c}" / "real.webp", real200[c, 0], 256)
    tp.shutdown(wait=True)
    np.savez(rd / "ctrl.npz", ctrl_embs=emb200, mu_ctrl=emb200.mean(0), anchor_imgs=real200)
    print(f"[phase-anchor] extended {have} → 200 anchors -> {rd/'ctrl.npz'}")
    return {"from": have, "to": 200}


def build_phase_topup(grain, targets, lo=45, hi=200, tree="viewer_assets_v5"):
    """200-cell top-up: for targets missing the top cell (hi-1), sample cell_range=(lo,hi) reusing the cached
    new-shap directions (cells 0..lo-1 kept). Idempotent (skips genes already topped up). The filter must check
    the SAME tree build_phase_shap writes to (tree, default viewer_assets_v5) — NOT _V5 (viewer_assets_v5_inv),
    or every round re-inverts all genes."""
    base = Path(C.OUT) / tree / "phase" / grain
    todo = [t for t in targets if not (base / slugify(t) / f"cell{hi - 1}" / "frame_00.webp").exists()]
    if not todo:
        return {"grain": grain, "done": 0, "note": "all topped up"}
    print(f"[phase-topup] {grain}: {len(todo)}/{len(targets)} targets → cells {lo}..{hi - 1}")
    return build_phase_shap(grain, todo, tree=tree, n_cells=hi, cell_range=(lo, hi))


def build_phase_shap_resume(grain, targets, cutoff, tree="viewer_assets_v5", ddim_steps=100):
    """Resume-aware phase-shap build for the preempted chain: regenerate only targets NOT already rebuilt this
    run — meta.json missing, older than `cutoff` (this rebuild's launch time), or not at `ddim_steps`. Idempotent
    across rounds, so a preempted round is picked up by the next without redoing finished targets."""
    base = Path(C.OUT) / tree / "phase" / grain
    todo = []
    for t in targets:
        m = base / slugify(t) / "meta.json"
        if not m.exists() or m.stat().st_mtime < cutoff:
            todo.append(t); continue
        try:
            if json.load(open(m)).get("ddim_steps") != ddim_steps:
                todo.append(t)
        except Exception:
            todo.append(t)
    if not todo:
        return {"grain": grain, "done": 0, "note": "all fresh"}
    print(f"[phase-shap resume] {grain}: {len(todo)}/{len(targets)} targets to (re)build → {ddim_steps} steps")
    return build_phase_shap(grain, todo, tree=tree, ddim_steps=ddim_steps)


def resume_marker(d, marker_channel, channel, target_steps=100):
    """Resume/upgrade a marker: KEEP the (step-independent) anchors and regenerate only the genes not already
    built at `target_steps`. A gene is regenerated if its meta.json is missing, was built at a different
    ddim_steps (e.g. old 50-step frames → the 100-step relaunch), or predates the anchor gather. Genes already
    at target_steps are skipped, so this is idempotent across preemption/timeout rounds — and it correctly
    distinguishes 50- vs 100-step frames via the stamped meta['ddim_steps'] (mtime alone cannot)."""
    frp = f"{FRP_DIR}/{slugify(marker_channel)}.parquet"
    if not os.path.exists(frp):
        return f"skip {marker_channel}: no fluor_rank_parquet"
    base = Path(C.OUT) / _V5 / slugify(marker_channel)
    ctrl = base / "_anchors" / "NTC" / "ctrl.npz"
    if not ctrl.exists():
        return build_marker(d, marker_channel, channel)     # no anchors yet → full build
    cutoff = ctrl.stat().st_mtime
    stale = []
    for g in _genes_for(frp):
        m = base / "geneKO" / slugify(g) / "meta.json"
        if not m.exists() or m.stat().st_mtime < cutoff:
            stale.append(g); continue
        try:
            steps = json.load(open(m)).get("ddim_steps")
        except Exception:
            steps = None
        if steps != target_steps:                            # built at a different step count (50→100) or unstamped (old)
            stale.append(g)
    if not stale:
        return {"marker": marker_channel, "stale": 0, "note": f"already {target_steps}-step fresh"}
    print(f"[resume] {marker_channel}: regenerating {len(stale)} genes → {target_steps} steps (keeping anchors)")
    return precompute_marker(grain="geneKO", targets=stale, marker_channel=marker_channel,
                             channel=channel, ckpt=f"{C.DD}/{d}/diffae_best.pt", out_root=C.OUT,
                             control="NTC", fluor_rank_parquet=frp, n_cells=40, invert_anchors=True,
                             w=1.5, force=True, v5_score=True, ddim_steps=target_steps)


def build_phase_anchor():
    """Pre-build phase _anchors/NTC ctrl.npz with the 45 MIXED anchors: 20 attention (phase geneKO parquet
    top-20) + 25 accuracy (SEL25). Per-anchor z0 embeddings + LOSSLESS anchor_imgs so inversion is faithful
    for BOTH pools (the accuracy cells are NOT in the attention-ranked embeddings, so this is required)."""
    import numpy as np, pandas as pd
    from pathlib import Path
    from concurrent.futures import ThreadPoolExecutor
    from .precompute import _gather_class, _save_webp
    from ..diffae.data import normalize
    from ..directions.config import DirConfig
    _use_v5()
    cfg = DirConfig(grain="geneKO", target="NTC", control="NTC", device="cuda")
    a_imgs, a_emb = _gather_class(cfg, "NTC", 20)                           # 20 attention
    sel = pd.read_csv(SEL25)
    parq = pd.DataFrame({"gene": "NTC", "experiment": sel.experiment, "well": sel.well,
                         "segmentation": sel.segmentation, "x_pheno": sel.x_pheno, "y_pheno": sel.y_pheno,
                         "pma_attention": sel.pma_attention, "rank": range(1, len(sel) + 1), "rank_type": "top"})
    tmp = f"{C.OUT}/{_V5}/_ntc25.parquet"; Path(tmp).parent.mkdir(parents=True, exist_ok=True); parq.to_parquet(tmp)
    b_imgs, b_emb = _gather_class(cfg, "NTC", 25, parquet=tmp)              # 25 accuracy (SEL25)
    embs = np.concatenate([a_emb[:20], b_emb[:25]], 0)
    real = normalize(np.concatenate([a_imgs[:20], b_imgs[:25]], 0))        # 45 lossless, cell0..44
    rd = Path(C.OUT) / _V5 / "phase" / "_anchors" / "NTC"; rd.mkdir(parents=True, exist_ok=True)
    tp = ThreadPoolExecutor(8)
    for c in range(len(real)):
        (rd / f"cell{c}").mkdir(parents=True, exist_ok=True); tp.submit(_save_webp, rd / f"cell{c}" / "real.webp", real[c, 0], 256)
    tp.shutdown(wait=True)
    np.savez(rd / "ctrl.npz", ctrl_embs=embs, mu_ctrl=embs.mean(0), anchor_imgs=real)
    print(f"[phase-anchor] pre-built 45 mixed anchors (20 attn + 25 acc): {real.shape}")


def build_phase(grain, targets):
    """Gen phase frames inverted into the fresh _V5 tree, REUSING cached v5 directions (via symlinked
    _directions) + the pre-built 45 anchors (symlinked _anchors). force=False → skip already-done targets
    (resume) and reuse cached directions; the fresh dir has no stale meta to drop."""
    _use_v5()
    return precompute_marker(grain=grain, targets=list(targets), ckpt=PHASE_CK, out_root=C.OUT,
                             control="NTC", n_cells=45, invert_anchors=True, w=1.5, force=False, v5_score=True)


def _use_v5emb():
    os.environ["OPS_DIFFEX_ASSETS"] = _V5EMB
    from . import precompute as P
    P._ASSETS = _V5EMB
    return _V5EMB


def setup_v5inv_emb():
    """Embeddings-only tree: symlink _directions + phase/_anchors from viewer_assets_v5 so directions and the
    45-cell lossless phase anchor are reused (no re-gather). Traversal dirs written fresh; we save gemb.npz only."""
    v5 = Path(C.OUT) / "viewer_assets_v5"
    emb = Path(C.OUT) / _V5EMB
    emb.mkdir(parents=True, exist_ok=True)
    if not (emb / "_directions").exists():
        (emb / "_directions").symlink_to(v5 / "_directions")
    (emb / "phase").mkdir(parents=True, exist_ok=True)
    if not (emb / "phase" / "_anchors").exists():
        (emb / "phase" / "_anchors").symlink_to(v5 / "phase" / "_anchors")
    print(f"[v5inv-emb] setup {emb}: _directions + phase/_anchors symlinked from v5")


def build_phase_embed(grain, targets):
    """Re-decode the inverted phase traversals and save the in-memory float CellDINO embeddings (gemb.npz),
    NO webp — the webp viewer assets come from the parallel _V5 run. Reuses cached directions + anchors."""
    _use_v5emb()
    return precompute_marker(grain=grain, targets=list(targets), ckpt=PHASE_CK, out_root=C.OUT,
                             control="NTC", n_cells=45, invert_anchors=True, w=1.5, force=False,
                             score=False, v5_score=False, save_gemb=True, skip_webp=True)


_V5EMBC = "viewer_assets_v5_emb_cmp"    # test tree: gemb.npz with BOTH float + webp-roundtrip embeddings


def build_phase_embed_cmp(grain, targets):
    """Same as build_phase_embed but also stores the webp-round-tripped embedding (webp_compare=True) so we can
    compare float-vs-webp mapping on the IDENTICAL inverted frames. Separate tree; force=True (small subset)."""
    os.environ["OPS_DIFFEX_ASSETS"] = _V5EMBC
    from . import precompute as P
    P._ASSETS = _V5EMBC
    v5 = Path(C.OUT) / "viewer_assets_v5"; emb = Path(C.OUT) / _V5EMBC
    emb.mkdir(parents=True, exist_ok=True)
    if not (emb / "_directions").exists():
        (emb / "_directions").symlink_to(v5 / "_directions")
    (emb / "phase").mkdir(parents=True, exist_ok=True)
    if not (emb / "phase" / "_anchors").exists():
        (emb / "phase" / "_anchors").symlink_to(v5 / "phase" / "_anchors")
    return precompute_marker(grain=grain, targets=list(targets), ckpt=PHASE_CK, out_root=C.OUT,
                             control="NTC", n_cells=45, invert_anchors=True, w=1.5, force=True,
                             score=False, v5_score=False, save_gemb=True, skip_webp=True, webp_compare=True)


def phase_test(grain, targets):
    build_phase_anchor()
    return build_phase(grain, targets)


def build_marker_complex(d, marker_channel, channel):
    """Complex traversals: REUSE the geneKO-built anchors (ctrl.npz already has lossless anchor_imgs) — do
    NOT drop it — and take complex directions from the per-marker complex ranking. Run after geneKO."""
    frp = f"{CFRP_DIR}/{slugify(marker_channel)}.parquet"
    if not os.path.exists(frp):
        return f"skip {marker_channel}: no complex ranking"
    return precompute_marker(grain="complex", targets=C.ebi_complexes(), marker_channel=marker_channel,
                             channel=channel, ckpt=f"{C.DD}/{d}/diffae_best.pt", out_root=C.OUT,
                             control="NTC", fluor_rank_parquet=frp, n_cells=40, invert_anchors=True,
                             w=1.5, force=True, v5_score=True)


def _submit(kind, func, after=None):
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    fdir = FRP_DIR if kind == "markers" else CFRP_DIR
    jobs = [{"name": f"v5{kind[:3]}_{slugify(mc)[:14]}", "func": func,
             "kwargs": {"d": d, "marker_channel": mc, "channel": ch}}
            for d, mc, ch in C.complete_markers()
            if os.path.exists(f"{fdir}/{slugify(mc)}.parquet")]
    sp = {"slurm_partition": "gpu", "gpus_per_node": 1, "cpus_per_task": 12, "mem_gb": 96,
          "timeout_min": 720, "slurm_constraint": "[a100_80|h100|h200|6000_blackwell]",
          "slurm_setup": ["export OPS_DIFFEX_ASSETS=viewer_assets_v5",
                          "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"]}
    if after:
        sp["slurm_additional_parameters"] = {"dependency": f"afterany:{after}"}
    print(f"[v5inv] submitting {len(jobs)} {kind} builds → viewer_assets_v5 (inverted, w=1.5)"
          + (f" [after {after}]" if after else ""))
    submit_parallel_jobs(jobs_to_submit=jobs, experiment="diffex_v5inv", slurm_params=sp,
                         log_dir="diffex_v5inv", wait_for_completion=False)


def _submit_phase():
    """Full phase: all ~1000 geneKO + 98 complexes, sharded. Anchor ctrl.npz (45 mixed) is already
    pre-built; each shard reuses it + the cached v5 accuracy directions (force=False), inverting."""
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    genes, cx = C.all_genes(), C.ebi_complexes()
    ch = lambda lst, n: [lst[i:i + n] for i in range(0, len(lst), n)]
    jobs = [{"name": f"v5ph_g{i}", "func": build_phase, "kwargs": {"grain": "geneKO", "targets": s}}
            for i, s in enumerate(ch(genes, 40))]
    jobs += [{"name": f"v5ph_c{i}", "func": build_phase, "kwargs": {"grain": "complex", "targets": s}}
             for i, s in enumerate(ch(cx, 40))]
    print(f"[v5inv] submitting phase: {len(genes)} geneKO + {len(cx)} complex → {len(jobs)} shards (inverted, w=1.5)")
    submit_parallel_jobs(
        jobs_to_submit=jobs, experiment="diffex_v5inv",
        slurm_params={"slurm_partition": "gpu", "gpus_per_node": 1, "cpus_per_task": 12, "mem_gb": 96,
                      "timeout_min": 720, "slurm_constraint": "[a100_80|h100|h200|6000_blackwell]",
                      "slurm_setup": ["export OPS_DIFFEX_ASSETS=viewer_assets_v5",
                                      "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"]},
        log_dir="diffex_v5inv", wait_for_completion=False)


def _submit_altanchors():
    """Rebuild the EXISTING phase A→B alt-anchors (80 geneKO + 80 complex) with inversion — reuse the
    sharded generator (V5G/V5C accuracy cells; precompute_anchors_marker now inverts by default). Sharded
    by anchor class so no two jobs write the same _anchors/<a> real-cell dir."""
    import glob, json
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    from ._altanchor_build import gen_shard
    jobs = []
    for grain in ["geneKO", "complex"]:
        by_anchor = {}
        for d in glob.glob(f"{C.OUT}/viewer_assets_v5/phase/{grain}/*__to__*"):
            m = json.load(open(f"{d}/meta.json")); by_anchor.setdefault(m["control"], []).append((m["control"], m["target"]))
        anchors = sorted(by_anchor); nsh = max(1, min(10, len(anchors)))
        for i in range(nsh):
            sh = anchors[i::nsh]
            if not sh:
                continue
            ps = [p for a in sh for p in by_anchor[a]]
            jobs.append({"name": f"v5alt_{grain}_{i}", "func": gen_shard,
                         "kwargs": {"grain": grain, "classes": sorted({c for p in ps for c in p}), "pairs": ps}})
        print(f"[v5alt] {grain}: {sum(len(v) for v in by_anchor.values())} pairs across {len(anchors)} anchors")
    print(f"[v5alt] submitting {len(jobs)} alt-anchor shards → viewer_assets_v5 (inverted, w=1.5)")
    submit_parallel_jobs(
        jobs_to_submit=jobs, experiment="diffex_v5inv",
        slurm_params={"slurm_partition": "gpu", "gpus_per_node": 1, "cpus_per_task": 12, "mem_gb": 96,
                      "timeout_min": 300, "slurm_constraint": "[a100_80|h100|h200|6000_blackwell]",
                      "slurm_setup": ["export OPS_DIFFEX_ASSETS=viewer_assets_v5",
                                      "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"]},
        log_dir="diffex_v5inv", wait_for_completion=False)


# ============ FAST ARCHITECTURE: pre-build lossless anchors ONCE, then gene-chunk gen-shards ============
CHUNK = 25           # small genes/shard: better parallelism packing + tiny tail; resume covers any timeout


def prebuild_marker_anchor(d, marker_channel, channel):
    """Explicitly write ctrl.npz with LOSSLESS anchor_imgs (top-40 accuracy NTC) + 200 embeddings (for
    directions) + real.webp. No reliance on drop/fresh-gather-branch — this IS the anchor build."""
    import numpy as np, pandas as pd
    from concurrent.futures import ThreadPoolExecutor
    from .precompute import _gather_class, _save_webp
    from ..diffae.data import normalize
    from ..directions.config import DirConfig
    frp = f"{FRP_DIR}/{slugify(marker_channel)}.parquet"
    if not os.path.exists(frp):
        return f"skip {marker_channel}"
    _use_v5()
    cfg = DirConfig(grain="geneKO", target="NTC", control="NTC", device="cuda")
    cfg.marker_channel = marker_channel; cfg.channel = channel; cfg._fluor_rows = pd.read_parquet(frp)
    imgs, embs = _gather_class(cfg, "NTC", 200)              # top-200 accuracy NTC (embs → directions)
    n = min(40, len(embs)); real = normalize(imgs[:n])       # 40 anchors, lossless
    rd = Path(C.OUT) / _V5 / slugify(marker_channel) / "_anchors" / "NTC"; rd.mkdir(parents=True, exist_ok=True)
    tp = ThreadPoolExecutor(8)
    for c in range(n):
        (rd / f"cell{c}").mkdir(parents=True, exist_ok=True); tp.submit(_save_webp, rd / f"cell{c}" / "real.webp", real[c, 0], 256)
    tp.shutdown(wait=True)
    np.savez(rd / "ctrl.npz", ctrl_embs=embs, mu_ctrl=embs.mean(0), anchor_imgs=real)
    return f"prebuilt {marker_channel}: {n} anchors"


def genshard(d, marker_channel, channel, grain, targets):
    """Gen frames for a chunk of targets, reusing the pre-built lossless anchors (invert) + accuracy dirs."""
    _use_v5()
    frp = f"{(FRP_DIR if grain == 'geneKO' else CFRP_DIR)}/{slugify(marker_channel)}.parquet"
    return precompute_marker(grain=grain, targets=list(targets), marker_channel=marker_channel, channel=channel,
                             ckpt=f"{C.DD}/{d}/diffae_best.pt", out_root=C.OUT, control="NTC",
                             fluor_rank_parquet=frp, n_cells=40, invert_anchors=True, w=1.5, force=False, v5_score=True)


def _gpu_sp(timeout, parallel=64):
    return {"slurm_partition": "gpu", "gpus_per_node": 1, "cpus_per_task": 12, "mem_gb": 96,
            "timeout_min": timeout, "slurm_constraint": "[a100_80|h100|h200|6000_blackwell]",
            "slurm_array_parallelism": parallel,
            "slurm_setup": ["export OPS_DIFFEX_ASSETS=viewer_assets_v5",
                            "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"]}


def _submit_prebuild():
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    jobs = [{"name": f"v5pre_{slugify(mc)[:14]}", "func": prebuild_marker_anchor,
             "kwargs": {"d": d, "marker_channel": mc, "channel": ch}}
            for d, mc, ch in C.complete_markers() if os.path.exists(f"{FRP_DIR}/{slugify(mc)}.parquet")]
    jobs.append({"name": "v5pre_phase", "func": build_phase_anchor, "kwargs": {}})
    print(f"[v5pre] {len(jobs)} anchor pre-builds → viewer_assets_v5 (lossless, fast)")
    submit_parallel_jobs(jobs_to_submit=jobs, experiment="diffex_v5inv", slurm_params=_gpu_sp(60),
                         log_dir="diffex_v5inv", wait_for_completion=False)


def setup_v5inv():
    """Stand up the viewer_assets_v5 output tree: symlink _directions + each modality's _anchors back
    to viewer_assets_v5 so cached directions (ckpt/w-independent) and prebuilt lossless anchors are reused with
    no re-gather. Traversal dirs (geneKO/, complex/) are written fresh. Idempotent."""
    v5 = Path(C.OUT) / "viewer_assets_v5"
    inv = Path(C.OUT) / _V5
    inv.mkdir(parents=True, exist_ok=True)
    dl = inv / "_directions"
    if not dl.exists():
        dl.symlink_to(v5 / "_directions")
    mods = [slugify(mc) for d, mc, ch in C.complete_markers() if os.path.exists(f"{FRP_DIR}/{slugify(mc)}.parquet")] + ["phase"]
    n = 0
    for m in mods:
        src = v5 / m / "_anchors"
        if not src.exists():
            print(f"  WARN no anchors in v5 for {m}"); continue
        (inv / m).mkdir(parents=True, exist_ok=True)
        al = inv / m / "_anchors"
        if not al.exists():
            al.symlink_to(src); n += 1
    print(f"[v5inv] setup {inv}: _directions symlinked, {n}/{len(mods)} modality _anchors symlinked")


def _submit_gen(after):
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    ch = lambda lst, n: [lst[i:i + n] for i in range(0, len(lst), n)]
    cx = C.ebi_complexes(); jobs = []
    for d, mc, chan in C.complete_markers():
        if not os.path.exists(f"{FRP_DIR}/{slugify(mc)}.parquet"):
            continue
        for i, s in enumerate(ch(_genes_for(f"{FRP_DIR}/{slugify(mc)}.parquet"), CHUNK)):
            jobs.append({"name": f"g_{slugify(mc)[:10]}_{i}", "func": genshard,
                         "kwargs": {"d": d, "marker_channel": mc, "channel": chan, "grain": "geneKO", "targets": s}})
        if os.path.exists(f"{CFRP_DIR}/{slugify(mc)}.parquet"):
            jobs.append({"name": f"c_{slugify(mc)[:10]}", "func": genshard,
                         "kwargs": {"d": d, "marker_channel": mc, "channel": chan, "grain": "complex", "targets": cx}})
    for i, s in enumerate(ch(C.all_genes(), CHUNK)):
        jobs.append({"name": f"ph_g{i}", "func": build_phase, "kwargs": {"grain": "geneKO", "targets": s}})
    jobs.append({"name": "ph_c", "func": build_phase, "kwargs": {"grain": "complex", "targets": cx}})
    sp = _gpu_sp(240)
    if after:
        sp["slurm_additional_parameters"] = {"dependency": f"afterany:{after}"}
    print(f"[v5gen] {len(jobs)} gen-shards (chunk={CHUNK}) → viewer_assets_v5, parallelism 64" + (f" [after {after}]" if after else ""))
    submit_parallel_jobs(jobs_to_submit=jobs, experiment="diffex_v5gen", slurm_params=sp,
                         log_dir="diffex_v5gen", wait_for_completion=False)


def _submit_fluor(partition="gpu"):
    """FLUOR buildout only (no phase): ONE genshard per marker per grain (geneKO all genes + complex), no
    chunking — matches the working 35083270/35083684 chains. genshard needs an 80GB+ GPU (weak a40/a6000/l40s
    OOM at ~1min), so keep the strong constraint from _gpu_sp. MUST be launched from a shared-fs cwd (repo),
    not /tmp, or compute nodes cannot write submitit logs and every task dies at ~1min."""
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    cx = C.ebi_complexes(); jobs = []
    for d, mc, chan in C.complete_markers():
        frp = f"{FRP_DIR}/{slugify(mc)}.parquet"
        if not os.path.exists(frp):
            continue
        jobs.append({"name": f"g_{slugify(mc)[:12]}", "func": genshard,
                     "kwargs": {"d": d, "marker_channel": mc, "channel": chan, "grain": "geneKO", "targets": _genes_for(frp)}})
        if os.path.exists(f"{CFRP_DIR}/{slugify(mc)}.parquet"):
            jobs.append({"name": f"c_{slugify(mc)[:12]}", "func": genshard,
                         "kwargs": {"d": d, "marker_channel": mc, "channel": chan, "grain": "complex", "targets": cx}})
    sp = dict(_gpu_sp(720)); sp["slurm_partition"] = partition
    print(f"[fluor] {len(jobs)} per-marker gen-shards ({partition}, strong GPU) → viewer_assets_v5")
    submit_parallel_jobs(jobs_to_submit=jobs, experiment="diffex_v5gen", slurm_params=sp,
                         log_dir="diffex_v5gen", wait_for_completion=False)


def _submit_topup(partition="gpu"):
    """PHASE 200-cell top-up: build_phase_topup per gene-chunk for genes still missing cell199 (idempotent).
    Light job → weak GPUs are fine. Launch from a shared-fs cwd (repo), not /tmp."""
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    genes = C.all_genes()
    jobs = [{"name": f"topup_{i}", "func": build_phase_topup, "kwargs": {"grain": "geneKO", "targets": genes[i:i + 8]}}
            for i in range(0, len(genes), 8)]
    sp = dict(_gpu_sp(300)); sp["slurm_partition"] = partition; sp["slurm_constraint"] = "[a40|a6000|l40s]"
    print(f"[topup] {len(jobs)} shards ({partition}, weak GPU ok) → viewer_assets_v5/phase/geneKO")
    submit_parallel_jobs(jobs_to_submit=jobs, experiment="diffex_v5inv", slurm_params=sp,
                         log_dir="diffex_v5inv", wait_for_completion=False)


if __name__ == "__main__":
    # NOTE: run from a shared-fs cwd (the repo), NOT /tmp — submitit writes logs relative to cwd and compute
    # nodes cannot read /tmp, so /tmp-launched jobs all die at ~1min with empty logs.
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    arg2 = sys.argv[2] if len(sys.argv) > 2 else None
    if cmd == "prebuild":
        _submit_prebuild()
    elif cmd == "setup":
        setup_v5inv()
    elif cmd == "gen":
        setup_v5inv()
        _submit_gen(arg2)
    elif cmd == "fluor":
        _submit_fluor(arg2 or "gpu")
    elif cmd == "topup":
        _submit_topup(arg2 or "gpu")
    elif cmd == "altanchors":
        _submit_altanchors()
    else:
        print("usage: _build_v5_inverted.py prebuild | setup | gen [after_jobid] | fluor [partition] | topup [partition] | altanchors")
