"""METRIC B production: per-cell per-α removal-based LOO score + rank-vs-real for EVERY traversal (phase + fluor).

Score = the validated Alex set-accuracy marginal, GEN variant: drop a generated cell into random bags of REAL
class cells, marginal contribution to P(true class), meaned over bag sizes {1,2,5,10,20,50,100,200,500} (see
[[reference_alex_setacc_score_replication]]). Gen is mapped into real space on the α=0 gen (cancel DiffAE
offset) so gen/real scores share one scale → percentile-vs-real is meaningful. Per-cell per-α cached to
loo_cache/{modality}/{grain}/{class}.npz for full resume (skip if present). Also emits peak-α f_B per traversal.

Shares the CellDINO embcache with f_centroid_recovery (fluor); phase reads gen_real_map_cache_v5new directly.
"""
import glob
import json
import os

import numpy as np

from ops_model.models.attention.diffex.gen_validation.metric_b_loo.rank_removal_test import BAGS, _gen_marg, _ipeak

ROOT = "/hpc/projects/icd.fast.ops/analysis/figure4_traversals"
CACHE_PHASE = f"{ROOT}/gen_real_map_cache_v5new"                       # phase gen per-α (classifier space)
PAPER_V2 = "/hpc/projects/icd.fast.ops/models/alex_lin_attention/v5/paper_v2_phase"   # FULL real pool (validated footing)
RANKPQ = "/hpc/projects/icd.fast.ops/models/diffex/viewer_assets_v5/_rankings"
LOO = f"{ROOT}/loo_cache"                                             # per-cell per-α output cache
V5A = "/hpc/projects/icd.fast.ops/models/diffex/viewer_assets_v5"
NCELL = 40                                                            # gen cells kept per traversal (viewer)
NREF = 150                                                           # real cells sampled for the ranking reference
REPS = 20


def _core(model, gen_by_a, real, tl, ci, alphas, device):
    """gen_by_a: list per α of (g,D) gen embeddings (or None). real: (R,D). Returns per-cell score/pct + f_B."""
    import torch
    real_t = torch.as_tensor(real, dtype=torch.float32, device=device)
    mu_r, sd_r = real.mean(0), real.std(0) + 1e-6
    a0 = int(np.argmin(np.abs(np.array(alphas)))); g0 = np.asarray(gen_by_a[a0], np.float32)
    mu_g, sd_g = g0.mean(0), g0.std(0) + 1e-6
    bags = [b for b in BAGS if b <= len(real)]
    rng = np.random.default_rng(0)
    refq = torch.as_tensor(real[rng.choice(len(real), min(NREF, len(real)), replace=False)], dtype=torch.float32, device=device)
    real_ref = np.sort(_gen_marg(model, refq, real_t, tl, ci, bags, REPS, 0, device))    # real add-in distribution (same protocol)
    na = len(alphas); score = np.full((NCELL, na), np.nan); pct = np.full((NCELL, na), np.nan)
    curve = []
    for ai in range(na):
        gv = gen_by_a[ai]
        if gv is None or not len(gv):
            curve.append(np.nan); continue
        G = np.asarray(gv, np.float32)[:NCELL]
        gm = (G - mu_g) / sd_g * sd_r + mu_r                          # map gen→real space (DiffAE offset cancel)
        s = _gen_marg(model, torch.as_tensor(gm, dtype=torch.float32, device=device), real_t, tl, ci, bags, REPS, 0, device)
        p = np.searchsorted(real_ref, s) / len(real_ref)             # percentile amongst real cells (0-1)
        score[:len(s), ai] = s; pct[:len(s), ai] = p
        curve.append(float(np.mean(p)))                              # α-curve = mean percentile-vs-real
    f = _ipeak(alphas, [np.nan if np.isnan(c) else c for c in curve])
    N = len(real); real_rank = (1 - pct) * N + 1                                  # position in the real ranking (1 = top)
    return dict(alphas=alphas, score=score, pct=pct, real_rank=real_rank, n_real=N,
                curve=curve, real_ref_median=float(np.median(real_ref)), f=round(float(f), 2))


def _save(modality, grain, cls, out):
    d = f"{LOO}/{modality}/{grain}"; os.makedirs(d, exist_ok=True)
    np.savez(f"{d}/{cls}.npz", **{k: np.asarray(v) for k, v in out.items() if k != "alphas"}, alphas=out["alphas"])


def _slug(s):
    from ops_model.models.attention.diffex.classifier.config import slugify
    return slugify(str(s))


def _paper_v2_real(gene):
    """FULL real pool for a geneKO class (train+val), classifier space — the validated footing (~39k cells)."""
    import torch
    embs, segs = [], []
    for split in ("train", "val"):
        fp = f"{PAPER_V2}/{split}/{gene}.pt"
        if not os.path.exists(fp):
            continue
        d = torch.load(fp, map_location="cpu")
        embs.append(d["embeddings"].numpy().astype(np.float32))
        segs += [int(s) for lst in d["cell_metadata"]["segmentation_id"] for s in lst]
    return (np.concatenate(embs, 0), np.array(segs)) if embs else (None, None)


def test_fix(gene="POLR1B", device="cuda"):
    """PROVE the full-run scoring reproduces the validated real scores before re-running the sweep.
    (1) a REAL cell's add-in score ≈ its pma_attention; (2) gen ranks move to the TOP of the real ranking with α."""
    import pandas as pd, torch
    from ops_model.models.attention.diffex.viewer.set_classifier import load_set_classifier, V5_CKPT_ROOT, V5_RUNS
    from ops_model.models.attention.diffex.gen_validation.metric_b_loo.rank_removal_test import _gen_marg
    model, cmap, c2i = load_set_classifier(run=V5_RUNS[("phase", "geneKO")], device=device, root=V5_CKPT_ROOT)
    tl = cmap[gene]; ci = c2i.get("Phase2D", 0)
    real, seg = _paper_v2_real(gene); real_t = torch.as_tensor(real, dtype=torch.float32, device=device)
    bags = [b for b in BAGS if b <= len(real)]
    print(f"{gene}: full real pool {len(real)}, bags={bags}", flush=True)
    # real ranking reference (Alex pma_attention)
    pq = pd.read_parquet(f"{RANKPQ}/pma_shap_phase_geneKO.parquet", columns=["gene", "pma_attention", "segmentation", "rank"])
    pqg = pq[pq.gene == gene]; pma_sorted = np.sort(pqg.pma_attention.to_numpy()); Nr = len(pma_sorted)
    # (1) reproduce a real cell's score
    tgt = 26471135 if gene == "POLR1B" else int(pqg.sort_values("rank").iloc[0].segmentation)
    row = int(np.where(seg == tgt)[0][0]) if tgt in seg else 0
    q = torch.as_tensor(real[row][None], dtype=torch.float32, device=device)
    s_real = float(_gen_marg(model, q, real_t, tl, ci, bags, 40, 0, device)[0])
    alex = float(pqg[pqg.segmentation == tgt].pma_attention.iloc[0]) if (pqg.segmentation == tgt).any() else float("nan")
    print(f"(1) REAL cell seg {tgt}: add-in score={s_real:.4f}  vs Alex pma_attention={alex:.4f}  diff={abs(s_real-alex):.4f}", flush=True)
    # real add-in REFERENCE distribution (same protocol as gen → one scale). Rank gen against THIS.
    rng = np.random.default_rng(0)
    refq = torch.as_tensor(real[rng.choice(len(real), min(400, len(real)), replace=False)], dtype=torch.float32, device=device)
    real_ref = np.sort(_gen_marg(model, refq, real_t, tl, ci, bags, 20, 0, device))
    pma_rank_of_tgt = int((pqg.segmentation == tgt).any() and pqg[pqg.segmentation == tgt]["rank"].iloc[0])
    tgt_pct = float((real_ref < s_real).mean())
    print(f"    real add-in ref: median={np.median(real_ref):+.4f} max={real_ref.max():+.4f}; top cell(pma rank {pma_rank_of_tgt}) sits at {tgt_pct*100:.1f} pct of real add-in", flush=True)
    # (2) gen rank curve vs the real add-in reference (self-consistent scale)
    d = np.load(f"{CACHE_PHASE}/geneKO/{gene}.npz", allow_pickle=True); al = [float(a) for a in d["alphas"]]
    a0 = int(np.argmin(np.abs(np.array(al)))); g0 = np.asarray(d["gen"][a0], np.float32)
    mu_g, sd_g = g0.mean(0), g0.std(0) + 1e-6; mu_r, sd_r = real.mean(0), real.std(0) + 1e-6
    print("(2) gen rank vs α (rank 1 = top; ranked against real add-in distribution):", flush=True)
    for ai, a in enumerate(al):
        if a < 0:
            continue
        G = (np.asarray(d["gen"][ai], np.float32)[:20] - mu_g) / sd_g * sd_r + mu_r
        sc = _gen_marg(model, torch.as_tensor(G, dtype=torch.float32, device=device), real_t, tl, ci, bags, 20, 0, device)
        pct = np.searchsorted(real_ref, sc, side="right") / len(real_ref)                 # gen percentile in real add-in
        rank = np.mean((1 - pct) * Nr + 1)
        print(f"    α {a:4.1f}   gen score {np.mean(sc):+.4f}   pct {np.mean(pct)*100:5.1f}   mean real-rank {rank:8.0f} / {Nr}", flush=True)


def run_phase(genes, grain="geneKO", device="cuda"):
    import torch
    from ops_model.models.attention.diffex.viewer.set_classifier import load_set_classifier, V5_CKPT_ROOT, V5_RUNS
    key = ("phase", "geneKO") if grain == "geneKO" else ("phase", "complex_ebionly")
    model, cmap, c2i = load_set_classifier(run=V5_RUNS[key], device=device, root=V5_CKPT_ROOT)
    ci = c2i.get("Phase2D", 0); done = 0
    cmap_by_slug = {_slug(k): k for k in cmap}                        # complex npz names are slugs, cmap keys are full names
    for g in genes:
        if os.path.exists(f"{LOO}/phase/{grain}/{_slug(g)}.npz"):
            done += 1; continue
        fp = f"{CACHE_PHASE}/{grain}/{g}.npz"                          # gen per-α (CellDINO, precomputed)
        name = g if g in cmap else cmap_by_slug.get(g)
        if name is None or not os.path.exists(fp):
            print(f"skip {g}", flush=True); continue
        real = (_paper_v2_real(name)[0] if grain == "geneKO" else _complex_real(name))   # FULL real pool (validated footing)
        if real is None or len(real) < min(BAGS):
            print(f"skip {g}: no/small real pool", flush=True); continue
        d = np.load(fp, allow_pickle=True); al = [float(a) for a in d["alphas"]]
        out = _core(model, list(d["gen"]), real, cmap[name], ci, al, device)
        _save("phase", grain, _slug(g), out); done += 1
        print(f"{g:14s} f_B={out['f']:.2f}  rank@f={int((1-max(c for c in out['curve'] if not np.isnan(c)))*out['n_real'])}/{out['n_real']}", flush=True)
    return {"grain": grain, "done": done, "n": len(genes)}


def _complex_real(complex_name, cap=40000):
    """Real pool for a complex = paper_v2 cells of its member genes (from the complex parquet), pooled + capped."""
    import pandas as pd
    pq = pd.read_parquet(f"{RANKPQ}/pma_shap_phase_complex.parquet", columns=["predicted_class", "gene"])
    members = pq[pq["predicted_class"] == complex_name]["gene"].unique()
    embs = [e for mg in members for e in [_paper_v2_real(mg)[0]] if e is not None]
    if not embs:
        return None
    R = np.concatenate(embs, 0)
    return R[np.random.default_rng(0).choice(len(R), cap, replace=False)] if len(R) > cap else R


def run_fluor(mod, grain, device="cuda"):
    import types, torch
    from ops_model.models.attention.diffex.viewer.set_classifier import load_set_classifier, V5_CKPT_ROOT, V5_RUNS
    from ops_model.models.attention.diffex.gen_validation.centroid_recovery.f_centroid_recovery import _emb   # shared embcache
    key = ("fluor", "geneKO") if grain == "geneKO" else ("fluor", "complex_ebionly")
    model, cmap, c2i = load_set_classifier(run=V5_RUNS[key], device=device, root=V5_CKPT_ROOT)
    # marker channel: match the classifier channel whose slug == mod
    chan = next((c for c in c2i if _slug(c) == mod), None)
    if chan is None:
        return {"mod": mod, "grain": grain, "skip": "no channel"}
    ci = c2i[chan]; cfg = types.SimpleNamespace(batch_size=128, celldino_z_score=True)
    block = "genes" if grain == "geneKO" else "complexes"
    idx = json.load(open(f"{V5A}/top_cells/markers/{mod}/index.json"))
    cropdir = f"{V5A}/top_cells/markers/{mod}/crops"; done = 0
    for cls, rec in idx.get(block, {}).items():
        if cls not in cmap or os.path.exists(f"{LOO}/{mod}/{grain}/{_slug(cls)}.npz"):
            done += os.path.exists(f"{LOO}/{mod}/{grain}/{_slug(cls)}.npz"); continue
        keys = (rec.get("accuracy") or rec.get("attention") or [])[:300]
        real = _emb([f"{cropdir}/{r['img']}" for r in keys], cfg, key=f"gal/{mod}/{grain}/{_slug(cls)}")
        gd = glob.glob(f"{V5A}/{mod}/{grain}/{_slug(cls)}*/meta.json") or glob.glob(f"{V5A}/{mod}/{grain}/*/meta.json")
        gd = next((p for p in gd if json.load(open(p)).get("target", os.path.basename(os.path.dirname(p))) == cls), gd[0] if gd else None)
        if gd is None or len(real) < min(BAGS):
            continue
        base = os.path.dirname(gd); meta = json.load(open(gd)); al = [float(a) for a in meta["alphas"]]
        cells = sorted(glob.glob(f"{base}/cell*/")); frames = sorted(os.path.basename(f) for f in glob.glob(f"{cells[0]}frame_*.webp"))
        gk = f"gen/{mod}/{grain}/{_slug(cls)}"
        gen_by_a = [_emb([f"{cd}{fr}" for cd in cells if os.path.exists(f"{cd}{fr}")], cfg, key=f"{gk}/a{ai}") for ai, fr in enumerate(frames)]
        out = _core(model, gen_by_a, np.asarray(real, np.float32), cmap[cls], ci, al, device)
        _save(mod, grain, _slug(cls), out); done += 1
        print(f"{mod}/{grain}/{cls[:24]:24s} f_B={out['f']:.2f}", flush=True)
    return {"mod": mod, "grain": grain, "done": done}


# ---- sharded submitters ----
def _phase_shards(grain, per=25):
    genes = sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{CACHE_PHASE}/{grain}/*.npz"))
    return [genes[i:i + per] for i in range(0, len(genes), per)]


def _fluor_jobs():
    import re
    man = json.load(open(f"{V5A}/manifest.json")); jobs = []
    for mk in man["markers"]:
        mc = mk.get("marker_channel")
        if not mc or re.match(r"(?i)phase", mc):
            continue
        mod = _slug(mc)
        for grain in ("geneKO", "complex"):
            if glob.glob(f"{V5A}/{mod}/{grain}/*/meta.json") and os.path.exists(f"{V5A}/top_cells/markers/{mod}/index.json"):
                jobs.append((mod, grain))
    return jobs


_SP = {"slurm_partition": "gpu", "slurm_gres": "gpu:1", "cpus_per_task": 8, "mem_gb": 64, "timeout_min": 240,
       "slurm_constraint": "[h100|a100|l40s]", "slurm_exclude": "gpu-b-4"}   # a6000 gpu-b-4 has a broken CUDA driver


def _guard():
    if os.environ.get("LOO_ENABLE") != "1":
        raise RuntimeError("loo submit disabled — set LOO_ENABLE=1 to re-enable (guard to stop a stray/peer resubmit loop)")


def submit_phase():
    _guard()
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    jobs = [{"name": f"looP_{grain[:4]}_{i}", "func": run_phase, "kwargs": {"genes": sh, "grain": grain}}
            for grain in ("geneKO", "complex") for i, sh in enumerate(_phase_shards(grain))]
    print(f"[loo-phase] {len(jobs)} shards")
    submit_parallel_jobs(jobs, experiment="loo", slurm_params=_SP, log_dir="loo", wait_for_completion=False)


def submit_fluor():
    _guard()
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    jobs = [{"name": f"looF_{mod[:14]}_{grain[:4]}", "func": run_fluor, "kwargs": {"mod": mod, "grain": grain}}
            for mod, grain in _fluor_jobs()]
    print(f"[loo-fluor] {len(jobs)} shards")
    submit_parallel_jobs(jobs, experiment="loo", slurm_params=_SP, log_dir="loo", wait_for_completion=False)


def submit():
    submit_phase(); submit_fluor()


if __name__ == "__main__":
    submit()
