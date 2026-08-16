"""Reproduce Alex Lin's per-cell set-accuracy `score` (PR #19 score.py) with OUR v5 model, to validate
the ranking (viewer 'accuracy confidence' = pma_attention). Score = mean over bag sizes of each cell's
LOO marginal P(class|bag) - P(class|bag-cell); bag1 = single-cell P(class). Deterministic (seed=0).
Our v5 checkpoint is no-mask, so LOO is done by physically removing the cell (identical to Alex's masking).
"""
import numpy as np

STORE = "/hpc/projects/icd.fast.ops/models/alex_lin_attention/v5/paper_v2_phase"   # train/ + val/ dumps
BAGS = (1, 2, 5, 10, 20, 50, 100, 200, 500)
_TAPER = {2: 1.0, 5: 0.6, 10: 0.3, 20: 0.1, 50: 0.04, 100: 0.02, 200: 0.01, 500: 0.002}


def _reps(bag, anchor=100, min_reps=10):
    if bag == 1:
        return 1
    frac = _TAPER.get(bag, (2.0 / bag) ** 0.85)
    return max(min_reps, round(anchor * frac))


def _P(model, feats, ci, tl):
    """P(class tl | each bag). feats: device tensor (B, k, D). Returns device tensor (B,) float64."""
    import torch, torch.nn.functional as F
    c = torch.full((feats.size(0),), int(ci), dtype=torch.long, device=feats.device)
    with torch.no_grad():
        return F.softmax(model(feats, c), dim=-1)[:, tl].double()


def _loo_index(bag, device):
    """(bag, bag-1) index: row j = all positions except j. Built once per bag size."""
    import torch
    cols = torch.arange(bag, device=device)
    m = cols[None, :] != cols[:, None]                                  # (bag, bag) off-diagonal
    return cols[None, :].expand(bag, bag)[m].view(bag, bag - 1)


def _marginal(model, embT, tl, ci, bag, n_reps, seed):
    """Mean LOO marginal per cell at one bag size over random partitions (Alex's scheme; LOO via gather, fully batched)."""
    import torch
    device = embT.device; n = embT.shape[0]; D = embT.shape[1]
    msum = torch.zeros(n, dtype=torch.float64, device=device)
    cnt = torch.zeros(n, dtype=torch.float64, device=device)
    li = _loo_index(bag, device)                                        # (bag, bag-1)
    block = max(1, 400_000 // (bag * bag))                              # bags/block → bound the (blk*bag, bag-1, D) expansion
    for rep in range(n_reps):
        g = torch.Generator().manual_seed(seed * 1_000_003 + rep)
        perm = torch.randperm(n, generator=g)
        nb = n // bag                                                   # drop remainder (covered across reps)
        idx = perm[: nb * bag].view(nb, bag).to(device)
        for b0 in range(0, nb, block):
            sub = idx[b0 : b0 + block]                                  # (blk, bag)
            e = embT[sub]                                               # (blk, bag, D)
            p_full = _P(model, e, ci, tl)                              # (blk,)
            loo = e[:, li].reshape(-1, bag - 1, D)                      # (blk*bag, bag-1, D) each cell removed once
            p_loo = _P(model, loo, ci, tl).view(sub.shape)             # (blk, bag)
            mg = (p_full[:, None] - p_loo).reshape(-1)                  # (blk*bag,)
            flat = sub.reshape(-1)
            msum.index_add_(0, flat, mg); cnt.index_add_(0, flat, torch.ones_like(mg))
    return (msum / cnt.clamp(min=1)).cpu().numpy()


def run(gene="POLR1B", device="cuda"):
    import json, torch
    from ops_model.models.attention.diffex.viewer.set_classifier import load_set_classifier, V5_CKPT_ROOT, V5_RUNS
    model, cmap, c2i = load_set_classifier(run=V5_RUNS[("phase", "geneKO")], device=device, root=V5_CKPT_ROOT)
    tl = cmap[gene]; ci = c2i.get("Phase2D", 0)
    emb_l, seg_l = [], []
    for split in ("train", "val"):
        d = torch.load(f"{STORE}/{split}/{gene}.pt", map_location="cpu")
        emb_l.append(d["embeddings"].numpy().astype(np.float32))
        seg_l += [int(s) for lst in d["cell_metadata"]["segmentation_id"] for s in lst]
    emb = np.concatenate(emb_l, 0); seg = np.array(seg_l)
    assert len(seg) == len(emb), (len(seg), len(emb))
    print(f"{gene}: {len(emb)} Phase2D cells pooled (train+val)", flush=True)
    embT = torch.as_tensor(emb, dtype=torch.float32, device=device)
    margs = {}
    for b in BAGS:
        r = _reps(b)
        margs[b] = _single(model, embT, tl, ci) if b == 1 else _marginal(model, embT, tl, ci, b, r, 0)
        print(f"  bag {b:4d} (reps {r:3d}): marg mean={margs[b].mean():+.4f}", flush=True)
    score = np.mean(np.stack([margs[b] for b in BAGS], 0), 0)
    order = np.argsort(-score)
    rank = {int(seg[i]): k + 1 for k, i in enumerate(order)}
    print(f"\n=== {gene} top-8 by score ===")
    for k in range(8):
        i = order[k]
        print(f"  rank {k+1:5d}  seg {int(seg[i]):>10d}  score {score[i]:+.5f}  bag1 {margs[1][i]:.4f}")
    tgt = 26471135
    if tgt in rank:
        i = int(np.where(seg == tgt)[0][0])
        print(f"\nTARGET seg {tgt}: my rank={rank[tgt]}  my score={score[i]:+.5f}   vs Alex pma_attention=0.266161 rank=1")
    out = "/hpc/projects/icd.fast.ops/analysis/figure4_traversals/alex_rank_POLR1B.json"
    json.dump({"gene": gene, "seg": seg.tolist(), "score": score.tolist(),
               "bag1": margs[1].tolist(), "rank": [rank[int(s)] for s in seg]}, open(out, "w"))
    print("wrote", out)


def _single(model, embT, tl, ci):
    """Deterministic bag-1 attribution P(class | cell alone), per cell."""
    import torch, torch.nn.functional as F
    out = np.empty(embT.shape[0])
    c = torch.full((embT.shape[0],), int(ci), dtype=torch.long, device=embT.device)
    with torch.no_grad():
        for i in range(0, embT.shape[0], 8192):
            out[i:i+8192] = F.softmax(model(embT[i:i+8192][:, None, :], c[i:i+8192]), -1)[:, tl].double().cpu().numpy()
    return out


def _gen_marg(model, genz, real, tl, ci, bags, reps, seed, device):
    """Per-gen-cell removal-based marginal: drop each gen cell into random REAL bags, measure P(true|real+gen)-P(true|real),
    averaged over reps and bag sizes (Alex's score form, gen-cell variant). genz,(real): (G,D),(R,D) device tensors."""
    import torch
    G = genz.shape[0]; R = real.shape[0]; D = real.shape[1]
    tot = torch.zeros(G, dtype=torch.float64, device=device); nb = 0
    for b in bags:
        g = torch.Generator().manual_seed(seed * 1_000_003 + b)
        for _ in range(reps):
            ridx = torch.randint(0, R, (b - 1,), generator=g) if b > 1 else torch.empty(0, dtype=torch.long)
            rb = real[ridx.to(device)]                                  # (b-1, D) shared real bag this rep
            p_base = _P(model, rb[None], ci, tl)[0] if b > 1 else torch.zeros((), dtype=torch.float64, device=device)
            full = torch.cat([genz[:, None, :], rb[None].expand(G, -1, -1)], dim=1)   # (G, b, D): gen + real bag
            p_full = _P(model, full, ci, tl)                            # (G,)
            tot += p_full - p_base; nb += 1
    return (tot / nb).cpu().numpy()


def gen_peak(gene="POLR1B", device="cuda"):
    """Metric B: peak-α of the generated cells' removal-based marginal (phase)."""
    import json, torch
    from ops_model.models.attention.diffex.viewer.set_classifier import load_set_classifier, V5_CKPT_ROOT, V5_RUNS
    model, cmap, c2i = load_set_classifier(run=V5_RUNS[("phase", "geneKO")], device=device, root=V5_CKPT_ROOT)
    tl = cmap[gene]; ci = c2i.get("Phase2D", 0)
    d = np.load(f"/hpc/projects/icd.fast.ops/analysis/figure4_traversals/gen_real_map_cache_v5new/geneKO/{gene}.npz", allow_pickle=True)
    al = [float(a) for a in d["alphas"]]
    real = torch.as_tensor(np.asarray(d["real"], np.float32), device=device)
    a0 = int(np.argmin(np.abs(np.array(al)))); g0 = np.asarray(d["gen"][a0], np.float32)
    mu, sd = g0.mean(0), g0.std(0) + 1e-6                               # z-standardize gen on α=0 gen (cancel DiffAE offset)
    bags = [b for b in BAGS if b <= len(real)]; curve = []
    for ai, a in enumerate(al):
        gv = np.asarray(d["gen"][ai], np.float32)
        genz = torch.as_tensor((gv - mu) / sd, device=device)
        m = _gen_marg(model, genz, real, tl, ci, bags, reps=20, seed=0, device=device)
        curve.append(float(np.mean(m)))
    f = _ipeak(al, curve)
    print(f"\n=== METRIC B {gene} (phase) peak-α f = {f:.2f} ===")
    for a, c in zip(al, curve):
        if a >= 0:
            print(f"  α {a:4.1f}  gen-marginal {c:+.4f}")
    json.dump({"gene": gene, "alphas": al, "gen_marginal": curve, "f": round(f, 2)},
              open(f"/hpc/projects/icd.fast.ops/analysis/figure4_traversals/genpeak_{gene}.json", "w"))


def _ipeak(al, y):
    al = np.asarray(al, float); y = np.asarray(y, float); pos = al > 0; a, v = al[pos], y[pos]
    i = int(np.argmax(v))
    if i == 0 or i == len(a) - 1:
        return float(a[i])
    x3, y3 = a[i - 1:i + 2], v[i - 1:i + 2]; c = np.polyfit(x3, y3, 2)
    return float(a[i]) if c[0] >= 0 else float(np.clip(-c[1] / (2 * c[0]), x3[0], x3[2]))


_FLUOR_CASES = {
    "MTOR": dict(run_key=("fluor", "geneKO"), mod="lysosome_LysoTracker_live_cell_dye", block="genes",
                 cls="MTOR", gendir="lysosome_LysoTracker_live_cell_dye/geneKO/MTOR",
                 channel="lysosome_LysoTracker live-cell dye"),
    "TIM23": dict(run_key=("fluor", "complex_ebionly"), mod="mitochondria_ChromaLIVE_561_excitation", block="complexes",
                  cls="TIM23 mitochondrial inner membrane pre-sequence translocase complex, TIM17A variant",
                  gendir="mitochondria_ChromaLIVE_561_excitation/complex/TIM23_mitochondrial_inner_membrane_pre_sequence_translocase_complex__TIM17A_variant",
                  channel="mitochondria_ChromaLIVE 561 excitation"),
}
V5A = "/hpc/projects/icd.fast.ops/models/diffex/viewer_assets_v5"


def gen_peak_fluor(case="MTOR", device="cuda"):
    """Metric B for a fluor traversal: peak-α of the generated cells' removal-based marginal (fluor classifier)."""
    import glob, json, os, types, torch
    from ops_model.models.attention.diffex.viewer.set_classifier import load_set_classifier, V5_CKPT_ROOT, V5_RUNS
    from ops_model.models.attention.diffex.gen_validation.centroid_recovery.centroid_recovery_fluor import _emb
    c = _FLUOR_CASES[case]
    model, cmap, c2i = load_set_classifier(run=V5_RUNS[c["run_key"]], device=device, root=V5_CKPT_ROOT)
    tl = cmap[c["cls"]]; ci = c2i[c["channel"]]
    cfg = types.SimpleNamespace(batch_size=64, celldino_z_score=True)
    idx = json.load(open(f"{V5A}/top_cells/markers/{c['mod']}/index.json"))
    cropdir = f"{V5A}/top_cells/markers/{c['mod']}/crops"
    keys = (idx[c["block"]][c["cls"]].get("accuracy") or idx[c["block"]][c["cls"]].get("attention") or [])[:300]
    real = torch.as_tensor(_emb([f"{cropdir}/{r['img']}" for r in keys], cfg).astype(np.float32), device=device)  # target-class real pool
    # gen frames per α
    gd = f"{V5A}/{c['gendir']}"; cells = sorted(glob.glob(f"{gd}/cell*/"))
    frames = sorted(os.path.basename(f) for f in glob.glob(f"{cells[0]}frame_*.webp"))
    al = [float(a) for a in json.load(open(f"{gd}/meta.json"))["alphas"]]; a0 = int(np.argmin(np.abs(np.array(al))))
    genA = {ai: [f"{cd}{fr}" for cd in cells if os.path.exists(f"{cd}{fr}")] for ai, fr in enumerate(frames)}
    g0 = _emb(genA[a0], cfg); mu, sd = g0.mean(0), g0.std(0) + 1e-6
    bags = [b for b in BAGS if b <= len(real)]; curve = []
    for ai in range(len(al)):
        e = _emb(genA[ai], cfg)
        if not len(e):
            curve.append(None); continue
        genz = torch.as_tensor(((e - mu) / sd).astype(np.float32), device=device)
        curve.append(float(np.mean(_gen_marg(model, genz, real, tl, ci, bags, 20, 0, device))))
    f = _ipeak(al, [x for x in curve])
    print(f"\n=== METRIC B {case} ({c['channel']}) peak-α f = {f:.2f}  (real pool {len(real)}) ===")
    for a, v in zip(al, curve):
        if a >= 0:
            print(f"  α {a:4.1f}  gen-marginal {('%.4f' % v) if v is not None else '  --'}")
    json.dump({"case": case, "alphas": al, "gen_marginal": curve, "f": round(f, 2)},
              open(f"/hpc/projects/icd.fast.ops/analysis/figure4_traversals/genpeak_fluor_{case}.json", "w"))


def submit_genpeak_fluor():
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    jobs = [{"name": f"genpeakF_{k}", "func": gen_peak_fluor, "kwargs": {"case": k}} for k in _FLUOR_CASES]
    submit_parallel_jobs(jobs, experiment="genpeakF",
                         slurm_params={"slurm_partition": "gpu", "slurm_gres": "gpu:1", "cpus_per_task": 8,
                                       "mem_gb": 64, "timeout_min": 60, "slurm_constraint": "[h100|a100|l40s|a6000]"},
                         log_dir="genpeakF", wait_for_completion=False)


def submit_genpeak():
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    submit_parallel_jobs([{"name": "genpeak", "func": gen_peak, "kwargs": {"gene": "POLR1B"}}], experiment="genpeak",
                         slurm_params={"slurm_partition": "gpu", "slurm_gres": "gpu:1", "cpus_per_task": 8,
                                       "mem_gb": 64, "timeout_min": 60, "slurm_constraint": "[h100|a100|l40s|a6000]"},
                         log_dir="genpeak", wait_for_completion=False)


def submit():
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    submit_parallel_jobs([{"name": "alexrank", "func": run, "kwargs": {"gene": "POLR1B"}}], experiment="alexrank",
                         slurm_params={"slurm_partition": "gpu", "slurm_gres": "gpu:1", "cpus_per_task": 8,
                                       "mem_gb": 64, "timeout_min": 60, "slurm_constraint": "[h100|a100|l40s|a6000]"},
                         log_dir="alexrank", wait_for_completion=False)


if __name__ == "__main__":
    submit()
