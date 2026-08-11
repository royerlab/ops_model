"""v5 SetTransformer accuracy of GENERATED traversal frames — the accuracy-vs-α validation overlay.

v5 is no-mask/160px, so generated frames feed the classifier directly:
  frame → embed_crops (no seg mask) → z-standardize on the NTC control (cached anchor ctrl.npz)
  → bag → v5 SetTransformer → softmax → P(target).
Writes scores_v5.json into each traversal dir: {alphas, p_target:[...], top1_target:[...]}.

geneKO target = the gene's class prob. complex target = sum of member-gene probs (ebionly model,
gene-labeled), matching the complex-centroid definition. A→B alt-anchor = the target (B) prob.
"""
from __future__ import annotations
import glob
import json
import os

import numpy as np
from PIL import Image

CACHE = os.environ.get("OPS_DIFFEX_ASSETS", "viewer_assets")
V5_BASE = f"/hpc/projects/icd.fast.ops/models/diffex/{CACHE}/phase"


def _emb_frames(cfg, trav, ai, embed_crops):
    """embed_crops the α=ai frame of every cell in a traversal (reverse _save_webp → [-1,1], no mask)."""
    imgs = []
    for c in range(len(glob.glob(f"{trav}/cell*"))):
        f = f"{trav}/cell{c}/frame_{ai:02d}.webp"
        if os.path.exists(f):
            imgs.append(np.asarray(Image.open(f).convert("L"), np.float32) / 255.0 * 2 - 1)
    if not imgs:
        return None
    return embed_crops(np.stack(imgs)[:, None].astype(np.float32), cfg, cache_path=None)


def diagnose(gene="TOMM20", device="cuda"):
    """Isolate the generated~0 issue: score REAL cells of `gene` through OUR embed_crops pipeline
    (raw, and z-standardized on NTC control), vs Alex's own val embeddings (positive control)."""
    import torch
    from .set_classifier import load_set_classifier, score_bags, V5_CKPT_ROOT, V5_RUNS
    from .precompute import _gather_class
    from ..directions.config import DirConfig
    m, g2i, c2i = load_set_classifier(run=V5_RUNS[("phase", "geneKO")], device=device, root=V5_CKPT_ROOT)
    ci = c2i.get("Phase2D", 0); gi = g2i[gene]
    # Alex val (already z-std) — positive control
    av = torch.load(f"/hpc/projects/icd.fast.ops/models/alex_lin_attention/v5/paper_v2_phase/val/{gene}.pt", map_location="cpu")["embeddings"].numpy()
    p_alex = float(score_bags(m, av[:100][None], channel_idx=ci, device=device)[0][gi])
    # our pipeline on REAL cells
    cfg = DirConfig(grain="geneKO", target=gene, device=device)
    _, real = _gather_class(cfg, gene, 100)          # materialize + embed_crops (our extraction)
    z = np.load(f"{V5_BASE}/_anchors/NTC/ctrl.npz"); mu = z["ctrl_embs"].mean(0); sd = z["ctrl_embs"].std(0) + 1e-6
    p_raw = float(score_bags(m, real[None], channel_idx=ci, device=device)[0][gi])
    p_zstd = float(score_bags(m, ((real - mu) / sd)[None], channel_idx=ci, device=device)[0][gi])
    print(f"[diagnose {gene}] Alex-val={p_alex:.3f} | OUR-real raw={p_raw:.3f} zstd-on-NTC={p_zstd:.3f}")
    print(f"  emb scale: Alex mean={av.mean():.2f} std={av.std():.2f} | our-real mean={real.mean():.2f} std={real.std():.2f} | our-zstd mean={((real-mu)/sd).mean():.2f} std={((real-mu)/sd).std():.2f}")
    import json
    json.dump({"gene": gene, "p_alex": p_alex, "p_raw": p_raw, "p_zstd": p_zstd},
              open("/hpc/projects/icd.fast.ops/models/diffex/viewer_assets_v5/_diag.json", "w"))


def diagnose2(gene="TOMM20", device="cuda"):
    """Test the domain-offset fix: standardize GENERATED frames by GENERATED-NTC(α0) stats instead of real NTC.
    Reports P(target) vs α for both references + whether generated α0 is recognized as NTC."""
    import torch, json
    from .set_classifier import load_set_classifier, score_bags, V5_CKPT_ROOT, V5_RUNS
    from ..directions.config import DirConfig
    from ..classifier.celldino_features import embed_crops
    m, g2i, c2i = load_set_classifier(run=V5_RUNS[("phase", "geneKO")], device=device, root=V5_CKPT_ROOT)
    ci = c2i.get("Phase2D", 0); gi = g2i[gene]; ni = g2i.get("NTC")
    trav = f"{V5_BASE}/geneKO/{gene}"; alphas = json.load(open(f"{trav}/meta.json"))["alphas"]
    embs = [_emb_frames(DirConfig(grain="geneKO", target=gene, device=device), trav, ai, embed_crops) for ai in range(len(alphas))]
    z0 = len(alphas) // 2
    zr = np.load(f"{V5_BASE}/_anchors/NTC/ctrl.npz"); mu_r, sd_r = zr["ctrl_embs"].mean(0), zr["ctrl_embs"].std(0) + 1e-6
    mu_g, sd_g = embs[z0].mean(0), embs[z0].std(0) + 1e-6         # generated-NTC (α0) stats
    def curve(mu, sd, idx):
        return [round(float(score_bags(m, ((e - mu) / sd)[None], channel_idx=ci, device=device)[0][idx]), 3) for e in embs]
    res = {"gene": gene, "alphas": alphas, "p_target_realNTC": curve(mu_r, sd_r, gi),
           "p_target_genNTC": curve(mu_g, sd_g, gi),
           "p_ntc_genNTC": curve(mu_g, sd_g, ni) if ni is not None else None}
    json.dump(res, open("/hpc/projects/icd.fast.ops/models/diffex/viewer_assets_v5/_diag2.json", "w"))
    print("DIAG2 DONE")


def _real_expectation(grain, target):
    """Alex real-cell top1_acc by bag size. geneKO → the gene's row; complex → MEAN over the complex's
    member-gene rows (grouped by Alex's label_name in the ebionly eval)."""
    import csv as _csv
    from collections import defaultdict
    E = "/hpc/projects/icd.fast.ops/models/alex_lin_attention/v5/phase"
    if grain == "geneKO":
        return {int(r["n_cells"]): float(r["top1_acc"]) for r in _csv.DictReader(open(f"{E}/eval_phase_e200_pergene_val.csv")) if r["gene_name"] == target}
    by = defaultdict(list)
    for r in _csv.DictReader(open(f"{E}/eval_phase_ebionly_e200_pergene_val.csv")):
        if r["label_name"] == target:
            by[int(r["n_cells"])].append(float(r["top1_acc"]))
    return {b: float(np.mean(v)) for b, v in by.items()}


def bag_experiment(grain="geneKO", target="MICOS13", n_max=200, sizes=(20, 50, 100, 150, 200), n_bags=30, device="cuda"):
    """Regenerate `target` (grain geneKO|complex) with n_max cells, then at the peak α sweep bag size →
    generated top1_acc + mean P(target), vs Alex's REAL top1_acc-by-bag (gene row, or mean-member for a
    complex). Writes _bagexp_<slug>.json. Isolated OPS_DIFFEX_ASSETS (bagtest dir set by caller)."""
    import json, torch
    from .precompute import precompute_marker
    from .submit import PHASE_CK
    from .set_classifier import load_set_classifier, score_bags, V5_CKPT_ROOT, V5_RUNS
    from ..directions.config import DirConfig
    from ..classifier.celldino_features import embed_crops
    from ..classifier.config import slugify
    OUT = "/hpc/projects/icd.fast.ops/models/diffex"
    run = V5_RUNS[("phase", "geneKO" if grain == "geneKO" else "complex_ebionly")]
    precompute_marker(grain=grain, targets=[target], ckpt=PHASE_CK, out_root=OUT, n_cells=n_max,
                      score=False, device=device, force=True)
    model, cmap, c2i = load_set_classifier(run=run, device=device, root=V5_CKPT_ROOT)
    if target not in cmap:
        print(f"[skip] {target} not in class map"); return
    gi = cmap[target]; ci = c2i.get("Phase2D", 0)
    slug = target if grain == "geneKO" else slugify(target)
    trav = f"{V5_BASE}/{grain}/{slug}"; alphas = json.load(open(f"{trav}/meta.json"))["alphas"]
    cfg = DirConfig(grain=grain, target=(target if grain == "geneKO" else "NTC"), device=device)
    embs = [_emb_frames(cfg, trav, ai, embed_crops) for ai in range(len(alphas))]
    z0 = len(alphas) // 2; mu = embs[z0].mean(0); sd = embs[z0].std(0) + 1e-6
    N = len(embs[z0])
    fullp = [float(score_bags(model, ((embs[ai] - mu) / sd)[None], channel_idx=ci, device=device)[0][gi]) for ai in range(len(alphas))]
    ai_pk = int(np.argmax(fullp)); Epk = (embs[ai_pk] - mu) / sd
    rng = np.random.default_rng(0)
    res = {"grain": grain, "target": target, "alphas": alphas, "n_generated": N, "peak_alpha": alphas[ai_pk],
           "real_expectation": _real_expectation(grain, target), "bag": {}}
    for sz in sizes:
        if sz > N:
            continue
        nb = 1 if sz >= N else n_bags
        ps, t1 = [], []
        for _ in range(nb):
            idx = rng.choice(N, sz, replace=False)
            prob = score_bags(model, Epk[idx][None], channel_idx=ci, device=device)[0]
            ps.append(float(prob[gi])); t1.append(int(int(np.argmax(prob)) == gi))
        res["bag"][str(sz)] = {"top1_acc": float(np.mean(t1)), "mean_p": float(np.mean(ps)), "n_bags": nb}
        print(f"[bag {sz}] {target[:30]} gen top1={np.mean(t1):.2f} | real={res['real_expectation'].get(sz,'-')}")
    os.makedirs(f"{OUT}/viewer_assets_v5_bagtest", exist_ok=True)
    json.dump(res, open(f"{OUT}/viewer_assets_v5_bagtest/_bagexp_{slug}.json", "w"))
    print(f"BAGEXP DONE {target} (peak α={alphas[ai_pk]:+g})")


def score_embs_v5(embs, alphas, tgt, model, g2i, ci, run, device="cuda", bag=None):
    """Score already-embedded per-α traversal frames with the v5 SetTransformer. embs[ai] = ncell×1024
    CellDINO embs for α=alphas[ai] (or None). Standardize on the α0 (middle) generated frames → removes
    the DiffAE domain offset. bag: score a FIXED-size bag (first `bag` cells) even if more are generated,
    for cross-approach comparability. Returns scores_v5.json dict, or None if tgt not in class map / no α0."""
    from .set_classifier import score_bags
    idxs = [g2i[tgt]] if tgt in g2i else []
    z0 = len(alphas) // 2
    if not idxs or embs[z0] is None:
        return None
    E = [None if e is None else (e[:bag] if bag else e) for e in embs]   # fixed bag → same statistic across approaches
    mu = E[z0].mean(0); sd = E[z0].std(0) + 1e-6
    tset = set(idxs)
    ptgt, top1, top5, ranks = [], [], [], []
    for emb in E:
        if emb is None:
            ptgt.append(None); top1.append(None); top5.append(None); ranks.append(None); continue
        prob = score_bags(model, ((emb - mu) / sd)[None], channel_idx=ci, device=device)[0]
        order = np.argsort(prob)[::-1].tolist()
        ptgt.append(float(prob[idxs].sum()))
        top1.append(int(order[0] in tset))
        top5.append(int(bool(tset & set(order[:5]))))                  # target among the 5 most-likely classes
        ranks.append(int(min(order.index(i) for i in idxs)) + 1)       # 1-indexed rank of the target class
    return {"alphas": alphas, "p_target": ptgt, "top1_target": top1, "top5_target": top5, "rank_target": ranks, "run": run, "bag": bag or len(E[z0])}


MAIN_V5 = "/hpc/projects/icd.fast.ops/models/diffex/viewer_assets_v5/phase"


def bag_scaling(grain, targets, n_max=500, sizes=(20, 50, 100, 200), n_bags=30, device="cuda"):
    """Lean bag-scaling pass. For each target: read its PEAK α from the existing main-build scores_v5.json,
    generate ONLY {α=0, peak α} × n_max cells (α=0 = standardization reference; peak = the pool), then sweep
    bag size sampling n_bags DISTINCT bags per size (resampled from the pool, like Alex's real-cell eval) →
    generated top1_acc + mean P(target) vs Alex's real expectation. Writes _bagexp_<slug>.json in bagtest."""
    import json, torch
    from .precompute import precompute_marker
    from .submit import PHASE_CK
    from .set_classifier import load_set_classifier, score_bags, V5_CKPT_ROOT, V5_RUNS
    from ..directions.config import DirConfig
    from ..classifier.celldino_features import embed_crops
    from ..classifier.config import slugify
    OUT = "/hpc/projects/icd.fast.ops/models/diffex"
    run = V5_RUNS[("phase", "geneKO" if grain == "geneKO" else "complex_ebionly")]
    model, cmap, c2i = load_set_classifier(run=run, device=device, root=V5_CKPT_ROOT)
    ci = c2i.get("Phase2D", 0); rng = np.random.default_rng(0)
    for tgt in targets:
      try:
        if tgt not in cmap:
            print(f"[skip] {tgt} not in class map"); continue
        gi = cmap[tgt]; slug = tgt if grain == "geneKO" else slugify(tgt)
        sc = json.load(open(f"{MAIN_V5}/{grain}/{slug}/scores_v5.json"))   # peak α from the main 20-cell build
        pk_a = sc["alphas"][int(np.nanargmax([-1 if v is None else v for v in sc["p_target"]]))]
        precompute_marker(grain=grain, targets=[tgt], ckpt=PHASE_CK, out_root=OUT, n_cells=n_max,
                          alphas=[0.0, pk_a], score=False, device=device, force=True)   # only α0 + peak α
        trav = f"{V5_BASE}/{grain}/{slug}"
        cfg = DirConfig(grain=grain, target=(tgt if grain == "geneKO" else "NTC"), device=device)
        al = sorted([0.0, pk_a]); pk_i = al.index(pk_a); z0_i = al.index(0.0)
        e0 = _emb_frames(cfg, trav, z0_i, embed_crops); ep = _emb_frames(cfg, trav, pk_i, embed_crops)
        mu = e0.mean(0); sd = e0.std(0) + 1e-6; E = (ep - mu) / sd; N = len(E)
        res = {"grain": grain, "target": tgt, "peak_alpha": pk_a, "n_generated": N,
               "real_expectation": _real_expectation(grain, tgt), "bag": {}}
        for sz in sizes:
            if sz > N:
                continue
            nb = 1 if sz >= N else n_bags
            ps, t1 = [], []
            for _ in range(nb):
                idx = rng.choice(N, sz, replace=False)                      # distinct resampled bag (Alex-style)
                prob = score_bags(model, E[idx][None], channel_idx=ci, device=device)[0]
                ps.append(float(prob[gi])); t1.append(int(int(np.argmax(prob)) == gi))
            res["bag"][str(sz)] = {"top1_acc": float(np.mean(t1)), "mean_p": float(np.mean(ps)), "n_bags": nb}
        os.makedirs(f"{OUT}/viewer_assets_v5_bagtest", exist_ok=True)
        json.dump(res, open(f"{OUT}/viewer_assets_v5_bagtest/_bagexp_{slug}.json", "w"))
        print(f"[bagscale {tgt[:30]}] peakα={pk_a:+g} N={N} @20={res['bag'].get('20',{}).get('top1_acc')} @200={res['bag'].get('200',{}).get('top1_acc')}")
      except Exception as e:
        import traceback; print(f"[ERR {tgt}] {repr(e)[:120]}"); traceback.print_exc()


def score_targets(grain, targets, device="cuda", run=None, members_map=None, bag=None):
    """Score a list of traversals (grain='geneKO'|'complex'). members_map: complex→member genes (complex only).
    Writes scores_v5.json per traversal dir. Returns {target: p_target-per-α}."""
    import torch  # noqa
    from .set_classifier import load_set_classifier, score_bags, V5_CKPT_ROOT, V5_RUNS
    from ..directions.config import DirConfig
    from ..classifier.celldino_features import embed_crops
    from ..classifier.config import slugify
    run = run or V5_RUNS[("phase", "geneKO" if grain == "geneKO" else "complex_ebionly")]
    model, g2i, c2i = load_set_classifier(run=run, device=device, root=V5_CKPT_ROOT)
    ci = c2i.get("Phase2D", 0)
    sub = "geneKO" if grain == "geneKO" else "complex"
    out = {}
    for tgt in targets:
      try:                                                 # per-target guard: one bad target must not kill the shard
        # geneKO → gene class; complex → the model classifies complexes directly (label_to_idx). g2i is the model's class map.
        idxs = [g2i[tgt]] if tgt in g2i else []
        trav = f"{V5_BASE}/{sub}/{tgt if grain == 'geneKO' else slugify(tgt)}"
        mp = f"{trav}/meta.json"
        if not idxs or not os.path.exists(mp):
            print(f"[skip] {tgt}: idxs={len(idxs)} meta={os.path.exists(mp)}"); continue
        alphas = json.load(open(mp))["alphas"]
        cfg = DirConfig(grain=grain, target=(tgt if grain == "geneKO" else "NTC"), device=device)
        embs = [_emb_frames(cfg, trav, ai, embed_crops) for ai in range(len(alphas))]
        d = score_embs_v5(embs, alphas, tgt, model, g2i, ci, run, device, bag)
        if d is None:
            print(f"[skip] {tgt}: no α0 frames / not in class map"); continue
        json.dump(d, open(f"{trav}/scores_v5.json", "w"))
        out[tgt] = d["p_target"]; z0 = len(alphas) // 2
        print(f"[score] {tgt}: a0={d['p_target'][z0]:.3f} a+5={d['p_target'][-1]:.3f} rise={d['p_target'][-1]-d['p_target'][z0]:+.3f}")
      except Exception as e:
        import traceback; print(f"[ERR] {tgt}: {repr(e)[:150]}"); traceback.print_exc()
    return out


def score_anchor_traversals(grain, device="cuda"):
    """Score the A→B alt-anchor traversals: P(target B) per α via the v5 SetTransformer, standardizing on the
    α0 (= anchor-A) frames — same recipe as the NTC traversals, just target=B. Writes scores_v5.json per dir."""
    import glob
    from .set_classifier import load_set_classifier, V5_CKPT_ROOT, V5_RUNS
    from ..directions.config import DirConfig
    from ..classifier.celldino_features import embed_crops
    run = V5_RUNS[("phase", "geneKO" if grain == "geneKO" else "complex_ebionly")]
    model, g2i, c2i = load_set_classifier(run=run, device=device, root=V5_CKPT_ROOT)
    ci = c2i.get("Phase2D", 0)
    sub = "geneKO" if grain == "geneKO" else "complex"
    out = {}
    for trav in sorted(glob.glob(f"{V5_BASE}/{sub}/*__to__*")):
        mp = f"{trav}/meta.json"
        if not os.path.exists(mp):
            continue
        m = json.load(open(mp)); B = m["target"]; alphas = m["alphas"]
        cfg = DirConfig(grain=grain, target=B, device=device)
        embs = [_emb_frames(cfg, trav, ai, embed_crops) for ai in range(len(alphas))]
        d = score_embs_v5(embs, alphas, B, model, g2i, ci, run, device)
        name = os.path.basename(trav)
        if d is None:
            print(f"[skip] {name}: B={B} not in class map / no α0"); continue
        json.dump(d, open(f"{trav}/scores_v5.json", "w"))
        z0 = len(alphas) // 2; p = d["p_target"]
        out[name] = p
        print(f"[anchor-score] {name}: a0={p[z0]:.3f} peak={max(v for v in p if v is not None):.3f}")
    print(f"ANCHOR SCORING DONE {grain}: {len(out)} traversals")
    return out
