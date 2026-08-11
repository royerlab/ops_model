"""DIAGNOSTIC (removable): score the EXISTING v4 traversals (viewer_assets) with the v5 SetTransformer,
to compare v4 vs v5 phenotype accuracy — are the v4 (in-distribution) traversals better phenotypes?
Reads v4 frames directly (no writes into the live v4 assets); collects peak/α0/α5 P(target) per traversal."""
import json
import os

V4_BASE = "/hpc/projects/icd.fast.ops/models/diffex/viewer_assets/phase"


def score_v4_shard(grain, targets, out_json):
    from .set_classifier import load_set_classifier, V5_CKPT_ROOT, V5_RUNS
    from .score_generated import _emb_frames, score_embs_v5
    from ops_model.models.interpretability.diffae.directions.config import DirConfig
    from ops_model.models.interpretability.diffae.classifier.celldino_features import embed_crops
    from ops_model.models.interpretability.diffae.classifier.config import slugify
    run = V5_RUNS[("phase", "geneKO" if grain == "geneKO" else "complex_ebionly")]
    model, g2i, ci_map = load_set_classifier(run=run, device="cuda", root=V5_CKPT_ROOT)
    ci = ci_map.get("Phase2D", 0)
    sub = "geneKO" if grain == "geneKO" else "complex"
    res = {}
    for tgt in targets:
        trav = f"{V4_BASE}/{sub}/{tgt if grain == 'geneKO' else slugify(tgt)}"
        mp = f"{trav}/meta.json"
        if not os.path.exists(mp):
            continue
        alphas = json.load(open(mp))["alphas"]
        cfg = DirConfig(grain=grain, target=(tgt if grain == "geneKO" else "NTC"), device="cuda")
        embs = [_emb_frames(cfg, trav, ai, embed_crops) for ai in range(len(alphas))]
        d = score_embs_v5(embs, alphas, tgt, model, g2i, ci, run, "cuda")
        if d:
            p = d["p_target"]; z0 = len(alphas) // 2
            fin = [v for v in p if v is not None]
            res[tgt] = {"alphas": d["alphas"], "p_target": p,   # full curve for the accuracy-vs-α plot
                        "peak": max(fin) if fin else None, "a0": p[z0], "a5": p[-1]}
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    json.dump(res, open(out_json, "w"))
    return {"grain": grain, "n": len(res), "out": out_json}
