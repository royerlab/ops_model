"""Build the PC ↔ morphometric-feature panel for the OPSin viewer 'PCs' tab.

The 'Features' toggle (next to tf-idf) swaps the ontology-enrichment bars for
morphometric-feature enrichment, sourced from
`organelle_profiler/scripts/pc_feature_correlation/pc_feature_correlation.py`
(PC × OP/CP phase-feature Pearson r + TF-IDF distinctiveness). Per PC we emit:
    - top +corr / -corr features (raw) and top TF-IDF-distinctive features
    - feature-class composition and organelle-group composition (both raw & tf-idf)

Sign alignment: the correlation matrix comes from a gene-level PCA
(gene_embedding_pca_optimized.h5ad), while the viewer strips are binned by Kyle's
own cell-level PCA (viewer geneData 'score' == his gene_pc_scores). Same axes up
to a per-PC sign flip, so we flip each PC's r by sign(pearson(kyle_score, P_corr))
to make "+corr features" line up with the strip's high bins. Compositions are
unsigned (|r| / tf-idf) so they need no flip.

  python -m ops_model.models.attention.diffex.viewer.build_pc_features
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os

import numpy as np
import pandas as pd

from . import catalog as C

PCS_OUT = f"{C.OUT}/viewer_assets/pcs"
CORR_DIR = ("/hpc/projects/icd.fast.ops/analysis/pc_feature_correlation/phase_only")
EMB_H5AD = ("/hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized_v0.3/"
            "cell_dino/zscore_per_exp/paper_v1/phase_only/fixed_80%/cosine/"
            "gene_embedding_pca_optimized.h5ad")
_SRC = ("/hpc/mydata/gav.sturm/ops_mono/organelle_profiler/scripts/"
        "pc_feature_correlation/pc_feature_correlation.py")

TOP_FEATS = 8       # +corr / -corr / distinctive features shown per PC
COMP_TOP_N = 50     # features whose group shares make up the composition bars


def _load_helpers():
    """Import _feat_class/_organelle_group/_composition + taxonomy from the analysis script."""
    spec = importlib.util.spec_from_file_location("pc_feature_correlation", _SRC)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _pc_flips(idx, n_pcs):
    """Per PC (1..n_pcs): +1 if the correlation-embedding axis matches the strip's
    low→high direction, else -1. Determined by pearson between Kyle's gene scores
    (viewer geneData, sparse top_pcs) and the correlation embedding's gene projections."""
    import anndata as ad
    emb = ad.read_h5ad(EMB_H5AD)
    genes = [str(g) for g in emb.obs_names]
    gi = {g: i for i, g in enumerate(genes)}
    var = {str(v): i for i, v in enumerate(emb.var_names)}
    X = np.asarray(emb.X)
    # sparse Kyle scores from geneData.top_pcs → {pc: [(kyle_score, proj)]}
    pairs = {p: [] for p in range(1, n_pcs + 1)}
    for g, gd in idx["geneData"].items():
        if g not in gi:
            continue
        for tp in gd["top_pcs"]:
            col = f"Phase_PC{tp['pc'] - 1}"
            if col in var:
                pairs[tp["pc"]].append((tp["score"], X[gi[g], var[col]]))
    flips, conf, weak = {}, {}, []
    for p in range(1, n_pcs + 1):
        a = np.array(pairs[p])
        if len(a) >= 8:
            r = np.corrcoef(a[:, 0], a[:, 1])[0, 1]
            flips[p] = -1.0 if r < 0 else 1.0
            conf[p] = abs(r) >= 0.5
        else:
            flips[p] = 1.0
            conf[p] = False
        if not conf[p]:
            weak.append(p)
    print(f"[pcfeat] sign flips resolved for {n_pcs} PCs; {len(weak)} low-confidence "
          f"(|pearson|<0.5 or sparse): {weak[:12]}{'...' if len(weak) > 12 else ''}")
    return flips, conf


def _source_group(name):   # feature-name prefix → profiling tool
    return "CellProfiler" if name.startswith("cp") else "OrganelleProfiler" if name.startswith("op") else "other"


# statistic/aggregation suffixes stripped (with trailing numeric param tokens) to get a feature's base name
STAT_TOKENS = {"mean", "median", "std", "max", "min", "sum", "var", "variance", "q1", "q3", "q25", "q75",
               "p25", "p75", "iqr", "mad", "sem", "cv", "range", "count", "total", "integrated", "avg"}


def _base_name(name):
    """Collapse variants of a measurement to one base: drop trailing stat suffixes and numeric params, so
    e.g. AngularSecondMoment_3_00_256 / _3_02_256 / eccentricity_mean / _std all group under one name."""
    toks = name.split("_")
    while len(toks) > 2 and (toks[-1].isdigit() or toks[-1].lower() in STAT_TOKENS):
        toks.pop()
    return "_".join(toks)


def _dedup_matrix(M):   # collapse columns sharing a base name → mean across the variants
    groups = {}
    for c in M.columns:
        groups.setdefault(_base_name(c), []).append(c)
    return pd.DataFrame({b: M[cols].mean(axis=1) for b, cols in groups.items()}, index=M.index)


def _composition_norm(M, group_of, groups, top_n):
    """Like the analysis _composition (share of a PC's top-N features per group) but corrected for how many
    features each group has: divide each group's top-N count by that group's total feature count, then
    renormalize to sum 1. Removes the base-rate advantage of groups we simply measure more of."""
    gsize = {g: sum(1 for f in M.columns if group_of.get(f) == g) for g in groups}
    out = pd.DataFrame(0.0, index=M.index, columns=groups)
    for pc in M.index:
        top = M.loc[pc].sort_values(ascending=False).head(top_n)
        cnt = {}
        for f in top.index:
            g = group_of.get(f)
            cnt[g] = cnt.get(g, 0) + 1
        vals = {g: (cnt.get(g, 0) / gsize[g] if gsize[g] else 0.0) for g in groups}
        s = sum(vals.values()) or 1.0
        for g in groups:
            out.loc[pc, g] = vals[g] / s
    return out


def _comp_dict(comp_row, groups):
    """PC composition row → {group: fraction} dropping zero groups, biggest first."""
    d = {g: round(float(comp_row[g]), 3) for g in groups if comp_row[g] > 0}
    return dict(sorted(d.items(), key=lambda kv: -kv[1]))


def _panels(Rm, TFm, H, flips, n_pcs):
    """Per-PC feature panel for one matrix pair: top +r/-r + distinctive features and the class/organelle/source
    compositions. Works identically on the full feature set or the deduped (base-name-collapsed) one."""
    feats = list(Rm.columns)
    cls_of = {f: H._feat_class(f) for f in feats}
    org_of = {f: H._organelle_group(f) for f in feats}
    src_of = {f: _source_group(f) for f in feats}
    cls_g = [g for g in H.FEATURE_CLASSES if any(v == g for v in cls_of.values())]
    org_g = [g for g in H.ORGANELLE_GROUPS if any(v == g for v in org_of.values())]
    src_g = [g for g in ("CellProfiler", "OrganelleProfiler", "other") if any(v == g for v in src_of.values())]
    Rabs = Rm.abs()
    comp = H._composition                              # raw share of top-N features per group
    cR = {"cls": comp(Rabs, cls_of, cls_g, COMP_TOP_N), "org": comp(Rabs, org_of, org_g, COMP_TOP_N), "src": comp(Rabs, src_of, src_g, COMP_TOP_N)}
    cT = {"cls": comp(TFm, cls_of, cls_g, COMP_TOP_N), "org": comp(TFm, org_of, org_g, COMP_TOP_N), "src": comp(TFm, src_of, src_g, COMP_TOP_N)}
    nR = {"cls": _composition_norm(Rabs, cls_of, cls_g, COMP_TOP_N), "org": _composition_norm(Rabs, org_of, org_g, COMP_TOP_N), "src": _composition_norm(Rabs, src_of, src_g, COMP_TOP_N)}
    nT = {"cls": _composition_norm(TFm, cls_of, cls_g, COMP_TOP_N), "org": _composition_norm(TFm, org_of, org_g, COMP_TOP_N), "src": _composition_norm(TFm, src_of, src_g, COMP_TOP_N)}
    out = {}
    for p in range(1, n_pcs + 1):
        rn = f"Phase_PC{p - 1}"
        s = (Rm.loc[rn] * flips[p]).sort_values(ascending=False)   # sign-aligned to strip low→high
        pos = [{"f": f, "r": round(float(v), 3)} for f, v in s.head(TOP_FEATS).items() if v > 0]
        neg = [{"f": f, "r": round(float(v), 3)} for f, v in s.tail(TOP_FEATS).items() if v < 0][::-1]
        td = TFm.loc[rn].sort_values(ascending=False).head(TOP_FEATS)
        dist = [{"f": f, "tfidf": round(float(v), 3), "r": round(float(Rm.loc[rn, f] * flips[p]), 3)} for f, v in td.items()]
        out[p] = {"raw": {"pos": pos, "neg": neg,
                          "cls": _comp_dict(cR["cls"].loc[rn], cls_g), "org": _comp_dict(cR["org"].loc[rn], org_g), "src": _comp_dict(cR["src"].loc[rn], src_g),
                          "clsN": _comp_dict(nR["cls"].loc[rn], cls_g), "orgN": _comp_dict(nR["org"].loc[rn], org_g), "srcN": _comp_dict(nR["src"].loc[rn], src_g)},
                  "tfidf": {"dist": dist,
                            "cls": _comp_dict(cT["cls"].loc[rn], cls_g), "org": _comp_dict(cT["org"].loc[rn], org_g), "src": _comp_dict(cT["src"].loc[rn], src_g),
                            "clsN": _comp_dict(nT["cls"].loc[rn], cls_g), "orgN": _comp_dict(nT["org"].loc[rn], org_g), "srcN": _comp_dict(nT["src"].loc[rn], src_g)}}
    return out, (cls_g, org_g, src_g)


def write_features_json(R, TF, flips, conf, n_pcs, out, H=None):
    """Assemble + write features.json from a PC×feature correlation matrix R and tf-idf matrix TF (rows named
    Phase_PC0..). Shared by the phase build and the per-marker build (build_pcs_marker)."""
    H = H or _load_helpers()
    full, (cls_g, org_g, src_g) = _panels(R, TF, H, flips, n_pcs)
    dedup, _ = _panels(_dedup_matrix(R), _dedup_matrix(TF), H, flips, n_pcs)   # base-name-collapsed (default view)
    out_data = {"meta": {"classes": cls_g, "orgGroups": org_g, "srcGroups": src_g,
                         "compTopN": COMP_TOP_N, "topFeats": TOP_FEATS}}
    for p in range(1, n_pcs + 1):
        out_data[str(p)] = {"dirConf": bool(conf[p]),
                            "raw": {"full": full[p]["raw"], "dedup": dedup[p]["raw"]},
                            "tfidf": {"full": full[p]["tfidf"], "dedup": dedup[p]["tfidf"]}}
    path = f"{out}/features.json"
    with open(path, "w") as f:
        json.dump(out_data, f)
    print(f"[pcfeat] {n_pcs} PCs feature panel -> {path} ({os.path.getsize(path) / 1024:.0f} KB)")
    return path


def build(out=PCS_OUT):
    H = _load_helpers()
    R = pd.read_parquet(f"{CORR_DIR}/raw/pc_feature_corr_matrix.parquet")   # PC × feature signed r
    TF = pd.read_parquet(f"{CORR_DIR}/tfidf/tfidf_matrix.parquet")          # PC × feature tf-idf (>=0)
    idx = json.load(open(f"{out}/index.json"))
    n_pcs = idx["overview"]["n_pcs"]
    flips, conf = _pc_flips(idx, n_pcs)
    return write_features_json(R, TF, flips, conf, n_pcs, out, H)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=PCS_OUT)
    build(ap.parse_args().out)
