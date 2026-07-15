"""Shared selection logic for the DiffEx viewer cache: distinctiveness matrices, per-marker
top-gene ranking, the complete-marker catalog (marker_channel + raw channel + generator ckpt),
and the dist/description maps the manifest attaches. Imported by `submit.py` so every cache
build uses the same targets — no duplicated one-off scripts.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..classifier.config import slugify

OUT = "/hpc/projects/icd.fast.ops/models/diffex"
DD = f"{OUT}/diffae"
_DIST_BASE = ("/hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized_v0.3/cell_dino/"
              "zscore_per_exp/paper_v2/with_cp/with_4i")
_DIST_RELS = ["all_livecell"]   # v2 with_cp/with_4i: single 56-reporter matrix (live + CP + 4i)
LAUNCH_JSON = f"{OUT}/directions/_ranking/fluor_marker_launch.json"
GENE_PANEL = "/hpc/projects/icd.fast.ops/configs/annotated_gene_panel_July2025.csv"
EBI_YAML = "/hpc/projects/icd.fast.ops/configs/gene_clusters/EBI_complexes_v1_updated_gene_names.yaml"
EBI_FLUOR_CSV = "/hpc/projects/icd.fast.ops/models/alex_lin_attention/v4/pma_fluorescent_cells_ebi_all.csv"
GENE_EMB_H5AD = ("/hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized_v0.3/cell_dino/"
                 "zscore_per_exp/paper_v2/phase_only/fixed_80%/cosine/gene_embedding_pca_optimized.h5ad")

# fixed-cell reporter (distinctiveness matrix col) for CP/4i marker_channels
FIXED_REP = {
    "Endoplasmic Reticulum_Concanavalin A": "ER, ConA (cp)", "F-actin_Phalloidin": "f-actin, Phalloidin (cp)",
    "Microtubules_Tubulin": "microtubules, Tubulin (cp)", "Mitochondria_TOMM20": "mitochondria, TOMM20 (cp)",
    "Nucleus_Hoechst": "nuclei, Hoechst (cp)", "Nucleoli_NPM1": "nucleoli, NPM1 (cp)",
    "Plasma Membrane_Wheat Germ Agglutinin": "plasma membrane, WGA (cp)", "NFkB_NFkB (mouse-488)": "NFkB (4i)",
    "p53_p53 (mouse-488)": "p53 (4i)", "pRb_pRb (rabbit-647)": "pRb (4i)", "pS6_pS6 (rabbit-647)": "pS6 (4i)",
    "p21_p21 (rabbit-647)": "p21 (4i)", "b-catenin_b-catenin (mouse-488)": "b-catenin (4i)", "c-Myc_c-Myc (mouse-488)": "c-Myc (4i)",
}
# launch-complete fluorescent markers (ep≥98) by generator-dir slug
COMPLETE_LAUNCH = {
    "fluor_autophagosome_ATG101", "fluor_lysosome_LAMP1", "fluor_nucleolus_DFC_FBL", "fluor_stress_granule_G3BP1",
    "fluor_microtubules_MAP4", "fluor_mitochondria_TOMM70A", "fluor_ER_NCLN", "fluor_Endoplasmic_Reticulum_Concanavalin_A",
    "fluor_Fe2__FeRhoNox_live_cell_dye", "fluor_Microtubules_Tubulin", "fluor_Mitochondria_TOMM20", "fluor_5xUPRE",
    "fluor_ChromaLIVE_488_excitation", "fluor_ER_Golgi_COPE", "fluor_ER_Golgi_COP_II_SEC23A", "fluor_ER_SEC61B",
    "fluor_ER_golgi_bridge_VAPA", "fluor_F_actin_Phalloidin", "fluor_cell_proliferation_marker_MKI67",
    "fluor_Nucleus_Hoechst", "fluor_Nucleoli_NPM1", "fluor_Plasma_Membrane_Wheat_Germ_Agglutinin", "fluor_NFkB_NFkB__mouse_488",
}
# no-PMA markers: NOT in Alex's attention CSV (built after it) → traversals built from the per_signal
# `features_processed` anndata, centroid-ranked (see precompute._rows_from_anndata).
# (generator dir, marker_channel [→ dist reporter], raw channel, features_processed reporter for the h5ad).
NO_PMA_H5AD = ("/hpc/projects/icd.fast.ops/ops0*/3-assembly/cell_dino_features_v2/"
               "anndata_objects/features_processed_{rep}.h5ad")
NO_PMA_MARKERS = [
    ("fluor_cisGolgi_mStayGold", "cis-Golgi_mStayGold-CENPRaltORF", "GFP", "mStayGold-CENPRaltORF"),
    ("fluor_VIM", "intermediate filaments_VIM", "GFP", "VIM"),
    ("fluor_LMNB1", "laminin_LMNB1", "GFP", "LMNB1"),
]
# early (pre-launch) fluorescent markers: (generator dir, marker_channel, raw channel) recovered from training pickles
EARLY_MARKERS = [
    ("fluor_NucleoLive", "nucleus_NucleoLIVE Live Cell dye", "mCherry"),
    ("fluor_NPM3", "nucleolus-GC_NPM3", "GFP"),
    ("fluor_FastAct", "actin filament_FastAct_SPY555 Live Cell Dye", "mCherry"),
    ("fluor_LysoTracker", "lysosome_LysoTracker live-cell dye", "GFP"),
    ("fluor_ChromaLIVE_mito", "mitochondria_ChromaLIVE 561 excitation", "mCherry"),
]


def ebi_complexes():
    """Canonical EBI complex names — from the config yaml (do NOT re-derive from cells)."""
    import yaml
    y = yaml.safe_load(open(EBI_YAML)) or {}
    return [v["name"] for v in y.values() if isinstance(v, dict) and v.get("name")]


def all_genes():
    """All ~1000 geneKO classes (dist-matrix index, excl NTC)."""
    return [g for g in dist_matrix().index if not str(g).startswith("NTC")]


def dist_matrix():
    return pd.concat([pd.read_csv(f"{_DIST_BASE}/{v}/fixed_80%/cosine/plots/marker_overlay/"
                                  "gene_reporter_distinctiveness_raw.csv", index_col=0) for v in _DIST_RELS],
                     axis=1)


COMPLEX_EBI_MAP_CSV = f"{OUT}/complex_reporter_ebi_map.csv"


def complex_dist():
    """Complex(name) × reporter EBI mAP — the copairs/EBI-Complex-Portal metric computed per-marker
    by `build_complex_ebi_map` (NOT the chad_consistency file). Columns = reporters (match dist_matrix)."""
    return pd.read_csv(COMPLEX_EBI_MAP_CSV, index_col=0)


def _assignment(dist):
    d = dist.drop(index=[g for g in dist.index if str(g).startswith("NTC")], errors="ignore")
    vals = d.values; order = np.argsort(-vals, axis=1, kind="stable")
    bmap = np.take_along_axis(vals, order[:, :1], axis=1).ravel()
    smap = np.take_along_axis(vals, order[:, 1:2], axis=1).ravel()
    return pd.DataFrame({"gene": d.index, "best": d.columns[order[:, 0]], "bestmap": bmap, "margin": bmap - smap}), d


def top_genes(dist, rep, n=8):
    """Top-n marker-specific genes for a distinctiveness reporter (best-marker assignment, margin-broken)."""
    asg, d = _assignment(dist)
    c = asg[asg.best == rep].sort_values(["bestmap", "margin"], ascending=False)
    g = list(c["gene"][:n])
    if len(g) < n and rep in d:
        g += [x for x in d[rep].dropna().sort_values(ascending=False).index if x not in g][:n - len(g)]
    return g


def rep_of(dist, marker_channel):
    """Distinctiveness-matrix column (reporter) for a marker_channel (live: normalized; fixed: hand map)."""
    return {c.replace(", ", "_"): c for c in dist.columns}.get(marker_channel) or FIXED_REP.get(marker_channel)


def complete_markers(min_ep=0):
    """[(generator_dir, marker_channel, raw_channel)] for fluorescent markers with a trained generator
    (diffae_best.pt + train_state on disk). Default min_ep=0 = no epoch gate — epoch count is NOT a quality
    signal (cond_ratio peaks ~ep40-55 then declines; diffae_best.pt banks the peak). Pass min_ep only to
    re-impose a floor. `submit seed` auto-includes any marker with a checkpoint."""
    import os
    import torch
    launch = json.load(open(LAUNCH_JSON))
    cand = {"fluor_" + slugify(e["marker"]): (e["marker"], e["channel"]) for e in launch}
    for d, mc, ch in EARLY_MARKERS:
        cand[d] = (mc, ch)
    out = []
    for d, (mc, ch) in cand.items():
        sp = f"{DD}/{d}/diffae_train_state.pt"
        if not os.path.exists(f"{DD}/{d}/diffae_best.pt") or not os.path.exists(sp):
            continue
        try:
            ep = torch.load(sp, map_location="cpu", mmap=True).get("epoch", 0)
        except Exception:
            ep = 0
        if ep >= min_ep:
            out.append((d, mc, ch))
    return out


def desc_map():
    """{class_name: description} — per-gene function + known systems (GO/Reactome/KEGG/CORUM) pulled from
    the gene-embedding h5ad (populated for all ~1000 genes, unlike the sparse panel) + complex members (EBI).
    Sections are joined with ' || ' so the viewer can render each as a labeled block."""
    out = {}
    try:
        import anndata as ad
        ob = ad.read_h5ad(GENE_EMB_H5AD, backed="r").obs
        flds = [("LongName", None), ("go_bp_term", "GO biological process"), ("go_cc_term", "GO cellular component"),
                ("reactome_term", "Reactome"), ("kegg_term", "KEGG"), ("corum_complex_term", "CORUM complex")]
        def _v(i, c):
            s = str(ob[c].iloc[i]) if c in ob.columns else ""
            return "" if s in ("nan", "None", "") else s.strip()
        for i, g in enumerate(ob["perturbation"].astype(str)):
            if g.startswith("NTC"):
                continue
            parts = [(v if lbl is None else f"{lbl}: {v}") for c, lbl in flds if (v := _v(i, c))]
            if parts:
                out[g] = " || ".join(parts)
    except Exception as e:
        print("gene desc failed:", e)
    try:
        import yaml
        for _, v in (yaml.safe_load(open(EBI_YAML)) or {}).items():
            if isinstance(v, dict) and v.get("name"):
                mem = list(v.get("genes", []) or [])
                out[v["name"]] = (f"Members ({len(mem)}): " + ", ".join(mem)) if mem else v["name"]
    except Exception as e:
        print("complex desc failed:", e)
    return out


def dist_map_for_assets(viewer_assets):
    """{(modality, grain, slug): distinctiveness mAP} for every rendered target (manifest sorting)."""
    import glob
    dist = dist_matrix()
    dd = dist.drop(index=[g for g in dist.index if str(g).startswith("NTC")], errors="ignore")
    try:
        cx = complex_dist()                                   # complex × reporter EBI mAP (complexes aren't in the gene matrix)
    except Exception:
        cx = None
    try:
        mb = json.load(open(f"{viewer_assets}/_minibinder_meta.json"))   # minibinders → per-binder cell_score
    except Exception:
        mb = {}
    out = {}
    for mj in glob.glob(f"{viewer_assets}/*/*/*/meta.json"):
        m = json.load(open(mj))
        rep = (rep_of(dist, m["marker_channel"]) if m.get("marker_channel") else "Phase")
        if m["grain"] == "minibinder":                        # minibinders → cell_score (no mAP)
            if m["slug"] in mb:
                out[(m["modality"], m["grain"], m["slug"])] = float(mb[m["slug"]]["cell_score"])
        elif m["grain"] == "complex":                         # complexes → EBI complex mAP
            if cx is not None and m["target"] in cx.index and rep in cx.columns:
                v = cx.at[m["target"], rep]
                if pd.notna(v):
                    out[(m["modality"], m["grain"], m["slug"])] = float(v)
        elif rep in dd.columns and m["target"] in dd.index:   # geneKO → distinctiveness mAP
            v = dd.at[m["target"], rep]
            if pd.notna(v):
                out[(m["modality"], m["grain"], m["slug"])] = float(v)
    return out
