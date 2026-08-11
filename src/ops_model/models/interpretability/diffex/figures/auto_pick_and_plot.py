"""For each batch target: scan full_features.json, pick the interpretable feature whose generated α3 lands
closest to real KO (same sign, |real KO| >= 12%), then generate its violin (SLURM) + line-graph. Prints the
picked feature + gen-α3-vs-KO per target so the choice is auditable."""
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("OPS_DIFFEX_ASSETS", "viewer_assets_v5")
from ops_model.models.interpretability.diffex.viewer.morpho_pipeline import MORPHO_TARGETS
from ops_model.models.interpretability.diffex.classifier.config import slugify

VA = "/hpc/projects/icd.fast.ops/models/diffex/viewer_assets_v5/_morphometrics"
BAD = re.compile("moment|hu_|inertia|eigval|intensity|haralick|zernike|glcm|orientation|centroid|_timing")

BATCH = ["KIF11_PHASE", "ATP6V1B2_PHASE", "HGS_PHASE", "RRM1_PHASE", "RRN3_PHASE", "SEC61A1_PHASE",
         "SON_PHASE", "AP2M1_PHASE", "GOLGA2_PHASE", "SMC2_PHASE", "NPC_PHASE", "PROTEASOME_PHASE",
         "HAUS_PHASE", "AP2M1_CLTA",
         "EIF2S2_SG", "AURKB_CHROMATIN", "NOP56_FBL", "ATG9A_AUTOPHAGO", "ATP6V1B2_LAMP1"]


def _dir(v):
    return f"{v['marker_dir']}/{v.get('grain', 'geneKO')}/{v['target']}"


NET_NOUN = {"skeleton_pixel_count": "Network length", "total_branch_length": "Network length",
            "num_branches": "Branch count", "num_endpoints": "Endpoint count", "average_degree": "Network connectivity",
            "num_skeleton_components": "Fragment count", "largest_connected_component_size": "Largest fragment",
            "network_length_density": "Network density", "branching_density": "Branching density",
            "num_nodes": "Node count", "euler_number": "Network topology"}
OBJ_NOUN = {"area": "size", "area_filled": "size", "extent": "compactness", "circularity": "roundness",
            "eccentricity": "elongation", "aspect_ratio": "elongation", "axis_minor_length": "width",
            "axis_major_length": "length", "equivalent_diameter_area": "diameter", "perimeter": "perimeter",
            "solidity": "solidity"}


def _label(f):
    """Intuitive y-axis label from a raw feature name (e.g. obj_extent_sum → 'Total compactness')."""
    m = re.match(r"network_.+?_seg_(.+)$", f)
    if m:
        return NET_NOUN.get(m.group(1), m.group(1).replace("_", " ").capitalize())
    if f.startswith("obj_"):
        b = f[4:]
        mm = re.match(r"(.+)_(sum|mean|median|std|min|max|count)$", b)
        prop, stat = (mm.group(1), mm.group(2)) if mm else (b, "")
        if stat == "count":
            return "Object count"
        noun = OBJ_NOUN.get(prop, prop.replace("_", " "))
        if stat == "std":
            return f"{noun.capitalize()} variability"
        s = ({"sum": "Total ", "min": "Min ", "max": "Max "}.get(stat, "") + noun).strip()
        return s[0].upper() + s[1:]
    return f.replace("_", " ")


def pick(dir_):
    """Best interpretable feature: gen α3 closest to real KO, same sign, |real KO| >= 12%."""
    d = json.load(open(f"{VA}/{dir_}/full_features.json"))
    al = d["alphas"]; agg = d["agg"]; rr = d.get("real_ref", {})
    z = min(range(len(al)), key=lambda i: abs(al[i]))
    i1 = min(range(len(al)), key=lambda i: abs(al[i] - 1)); i3 = min(range(len(al)), key=lambda i: abs(al[i] - 3))
    best = None
    for f, ser in agg.items():
        if not f.startswith(("obj_", "network_")) or BAD.search(f):
            continue
        r = rr.get(f)
        if not r or not r.get("ko") or r["ntc"][0] is None or ser[z] is None or ser[i1] is None:
            continue
        b = abs(ser[z]) or 1e-9; nb = abs(r["ntc"][0]) or 1e-9
        g1 = (ser[i1] - ser[z]) / b * 100; g3 = (ser[i3] - ser[z]) / b * 100; ko = (r["ko"][0] - r["ntc"][0]) / nb * 100
        # require BOTH α1 and α3 to track the real-KO direction (α1 not flat) — not just α3
        if abs(ko) < 12 or np.sign(g3) != np.sign(ko) or np.sign(g1) != np.sign(ko) or abs(g1) < 4:
            continue
        gap = abs(g3 - ko)
        if best is None or gap < best[0]:
            best = (gap, f, round(ko), round(g3))
    return best


def main():
    keys = sys.argv[1:] or BATCH
    figs = []
    for k in keys:
        v = MORPHO_TARGETS[k]; dir_ = _dir(v)
        try:
            b = pick(dir_)
        except FileNotFoundError:
            print(f"  {k}: no full_features.json"); continue
        if not b:
            print(f"  {k}: no interpretable feature tracks KO (skip)"); continue
        gap, feat, ko, g3 = b
        print(f"  {k:18} feat={feat:52} KO {ko:+4d}%  gen α3 {g3:+4d}%")
        figs.append({"group": k, "dir": dir_, "feature": feat, "simple": _label(feat),
                     "out_stem": f"{k}_{slugify(feat)[:24]}", "label": f"{k} · {feat}"})
    # violins (SLURM array) + line-graphs (local)
    from figure4_morpho_violin import submit
    submit(figs)
    from figure4_morpho_traversal import make_figure
    for f in figs:
        for c in range(2):
            try:
                make_figure(f["dir"], f["dir"], f["feature"], c, [0, 1, 3], f["label"], f["simple"],
                            f"{f['group']}/{f['out_stem']}_cell{c}")
            except Exception as e:
                print(f"  line skip {f['group']} c{c}: {type(e).__name__}")


if __name__ == "__main__":
    main()
