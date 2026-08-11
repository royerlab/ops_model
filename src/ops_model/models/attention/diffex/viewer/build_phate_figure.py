"""Figure 4 embedding — 3-panel reproduction of the paper's phase gene-PHATE (paper_v2):
  E  major biological-process arms
  F  mitochondrial sub-groups (membrane translocation/folding, electron transport, 39S mito ribosome)
  G  transcription / RNA-processing sub-groups (spliceosome U-snRNP & Prp19-LSm, Pol I/II/III)
Each panel: the same PHATE scatter (grey), that panel's groups colored + leader-labelled with the
single-cell generated morph (NTC cell1 → group, alpha=+5). NTC original shown top-left of panel E.

  python -m ops_model.models.attention.diffex.viewer.build_phate_figure
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.offsetbox import AnnotationBbox, OffsetImage, TextArea, VPacker

plt.rcParams["pdf.fonttype"] = 42

VA = "/hpc/projects/icd.fast.ops/models/diffex/viewer_assets"
LAYOUT = f"{VA}/_montage/layout_phate.json"
MORPH = f"{VA}/phase/geneKO/{{gene}}/cell1/frame_16.webp"   # alpha=+5 (index 16 of 17)
# NTC original = the SAME cell1 at alpha=0 (frame_08, the traversal midpoint = unmorphed base).
# Every morph above is this exact cell pushed toward its gene, so this is the honest reference.
NTC_IMG = f"{VA}/phase/geneKO/FANCC/cell1/frame_08.webp"
OUT_DIR = "/hpc/projects/icd.fast.ops/analysis/figure4_embedding"

# each panel: {label: (color, [match substrings])}. First matching label wins within a panel.
PANEL_E = {
    "spliceosome":            ("#e15759", ["spliceosome", "snRNP", "snRNA", "RNA splicing", "Prp19", "LSm"]),
    "RNA transcription":      ("#7b4173", ["rna polymerase", "mediator", "transcription factor TFII"]),
    "DNA replication":        ("#bcbd22", ["DNA replication", "MCM", "replicative", "origin recognition", "replisome"]),
    "mitochondria & ox. phos.": ("#8c6d31", ["mitochondrial", "electron transport", "respiratory chain", "39S mito"]),
    "ER-Golgi transport":     ("#2ca02c", ["ER-Golgi", "COPI", "COPII", "COP-II", "golgi", "endoplasmic reticulum", "SEC23", "SEC61"]),
    "dynein motors":          ("#17becf", ["dynein", "dynactin", "microtubule motor"]),
    "translation initiation": ("#4c78a8", ["translational initiation", "translation factor", "eIF", "eukaryotic initiation"]),
    "ribosome biogenesis":    ("#e377c2", ["ribosomal subunit processome", "ribosome biogenesis", "rRNA processing", "nucleolar"]),
    "60S ribosome":           ("#ff7f0e", ["60S cytosolic large ribosomal", "large ribosomal subunit"]),
    "40S ribosome":           ("#6baed6", ["40S cytosolic small ribosomal", "small ribosomal subunit"]),
    "proteasome":             ("#9467bd", ["proteasome", "PA700", "ubiquitin-dependent protein catabolic"]),
    "mTORC1":                 ("#1b9e77", ["mtorc1", "mtorc2", "mtor complex"]),
}
PANEL_F = {
    "mito. membrane translocation & folding": ("#8c6d31", ["tim23", "tom complex", "mitochondrial import", "presequence translocase", "translocase of the", "chaperonin", "hsp60"]),
    "electron transport chain":               ("#4c78a8", ["electron transport", "respiratory chain", "atp synthase", "cytochrome c oxidase", "nadh dehydrogenase"]),
    "39S mito. ribosome":                     ("#8bc34a", ["39s mitochondrial", "mitochondrial large ribosomal", "55S ribosome, mitochondrial"]),
}
PANEL_G = {
    "spliceosome U1-5 snRNPs":  ("#4c78a8", ["u1 snrnp", "u2 snrnp", "u4", "u5 snrnp", "u1-5", "u11/u12", "u2-type spliceosomal", "snrnp"]),
    "spliceosome Prp19 / LSm":  ("#8bc34a", ["prp19", "lsm", "nineteen complex", "intron lariat"]),
    "Pol-I RNA polymerase":     ("#c9a227", ["rna polymerase i complex", "polymerase i "]),
    "Pol-II RNA polymerase":    ("#8c6d31", ["rna polymerase ii", "polymerase ii complex"]),
    "Pol-III RNA polymerase":   ("#e15759", ["rna polymerase iii", "polymerase iii complex"]),
}


def _img(path, frac=0.80):
    from PIL import Image
    if not os.path.exists(path):
        return None
    a = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    h, w = a.shape; ch, cw = int(h * (1 - frac) / 2), int(w * (1 - frac) / 2)
    return a[ch:h - ch, cw:w - cw]


def _framed(a, color, bw_frac=0.045):
    """Grey image -> RGB with a solid group-colored border around the FOV."""
    from matplotlib.colors import to_rgb
    if a is None:
        return None
    rgb = np.repeat(a[:, :, None], 3, axis=2)
    c = np.array(to_rgb(color)); b = max(2, int(a.shape[0] * bw_frac))
    rgb[:b, :] = c; rgb[-b:, :] = c; rgb[:, :b] = c; rgb[:, -b:] = c
    return rgb


# hand-picked representative gene per arm (must be an EBI member of that arm with a morph)
REP_OVERRIDE = {
    "mitochondria & ox. phos.": "TIMM23",
    "RNA transcription":        "POLR1B",
    "40S ribosome":             "RPS16",
    "mTORC1":                   "MTOR",
}

# explicit arm angle (radians, 0=+x CCW) for arms whose cluster is too central to have a natural direction
ANG_OVERRIDE = {"mTORC1": math.radians(205)}

# per-label directional bias (span units, +x right / +y up), applied before declutter
NUDGE = {
    "ER-Golgi transport":  (0.34, 0.16),
    "proteasome":          (0.10, 0.32),
    "RNA transcription":   (0.22, 0.05),
    "40S ribosome":        (0.06, 0.00),
    "DNA replication":     (0.00, 0.12),
}


EBI_YAML = "/hpc/projects/icd.fast.ops/configs/gene_clusters/EBI_complexes_v1_updated_gene_names.yaml"
_G2C = None


def _ebi_complex(sym):
    """Authoritative gene -> EBI complex name (310 genes); None if the gene is in no EBI complex."""
    global _G2C
    if _G2C is None:
        import yaml
        _G2C = {}
        for v in yaml.safe_load(open(EBI_YAML)).values():
            for g in (v.get("genes") or []):
                _G2C.setdefault(g, v["name"])
    return _G2C.get(sym)


def _assign(g, panel):
    ebi = _ebi_complex(g.get("g"))                       # EBI complex membership is the only source of truth
    if not ebi:
        return None                                      # gene in no EBI complex -> grey/unassigned
    e = ebi.lower()
    for label, (_, subs) in panel.items():
        if any(s.lower() in e for s in subs):
            return label
    return None


def _arm_tip(pts, cen):
    """Robust outer tip of an arm = mean of its 5 points farthest from the cloud center."""
    rad = np.linalg.norm(pts - cen, axis=1)
    return pts[np.argsort(-rad)[:min(5, len(pts))]].mean(0)


def _declutter(box, cen, minsep, P, pad, iters=800, fixed=()):
    """Force-separate insets; keep each box just outside the scatter's edge *in its own direction*
    (hug the cloud, minimal whitespace). `fixed` boxes never move but still push the others away."""
    keys = list(box); fixed = set(fixed)
    for _ in range(iters):
        moved = False
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = keys[i], keys[j]
                v = box[a] - box[b]; d = np.hypot(*v)
                if d < minsep:
                    u = v / (d or 1); sh = (minsep - d) / 2
                    fa, fb = a in fixed, b in fixed
                    if fa and fb:
                        continue
                    if fb:
                        box[a] = box[a] + u * 2 * sh
                    elif fa:
                        box[b] = box[b] - u * 2 * sh
                    else:
                        box[a] = box[a] + u * sh; box[b] = box[b] - u * sh
                    moved = True
        for l in keys:                                   # keep box just past the cloud edge along its dir
            if l in fixed:
                continue
            dirv = box[l] - cen; rb = np.hypot(*dirv); u = dirv / (rb or 1)
            rmin = ((P - cen) @ u).max() + pad
            if rb < rmin:
                box[l] = cen + u * rmin
        if not moved:
            break
    return box


def build(out_dir=None, thumb=130, minsep=0.42, ntc_scale=1.7, alpha=5):
    out_dir = out_dir or OUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    frame = max(0, min(16, int(round(8 + alpha / 5.0 * 8))))   # 17 frames span alpha=-5..+5; 8 = base
    morph = MORPH.replace("frame_16", f"frame_{frame:02d}")
    genes = json.load(open(LAYOUT))["genes"]
    xy = {g["g"]: (g["nx"], g["ny"]) for g in genes}
    ntc = np.array([[g["nx"], g["ny"]] for g in genes if str(g["g"]).startswith("NTC")])
    real = [g for g in genes if not str(g["g"]).startswith("NTC")]
    P = np.array([[g["nx"], g["ny"]] for g in real])
    panel = PANEL_E
    grp = {g["g"]: _assign(g, panel) for g in real}
    labels = [l for l in panel if any(v == l for v in grp.values())]

    fig, ax = plt.subplots(figsize=(18, 16))
    ax.scatter(P[:, 0], P[:, 1], s=15, c="#dcdcdc", edgecolors="none", zorder=1)
    for l in labels:
        pts = np.array([xy[g] for g in grp if grp[g] == l])
        ax.scatter(pts[:, 0], pts[:, 1], s=43, c=panel[l][0], edgecolors="white", linewidths=0.4, zorder=3)
    if len(ntc):
        c = ntc.mean(0); r = 1.6 * np.percentile(np.linalg.norm(ntc - c, axis=1), 90) + 0.01
        ax.scatter(ntc[:, 0], ntc[:, 1], s=18, c="#000", edgecolors="none", zorder=5)
        ax.annotate("NTCs", c + [0, -r * 1.4], ha="center", va="top", fontsize=11, color="#000", weight="bold")

    cen = P.mean(0); span = max(np.ptp(P[:, 0]), np.ptp(P[:, 1]))
    pad = 0.12 * span                                    # gap from cloud edge to inset
    tip, box, rep, img, ang = {}, {}, {}, {}, {}
    for l in labels:
        members = [g for g in grp if grp[g] == l]
        pts = np.array([xy[g] for g in members]); gc = pts.mean(0)
        tip[l] = _arm_tip(pts, cen)
        withm = [g for g in members if os.path.exists(morph.format(gene=g))]
        ov = REP_OVERRIDE.get(l)
        rep[l] = ov if ov in members and os.path.exists(morph.format(gene=ov)) \
            else min(withm or members, key=lambda g: np.hypot(*(np.array(xy[g]) - gc)))
        img[l] = _img(morph.format(gene=rep[l]))
        v = tip[l] - cen
        ang[l] = ANG_OVERRIDE.get(l, math.atan2(v[1], v[0]) if np.hypot(*v) > 1e-6 else 0.0)
        u = np.array([math.cos(ang[l]), math.sin(ang[l])])
        box[l] = cen + u * (((P - cen) @ u).max() + pad)  # just beyond the cloud edge along this arm
    # pin 40S & 60S together: 40S at its slot, 60S stacked just below (close, non-overlapping)
    pair = [l for l in ("40S ribosome", "60S ribosome") if l in labels]
    if len(pair) == 2:
        box["60S ribosome"] = box["40S ribosome"] + np.array([0.0, -0.40 * span])
    for l, (dx, dy) in NUDGE.items():                    # directional bias before declutter resolves overlaps
        if l in box and l not in pair:
            box[l] = box[l] + np.array([dx, dy]) * span
    box = _declutter(box, cen, minsep * span, P, pad, fixed=pair)
    for l in labels:
        col = panel[l][0]
        kids = [TextArea(l, textprops=dict(color=col, size=12, weight="bold", ha="center"))]
        fr = _framed(img[l], col)
        if fr is not None:
            kids += [OffsetImage(fr, zoom=thumb / fr.shape[0]),
                     TextArea(rep[l], textprops=dict(color=col, size=13, weight="bold", ha="center"))]
        ax.add_artist(AnnotationBbox(VPacker(children=kids, align="center", pad=0, sep=2), tip[l], xybox=box[l],
                                     xycoords="data", boxcoords="data", frameon=False, annotation_clip=False,
                                     arrowprops=dict(arrowstyle="-", color=col, lw=1.3, shrinkA=0, shrinkB=3)))
    nimg = _framed(_img(NTC_IMG), "#111")   # original NTC — bigger, top-left corner reference
    if nimg is not None:
        kids = [TextArea("NTC (original)", textprops=dict(color="#111", size=14, weight="bold", ha="center")),
                OffsetImage(nimg, zoom=ntc_scale * thumb / nimg.shape[0])]
        ax.add_artist(AnnotationBbox(VPacker(children=kids, align="center", pad=0, sep=3), (0.005, 0.995),
                                     xycoords="axes fraction", box_alignment=(0, 1), frameon=False))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.margins(0.42)
    stem = f"phate_arms_morph_a{alpha:g}"
    for ext in ("png", "svg"):
        fig.savefig(f"{out_dir}/{stem}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[phate-fig] single-panel figure (alpha={alpha}, frame_{frame:02d}) -> {out_dir}/{stem}.png / .svg")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--thumb", type=int, default=130)
    ap.add_argument("--alpha", type=float, default=5)
    a = ap.parse_args()
    build(a.out, a.thumb, alpha=a.alpha)
