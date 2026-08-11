"""Per-marker SetTransformer real-cell set-accuracy (top1_acc @ bag=100) for the viewer's Perturbation
'by SET ACC' sort. Keys = slugified marker channel (matches attnModality()/VS slug) + "phase".
Each value = {perturbation_name: top1_acc} covering geneKO (gene_name) and complex (label_name, mean over members).
Written to viewer_assets_v5/_montage/setacc_bymarker.json."""
import json, os, re
import pandas as pd

V5 = "/hpc/projects/icd.fast.ops/models/alex_lin_attention/v5"
OUT = "/hpc/projects/icd.fast.ops/models/diffex/viewer_assets_v5/_montage/setacc_bymarker.json"
BAG = 100
slug = lambda s: re.sub(r"[^A-Za-z0-9]", "_", str(s))


def build():
    out = {}

    # --- fluor per-marker geneKO: {channel_slug: {gene: acc}} ---
    g = pd.read_csv(f"{V5}/fluorescence/fluor_bychannel_paperv2gene_cps_pergene.csv")
    g = g[g.n_cells == BAG]
    for ch, sub in g.groupby("channel_name"):
        out.setdefault(slug(ch), {}).update(dict(zip(sub.gene_name, sub.top1_acc.round(4))))

    # --- fluor per-marker complex: mean member-gene acc per (channel, label_name) ---
    ce = pd.read_csv(f"{V5}/fluorescence/fluor_ebi_bychannel_pergene.csv")
    ce = ce[ce.n_cells == BAG]
    for (ch, lab), sub in ce.groupby(["channel_name", "label_name"]):
        out.setdefault(slug(ch), {})[lab] = round(float(sub.top1_acc.mean()), 4)

    # --- phase geneKO + complex ---
    p = pd.read_csv(f"{V5}/phase/eval_phase_e200_pergene_val.csv")
    p = p[p.n_cells == BAG]
    out["phase"] = dict(zip(p.gene_name, p.top1_acc.round(4)))
    pe = pd.read_csv(f"{V5}/phase/eval_phase_ebionly_e200_pergene_val.csv")
    pe = pe[pe.n_cells == BAG]
    for lab, sub in pe.groupby("label_name"):
        out["phase"][lab] = round(float(sub.top1_acc.mean()), 4)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"))
    nmk = len(out) - 1
    print(f"[setacc_bymarker] {nmk} markers + phase; e.g. phase geneKO={len(out['phase'])} entries -> {OUT}")
    ex = next(k for k in out if k != "phase")
    print(f"  sample marker '{ex}': {list(out[ex].items())[:3]}")
    return out


if __name__ == "__main__":
    build()
