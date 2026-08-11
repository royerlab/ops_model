"""Per-marker SetTransformer real-cell set-accuracy across ALL bag sizes (bins) for the viewer's
Top-cells 'show classification accuracy' overlay (adaptive bin dropdown) + the 'by SET ACC' sort.

Same sources as build_setacc_bymarker.py but keeps every n_cells (bag size) instead of only 100.
Bins differ by modality (phase has up to 5000; fluor up to 500), so the viewer reads the per-marker
`bins` list and adapts the dropdown.

Output (compact, bins + aligned arrays):
  { "<marker_slug>"|"phase": { "bins": [10,20,...], "acc": { "<perturbation>": [acc@10, acc@20, ...] } } }
Written to viewer_assets_v5/_montage/setacc_bins_bymarker.json (new file; leaves the @100-only
setacc_bymarker.json in place so older app builds keep working)."""
import json, os, re
import pandas as pd

V5 = "/hpc/projects/icd.fast.ops/models/alex_lin_attention/v5"
OUT = "/hpc/projects/icd.fast.ops/models/diffex/viewer_assets_v5/_montage/setacc_bins_bymarker.json"
slug = lambda s: re.sub(r"[^A-Za-z0-9]", "_", str(s))


def build():
    raw = {}   # marker -> perturbation -> {bin: acc}

    def add(marker, pert, nb, acc):
        raw.setdefault(marker, {}).setdefault(pert, {})[int(nb)] = round(float(acc), 4)

    # fluor per-marker geneKO
    g = pd.read_csv(f"{V5}/fluorescence/fluor_bychannel_paperv2gene_cps_pergene.csv")
    for _, r in g.iterrows():
        add(slug(r.channel_name), r.gene_name, r.n_cells, r.top1_acc)

    # fluor per-marker complex: mean member-gene acc per (channel, label, bin)
    ce = pd.read_csv(f"{V5}/fluorescence/fluor_ebi_bychannel_pergene.csv")
    for r in ce.groupby(["channel_name", "label_name", "n_cells"]).top1_acc.mean().reset_index().itertuples(index=False):
        add(slug(r[0]), r[1], r[2], r[3])

    # phase geneKO
    p = pd.read_csv(f"{V5}/phase/eval_phase_e200_pergene_val.csv")
    for _, r in p.iterrows():
        add("phase", r.gene_name, r.n_cells, r.top1_acc)

    # phase complex: mean member-gene acc per (label, bin)
    pe = pd.read_csv(f"{V5}/phase/eval_phase_ebionly_e200_pergene_val.csv")
    for r in pe.groupby(["label_name", "n_cells"]).top1_acc.mean().reset_index().itertuples(index=False):
        add("phase", r[0], r[1], r[2])

    out = {}
    for marker, perts in raw.items():
        bins = sorted({b for d in perts.values() for b in d})
        out[marker] = {"bins": bins, "acc": {p: [d.get(b) for b in bins] for p, d in perts.items()}}

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"))
    nmk = len(out) - 1
    print(f"[setacc_bins] {nmk} markers + phase -> {OUT}")
    print(f"  phase bins={out['phase']['bins']} ({len(out['phase']['acc'])} perturbations)")
    ex = next(k for k in out if k != "phase")
    print(f"  fluor '{ex}' bins={out[ex]['bins']} ({len(out[ex]['acc'])} perturbations)")
    return out


if __name__ == "__main__":
    build()
