"""Build PC Strip Explorer assets for the OPSin viewer 'PCs' tab.

Kyle's `kyle_pcs/build_static_explorer.py` bakes everything (inline JSON + base64 crops) into one
41 MB HTML. This splits that into the static-asset layout the OPSin app uses:
    viewer_assets/pcs/index.json        {overview, pcData, geneData, geneNames}
    viewer_assets/pcs/crops/<img>.png   the per-cell strip crops (pc{NNN}_bin{NN}_row{N}.png)

Source = the self-contained `pc_explorer_static.html` (the assembled data + crops live inline there;
the raw artifacts dir is gone). If Kyle regenerates artifacts, re-run his build_static_explorer.py
then point --html at the fresh output.

  python -m ops_model.models.attention.diffex.viewer.build_pcs
  python -m ops_model.models.attention.diffex.viewer.build_pcs --html /path/to/pc_explorer_static.html
"""
from __future__ import annotations

import argparse
import base64
import json
import os

from . import catalog as C

DEFAULT_HTML = os.path.join(os.path.dirname(__file__), "..", "kyle_pcs", "pc_explorer_static.html")
PCS_OUT = f"{C.OUT}/viewer_assets/pcs"


def build_from_html(html_path=DEFAULT_HTML, out=PCS_OUT):
    """Extract the inline `const overview/pcData/geneData/geneNames/cropB64 = …;` blocks (one per line)
    → index.json + decoded crop PNGs. No re-computation; the HTML is the assembled source of truth."""
    consts, crop_b64 = {}, {}
    for line in open(html_path):
        for v in ("overview", "pcData", "geneData", "geneNames"):
            pre = f"const {v} = "
            if line.startswith(pre):
                consts[v] = json.loads(line[len(pre):].rstrip()[:-1])   # strip trailing ';'
        if line.startswith("const cropB64 = "):
            crop_b64 = json.loads(line[len("const cropB64 = "):].rstrip()[:-1])
    missing = [v for v in ("overview", "pcData", "geneData", "geneNames") if v not in consts]
    if missing:
        raise SystemExit(f"could not extract {missing} from {html_path}")

    os.makedirs(f"{out}/crops", exist_ok=True)
    for name, b64 in crop_b64.items():
        with open(f"{out}/crops/{name}", "wb") as f:
            f.write(base64.b64decode(b64))
    with open(f"{out}/index.json", "w") as f:
        json.dump({k: consts[k] for k in ("overview", "pcData", "geneData", "geneNames")}, f)
    ov = consts["overview"]
    print(f"[pcs] {ov['n_pcs']} PCs · {ov['n_genes']} genes · {len(crop_b64)} crops -> {out}/index.json")
    return f"{out}/index.json"


def _pc_gene_sets(gene_data, top_n=50, tfidf=False):
    """Per PC, split genes by the SIGN of their loading (from geneData top_pcs) → high/low gene lists,
    each capped at top_n. Default ranks by |loading|. With tfidf=True, rank by |loading|·idf where
    idf = log(nPCs/(1+df)) and df = # PC-directions the gene is in the raw top-n — so genes shared
    across many PCs sink and PC-unique genes rise (→ the enrichment reflects PC-unique biology)."""
    import math
    high, low = {}, {}
    for gene, gd in gene_data.items():
        for tp in gd["top_pcs"]:
            (high if tp["score"] > 0 else low).setdefault(tp["pc"], []).append((gene, abs(tp["score"])))
    n_pcs = max([*high, *low], default=1)
    if not tfidf:
        top = lambda d: {p: [g for g, _ in sorted(v, key=lambda x: -x[1])[:top_n]] for p, v in d.items()}
        return top(high), top(low)
    df = {}   # document frequency over the raw top-n sets (both directions)
    for d in (high, low):
        for p, v in d.items():
            for g, _ in sorted(v, key=lambda x: -x[1])[:top_n]:
                df[g] = df.get(g, 0) + 1
    idf = lambda g: math.log(n_pcs / (1 + df.get(g, 0)))
    rr = lambda d: {p: [g for g, _ in sorted([(g, s * idf(g)) for g, s in v], key=lambda x: -x[1])[:top_n]] for p, v in d.items()}
    return rr(high), rr(low)


# the 4 the viewer shows (labels below); GO_BP, GO_compartments, KEGG, Reactome
ENRICH_LIBS = ("GO_Biological_Process_2025", "GO_Cellular_Component_2025", "Reactome_2022", "KEGG_2026")
LIB_LABEL = {"GO_Biological_Process_2025": "GO BP", "GO_Cellular_Component_2025": "GO compartment",
             "Reactome_2022": "Reactome", "KEGG_2026": "KEGG"}


def _n_overlap(overlap):
    return len([x for x in str(overlap).strip("[]").split(",") if x.strip()])


def annotate_term_sizes(out=PCS_OUT):
    """Add K (total genes in each ontology term) to every enrichment record so the app can show k/K %.
    Enrichr's API returns only the overlap gene list, not K — it lives in the library GMT. Fetch each
    library's GMT once, key term sizes by the same display name we stored (GO id stripped), annotate."""
    import urllib.request
    sizes = {}
    for lib in ENRICH_LIBS:
        url = f"https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName={lib}"
        try:
            txt = urllib.request.urlopen(url, timeout=120).read().decode()
        except Exception as e:
            print(f"[pcs] GMT fetch failed for {lib}: {e}"); continue
        m = {}
        for line in txt.splitlines():
            p = line.split("\t")
            if len(p) < 3:
                continue
            m[p[0].split(" (GO:")[0]] = len([g for g in p[2:] if g.strip()])   # strip GO id to match stored term
        sizes[LIB_LABEL[lib]] = m
        print(f"[pcs] {lib}: {len(m)} term sizes")
    enr = json.load(open(f"{out}/enrichment.json"))
    for dirs in enr.values():
        for libs in dirs.values():
            for lib, terms in libs.items():
                sm = sizes.get(lib, {})
                for t in terms:
                    t["K"] = sm.get(t["term"])
    with open(f"{out}/enrichment.json", "w") as f:
        json.dump(enr, f)
    print(f"[pcs] annotated term sizes -> {out}/enrichment.json")


def build_enrichment(out=PCS_OUT, top_n=50, top_terms=8):
    """Enrichr (speedrichr) per PC × direction × library on the PC-loading gene sets → enrichment.json:
    {pc: {high: {lib_label: [{term, adjp, n_overlap}]}, low: {...}}}, terms ranked by adjusted p-value.
    Reuses embedding_overlays._run_cluster_enrichment (GO BP + GO compartments + Reactome + KEGG)."""
    from ops_model.post_process.combination.embedding_overlays import _run_cluster_enrichment
    idx = json.load(open(f"{out}/index.json"))
    bg = idx["geneNames"]
    rawH, rawL = _pc_gene_sets(idx["geneData"], top_n, tfidf=False)
    tfH, tfL = _pc_gene_sets(idx["geneData"], top_n, tfidf=True)
    c2g = {}    # raw high/low (H/L) + tf-idf high/low (h/l) per PC — cache both so the toggle needs no API
    for p, g in rawH.items(): c2g[f"H{p}"] = g
    for p, g in rawL.items(): c2g[f"L{p}"] = g
    for p, g in tfH.items(): c2g[f"h{p}"] = g
    for p, g in tfL.items(): c2g[f"l{p}"] = g
    print(f"[pcs] enrichment on {len(c2g)} PC×direction sets (raw + tf-idf) × {len(ENRICH_LIBS)} libraries...")
    res = _run_cluster_enrichment(c2g, background_genes=bg, libraries=ENRICH_LIBS, top_n_terms=top_terms)

    def compact(key):
        bl = (res.get(key) or {}).get("by_library", {})
        return {LIB_LABEL[lib]: [{"term": t["term"].split(" (GO:")[0], "adjp": t["adj_pvalue"], "n": _n_overlap(t["overlap"])}
                                 for t in bl.get(lib, [])] for lib in ENRICH_LIBS if lib in bl}
    enr = {str(p): {"high": compact(f"H{p}"), "low": compact(f"L{p}"),
                    "high_tfidf": compact(f"h{p}"), "low_tfidf": compact(f"l{p}")}
           for p in range(1, idx["overview"]["n_pcs"] + 1)}
    with open(f"{out}/enrichment.json", "w") as f:
        json.dump(enr, f)
    print(f"[pcs] enrichment for {len(enr)} PCs -> {out}/enrichment.json")
    annotate_term_sizes(out)     # add K (term sizes) for k/K %
    return f"{out}/enrichment.json"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", default=DEFAULT_HTML)
    ap.add_argument("--out", default=PCS_OUT)
    ap.add_argument("--enrich", action="store_true", help="also run GO/KEGG/Reactome enrichment per PC")
    ap.add_argument("--enrich-only", action="store_true", help="skip html extract; just (re)run enrichment")
    ap.add_argument("--sizes-only", action="store_true", help="just annotate term sizes (K) onto existing enrichment.json")
    args = ap.parse_args()
    if args.sizes_only:
        annotate_term_sizes(args.out)
    else:
        if not args.enrich_only:
            build_from_html(args.html, args.out)
        if args.enrich or args.enrich_only:
            build_enrichment(args.out)
