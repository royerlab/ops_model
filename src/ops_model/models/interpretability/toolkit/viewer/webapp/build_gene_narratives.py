#!/usr/bin/env python3
"""Pre-fetch affinage.wi.mit.edu mechanistic narratives for every gene in gene_desc.json
into gene_narrative.json, so the static viewer can show them without a runtime cross-origin
call to the (internal) affinage API.

    python build_gene_narratives.py            # fetch all genes in gene_desc.json
    python build_gene_narratives.py HSPA5 TP53 # fetch just these (for testing)

Output: {gene: mechanistic_narrative_str} for genes affinage has; missing/404 genes are skipped.
"""
import json, sys, os, urllib.request, concurrent.futures as cf

ASSETS = "/hpc/projects/icd.fast.ops/models/diffex/viewer_assets_v5"
API = "https://affinage.wi.mit.edu/api/gene/{}"
OUT = os.path.join(ASSETS, "gene_narrative.json")


def fetch(gene):
    try:
        req = urllib.request.Request(API.format(gene), headers={"User-Agent": "diffex-viewer/1.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            d = json.load(r)
        nar = (d.get("narrative") or {}).get("mechanistic_narrative")
        return gene, (nar.strip() if isinstance(nar, str) and nar.strip() else None)
    except Exception as e:
        return gene, None


def main():
    genes = sys.argv[1:] or sorted(json.load(open(os.path.join(ASSETS, "gene_desc.json"))).keys())
    print(f"fetching {len(genes)} genes from affinage…", flush=True)
    out, miss = {}, []
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        for i, (gene, nar) in enumerate(ex.map(fetch, genes), 1):
            if nar:
                out[gene] = nar
            else:
                miss.append(gene)
            if i % 100 == 0:
                print(f"  {i}/{len(genes)} · {len(out)} hit · {len(miss)} miss", flush=True)
    json.dump(out, open(OUT, "w"))
    print(f"wrote {OUT}: {len(out)} narratives ({len(miss)} missing)", flush=True)
    if miss:
        print("missing (first 40):", ", ".join(miss[:40]), flush=True)


if __name__ == "__main__":
    main()
