"""Build the 'Top Cells' tab assets: per-gene top-ranked phenotype cells by attention or accuracy.

Two rankings (phase, per gene, top-N):
  Top Attention  alex_lin_attention/v4/pma_phase_cells_all_v2.csv         ranked by pma_attention (incl. NTC)
  Top Accuracy   alex_lin_attention/v4/accuracy_ranking/pergene_phase_cell_rankings.csv   ranked by classifier score

Each ranked cell has experiment/well/x_pheno/y_pheno/segmentation → we crop it from phenotyping_v3.zarr with
the SAME 150px + blue negative cell-mask overlay as the PC tab (reuses build_pc_crops_masked), dedup crops by
position (a cell ranked in both lists shares one PNG), and emit:
  viewer_assets/top_cells/index.json   {"top_n", "genes": {GENE: {"attention": [rec...], "accuracy": [rec...]}}}
  viewer_assets/top_cells/crops/<pos>.png

  python -m ops_model.models.attention.diffex.viewer.build_top_cells --genes HSPA5 NTC AARS   # validate subset
  python -m ops_model.models.attention.diffex.viewer.build_top_cells                          # full (~40k crops)
"""
from __future__ import annotations

import argparse
import csv
import json
import os

from . import catalog as C
from .build_pc_crops_masked import BASE, CROP_SIZE, PHASE_CHANNEL, _crop, _is_blank, _render, _zarr_patch

V4 = "/hpc/projects/icd.fast.ops/models/alex_lin_attention/v4"
ATTN_CSV = f"{V4}/pma_phase_cells_all_v2.csv"
ACC_CSV = f"{V4}/accuracy_ranking/pergene_phase_cell_rankings.csv"
ATTN_EBI = f"{V4}/pma_phase_cells_ebi_all.csv"                              # complex-level (EBI) attention
ACC_EBI = f"{V4}/accuracy_ranking/ebi_pergene_phase_cell_rankings.csv"      # complex-level (EBI) accuracy
OUT = f"{C.OUT}/viewer_assets/top_cells"
TOP_N = 20


def _pos_key(exp, well, x, y):
    return f"{exp}__{well}__{int(round(float(x)))}__{int(round(float(y)))}"


def _read_ranking(path, *, gene_c, exp_c, well_c, x_c, y_c, seg_c, score_c, rank_c, top_n, names=None, base=False):
    """Stream a ranking CSV, keep rows with rank<=top_n (optionally only `names`) → {name: [record...]}.
    base=True keys/filters by the complex BASE name (before the first comma) since the EBI label variant
    text is inconsistent between the attention and accuracy CSVs."""
    keep, want = {}, set(names) if names else None
    with open(path, newline="") as f:
        r = csv.reader(f)
        next(r, None)   # header
        for row in r:
            try:
                if int(float(row[rank_c])) > top_n:
                    continue
            except (ValueError, IndexError):
                continue
            g = row[gene_c].strip().strip('"')
            if base:
                g = g.split(",")[0].strip()
            if want is not None and g not in want:
                continue
            keep.setdefault(g, []).append({
                "exp": row[exp_c], "well": row[well_c], "x": float(row[x_c]), "y": float(row[y_c]),
                "seg": row[seg_c] if seg_c is not None else "", "rank": int(float(row[rank_c])),
                "score": round(float(row[score_c]), 5),
            })
    for g in keep:
        keep[g].sort(key=lambda d: d["rank"])
    return keep


def _crop_cells(attn, acc, crops_dir):
    """Crop every unique cell (dedup by position across both rankings) from the zarr with the blue mask."""
    import zarr
    from PIL import Image
    _zarr_patch()
    uniq = {}
    for d in (attn, acc):
        for recs in d.values():
            for c in recs:
                uniq[_pos_key(c["exp"], c["well"], c["x"], c["y"])] = c
    print(f"[topcells] {len(uniq)} unique cells to crop")
    os.makedirs(crops_dir, exist_ok=True)
    half = CROP_SIZE // 2
    cache, ok, blank, fail, valid = {}, 0, 0, 0, set()
    for i, (pk, c) in enumerate(uniq.items()):
        exp, well = c["exp"], c["well"]
        key = (exp, well)
        if key not in cache:
            pos = f"{BASE}/{exp}/3-assembly/phenotyping_v3.zarr/{well[0]}/{well[1:]}/0"
            try:
                cache[key] = (zarr.open(f"{pos}/0", mode="r"), zarr.open(f"{pos}/labels/cell_seg/0", mode="r"))
            except Exception as e:
                cache[key] = None; print(f"[topcells] open failed {exp}/{well}: {e}")
        if cache[key] is None:
            fail += 1; continue
        img, seg = cache[key]
        x, y = int(round(c["x"])), int(round(c["y"]))
        try:
            phase = _crop(img, PHASE_CHANNEL, x, y, half)
            if _is_blank(phase):
                blank += 1; continue
            Image.fromarray(_render(phase, _crop(seg, None, x, y, half), half)).save(f"{crops_dir}/{pk}.png")
            ok += 1; valid.add(pk)
        except Exception as e:
            fail += 1
            if fail <= 5:
                print(f"[topcells] crop failed {pk}: {e}")
        if (i + 1) % 2000 == 0:
            print(f"[topcells] {i + 1}/{len(uniq)}  ok={ok} blank={blank} fail={fail}")
    print(f"[topcells] crops: {ok} written, {blank} blank, {fail} failed")
    return valid


def _recs(d, extra, valid):   # attach crop filename; drop cells without a valid crop
    out_d = {}
    for g, cells in d.items():
        lst = [{"img": f"{_pos_key(c['exp'], c['well'], c['x'], c['y'])}.png", "gene": g, "exp": c["exp"],
                "well": c["well"], "x": round(c["x"], 1), "y": round(c["y"], 1), "rank": c["rank"], extra: c["score"]}
               for c in cells if _pos_key(c["exp"], c["well"], c["x"], c["y"]) in valid]
        if lst:
            out_d[g] = lst
    return out_d


def _merge_index(out, top_n, key, entries):
    """Merge entries under index[key] ('genes' or 'complexes'), preserving the other key."""
    path = f"{out}/index.json"
    idx = json.load(open(path)) if os.path.exists(path) else {"marker": "phase"}
    idx["marker"] = "phase"; idx["top_n"] = top_n
    idx.setdefault(key, {}).update(entries)
    with open(path, "w") as f:
        json.dump(idx, f)
    print(f"[topcells] {key}: +{len(entries)} → {len(idx[key])} total; {os.path.getsize(path) / 1024:.0f} KB")
    return path


def build(genes=None, out=OUT, top_n=TOP_N):
    print(f"[topcells] GENE rankings (top {top_n}/gene)…")
    attn = _read_ranking(ATTN_CSV, gene_c=0, exp_c=4, well_c=5, x_c=7, y_c=8, seg_c=6, score_c=9, rank_c=10, top_n=top_n, names=genes)
    acc = _read_ranking(ACC_CSV, gene_c=0, exp_c=2, well_c=3, x_c=4, y_c=5, seg_c=6, score_c=8, rank_c=9, top_n=top_n, names=genes)
    print(f"[topcells]   {len(attn)} attention, {len(acc)} accuracy genes")
    valid = _crop_cells(attn, acc, f"{out}/crops")
    a, c = _recs(attn, "attn", valid), _recs(acc, "conf", valid)
    entries = {g: {"attention": a.get(g, []), "accuracy": c.get(g, [])} for g in sorted(set(a) | set(c))}
    return _merge_index(out, top_n, "genes", entries)


def build_complexes(labels, out=OUT, top_n=TOP_N):
    """Complexes use the EBI rankings (label_name keyed, base complex names). Crops share the same dir."""
    labels = [l.split(",")[0].strip() for l in labels]   # match by base complex name
    print(f"[topcells] COMPLEX (EBI) rankings for {labels} (top {top_n})…")
    attn = _read_ranking(ATTN_EBI, gene_c=3, exp_c=4, well_c=5, x_c=7, y_c=8, seg_c=6, score_c=9, rank_c=10, top_n=top_n, names=labels, base=True)   # complex label = predicted_class (col3)
    acc = _read_ranking(ACC_EBI, gene_c=0, exp_c=4, well_c=5, x_c=6, y_c=7, seg_c=8, score_c=10, rank_c=11, top_n=top_n, names=labels, base=True)
    print(f"[topcells]   {len(attn)} attention, {len(acc)} accuracy complexes: {sorted(set(attn) | set(acc))}")
    valid = _crop_cells(attn, acc, f"{out}/crops")
    a, c = _recs(attn, "attn", valid), _recs(acc, "conf", valid)
    entries = {g: {"attention": a.get(g, []), "accuracy": c.get(g, [])} for g in sorted(set(a) | set(c))}
    return _merge_index(out, top_n, "complexes", entries)


COMPLEX_RECORDS = f"{OUT}/_complex_records.json"


def prepare_complex_records(out=OUT, top_n=TOP_N, labels=None):
    """Single pass over the EBI rankings → {base_complex: {attention:[cells], accuracy:[cells]}} (positions,
    no crops). Written once so parallel crop shards don't each re-stream the multi-GB CSVs."""
    bases = [l.split(",")[0].strip() for l in labels] if labels else None
    attn = _read_ranking(ATTN_EBI, gene_c=3, exp_c=4, well_c=5, x_c=7, y_c=8, seg_c=6, score_c=9, rank_c=10, top_n=top_n, names=bases, base=True)
    acc = _read_ranking(ACC_EBI, gene_c=0, exp_c=4, well_c=5, x_c=6, y_c=7, seg_c=8, score_c=10, rank_c=11, top_n=top_n, names=bases, base=True)
    recs = {c: {"attention": attn.get(c, []), "accuracy": acc.get(c, [])} for c in sorted(set(attn) | set(acc))}
    os.makedirs(out, exist_ok=True)
    with open(f"{out}/_complex_records.json", "w") as f:
        json.dump(recs, f)
    print(f"[topcells] prepared records for {len(recs)} complexes -> {out}/_complex_records.json")
    return list(recs)


def crop_complex_shard(labels, out=OUT):
    """SLURM job: crop this shard's complex cells (from _complex_records.json) into the shared crops/ dir.
    Crops are additive (unique per position) so parallel shards never conflict; the index is built later."""
    recs = json.load(open(f"{out}/_complex_records.json"))
    attn = {c: recs[c]["attention"] for c in labels if c in recs}
    acc = {c: recs[c]["accuracy"] for c in labels if c in recs}
    _crop_cells(attn, acc, f"{out}/crops")
    return {"labels": labels, "n": sum(len(v) for v in attn.values()) + sum(len(v) for v in acc.values())}


def _cap(cells, top_n):
    """Base-name aggregation can merge several variant labels → dedup by position, keep the top_n by score, re-rank."""
    seen, uniq = set(), []
    for c in sorted(cells, key=lambda d: -d["score"]):
        pk = _pos_key(c["exp"], c["well"], c["x"], c["y"])
        if pk in seen:
            continue
        seen.add(pk); uniq.append(c)
        if len(uniq) >= top_n:
            break
    return [{**c, "rank": i} for i, c in enumerate(uniq, 1)]


def finalize_complex_index(out=OUT, top_n=TOP_N):
    """Build index['complexes'] from the prepared records + whichever crops now exist; merge (preserve genes)."""
    recs = json.load(open(f"{out}/_complex_records.json"))
    valid = {f[:-4] for f in os.listdir(f"{out}/crops") if f.endswith(".png")}
    attn = {c: _cap(d["attention"], top_n) for c, d in recs.items()}
    acc = {c: _cap(d["accuracy"], top_n) for c, d in recs.items()}
    a, c = _recs(attn, "attn", valid), _recs(acc, "conf", valid)
    entries = {g: {"attention": a.get(g, []), "accuracy": c.get(g, [])} for g in sorted(set(a) | set(c))}
    return _merge_index(out, top_n, "complexes", entries)


def submit_complexes(out=OUT, top_n=TOP_N, n_shards=16):
    """Prepare records, fan out crop shards on SLURM, then finalize the complex index."""
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    names = prepare_complex_records(out, top_n)
    shards = [names[i::n_shards] for i in range(n_shards)]
    shards = [s for s in shards if s]
    jobs = [{"name": f"topcells_cx_{i}", "func": crop_complex_shard, "kwargs": {"labels": s, "out": out}}
            for i, s in enumerate(shards)]
    print(f"[topcells] submitting {len(jobs)} SLURM shards for {len(names)} complexes")
    submit_parallel_jobs(jobs, experiment="topcells_complexes",
                         slurm_params={"slurm_partition": "cpu", "cpus_per_task": 8, "mem_gb": 32, "timeout_min": 90},
                         log_dir="topcells_complexes", wait_for_completion=True)
    finalize_complex_index(out, top_n)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--genes", nargs="*", default=None, help="only these genes (subset; omit for all)")
    ap.add_argument("--complexes", nargs="*", default=None, help="EBI complex label_name(s) → local build_complexes")
    ap.add_argument("--all-complexes", action="store_true", help="all EBI complexes via SLURM parallel shards")
    ap.add_argument("--finalize-complexes", action="store_true", help="rebuild complex index from existing crops")
    ap.add_argument("--top-n", type=int, default=TOP_N)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    if a.all_complexes:
        submit_complexes(a.out, a.top_n)
    elif a.finalize_complexes:
        finalize_complex_index(a.out, a.top_n)
    elif a.complexes:
        build_complexes(a.complexes, a.out, a.top_n)
    else:
        build(a.genes, a.out, a.top_n)
