"""Build v5 fluorescence COMPLEX-level (EBI) Top Cells for all 55 markers.

Alex supplied only a per-GENE EBI cell ranking (gene_marker_ebi_complexqual.compact.parquet, cells labeled by
member gene) plus a gene->complex label map (fluor_ebi_bychannel_pergene.csv: gene_name -> label_name). There is
no complex-labeled cell ranking, so we build one: pool each complex's member-gene TOP cells (saturation bag),
re-rank by model score, keep top-N. Reuses crop_marker_shard (key="complexes") to merge a `complexes` block into
each marker's existing top-cells index.json; register_complex() adds grain="complex" targets to the manifest.
"""
import os, re, json
import pandas as pd
from . import catalog as C
from ops_model.models.interpretability.diffae.classifier.config import slugify
from ops_model.models.interpretability.diffae.traversal._fluor_topcells import crop_marker_shard, TOP_N, OUT

F = "/hpc/projects/icd.fast.ops/models/alex_lin_attention/v5/fluorescence"
EBI = f"{F}/misc/gene_marker_ebi_complexqual.compact.parquet"     # per-GENE EBI cells (all 55 channels)
G2C = f"{F}/fluor_ebi_bychannel_pergene.csv"                      # gene_name -> label_name (complex)
ASSETS = "viewer_assets_v5"
RANKDIR = f"{C.OUT}/{ASSETS}/_rankings/fluor/complex"
_COLS = ["channel_name", "gene", "bag_size", "rank", "score", "experiment", "well", "x_pheno", "y_pheno", "segmentation_id", "_pool"]


def build_rankings():
    """Per-channel complex ranking parquets (geneKO schema, `gene` col = complex name) in RANKDIR. → {channel: (parquet, [complexes])}."""
    os.makedirs(RANKDIR, exist_ok=True)
    g2c = (pd.read_csv(G2C, usecols=["gene_name", "label_name"]).dropna()
           .drop_duplicates("gene_name").set_index("gene_name")["label_name"].to_dict())
    df = pd.read_parquet(EBI, columns=_COLS)
    df = df[df["_pool"] == "top"]                                                    # top cells (not the random pool)
    mb = df.groupby(["channel_name", "gene"])["bag_size"].transform("max")           # saturation bag per member gene
    df = df[df["bag_size"] == mb].copy()
    df["complex"] = df["gene"].map(g2c)
    df = df.dropna(subset=["complex"])
    out = {}
    for ch, gch in df.groupby("channel_name"):
        parts = []
        for cx, gcx in gch.groupby("complex"):
            g = (gcx.drop_duplicates(["experiment", "well", "x_pheno", "y_pheno"])   # one row per cell
                    .sort_values("score", ascending=False).head(TOP_N).copy())       # re-rank member cells by score
            g["rank"] = range(1, len(g) + 1)
            g["gene"] = cx                                                            # grouping key -> complex name
            parts.append(g)
        if not parts:
            continue
        o = (pd.concat(parts, ignore_index=True)[["channel_name", "gene", "rank", "score", "experiment", "well",
                                                   "x_pheno", "y_pheno", "segmentation_id"]]
             .rename(columns={"segmentation_id": "segmentation", "score": "pma_attention"}))
        o["rank_type"] = "top"
        o["predicted_class"] = o["gene"]        # complex class_col for grain="complex" (top-cells crop uses `gene`)
        p = f"{RANKDIR}/{slugify(ch)}.parquet"; o.to_parquet(p)
        out[ch] = (p, sorted(o["gene"].unique()))
    print(f"[complex-rank] {len(out)} channels; complexes/channel: {[len(v[1]) for v in out.values()][:5]}...")
    return out


def register_complex():
    """Register grain='complex' targets for every fluor marker (from the `complexes` block of its index.json).
    Complexes with a generated traversal (meta.json) get its asset_dir/alphas; the rest are top-cells-only."""
    V5 = f"{C.OUT}/{ASSETS}"
    man = json.load(open(f"{V5}/manifest.json"))
    total = full = 0
    for mk in man["markers"]:
        mc = mk.get("marker_channel")
        if not mc or re.match(r"(?i)phase", mc):
            continue
        mod = slugify(mc)
        tci = f"{V5}/top_cells/markers/{mod}/index.json"
        cx = sorted(json.load(open(tci)).get("complexes", {})) if os.path.exists(tci) else []
        keep = [t for t in mk["targets"] if t["grain"] != "complex"]                  # keep geneKO/PC; rebuild complex
        cxt = []
        for c in cx:
            mp = f"{V5}/{mod}/complex/{slugify(c)}/meta.json"
            if os.path.exists(mp):                                                    # traversal generated → full target
                m = json.load(open(mp))
                cxt.append({"grain": "complex", "target": c, "slug": c, "control": None, "has_real": m.get("has_real", True),
                            "real_dir": m.get("real_dir"), "n_cells": m.get("n_cells"), "asset_dir": m["asset_dir"],
                            "alphas": m["alphas"], "dist_map": None, "desc": ""}); full += 1
            else:                                                                     # top-cells only
                cxt.append({"grain": "complex", "target": c, "slug": c, "control": None, "has_real": False,
                            "real_dir": None, "n_cells": 0, "asset_dir": None, "alphas": [], "dist_map": None, "desc": ""})
        mk["targets"] = keep + cxt
        total += len(cxt)
    json.dump(man, open(f"{V5}/manifest.json", "w"))
    print(f"[register] {total} fluor complex targets ({full} with traversals) across markers")


N_CELLS = 40   # NTC accuracy anchor cells (matches the geneKO fleet)


def gen_complex_shard(mc, d, ch, targets, parq, force=False):
    """One marker's complex traversals: NTC-anchored, complex-KD direction, fluor complex SetTransformer scoring."""
    os.environ["OPS_DIFFEX_ASSETS"] = ASSETS
    from . import precompute as P
    P._ASSETS = ASSETS
    P.precompute_marker(grain="complex", targets=targets, ckpt=f"{C.DD}/{d}/diffae_best.pt", out_root=C.OUT,
                        marker_channel=mc, channel=ch, control="NTC", n_cells=N_CELLS,
                        fluor_rank_parquet=parq, v5_score=True, load_workers=12, force=force)


def launch_traversals():
    """Fan out one GPU shard per marker over its complex ranking parquet (built by build_rankings)."""
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    os.environ["OPS_DIFFEX_ASSETS"] = ASSETS
    cm = {mc: (d, ch) for d, mc, ch in C.complete_markers()}
    jobs = []
    for mc, (d, ch) in cm.items():
        p = f"{RANKDIR}/{slugify(mc)}.parquet"
        if not os.path.exists(p):
            continue
        cxs = sorted(pd.read_parquet(p, columns=["gene"]).gene.unique())
        jobs.append({"name": f"fluorcxv5_{slugify(mc)[:14]}", "func": gen_complex_shard,
                     "kwargs": {"mc": mc, "d": d, "ch": ch, "targets": cxs, "parq": p}})
    print(f"[fluor-complex-v5] {len(jobs)} marker traversal shards")
    submit_parallel_jobs(jobs, experiment="diffex_fluor_cx_v5",
                         slurm_params={"slurm_partition": "gpu", "slurm_gres": "gpu:1", "cpus_per_task": 12,
                                       "mem_gb": 96, "timeout_min": 600},
                         log_dir="diffex_fluor_cx_v5", wait_for_completion=False)


def main():
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    build_rankings()
    cm = {mc: (d, ch) for d, mc, ch in C.complete_markers()}
    jobs = []
    for mc, (d, ch) in cm.items():
        if os.path.exists(f"{RANKDIR}/{slugify(mc)}.parquet"):
            jobs.append({"name": f"ftcx_{slugify(mc)[:20]}", "func": crop_marker_shard,
                         "kwargs": {"mc": mc, "ch": ch, "rankdir": RANKDIR, "block": "complexes"}})
    print(f"[fluor-complex-topcells] {len(jobs)} marker crop jobs")
    submit_parallel_jobs(jobs, experiment="diffex_fluor_cx_topcells",
                         slurm_params={"slurm_partition": "cpu", "cpus_per_task": 8, "mem_gb": 32, "timeout_min": 150},
                         log_dir="diffex_fluor_cx_topcells", wait_for_completion=False)


if __name__ == "__main__":
    main()
