"""Build v5 fluorescence geneKO NTC traversals for all 55 markers:
  - anchor = top-40 accuracy NTC cells for that channel (from Alex's v5 gene_marker_1K_qualifying)
  - direction = v5-accuracy KD centroid (same ranking), per-marker DiffAE checkpoint
  - inline v5 FLUOR SetTransformer scoring (P(target)+rank) via the fixed modality-aware v5ctx
Complexes are a separate follow-up (per-gene table needs gene->complex pooling). geneKO only here.
"""
import os, re, json
import pandas as pd
from . import catalog as C
from ops_model.models.interpretability.diffae.classifier.config import slugify

ASSETS = "viewer_assets_v5"
F = "/hpc/projects/icd.fast.ops/models/alex_lin_attention/v5/fluorescence"
# Alex Lin's celldino rankings (src /bio/projects/katamari/alex.lin/paper_celldino_rankings_v2/fluorescence):
#   - accuracies exist for ALL 1000 gene × 55 marker combos (fluor_bychannel_*_pergene.csv, all 1001 genes).
#   - per-CELL rankings (coords below) only generated where top1_acc > 0.5 @ 100 cells  -> this is the ONLY
#     reason a marker has 35-423 geneKO (not 1000): a DISTINCTIVENESS filter, NOT missing cells (the 65M-cell
#     screen has thousands/class·marker). Alex can generate rankings for lower-acc combos on request.
GENE1K = f"{F}/misc/gene_marker_1K_qualifying.compact.parquet"
CP_CSV = f"{F}/misc/gene_marker_1K_CP.csv"          # 7 Cell-Painting markers (TOMM20, Tubulin, ...) — authoritative ranking
RANKDIR = f"{C.OUT}/{ASSETS}/_rankings/fluor/geneKO"
N_CELLS = 40
_COLS = ["channel_name", "gene", "rank", "score", "experiment", "well", "x_pheno", "y_pheno", "segmentation_id"]


def build_rankings():
    """Per-channel geneKO ranking parquets in the _fluor_rows schema (incl. NTC). → {channel: (parquet, [genes])}.
    Cell-Painting markers (CP_CSV) override the qualifying-set ranking for their channels; 4i markers stay in qualifying."""
    os.makedirs(RANKDIR, exist_ok=True)
    q = pd.read_parquet(GENE1K, columns=_COLS)
    cp = pd.read_csv(CP_CSV, usecols=_COLS)
    cp_ch = set(cp["channel_name"].unique())
    df = pd.concat([q[~q["channel_name"].isin(cp_ch)], cp], ignore_index=True)   # CP CSV wins for its 7 channels
    df = df.rename(columns={"segmentation_id": "segmentation", "score": "pma_attention"})
    df["rank_type"] = "top"
    out = {}
    for ch, g in df.groupby("channel_name"):
        p = f"{RANKDIR}/{slugify(ch)}.parquet"
        g.to_parquet(p)
        genes = sorted(x for x in g["gene"].unique() if not str(x).startswith("NTC"))
        out[ch] = (p, genes)
    return out


def gen_marker_shard(mc, d, ch, targets, parq, force=False):
    os.environ["OPS_DIFFEX_ASSETS"] = ASSETS
    from . import precompute as P
    P._ASSETS = ASSETS
    P.precompute_marker(grain="geneKO", targets=targets, ckpt=f"{C.DD}/{d}/diffae_best.pt", out_root=C.OUT,
                        marker_channel=mc, channel=ch, control="NTC", n_cells=N_CELLS,
                        fluor_rank_parquet=parq, v5_score=True, load_workers=12, force=force)


CP_MARKERS = ["Endoplasmic Reticulum_Concanavalin A", "F-actin_Phalloidin", "Microtubules_Tubulin",
              "Mitochondria_TOMM20", "Nucleoli_NPM1", "Nucleus_Hoechst", "Plasma Membrane_Wheat Germ Agglutinin"]


def regen_cp_markers():
    """Force-regenerate the 7 Cell-Painting markers' geneKO traversals with the CP-CSV ranking (their original
    jobs read the qualifying parquets before the CP fix). Submits force=True jobs."""
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    os.environ["OPS_DIFFEX_ASSETS"] = ASSETS
    rk = build_rankings()
    cm = {mc: (d, ch) for d, mc, ch in C.complete_markers()}
    jobs = []
    for mc in CP_MARKERS:
        if mc in rk and mc in cm:
            d, ch = cm[mc]; parq, genes = rk[mc]
            jobs.append({"name": f"fluorv5cp_{slugify(mc)[:16]}", "func": gen_marker_shard,
                         "kwargs": {"mc": mc, "d": d, "ch": ch, "targets": genes, "parq": parq, "force": True}})
    print(f"[fluor-v5-cp] force-regen {len(jobs)} CP markers")
    submit_parallel_jobs(jobs, experiment="diffex_fluor_v5",
                         slurm_params={"slurm_partition": "gpu", "slurm_gres": "gpu:1",
                                       "cpus_per_task": 12, "mem_gb": 96, "timeout_min": 600},
                         log_dir="diffex_fluor_v5", wait_for_completion=False)


def main():
    import re  # noqa
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs


def register_manifest():
    """Register every fluor marker's FULL qualifying geneKO set (from its top-cells index) as targets, so the
    perturbation dropdown / Top Cells is complete regardless of traversal-fleet progress. Genes with a generated
    traversal get its asset_dir/alphas; the rest are top-cells-only (n_cells=0, empty Traversal). Idempotent."""
    V5 = f"{C.OUT}/{ASSETS}"
    man = json.load(open(f"{V5}/manifest.json"))
    acc = pd.read_csv(f"{F}/fluor_bychannel_paperv2gene_cps_pergene.csv", usecols=["channel_name", "gene_name", "top1_acc"])
    accmap = {(r.channel_name, r.gene_name): float(r.top1_acc) for r in acc.itertuples()}
    desc = json.load(open(f"{V5}/gene_desc.json")) if os.path.exists(f"{V5}/gene_desc.json") else {}
    total = 0
    for mk in man["markers"]:
        mc = mk.get("marker_channel")
        if not mc or re.match(r"(?i)phase", mc):
            continue
        mod = slugify(mc)
        tci = f"{V5}/top_cells/markers/{mod}/index.json"
        genes = sorted(g for g in json.load(open(tci))["genes"] if g != "NTC") if os.path.exists(tci) else []   # viewer adds NTC itself
        keep = [t for t in mk["targets"] if t["grain"] != "geneKO"]     # keep PC; rebuild geneKO
        gk = []
        for g in genes:
            mp = f"{V5}/{mod}/geneKO/{g}/meta.json"
            if os.path.exists(mp):                                       # traversal generated → full target
                m = json.load(open(mp))
                gk.append({"grain": "geneKO", "target": g, "slug": g, "control": None, "has_real": m.get("has_real", True),
                           "real_dir": m.get("real_dir"), "n_cells": m.get("n_cells"), "asset_dir": m["asset_dir"],
                           "alphas": m["alphas"], "dist_map": accmap.get((mc, g)), "desc": desc.get(g, "")})
            else:                                                        # top-cells only (traversal not built yet)
                gk.append({"grain": "geneKO", "target": g, "slug": g, "control": None, "has_real": False,
                           "real_dir": None, "n_cells": 0, "asset_dir": None, "alphas": [],
                           "dist_map": accmap.get((mc, g)), "desc": desc.get(g, "")})
        mk["targets"] = keep + gk
        total += len(gk)
    json.dump(man, open(f"{V5}/manifest.json", "w"))
    print(f"[register] {total} fluor geneKO targets (full qualifying set) across {len(man['markers']) - 1} markers")


def build_real_acc20():
    """Extend real_acc20.json with fluor real-cell top1_acc@bag20, keyed by traversal asset_dir so the viewer's
    real-ceiling reference lights up for the 55 markers (like phase). geneKO = per-(channel,gene); complex = mean of
    member-gene acc grouped by Alex's EBI label_name. Idempotent (overwrites the fluor keys)."""
    V5 = f"{C.OUT}/{ASSETS}"
    ra = json.load(open(f"{V5}/real_acc20.json"))
    gk = pd.read_csv(f"{F}/fluor_bychannel_paperv2gene_cps_pergene.csv", usecols=["channel_name", "n_cells", "gene_name", "top1_acc"])
    gk = gk[gk["n_cells"] == 20]
    for r in gk.itertuples():
        ra[f"{slugify(r.channel_name)}/geneKO/{r.gene_name}"] = float(r.top1_acc)
    cx = pd.read_csv(f"{F}/fluor_ebi_bychannel_pergene.csv", usecols=["channel_name", "n_cells", "label_name", "top1_acc"])
    cx = cx[cx["n_cells"] == 20]
    nc = 0
    for (ch, lbl), g in cx.groupby(["channel_name", "label_name"]):
        ra[f"{slugify(ch)}/complex/{lbl}"] = float(g["top1_acc"].mean()); nc += 1
    json.dump(ra, open(f"{V5}/real_acc20.json", "w"))
    print(f"[real_acc20] +{len(gk)} fluor geneKO, +{nc} fluor complex keys (total {len(ra)})")


def main():
    import re  # noqa
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    os.environ["OPS_DIFFEX_ASSETS"] = ASSETS
    rk = build_rankings()
    cm = {mc: (d, ch) for d, mc, ch in C.complete_markers()}          # channel -> (diffae dir, raw channel)
    jobs, skipped = [], []
    for ch_name, (parq, genes) in rk.items():
        if ch_name not in cm:
            skipped.append(ch_name); continue
        d, rawch = cm[ch_name]
        jobs.append({"name": f"fluorv5_{slugify(ch_name)[:18]}", "func": gen_marker_shard,
                     "kwargs": {"mc": ch_name, "d": d, "ch": rawch, "targets": genes, "parq": parq}})
    print(f"[fluor-v5] {len(jobs)} marker jobs ({sum(len(g) for _, (p, g) in rk.items())} gene×channel pairs)")
    print(f"[fluor-v5] skipped (no complete DiffAE / name mismatch): {skipped}")
    submit_parallel_jobs(jobs, experiment="diffex_fluor_v5",
                         slurm_params={"slurm_partition": "gpu", "slurm_gres": "gpu:1",
                                       "cpus_per_task": 12, "mem_gb": 96, "timeout_min": 600},
                         log_dir="diffex_fluor_v5", wait_for_completion=False)


if __name__ == "__main__":
    main()
