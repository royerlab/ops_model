"""Sidecar metadata patch for stale ISS labels in features_processed_*.h5ad.

Background
----------
``move_links.py`` snapshots ``<exp>/3-assembly/<W>_linked_pheno_iss.csv`` into
the frozen ``/hpc/projects/icd.fast.ops/models/link_csvs/<exp>/`` location once
per experiment. When the ISS calling pipeline is later re-run, only the
``3-assembly/`` copy is refreshed; the frozen snapshot stays stale.

Every feature-extraction model (cell-dino, dinov3, subcell, cp_features,
cp_extraction) loads the frozen copy via ``OpsPaths(...).links["training"]``
([paths.py:50]), so cells extracted after the ISS re-run carry **the old**
gene/sgRNA assignments in ``obs["sgRNA"]`` / ``obs["perturbation"]``.

Diagnosis confirmed that ``bbox`` is bit-identical across the two CSVs for all
matched cells, so the extracted features themselves remain correct. Only the
metadata labels are wrong, which is fixable without re-running inference.

What this script does
---------------------
For each affected experiment, for each ``features_processed_<channel>.h5ad``,
writes a sibling parquet named
``features_processed_<channel>_obs_corrected.parquet`` containing the corrected
metadata. The original h5ad is **never modified**.

Two-step join (because the h5ad obs has no segmentation_id):
    1. ``(well, x_position, y_position)`` --> FROZEN CSV --> ``segmentation_id``
    2. ``segmentation_id``                --> CURRENT CSV --> ``sgRNA``, ``gene_name``

Sidecar schema (one row per obs row, indexed by positional ``obs_idx``):
    obs_idx, sgRNA, perturbation, label_str, segmentation_id, correction_status

``correction_status`` is one of:
    ``matched``               cell matched in both frozen + current ISS; labels
                              updated to current values.
    ``orphan_in_h5ad``        cell was in frozen ISS (and h5ad) but the current
                              ISS run dropped it; original (stale) labels kept.
    ``position_unresolved``   could not even resolve the obs row to a frozen
                              CSV row (rare; floating-point drift or missing
                              well CSV); original labels kept.

Downstream consumers load via
``ops_model.features.anndata_utils.load_features_corrected`` which transparently
applies the sidecar.

Usage
-----
    # Build for one experiment locally:
    python -m ops_model.data.iss_drift_fix --exp ops0031_20250424

    # Build for all stale experiments via SLURM:
    python -m ops_model.data.iss_drift_fix --slurm

    # Build for all stale experiments locally (slow):
    python -m ops_model.data.iss_drift_fix
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

BASE = Path("/hpc/projects/icd.fast.ops")
FEATURE_DIR_NAME = "cell_dino_features"
ANNDATA_SUBDIR = "anndata_objects"
SIDECAR_SUFFIX = "_obs_corrected.parquet"

# Position-join tolerance: x_position in h5ad obs is x_pheno from the frozen
# CSV ([cell_dino.py:166]), so we expect exact equality up to CSV-roundtrip
# float precision. Rounding to 3 decimals (0.001 px) catches roundtrip noise
# without colliding cells (smallest cell-to-cell distance ~ several pixels).
POS_ROUND = 3


def _well_prefix_from_obs_well(well: str) -> str | None:
    """obs["well"] looks like ``A/1/0_ops0031_20250424`` -> ``A1``."""
    path = str(well).split("_", 1)[0]  # ``A/1/0``
    if len(path) < 3 or path[1] != "/":
        return None
    return path[0] + path[2]


def _process_experiment(exp: str, base: str = str(BASE),
                         feature_dir: str = FEATURE_DIR_NAME,
                         overwrite: bool = False) -> dict:
    """Build sidecar parquets for every ``features_processed_*.h5ad`` in one
    experiment.

    Self-contained so submitit can pickle it for SLURM dispatch (all imports
    inside the function).
    """
    import anndata as ad
    import numpy as np
    import pandas as pd

    base_p = Path(base)
    h5ad_dir = base_p / exp / "3-assembly" / feature_dir / ANNDATA_SUBDIR
    if not h5ad_dir.exists():
        return {"exp": exp, "status": "no_anndata_objects"}

    h5ads = sorted(p for p in h5ad_dir.glob("features_processed_*.h5ad")
                    if SIDECAR_SUFFIX not in p.name)

    results = []
    t0 = time.time()
    for h5 in h5ads:
        channel = h5.stem.replace("features_processed_", "")
        sidecar = h5.with_name(h5.stem + SIDECAR_SUFFIX)
        if sidecar.exists() and not overwrite:
            results.append({"channel": channel, "status": "skipped_exists",
                             "sidecar": str(sidecar)})
            continue

        try:
            a = ad.read_h5ad(h5, backed="r")
            obs = a.obs[["well", "x_position", "y_position", "sgRNA",
                          "perturbation"]].copy()
            obs["_obs_idx"] = np.arange(len(obs), dtype=np.int64)
            obs["_well_prefix"] = obs["well"].map(_well_prefix_from_obs_well)

            blocks = []
            for wp, obs_w in obs.groupby("_well_prefix", dropna=False):
                if wp is None or not isinstance(wp, str):
                    logger.warning(f"{exp} {channel}: bad well prefix; "
                                    f"{len(obs_w)} cells")
                    blocks.append(_unresolved_block(obs_w))
                    continue
                cur_p = base_p / exp / "3-assembly" / f"{wp}_linked_pheno_iss.csv"
                frz_p = (base_p / "models" / "link_csvs" / exp
                          / f"{wp}_linked_pheno_iss.csv")
                if not (cur_p.exists() and frz_p.exists()):
                    logger.warning(f"{exp} {wp}: missing link CSV "
                                    f"(cur={cur_p.exists()}, frz={frz_p.exists()})")
                    blocks.append(_unresolved_block(obs_w))
                    continue
                blocks.append(_correct_well(obs_w, frz_p, cur_p))

            patch = pd.concat(blocks, ignore_index=True)
            patch = patch.sort_values("obs_idx").reset_index(drop=True)
            if len(patch) != len(obs):
                raise RuntimeError(
                    f"size mismatch: patch={len(patch)} vs h5ad={len(obs)}")

            sidecar.parent.mkdir(parents=True, exist_ok=True)
            tmp = sidecar.with_suffix(sidecar.suffix + ".tmp")
            patch.to_parquet(tmp, index=False)
            tmp.replace(sidecar)

            counts = patch["correction_status"].value_counts().to_dict()
            results.append({
                "channel": channel,
                "n_total":      len(patch),
                "n_matched":    int(counts.get("matched", 0)),
                "n_orphan":     int(counts.get("orphan_in_h5ad", 0)),
                "n_unresolved": int(counts.get("position_unresolved", 0)),
                "sidecar":      str(sidecar),
                "status":       "ok",
            })
        except Exception as e:
            results.append({"channel": channel, "status": "error",
                             "error": str(e)})

    return {"exp": exp, "elapsed_s": round(time.time() - t0, 1),
            "channels": results}


def _unresolved_block(obs_w):
    """Build a sidecar block where every row is position_unresolved (keep
    original labels)."""
    import pandas as pd
    return pd.DataFrame({
        "obs_idx":          obs_w["_obs_idx"].to_numpy(),
        "sgRNA":            obs_w["sgRNA"].astype(str).to_numpy(),
        "perturbation":     obs_w["perturbation"].astype(str).to_numpy(),
        "label_str":        obs_w["perturbation"].astype(str).to_numpy(),
        "segmentation_id":  pd.array([pd.NA] * len(obs_w), dtype="Int64"),
        "correction_status": "position_unresolved",
    })


def _correct_well(obs_w, frz_p: Path, cur_p: Path):
    """Two-step join for one well's obs rows.

    Step 1: position --> frozen CSV --> segmentation_id.
    Step 2: segmentation_id --> current CSV --> corrected sgRNA + gene_name.
    """
    import numpy as np
    import pandas as pd

    cols = ["segmentation_id", "x_pheno", "y_pheno", "gene_name", "sgRNA"]
    frz = pd.read_csv(frz_p, usecols=cols, low_memory=False)
    cur = pd.read_csv(cur_p, usecols=cols, low_memory=False)
    frz = frz.dropna(subset=["segmentation_id", "x_pheno", "y_pheno"])
    cur = cur.dropna(subset=["segmentation_id"])
    frz["segmentation_id"] = frz["segmentation_id"].astype("int64")
    cur["segmentation_id"] = cur["segmentation_id"].astype("int64")

    # Step 1: position --> segmentation_id via frozen CSV.
    frz["_x"] = frz["x_pheno"].astype(float).round(POS_ROUND)
    frz["_y"] = frz["y_pheno"].astype(float).round(POS_ROUND)
    obs_w = obs_w.assign(
        _x=obs_w["x_position"].astype(float).round(POS_ROUND),
        _y=obs_w["y_position"].astype(float).round(POS_ROUND),
    )
    # Dedup the frozen position table (extremely rare collisions at sub-pixel
    # rounding; keep first).
    frz_lookup = (frz[["_x", "_y", "segmentation_id"]]
                    .drop_duplicates(subset=["_x", "_y"], keep="first"))

    s1 = obs_w.merge(frz_lookup, on=["_x", "_y"], how="left")

    # Step 2: segmentation_id --> corrected labels via current CSV. ISS can
    # write multiple rows per segmentation_id (one per barcode read), some
    # with NaN gene_name where the call failed; sort so a real gene_name
    # wins over a NaN one for the same cell.
    cur_sorted = cur.sort_values(
        ["segmentation_id", "gene_name"], na_position="last"
    )
    cur_lookup = cur_sorted.drop_duplicates(
        subset="segmentation_id", keep="first"
    )[["segmentation_id", "gene_name", "sgRNA"]]
    # NaN gene_name --> "NTC" (matches data_loader.get_labels' fillna). NaN
    # sgRNA --> "" (matches what a fresh extraction would record).
    cur_lookup = cur_lookup.assign(
        _cur_gene_name=cur_lookup["gene_name"].fillna("NTC").astype(str),
        _cur_sgRNA=cur_lookup["sgRNA"].fillna("").astype(str),
    )[["segmentation_id", "_cur_gene_name", "_cur_sgRNA"]]

    s2 = s1.merge(cur_lookup, on="segmentation_id", how="left")

    # Classify status. A row counts as "matched" whenever the cell's
    # segmentation_id is present in the current ISS calls, even if the new
    # call is no-barcode-found ("NTC"). True orphans are seg_ids the ISS
    # re-run removed entirely.
    has_seg = s2["segmentation_id"].notna().to_numpy()
    has_cur = s2["_cur_gene_name"].notna().to_numpy()
    status = np.where(
        ~has_seg, "position_unresolved",
        np.where(~has_cur, "orphan_in_h5ad", "matched"),
    )

    # For matched rows, use current labels; otherwise keep stale.
    cur_gene  = s2["_cur_gene_name"].fillna("").astype(str).to_numpy()
    cur_sgRNA = s2["_cur_sgRNA"].fillna("").astype(str).to_numpy()
    orig_pert  = s2["perturbation"].astype(str).to_numpy()
    orig_sgRNA = s2["sgRNA"].astype(str).to_numpy()
    matched_mask = (status == "matched")

    new_pert  = np.where(matched_mask, cur_gene,  orig_pert)
    new_sgRNA = np.where(matched_mask, cur_sgRNA, orig_sgRNA)

    return pd.DataFrame({
        "obs_idx":           s2["_obs_idx"].to_numpy(),
        "sgRNA":             new_sgRNA,
        "perturbation":      new_pert,
        "label_str":         new_pert,
        "segmentation_id":   s2["segmentation_id"].astype("Int64"),
        "correction_status": status,
    })


# ----------------------------------------------------------------------------
# Stale-experiment discovery (content-based, no file-mtime heuristic)
# ----------------------------------------------------------------------------

def discover_stale_experiments(base: Path = BASE,
                                feature_dir: str = FEATURE_DIR_NAME) -> list[str]:
    """Return experiments where the 3-assembly link CSVs differ in content
    from the frozen ``models/link_csvs/<exp>/`` snapshot."""
    import pandas as pd

    import hashlib
    exps = sorted({p.parent.parent.parent.name
                    for p in base.glob(f"ops0*/3-assembly/{feature_dir}/{ANNDATA_SUBDIR}")})

    def _gene_col_hash(p: Path) -> str:
        col = pd.read_csv(p, usecols=["gene_name"], low_memory=False)["gene_name"]
        return hashlib.md5(col.fillna("").astype(str).str.cat().encode()).hexdigest()

    stale = []
    for exp in exps:
        for w in ("A1", "A2", "A3"):
            cur_p = base / exp / "3-assembly" / f"{w}_linked_pheno_iss.csv"
            frz_p = base / "models" / "link_csvs" / exp / f"{w}_linked_pheno_iss.csv"
            if not (cur_p.exists() and frz_p.exists()):
                continue
            n_cur = sum(1 for _ in open(cur_p)) - 1
            n_frz = sum(1 for _ in open(frz_p)) - 1
            if n_cur != n_frz:
                stale.append(exp); break
            # Same row count -> hash the full gene_name column to catch drift
            # that's spread throughout the file (200-row samples miss this).
            if _gene_col_hash(cur_p) != _gene_col_hash(frz_p):
                stale.append(exp); break
    return stale


# ----------------------------------------------------------------------------
# CLI / SLURM orchestration
# ----------------------------------------------------------------------------

def _run_slurm(exps: list[str], overwrite: bool) -> dict:
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs

    jobs = []
    for exp in exps:
        jobs.append({
            "name": f"iss_fix_{exp.split('_')[0]}",
            "func": _process_experiment,
            "kwargs": {"exp": exp, "overwrite": overwrite},
        })
    return submit_parallel_jobs(
        jobs_to_submit=jobs,
        experiment="iss_drift_fix",
        slurm_params={
            "timeout_min":     30,
            "mem":             "16GB",
            "cpus_per_task":   2,
            "slurm_partition": "cpu",
        },
        log_dir="iss_drift_fix",
        manifest_prefix="iss_fix",
        wait_for_completion=True,
    )


def main():
    logging.basicConfig(level=logging.INFO,
                         format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", help="Single experiment to process")
    ap.add_argument("--exps", nargs="+", help="Explicit list of experiments")
    ap.add_argument("--slurm", action="store_true",
                    help="Fan out across experiments via SLURM")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-build sidecars even if they already exist")
    ap.add_argument("--feature-dir", default=FEATURE_DIR_NAME,
                    help=f"Feature directory under 3-assembly/ "
                         f"(default: {FEATURE_DIR_NAME})")
    args = ap.parse_args()

    if args.exp:
        r = _process_experiment(args.exp, feature_dir=args.feature_dir,
                                  overwrite=args.overwrite)
        for c in r.get("channels", []):
            logger.info(f"  {args.exp} {c}")
        return

    if args.exps:
        exps = args.exps
    else:
        logger.info("Discovering stale experiments (content-based)…")
        exps = discover_stale_experiments(feature_dir=args.feature_dir)
        logger.info(f"  found {len(exps)} stale experiments")

    if args.slurm:
        result = _run_slurm(exps, overwrite=args.overwrite)
        failed = result.get("failed", [])
        logger.info(f"SLURM done: {len(failed)} failed")
        if failed:
            logger.warning(f"  failed jobs: {failed}")
    else:
        for exp in exps:
            logger.info(f"--- {exp} ---")
            r = _process_experiment(exp, feature_dir=args.feature_dir,
                                      overwrite=args.overwrite)
            for c in r.get("channels", []):
                logger.info(f"  {c}")


if __name__ == "__main__":
    main()
