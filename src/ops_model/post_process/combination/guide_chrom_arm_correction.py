"""Chromosome-arm correction for guide-level PCA-optimized features.

Ported from ``guide_chrom_arm_correction.ipynb`` into a maintainable module.

The correction is motivated by the observation that guides whose target genes
sit on the same chromosome arm sometimes form artificial clusters in the
guide-level PCA embedding — most likely because of large-scale CNV effects or
co-regulated arm-wide expression programs that leak into the morphology /
fluorescence features. The pipeline below:

1. **Annotates** each guide with its target gene's chromosome arm via
   :mod:`mygene` (e.g. ``GAK -> chr4p``).
2. **Tests** each guide for kNN-cohesion against its arm using a one-sided
   hypergeometric: if a guide's K nearest non-same-gene neighbours over-share
   its chrom arm, it's flagged ``knn_sig``.
3. **Regresses** out the median of the significant-cohesion guides on each
   arm — subtracted from every guide annotated with that arm. Arms with fewer
   than ``min_genes_for_regression`` significant guides are skipped.

The corrected guide-level h5ad is written next to the input with a
``_chrom_arm_corr`` suffix; the standard 2nd-pass PCA can then be run on it
without clobbering existing outputs (the 2nd-pass writes go to a
``second_pca_*_chrom_arm_corr/`` subdir, mirroring the suffix convention).

Top-level entry point: :func:`run_chrom_arm_correction`.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Default kNN test parameters. The notebook uses K=15, q<0.01; we run a bit
# wider (K=25, q<0.05) to cover more guides per arm — that catches the same
# arms but with deeper sig populations, producing a stronger correction.
DEFAULT_K_NN = 25
DEFAULT_PV_THRESHOLD = 0.05
DEFAULT_MIN_GENES_FOR_REGRESSION = 10
CORRECTED_SUFFIX = "_chrom_arm_corr"

# Three correction strategies; see ``run_chrom_arm_correction``:
#   "cohesion"        (default, notebook) — kNN test flags sig-cohesive
#                      guides, only those guides have the per-arm median
#                      subtracted. Surgical: ~5% of guides modified.
#   "centering"       — every gene on an arm with ≥min_genes annotated
#                      members has the arm-mean offset removed and the
#                      global mean added back. Global: ~all annotated guides.
#   "scanpy_regress"  — calls ``sc.pp.regress_out(keys='chrom_arm')`` to
#                      fit a linear regression of each feature on the
#                      categorical chrom_arm and return the residuals.
#                      Mathematically: each arm's group mean is subtracted
#                      (no global-mean add-back), so all arms are centered
#                      at zero.
CORRECTION_METHODS = ("cohesion", "centering", "scanpy_regress")
METHOD_SUFFIX = {
    "cohesion":       "",          # no extra infix → preserves existing _chrom_arm_corr suffix
    "centering":      "_centering",
    "scanpy_regress": "_scanpy_regress",
}
# Kept for backwards compatibility with callers that imported the old name.
CENTERING_SUFFIX_INFIX = METHOD_SUFFIX["centering"]

# Shared symbol→arm cache. Lives under the icd.fast.ops shared config tree
# so every experiment + every distance variant can reuse the same lookup.
# Authoritative: this is the *only* default lookup path — no per-input
# sibling fallback. Use ``--chrom-arm-map-csv`` to override per-run.
#
# Schema (3 columns, TSV): ``gene\tupdated_gene\tchrom_arm`` (e.g.
# ``AARS\tAARS1\tchr16q``). Both legacy and updated symbols map to the
# same arm so deprecated HUGO names still resolve.
SHARED_MAP_CSV_PATH = Path(
    "/hpc/projects/icd.fast.ops/configs/library/gene_chrom_arm_mapping.tsv"
)


# ---------------------------------------------------------------------------
# Step 1: chromosome-arm annotation via mygene
# ---------------------------------------------------------------------------


def _parse_arm(row: pd.Series) -> float:
    """Build a ``chr<N><p|q>`` label from mygene's chromosome + cytoband.

    ``map_location`` strings look like ``'17q21.31'``, ``'Xp22.2'``, or
    ``'1p36.33-p36.32'``. We take the first ``p``/``q`` after the chromosome
    label and combine with the canonical chromosome from ``genomic_pos.chr``
    (falling back to whatever prefixed the cytoband if that column is empty).
    """
    chrom = row.get("genomic_pos.chr")
    if isinstance(chrom, list):
        chrom = chrom[0] if chrom else None
    band = row.get("map_location")
    if not isinstance(band, str):
        return np.nan
    for i, ch in enumerate(band):
        if ch in ("p", "q") and i > 0:
            chrom_part = band[:i]
            chrom_label = chrom if isinstance(chrom, str) and chrom else chrom_part
            return f"chr{chrom_label}{ch}"
    return np.nan


def _query_mygene_symbol_to_arm(symbols: list) -> Dict[str, str]:
    """Look up symbol → chrom-arm via mygene.info. Requires network on the
    caller's node and the ``mygene`` package. Falls through to a friendly
    error if either is unavailable so the script doesn't deep-stack-trace
    inside a SLURM worker.
    """
    try:
        import mygene  # local import — heavy and only needed when this runs
    except ImportError as exc:
        raise RuntimeError(
            "mygene is not installed in this environment. Either:\n"
            "  (a) install it (`uv pip install mygene` or `pip install mygene`),\n"
            "  (b) pre-compute the symbol→arm mapping CSV on a node with "
            "network access and pass it via ``map_csv_path`` / "
            "``--chrom-arm-map-csv``."
        ) from exc

    logger.info(f"Querying mygene for {len(symbols)} unique gene symbols...")
    mg = mygene.MyGeneInfo()
    res = mg.querymany(
        symbols,
        scopes="symbol",
        fields="symbol,genomic_pos.chr,map_location",
        species="human",
        returnall=False,
        as_dataframe=True,
    )
    res = res[~res.index.duplicated(keep="first")].copy()
    res["chrom_arm"] = res.apply(_parse_arm, axis=1)
    mapped = int(res["chrom_arm"].notna().sum())
    logger.info(f"  Mapped {mapped}/{len(res)} symbols to a chromosome arm")
    return res["chrom_arm"].to_dict()


def _load_symbol_to_arm_from_csv(path: Path) -> Dict[str, str]:
    """Read a cached symbol → arm mapping from disk.

    Two schemas accepted, auto-detected by columns:

    1. ``gene, updated_gene, chrom_arm`` (TSV — current shared format):
       both legacy and updated symbols map to the same arm so deprecated
       HUGO names still resolve.
    2. ``symbol, chrom_arm`` (CSV — legacy format produced by the helper
       on first mygene query).

    File format (CSV vs TSV) is inferred from extension. NaN / blank arm
    values stay NaN so ``map()`` returns NaN for unmapped entries.
    """
    path = Path(path)
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    df = pd.read_csv(path, sep=sep)
    cols = set(df.columns)
    if {"gene", "updated_gene", "chrom_arm"}.issubset(cols):
        # New schema: register both legacy and updated symbol keys so the
        # lookup is robust to whichever name a downstream caller uses.
        df = df[["gene", "updated_gene", "chrom_arm"]].copy()
        df["chrom_arm"] = df["chrom_arm"].astype(str).str.strip()
        df.loc[df["chrom_arm"].isin({"", "nan", "None", "NaN"}), "chrom_arm"] = np.nan
        mapping: Dict[str, str] = {}
        for _, row in df.iterrows():
            arm = row["chrom_arm"]
            for key in (row["gene"], row["updated_gene"]):
                if isinstance(key, str) and key.strip():
                    mapping[key.strip()] = arm
        return mapping
    if {"symbol", "chrom_arm"}.issubset(cols):
        df = df[["symbol", "chrom_arm"]].copy()
        df["chrom_arm"] = df["chrom_arm"].replace({"": np.nan})
        return df.set_index("symbol")["chrom_arm"].to_dict()
    raise ValueError(
        f"{path} must have either columns (gene, updated_gene, chrom_arm) "
        f"or (symbol, chrom_arm); got: {sorted(cols)}"
    )


def _save_symbol_to_arm_to_csv(
    symbol_to_arm: Dict[str, str], path: Path,
) -> None:
    """Persist a symbol → arm mapping CSV so subsequent runs can skip the
    mygene network call.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(symbol_to_arm.items(), key=lambda kv: str(kv[0]))
    df = pd.DataFrame(rows, columns=["symbol", "chrom_arm"])
    df.to_csv(path, index=False)
    logger.info(f"  Cached symbol→arm mapping to {path}")


def annotate_chrom_arms(
    adata: ad.AnnData,
    gene_column: str = "geneKO_name",
    map_csv_path: Optional[Path] = None,
) -> ad.AnnData:
    """Add ``adata.obs['chrom_arm']`` keyed by ``gene_column``.

    Lookup source priority:

    1. ``map_csv_path``, if it exists — load the cached mapping (no network
       needed). This is the recommended path for SLURM workers without
       outbound internet.
    2. Otherwise, query ``mygene.info`` for every unique symbol in
       ``adata.obs[gene_column]``. If a non-existent ``map_csv_path`` was
       provided, the freshly-queried mapping is *written* there so future
       runs hit the cache instead.

    Unknown / unmapped gene symbols (e.g. ``NTC``) get ``NaN`` and are
    excluded from the downstream kNN test and regression. Mutates ``adata``
    in place and returns it.
    """
    if gene_column not in adata.obs.columns:
        raise ValueError(
            f"adata.obs missing required column {gene_column!r}; "
            f"set ``gene_column`` to point at your gene-symbol column"
        )

    map_csv_path = Path(map_csv_path) if map_csv_path is not None else None
    if map_csv_path is not None and map_csv_path.is_file():
        logger.info(f"Loading cached symbol→arm map: {map_csv_path}")
        symbol_to_arm = _load_symbol_to_arm_from_csv(map_csv_path)
    else:
        symbols = adata.obs[gene_column].astype(str).unique().tolist()
        symbol_to_arm = _query_mygene_symbol_to_arm(symbols)
        if map_csv_path is not None:
            _save_symbol_to_arm_to_csv(symbol_to_arm, map_csv_path)

    adata.obs["chrom_arm"] = (
        adata.obs[gene_column].astype(str).map(symbol_to_arm).astype("category")
    )
    n_missing = int(adata.obs["chrom_arm"].isna().sum())
    logger.info(
        f"  {adata.n_obs - n_missing}/{adata.n_obs} guides annotated; "
        f"{n_missing} unannotated (e.g. NTCs)"
    )
    return adata


# ---------------------------------------------------------------------------
# Step 2: per-guide kNN arm-cohesion test (hypergeometric, BH-corrected)
# ---------------------------------------------------------------------------


def compute_knn_arm_cohesion(
    adata: ad.AnnData,
    k_nn: int = DEFAULT_K_NN,
    pv_threshold: float = DEFAULT_PV_THRESHOLD,
    gene_column: str = "geneKO_name",
    metric: str = "euclidean",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """For each annotated guide, count how many of its k nearest annotated
    neighbours share its chrom arm and one-sided hypergeometric-test against
    a null where arms are randomly assigned (excluding self + same-gene
    guides, since those trivially share an arm).

    Writes ``knn_p``, ``knn_neg_log10_p``, ``knn_q`` (BH-FDR), ``knn_sig`` and
    ``sig_chrom_arm`` (= chrom_arm where knn_sig else NaN) columns onto
    ``adata.obs``. Returns ``(per_gene_df, arm_summary_df)`` for downstream
    inspection.
    """
    from scipy.stats import hypergeom
    from sklearn.neighbors import NearestNeighbors
    from statsmodels.stats.multitest import multipletests

    if "chrom_arm" not in adata.obs.columns:
        raise ValueError("Run annotate_chrom_arms() before compute_knn_arm_cohesion()")

    X_full = adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray()
    arm_obs = adata.obs["chrom_arm"]
    annotated = arm_obs.notna().values
    X_ann = X_full[annotated]
    arm_ann = arm_obs[annotated].astype(str).values
    ann_obs_names = adata.obs_names[annotated].values
    ann_gene_names = adata.obs.loc[annotated, gene_column].values
    n_ann = len(arm_ann)
    if n_ann == 0:
        raise ValueError("No annotated guides — annotate_chrom_arms returned all NaN")

    logger.info(
        f"Fitting kNN (k={k_nn}, metric={metric}) on {n_ann} annotated guides..."
    )
    nn = NearestNeighbors(n_neighbors=k_nn + 1, metric=metric).fit(X_ann)
    _, knn_idx = nn.kneighbors(X_ann)
    knn_idx = knn_idx[:, 1:]  # drop self

    arm_to_count = pd.Series(arm_ann).value_counts().to_dict()
    gene_to_guidecount = pd.Series(ann_gene_names).value_counts().to_dict()

    rows = []
    for i in range(n_ann):
        a = arm_ann[i]
        gene_name = ann_gene_names[i]
        n_a = arm_to_count[a]
        # Drop neighbours that target the same gene (they trivially share arm).
        knn_gene_names = ann_gene_names[knn_idx[i]]
        kept_idx = knn_idx[i][knn_gene_names != gene_name]
        knn = len(kept_idx)
        shared = int((arm_ann[kept_idx] == a).sum())
        # Same-gene guides leave the population too (and they were all on arm a).
        g = gene_to_guidecount[gene_name] - 1
        M = n_ann - 1 - g
        K = n_a - 1 - g
        if M > 0 and knn > 0:
            p = float(hypergeom.sf(shared - 1, M, K, knn))
            expected = knn * K / M
        else:
            p = 1.0
            expected = 0.0
        rows.append({
            "obs_name": ann_obs_names[i],
            "geneKO_name": gene_name,
            "arm": a,
            "n_arm": n_a,
            "n_same_gene_excluded": g,
            "knn_after_filter": knn,
            "shared_neighbors": shared,
            "expected_shared": expected,
            "p_knn": p,
        })
    per_gene = pd.DataFrame(rows)
    per_gene["q_knn"] = multipletests(per_gene["p_knn"].values, method="fdr_bh")[1]
    per_gene["sig_knn"] = per_gene["q_knn"] < pv_threshold
    per_gene["neg_log10_p"] = -np.log10(per_gene["p_knn"].clip(lower=1e-300))

    n_sig = int(per_gene["sig_knn"].sum())
    logger.info(
        f"  Per-guide significant (q < {pv_threshold}): {n_sig} / {len(per_gene)}"
    )

    arm_summary = (
        per_gene.groupby("arm")
        .agg(
            n_genes_arm=("obs_name", "size"),
            n_sig_knn=("sig_knn", "sum"),
            median_shared=("shared_neighbors", "median"),
            median_neg_log10_p=("neg_log10_p", "median"),
        )
        .reset_index()
    )
    arm_summary["frac_sig_knn"] = (
        arm_summary["n_sig_knn"] / arm_summary["n_genes_arm"]
    )
    arm_summary = arm_summary.sort_values("frac_sig_knn", ascending=False)

    # Stamp results onto adata.obs (NaN for unannotated rows).
    lookup = per_gene.set_index("obs_name")
    adata.obs["knn_p"] = lookup["p_knn"].reindex(adata.obs_names).astype(float).values
    adata.obs["knn_neg_log10_p"] = (
        lookup["neg_log10_p"].reindex(adata.obs_names).astype(float).values
    )
    adata.obs["knn_q"] = lookup["q_knn"].reindex(adata.obs_names).astype(float).values
    adata.obs["knn_sig"] = (
        lookup["sig_knn"].reindex(adata.obs_names).fillna(False).astype(bool).values
    )
    # Mirror the notebook: direct copy of the category column, then .loc-set
    # NaN on non-significant rows. Don't round-trip through object dtype —
    # the result is a category column with NaN wherever knn_sig is False,
    # exactly as the notebook produces it.
    adata.obs["sig_chrom_arm"] = adata.obs["chrom_arm"]
    adata.obs.loc[~adata.obs["knn_sig"], "sig_chrom_arm"] = np.nan

    return per_gene, arm_summary


# ---------------------------------------------------------------------------
# Step 3: regression — subtract median of sig-cohesion guides per arm
# ---------------------------------------------------------------------------


def apply_arm_regression(
    adata: ad.AnnData,
    min_genes_for_regression: int = DEFAULT_MIN_GENES_FOR_REGRESSION,
) -> ad.AnnData:
    """Subtract the median feature vector of significant-cohesion guides on
    each arm *from those same significant guides*. Returns a new copy of
    ``adata`` with corrected ``.X``; caches per-arm null vectors at
    ``adata.uns['chrom_arm_correction_nulls']`` for reproducibility.

    Matches the notebook (``guide_chrom_arm_correction.ipynb``) exactly: the
    null is computed from ``sig_chrom_arm == arm`` and the subtraction also
    targets those rows — non-significant guides on the same arm are left
    alone. Arms with fewer than ``min_genes_for_regression`` significant
    guides are skipped (regression null would be too unstable to trust).
    """
    if "sig_chrom_arm" not in adata.obs.columns:
        raise ValueError(
            "Run compute_knn_arm_cohesion() before apply_arm_regression()"
        )

    sig_arms = adata.obs["sig_chrom_arm"].dropna().unique()
    logger.info(
        f"Correcting the following chromosome arms: "
        f"{', '.join(map(str, sig_arms))}"
    )

    adata_reg = adata.copy()
    X = (
        adata_reg.X if isinstance(adata_reg.X, np.ndarray)
        else adata_reg.X.toarray().astype(np.float32)
    )
    if not isinstance(X, np.ndarray):
        X = np.asarray(X)
    adata_reg.X = X  # ensure dense + writable

    arm_nulls: Dict[str, np.ndarray] = {}
    arms_corrected, arms_skipped = [], []
    for arm in sig_arms:
        mask = (adata_reg.obs["sig_chrom_arm"] == arm).to_numpy()
        n_sig = int(mask.sum())
        if n_sig < min_genes_for_regression:
            logger.info(
                f"  Skipping {arm} (only {n_sig} genes)"
            )
            arms_skipped.append((str(arm), n_sig))
            continue
        logger.info(
            f"  Calculating mean of clustered genes (n={n_sig}) on {arm} for regression"
        )
        null_vec = np.median(adata_reg.X[mask], axis=0)
        logger.info(
            f"  Subtracting mean from genes (n={n_sig}) on {arm}"
        )
        # NOTE: notebook subtracts ONLY from the significant-cohesion guides
        # on this arm (``mask``), not from every guide on the arm. The
        # notebook computes ``arm_mask = obs.chrom_arm == arm`` but never
        # uses it; we mirror that exactly.
        adata_reg.X[mask] = adata_reg.X[mask] - null_vec
        arm_nulls[str(arm)] = null_vec.astype(np.float32)
        arms_corrected.append((str(arm), n_sig))

    adata_reg.uns["chrom_arm_correction"] = {
        "min_genes_for_regression": int(min_genes_for_regression),
        "arms_corrected": arms_corrected,
        "arms_skipped": arms_skipped,
        "arm_nulls_shape": {a: list(v.shape) for a, v in arm_nulls.items()},
    }
    # Stash nulls under a separate key so AnnData write_h5ad can serialize them.
    adata_reg.uns["chrom_arm_correction_nulls"] = arm_nulls
    return adata_reg


# ---------------------------------------------------------------------------
# Step 3b (alternative): "centering" — arm-mean → global-mean shift
# ---------------------------------------------------------------------------


def apply_arm_centering(
    adata: ad.AnnData,
    min_genes_for_regression: int = DEFAULT_MIN_GENES_FOR_REGRESSION,
) -> ad.AnnData:
    """Alternative correction: center each arm at the global feature mean.

    For each arm with ≥ ``min_genes_for_regression`` annotated members::

        X[arm_mask] := X[arm_mask] - mean(X[arm_mask]) + mean(X[all])

    This shifts every gene on the arm by ``(global_mean - arm_mean)``,
    removing the arm-level offset while preserving within-arm relative
    positions. Unlike :func:`apply_arm_regression`, this:

      * Does NOT use the kNN cohesion test — every arm with enough genes
        gets corrected, regardless of whether its members were kNN-flagged.
      * Touches every gene on every qualifying arm (≈ all annotated guides),
        not just the ~5% flagged as sig-cohesive.
      * Uses ``mean`` not ``median`` for the per-arm null.
      * Adds the global mean back so the data stays centered near its
        original location instead of pulling each arm to zero.

    Use this when you believe the arm-cohesion artefact is uniform across
    members (e.g. CNV-driven dosage), not concentrated in a sig-cohesive
    sub-cluster.
    """
    if "chrom_arm" not in adata.obs.columns:
        raise ValueError(
            "Run annotate_chrom_arms() before apply_arm_centering()"
        )

    adata_reg = adata.copy()
    X = (
        adata_reg.X if isinstance(adata_reg.X, np.ndarray)
        else adata_reg.X.toarray().astype(np.float32)
    )
    if not isinstance(X, np.ndarray):
        X = np.asarray(X)
    adata_reg.X = X

    overall_mean = np.mean(adata_reg.X, axis=0)
    arms = adata_reg.obs["chrom_arm"].dropna().unique()
    arms_corrected, arms_skipped = [], []
    arm_nulls: Dict[str, np.ndarray] = {}
    for arm in arms:
        mask = (adata_reg.obs["chrom_arm"] == arm).to_numpy()
        n = int(mask.sum())
        if n < min_genes_for_regression:
            logger.info(f"  Skipping {arm} (only {n} genes)")
            arms_skipped.append((str(arm), n))
            continue
        arm_mean = np.mean(adata_reg.X[mask], axis=0)
        logger.info(
            f"  Centering {arm} (n={n}) — subtracting arm_mean, adding overall_mean"
        )
        adata_reg.X[mask] = adata_reg.X[mask] - arm_mean + overall_mean
        # Persist the per-arm offset that was applied so the operation is
        # reproducible from the saved h5ad alone.
        arm_nulls[str(arm)] = (overall_mean - arm_mean).astype(np.float32)
        arms_corrected.append((str(arm), n))

    adata_reg.uns["chrom_arm_correction"] = {
        "method": "centering",
        "min_genes_for_regression": int(min_genes_for_regression),
        "arms_corrected": arms_corrected,
        "arms_skipped": arms_skipped,
        "arm_nulls_shape": {a: list(v.shape) for a, v in arm_nulls.items()},
    }
    adata_reg.uns["chrom_arm_correction_nulls"] = arm_nulls
    return adata_reg


# ---------------------------------------------------------------------------
# Step 3c (alternative): scanpy.pp.regress_out on chrom_arm
# ---------------------------------------------------------------------------


def apply_arm_scanpy_regress(
    adata: ad.AnnData,
    min_genes_for_regression: int = DEFAULT_MIN_GENES_FOR_REGRESSION,
) -> ad.AnnData:
    """Use ``sc.pp.regress_out(keys='chrom_arm')`` to regress chrom_arm out
    of the feature matrix.

    For a single categorical covariate, scanpy's ``regress_out`` fits a
    linear regression per feature where the predictor is the chrom_arm
    dummy variables, and replaces ``X`` with the residuals. Mathematically:
    each arm's group mean is subtracted to zero (no global-mean add-back),
    so the corrected data has every arm centered at zero.

    Differences vs :func:`apply_arm_centering`:
      * Same per-arm-mean subtraction.
      * No global-mean re-add (centering's ``+ overall_mean``).
      * Drives the regression through scanpy's well-tested code path
        (handles multi-thread/sparsity edge cases for us).

    Genes with a missing ``chrom_arm`` annotation get a sentinel
    ``"unmapped"`` category so the regression doesn't crash on NaN. Arms
    with fewer than ``min_genes_for_regression`` members keep their data
    untouched (small-group estimates would be too noisy) — they're folded
    into the same ``"unmapped"`` sentinel so they don't contribute to the
    regression.
    """
    import scanpy as sc  # local import — scanpy is heavy

    if "chrom_arm" not in adata.obs.columns:
        raise ValueError(
            "Run annotate_chrom_arms() before apply_arm_scanpy_regress()"
        )

    adata_in = adata.copy()
    # Fold under-quorum arms + NaNs into one sentinel category so regress_out
    # doesn't fit noisy per-group means for tiny populations.
    arm_counts = adata_in.obs["chrom_arm"].value_counts(dropna=False)
    too_small = set(arm_counts[arm_counts < min_genes_for_regression].index)
    too_small.discard(np.nan)  # NaN is handled by fillna below
    sentinel = "unmapped"
    chrom = (
        adata_in.obs["chrom_arm"].astype(object)
        .where(~adata_in.obs["chrom_arm"].isin(too_small), other=sentinel)
        .fillna(sentinel)
        .astype("category")
    )
    adata_in.obs["chrom_arm"] = chrom

    n_corrected = int((adata_in.obs["chrom_arm"] != sentinel).sum())
    arms_corrected = [
        (str(a), int(arm_counts.get(a, 0)))
        for a in adata_in.obs["chrom_arm"].cat.categories
        if a != sentinel
    ]
    arms_skipped = [(str(a), int(arm_counts.get(a, 0))) for a in too_small]
    logger.info(
        f"[scanpy_regress] {len(arms_corrected)} arms used as covariates, "
        f"{len(arms_skipped)} small/unmapped folded into '{sentinel}'; "
        f"{n_corrected}/{adata_in.n_obs} guides on covariate arms"
    )

    logger.info("[scanpy_regress] sc.pp.regress_out(keys='chrom_arm', copy=True)...")
    adata_reg = sc.pp.regress_out(adata_in, keys="chrom_arm", copy=True)
    logger.info(
        f"[scanpy_regress] residuals computed; X stats: "
        f"mean={float(np.mean(adata_reg.X)):.4f}, "
        f"std={float(np.std(adata_reg.X)):.4f}"
    )

    adata_reg.uns["chrom_arm_correction"] = {
        "method": "scanpy_regress",
        "min_genes_for_regression": int(min_genes_for_regression),
        "arms_corrected": arms_corrected,
        "arms_skipped": arms_skipped,
        "sentinel_category": sentinel,
        "n_obs_corrected": n_corrected,
        "n_obs_total": int(adata_in.n_obs),
    }
    return adata_reg


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


def corrected_h5ad_path(input_path: Path, method: str = "cohesion") -> Path:
    """Default output filename for the corrected h5ad.

    Method-aware suffix from :data:`METHOD_SUFFIX`:
      * ``cohesion``       → ``…_chrom_arm_corr.h5ad`` (default)
      * ``centering``      → ``…_chrom_arm_corr_centering.h5ad``
      * ``scanpy_regress`` → ``…_chrom_arm_corr_scanpy_regress.h5ad``

    Lives in the same directory as the input. Never overwrites the input.
    """
    input_path = Path(input_path)
    method_infix = METHOD_SUFFIX.get(method, "")
    return input_path.with_name(
        input_path.stem + CORRECTED_SUFFIX + method_infix + input_path.suffix
    )


def default_map_csv_path(input_path: Path) -> Path:
    """Authoritative location for the symbol→arm mapping.

    Returns :data:`SHARED_MAP_CSV_PATH` unconditionally — no per-input
    sibling fallback. Callers who need a different file must pass
    ``map_csv_path`` explicitly.
    """
    return SHARED_MAP_CSV_PATH


def run_chrom_arm_correction(
    input_path: str | Path,
    output_path: Optional[str | Path] = None,
    *,
    method: str = "cohesion",
    k_nn: int = DEFAULT_K_NN,
    pv_threshold: float = DEFAULT_PV_THRESHOLD,
    min_genes_for_regression: int = DEFAULT_MIN_GENES_FOR_REGRESSION,
    gene_column: str = "geneKO_name",
    save_diagnostics: bool = True,
    map_csv_path: Optional[str | Path] = None,
) -> Path:
    """Annotate → (kNN cohesion test if cohesion) → correct → write.

    ``method``:
      * ``"cohesion"`` (default, notebook): kNN cohesion test flags sig
        guides, per-arm median is subtracted only from sig members.
        ``k_nn`` and ``pv_threshold`` control the kNN test.
      * ``"centering"``: every arm with ≥ ``min_genes_for_regression``
        annotated members gets its arm-mean offset replaced with the
        overall mean. ``k_nn`` / ``pv_threshold`` are unused.

    Output filename auto-suffixes with ``_centering`` for the new method so
    runs with different methods land in different files. Diagnostics for
    the cohesion method (per-guide kNN table, per-arm summary) land beside
    the output h5ad with matching basenames; the centering method has no
    per-guide diagnostics to save.

    ``map_csv_path`` (default: shared cache at
    ``/hpc/projects/icd.fast.ops/configs/library/chrom_arm_mapping.csv``) is
    the symbol→arm lookup. If it exists, mygene is skipped entirely.
    """
    if method not in CORRECTION_METHODS:
        raise ValueError(
            f"method must be one of {CORRECTION_METHODS}, got {method!r}"
        )
    input_path = Path(input_path)
    if output_path is None:
        output_path = corrected_h5ad_path(input_path, method=method)
    else:
        output_path = Path(output_path)
    if output_path.resolve() == input_path.resolve():
        raise ValueError(
            "output_path must differ from input_path to avoid clobbering the "
            f"upstream artefact: {input_path}"
        )

    if map_csv_path is None:
        map_csv_path = default_map_csv_path(input_path)
    else:
        map_csv_path = Path(map_csv_path)

    logger.info(
        f"[chrom-arm correction] method={method}; loading {input_path}"
    )
    adata = ad.read_h5ad(input_path)
    annotate_chrom_arms(adata, gene_column=gene_column, map_csv_path=map_csv_path)

    per_gene = arm_summary = None
    if method == "cohesion":
        per_gene, arm_summary = compute_knn_arm_cohesion(
            adata,
            k_nn=k_nn,
            pv_threshold=pv_threshold,
            gene_column=gene_column,
        )
        adata_reg = apply_arm_regression(
            adata, min_genes_for_regression=min_genes_for_regression,
        )
    elif method == "centering":
        adata_reg = apply_arm_centering(
            adata, min_genes_for_regression=min_genes_for_regression,
        )
    elif method == "scanpy_regress":
        adata_reg = apply_arm_scanpy_regress(
            adata, min_genes_for_regression=min_genes_for_regression,
        )
    else:
        raise ValueError(f"Unhandled method {method!r}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata_reg.write_h5ad(output_path)
    logger.info(f"[chrom-arm correction] Wrote {output_path}")

    if save_diagnostics:
        # Per-guide kNN + per-arm summary CSVs only exist for the cohesion
        # method (centering skips the kNN cohesion test).
        if per_gene is not None:
            per_gene_csv = output_path.with_suffix("").with_suffix(".per_gene_knn.csv")
            per_gene.to_csv(per_gene_csv, index=False)
            logger.info(f"[chrom-arm correction] Saved {per_gene_csv.name}")
        if arm_summary is not None:
            arm_csv = output_path.with_suffix("").with_suffix(".arm_summary.csv")
            arm_summary.to_csv(arm_csv, index=False)
            logger.info(f"[chrom-arm correction] Saved {arm_csv.name}")

    return output_path
