"""Run the v3 ``pca_optimization`` pipeline on Alex Lin's v4 cell-dino features
with **attention-weighted cell→guide aggregation** instead of mean.

Approach
--------
v3 aggregates cell→guide via ``mean(X_cells)`` per (sgRNA, experiment). We want
``sum(w * X_cells) / sum(w)`` where w is derived from the per-cell PMA attention
scalars.

Trick to avoid editing v3 core code: at load time, pre-multiply each cell's
feature vector by its attention weight, NORMALIZED PER (sgRNA, experiment)
group so that ``mean(w_j) == 1`` within each group. Then v3's standard ``mean``
aggregation downstream produces the weighted mean.

Trade-off: PCA is fitted on the pre-multiplied X — implicitly a weighted PCA.
Within-sgRNA, the per-group normalization preserves total energy (mean(w)=1) so
the impact on the PCA basis is bounded.

Strategies
----------
Single-head:
    ebi          — w = attn_ebi  (NaN→1)
    geneko       — w = attn_geneko  (NaN→1)
    max          — w = max(ebi, geneko)  (NaN-safe)

Multi-head combinations:
    min          — w = min(ebi, geneko)  (NaN-safe)
    product      — w = ebi * geneko  (NaN→1 per factor)
    concordance_50 — binary: w=1 if both heads in their top-50% (NaN counts as
                     concordant so non-panel cells aren't dropped), else 0.01
    softmax_K100 — w = exp(100  * max(ebi, geneko))
    softmax_K1k  — w = exp(1000 * max(ebi, geneko))
    softmax_K10k — w = exp(10000 * max(ebi, geneko))

Output run-tag: ``attention/<strategy>`` for attention-only strategies, or
``sister/<strategy>`` for sister-coherence strategies (sister*, attn_*_x_sister,
attn_ebi_plus_sister). Results land under
``<root>/cell_dino/zscore_per_exp/paper_v1/{attention|sister}/<strategy>/phase_only/...``.

SLURM compatibility: monkey-patches in the submitter process do not propagate
to submitit workers. We wrap each SLURM worker function with a picklable
closure in ``_weighted_pca_worker.make_patched_phase1_worker`` so the patches are
re-applied on the worker side. cloudpickle's ``register_pickle_by_value``
ensures the closure's bytecode is embedded in the pickle.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_KO_SHAP_DIR = Path(__file__).resolve().parent
if str(_KO_SHAP_DIR) not in sys.path:
    sys.path.insert(0, str(_KO_SHAP_DIR))

import _weighted_pca_worker as _weighted_worker  # noqa: E402
from _weighted_pca_worker import make_patched_phase1_worker  # noqa: E402

import cloudpickle  # noqa: E402
cloudpickle.register_pickle_by_value(_weighted_worker)

V4_PER_EXP = Path(
    "/hpc/projects/icd.fast.ops/models/alex_lin_attention/v4/expansion_v1/"
    "per_experiment_v4"
)
ATTN_SIDECAR = Path(
    "/hpc/projects/icd.fast.ops/models/alex_lin_attention/v4/expansion_v1/"
    "per_experiment_v4_attn.parquet"
)
# Combined sidecar (base + set_accuracy per-cell pct from Alex's
# accuracy_ranking/pergene_phase_cell_rankings.csv). Used only by the
# set_accuracy strategy.
ATTN_SIDECAR_WITH_SET_ACC = Path(
    "/hpc/projects/icd.fast.ops/models/alex_lin_attention/v4/expansion_v1/"
    "per_experiment_v4_attn_with_set_accuracy.parquet"
)
# Extended sidecar with both set_accuracy (per-gene) and set_accuracy_ebi
# (per-complex) columns.
ATTN_SIDECAR_WITH_SET_ACC_EBI = Path(
    "/hpc/projects/icd.fast.ops/models/alex_lin_attention/v4/expansion_v1/"
    "per_experiment_v4_attn_with_set_accuracy_and_ebi.parquet"
)

# ---- v5 (paper_v2, Alex's new SetTransformer) ---------------------------
V5_PER_EXP = Path(
    "/hpc/projects/icd.fast.ops/models/alex_lin_attention/v5/expansion_v1/"
    "per_experiment_v5"
)
V5_SIDECAR_GKO = Path(
    "/hpc/projects/icd.fast.ops/models/alex_lin_attention/v5/expansion_v1/"
    "per_experiment_v5_gko.parquet"
)
V5_SIDECAR_EBIONLY = Path(
    "/hpc/projects/icd.fast.ops/models/alex_lin_attention/v5/expansion_v1/"
    "per_experiment_v5_ebionly.parquet"
)
V5_SIDECAR_EBIFB = Path(
    "/hpc/projects/icd.fast.ops/models/alex_lin_attention/v5/expansion_v1/"
    "per_experiment_v5_ebifb.parquet"
)
# v5-multibag: rankings from the multi-bag SHAP screen (v5/multi_rank/*).
# Same 3 heads (gko/ebionly/ebifb) as v5, just re-scored with multi-bag SHAP
# instead of single-bag set-accuracy. Same checkpoints; only ranking differs.
V5M_SIDECAR_GKO = Path(
    "/hpc/projects/icd.fast.ops/models/alex_lin_attention/v5/expansion_v1/"
    "per_experiment_v5_multibag_gko.parquet"
)
V5M_SIDECAR_EBIONLY = Path(
    "/hpc/projects/icd.fast.ops/models/alex_lin_attention/v5/expansion_v1/"
    "per_experiment_v5_multibag_ebionly.parquet"
)
V5M_SIDECAR_EBIFB = Path(
    "/hpc/projects/icd.fast.ops/models/alex_lin_attention/v5/expansion_v1/"
    "per_experiment_v5_multibag_ebifb.parquet"
)
# Combined fluor sidecar — per (exp, well, seg, channel_name) with both
# gko + ebionly weight columns. Consumed on --signal-set no_phase runs.
V5M_SIDECAR_FLUOR = Path(
    "/hpc/projects/icd.fast.ops/models/alex_lin_attention/v5/expansion_v1/"
    "per_experiment_v5_multibag_fluor.parquet"
)
V5_STRATEGIES = {
    "v5_gko", "v5_ebionly", "v5_ebifb",
    "v5_gko_cutoff_20k", "v5_ebionly_cutoff_20k", "v5_ebifb_cutoff_20k",
    "v5_ebifb_cutoff_15k",
    # multi-bag SHAP-ranked variants (same checkpoints, new ranking method)
    "v5m_gko", "v5m_ebionly", "v5m_ebifb",
    "v5m_gko_cutoff_20k", "v5m_ebionly_cutoff_20k", "v5m_ebifb_cutoff_20k",
    "v5m_gko_cutoff_10k", "v5m_ebionly_cutoff_10k", "v5m_ebifb_cutoff_10k",
    # multi-bag SHAP fluor variants (per-cell, per-channel; use --signal-set no_phase)
    "v5m_gko_fluor_cutoff_500",      "v5m_gko_fluor_cutoff_100",
    "v5m_ebionly_fluor_cutoff_500",  "v5m_ebionly_fluor_cutoff_100",
}
FLUOR_ATTN_SIDECAR = Path(
    "/hpc/projects/icd.fast.ops/models/alex_lin_attention/v4/expansion_v1/"
    "per_experiment_v4_attn_fluor.parquet"
)
V4_PER_EXP_FLUOR = Path(
    "/hpc/projects/icd.fast.ops/models/alex_lin_attention/v4/expansion_v1/"
    "per_experiment_v4_fluor"
)
# v3 cdino classification accuracy tables — used by acc_select strategies.
# Per (gene, n_cells), top1_acc / top5_acc; the bin is the smallest n_cells
# at which top1_acc >= 0.95.
ACC_CSV_GENEKO = Path(
    "/home/gav.sturm/linked_folders/icd.fast.ops/models/alex_lin_attention/v3/"
    "attention_v3/cdino/cdino_eval_phase_50.csv"
)
ACC_CSV_CHAD = Path(
    "/home/gav.sturm/linked_folders/icd.fast.ops/models/alex_lin_attention/v3/"
    "attention_v3/cdino/cdino_eval_phase_chad_50.csv"
)
ACC_THRESHOLD = 0.95


def _build_geneko_gene_to_K() -> dict:
    """Per-gene K cap from v3 cdino classification accuracy.

    K = bin_n_cells // 4, where bin_n_cells is the smallest ``n_cells`` row at
    which top1_acc reaches ``ACC_THRESHOLD``. Genes that never reach the
    threshold get K=-1 ('keep all cells').
    """
    import pandas as pd
    df = pd.read_csv(ACC_CSV_GENEKO)
    out = {}
    for gene, sub in df.groupby("gene_name"):
        sub = sub.sort_values("n_cells")
        meets = sub[sub["top1_acc"] >= ACC_THRESHOLD]
        if meets.empty:
            out[str(gene)] = -1
        else:
            bin_n = int(meets.iloc[0]["n_cells"])
            out[str(gene)] = max(1, bin_n // 4)
    return out


def _build_chad_gene_to_K() -> dict:
    """Per-gene K cap derived from v3 cdino CHAD complex accuracies.

    Each gene → its CHAD complex → complex bin (smallest n_cells where the
    *mean of the complex's member-gene top1_acc* reaches ACC_THRESHOLD).
    Genes not assigned to any CHAD complex are absent from the returned dict
    (the worker treats missing genes as K=-1, i.e. keep all cells).
    """
    import pandas as pd
    df = pd.read_csv(ACC_CSV_CHAD)
    # complex → n_cells → mean(top1_acc) across constituent genes
    agg = (
        df.groupby(["label_name", "n_cells"], observed=True)["top1_acc"]
        .mean()
        .reset_index()
    )
    complex_to_bin = {}
    for cmpx, sub in agg.groupby("label_name"):
        sub = sub.sort_values("n_cells")
        meets = sub[sub["top1_acc"] >= ACC_THRESHOLD]
        if meets.empty:
            complex_to_bin[str(cmpx)] = -1
        else:
            complex_to_bin[str(cmpx)] = int(meets.iloc[0]["n_cells"])

    # Per-gene K = complex's bin // 4
    out = {}
    for gene, sub in df.groupby("gene_name"):
        cmpx = sub["label_name"].iloc[0]
        bin_n = complex_to_bin.get(str(cmpx), -1)
        if bin_n < 0:
            out[str(gene)] = -1
        else:
            out[str(gene)] = max(1, bin_n // 4)
    return out


def _resolve_strategy(name: str) -> dict:
    """Map --attn-strategy name → strategy_spec dict consumed by _weighted_pca_worker."""
    if name == "ebi":
        return {"op": "column", "col": "attn_ebi"}
    if name == "geneko":
        return {"op": "column", "col": "attn_geneko"}
    if name == "max":
        return {"op": "column", "col": "attn_max"}
    if name == "min":
        return {"op": "min", "cols": ["attn_ebi", "attn_geneko"]}
    if name == "product":
        return {"op": "product", "cols": ["attn_ebi", "attn_geneko"]}
    if name == "concordance_50":
        return {"op": "concordance", "cols": ["attn_ebi", "attn_geneko"], "percentile": 50.0}
    if name == "softmax_K100":
        return {"op": "softmax", "col": "attn_max", "K": 100}
    if name == "softmax_K1k":
        return {"op": "softmax", "col": "attn_max", "K": 1000}
    if name == "softmax_K10k":
        return {"op": "softmax", "col": "attn_max", "K": 10000}
    if name == "ebi_then_geneko":
        # Fallback hierarchy: use EBI attention where present (panel-member
        # cells get EBI-trained weighting), else geneKO attention, else 1.0.
        return {"op": "fallback", "cols": ["attn_ebi", "attn_geneko"]}
    if name == "set_accuracy":
        # Per-cell pct from Alex's set-accuracy classifier ranking (each cell
        # scored under its own assigned gene). NaN → 1 (cells not in the CSV
        # get uniform weight, matches the "column" op semantics). Sidecar
        # column added to the combined parquet (ATTN_SIDECAR_WITH_SET_ACC).
        return {"op": "column", "col": "set_accuracy"}
    if name == "set_accuracy_ebi":
        # Per-cell pct from Alex's set-accuracy classifier — but scored under
        # each cell's own EBI COMPLEX (not gene). NaN → 1 (cells outside the
        # EBI panel get uniform weight). Sidecar column added to the extended
        # parquet (ATTN_SIDECAR_WITH_SET_ACC_EBI).
        return {"op": "column", "col": "set_accuracy_ebi"}
    # ---- v5 heads: Alex's new SetTransformer rankings on paper_v2 -----
    if name == "v5_gko":
        return {"op": "column", "col": "v5_gko"}
    if name == "v5_ebionly":
        return {"op": "column", "col": "v5_ebionly"}
    if name == "v5_ebifb":
        return {"op": "column", "col": "v5_ebifb"}
    # Cutoff variants: keep only top-20K cells per gene (below the sliding
    # window's cliff at K=20000). Cells with rank > 20000 → w=0 (excluded).
    if name == "v5_gko_cutoff_20k":
        return {"op": "column", "col": "v5_gko_cutoff_20k"}
    if name == "v5_ebionly_cutoff_20k":
        return {"op": "column", "col": "v5_ebionly_cutoff_20k"}
    if name == "v5_ebifb_cutoff_20k":
        return {"op": "column", "col": "v5_ebifb_cutoff_20k"}
    if name == "v5_ebifb_cutoff_15k":
        return {"op": "column", "col": "v5_ebifb_cutoff_15k"}
    # ---- v5 multi-bag SHAP variants (same 3 heads, new ranking) --------
    if name == "v5m_gko":
        return {"op": "column", "col": "v5m_gko"}
    if name == "v5m_ebionly":
        return {"op": "column", "col": "v5m_ebionly"}
    if name == "v5m_ebifb":
        return {"op": "column", "col": "v5m_ebifb"}
    if name == "v5m_gko_cutoff_20k":
        return {"op": "column", "col": "v5m_gko_cutoff_20k"}
    if name == "v5m_ebionly_cutoff_20k":
        return {"op": "column", "col": "v5m_ebionly_cutoff_20k"}
    if name == "v5m_ebifb_cutoff_20k":
        return {"op": "column", "col": "v5m_ebifb_cutoff_20k"}
    if name == "v5m_gko_cutoff_10k":
        return {"op": "column", "col": "v5m_gko_cutoff_10k"}
    if name == "v5m_ebionly_cutoff_10k":
        return {"op": "column", "col": "v5m_ebionly_cutoff_10k"}
    if name == "v5m_ebifb_cutoff_10k":
        return {"op": "column", "col": "v5m_ebifb_cutoff_10k"}
    # ---- v5-multibag FLUOR variants -----------------------------------
    # Fluor SHAP coverage per (exp, channel) is thin; set nan_default=0 so
    # unranked cells contribute nothing to the weighted mean, cap w_norm at 5
    # so a few ranked cells can't dominate. Guides with 0 ranked cells at an
    # (exp, channel) fall back to uniform w=1 → gene keeps a baseline profile
    # on every marker (fair-coverage; no marker silently dropped for that gene).
    # cutoff_500 ≈ phase cutoff_20k proportion (top ~31% of median ~1600 cells).
    if name == "v5m_gko_fluor_cutoff_500":
        return {"op": "column", "col": "v5m_gko_fluor_cutoff_500",
                "nan_default": 0.0, "w_norm_max": 5.0}
    if name == "v5m_gko_fluor_cutoff_100":
        return {"op": "column", "col": "v5m_gko_fluor_cutoff_100",
                "nan_default": 0.0, "w_norm_max": 5.0}
    if name == "v5m_ebionly_fluor_cutoff_500":
        return {"op": "column", "col": "v5m_ebionly_fluor_cutoff_500",
                "nan_default": 0.0, "w_norm_max": 5.0}
    if name == "v5m_ebionly_fluor_cutoff_100":
        return {"op": "column", "col": "v5m_ebionly_fluor_cutoff_100",
                "nan_default": 0.0, "w_norm_max": 5.0}
    if name == "acc_select_geneko_raw":
        gene_to_K = _build_geneko_gene_to_K()
        return {"op": "acc_select", "col": "attn_geneko", "mode": "raw",
                "gene_to_K": gene_to_K}
    if name == "acc_select_geneko_weighted":
        gene_to_K = _build_geneko_gene_to_K()
        return {"op": "acc_select", "col": "attn_geneko", "mode": "weighted",
                "gene_to_K": gene_to_K}
    if name == "acc_select_chad_raw":
        gene_to_K = _build_chad_gene_to_K()
        return {"op": "acc_select", "col": "attn_chad", "mode": "raw",
                "gene_to_K": gene_to_K}
    if name == "acc_select_chad_weighted":
        gene_to_K = _build_chad_gene_to_K()
        return {"op": "acc_select", "col": "attn_chad", "mode": "weighted",
                "gene_to_K": gene_to_K}
    # Fluorescent-marker attention strategies — these reuse the same op shapes
    # as the phase strategies but read attention from the per-(cell, channel)
    # FLUOR sidecar (per_experiment_v4_attn_fluor.parquet). Run with
    # ``--signal-set no_phase`` (fluor only) or ``all_with_autofluorescence``
    # (phase + fluor) to actually load fluor channels.
    if name == "fluor_ebi":
        return {"op": "column", "col": "attn_ebi"}
    if name == "fluor_geneko":
        return {"op": "column", "col": "attn_geneko"}
    if name == "fluor_max":
        return {"op": "column", "col": "attn_max"}
    if name == "fluor_ebi_then_geneko":
        return {"op": "fallback", "cols": ["attn_ebi", "attn_geneko"]}
    # Sister-coherence strategies (sidecar columns added by build_v4_attn_sidecar
    # --add-sister-coherence). sister_ratio is the KDTree-derived per-cell
    # spatial co-localization metric; orthogonal to attention (Spearman ~ -0.1
    # to -0.2 across heads — see sister_vs_attention_corr.csv).
    if name == "sister":
        return {"op": "column", "col": "sister_ratio"}
    if name == "sister_pow2":
        return {"op": "power", "col": "sister_ratio", "p": 2.0}
    if name == "sister_pow4":
        return {"op": "power", "col": "sister_ratio", "p": 4.0}
    if name == "sister_floored_01":
        # w = max(sister_ratio, 0.1); cells with sister_ratio=0 still get 0.1
        # (i.e. ~10% of a typical cell's weight). Tests "does dropping zeros hurt?"
        return {"op": "floor", "col": "sister_ratio", "floor": 0.1}
    if name == "sister_smoothed_01":
        # w = sister_ratio + 0.1; additive smoothing — high-sister still dominates
        # but the dynamic range is compressed.
        return {"op": "shift", "col": "sister_ratio", "shift": 0.1}
    if name == "sister_filter_gt0":
        # Drop only the zero-sister tail: sister_ratio > 0 (~4.2M cells, 7.1%).
        # 1e-9 acts as ">0" since min nonzero ratio is 1/max_neighbor ≈ 0.01.
        # NaN-sister cells (~3.67M) also drop → 13.3% total drop.
        return {"op": "filter", "col": "sister_ratio", "threshold": 1e-9}
    if name == "sister_filter_gt0_nankeep":
        # Same as gt0 but NaN cells (no spatial coherence data) kept at w=1.
        # Drops only the 4.2M sr=0 cells (7.1%) — gentler than gt0.
        return {"op": "filter", "col": "sister_ratio", "threshold": 1e-9, "nan_keep": True}
    if name == "sister_filter_005":
        return {"op": "filter", "col": "sister_ratio", "threshold": 0.05}
    if name == "sister_filter_010":
        return {"op": "filter", "col": "sister_ratio", "threshold": 0.10}
    if name == "sister_filter_015":
        return {"op": "filter", "col": "sister_ratio", "threshold": 0.15}
    if name == "sister_filter_020":
        return {"op": "filter", "col": "sister_ratio", "threshold": 0.20}
    if name == "sister_filter_03":
        return {"op": "filter", "col": "sister_ratio", "threshold": 0.3}
    if name == "sister_filter_05":
        return {"op": "filter", "col": "sister_ratio", "threshold": 0.5}
    if name == "sister_filter_08":
        # Extreme tail: median sister_ratio ~0.16, so 0.8 keeps only the
        # extreme high-coherence tail (a few percent of cells).
        return {"op": "filter", "col": "sister_ratio", "threshold": 0.8}
    if name == "attn_ebi_x_sister":
        return {"op": "product", "cols": ["attn_ebi", "sister_ratio"]}
    if name == "attn_geneko_x_sister":
        return {"op": "product", "cols": ["attn_geneko", "sister_ratio"]}
    # Region-homogeneity-based "barcode miscall" weighting / filtering.
    # miscall_score = max(0, region_homogeneity - sister_ratio) — high iff
    # the local region is dominated by a single sgRNA and THIS cell does NOT
    # share it. See compute_region_homogeneity.py for derivation.
    if name == "region_miscall_filter_03":
        # Drop cells with miscall_score > 0.3 (likely barcode miscalls in
        # coherent regions). NaN cells (no sister data) keep at w=1.
        return {"op": "filter_le", "col": "miscall_score", "threshold": 0.3,
                "nan_keep": True}
    if name == "region_miscall_filter_05":
        return {"op": "filter_le", "col": "miscall_score", "threshold": 0.5,
                "nan_keep": True}
    if name == "region_miscall_w_a03":
        # Continuous: w = 1 / (1 + miscall_score / 0.3). NaN → w=1.
        return {"op": "region_recip", "col": "miscall_score", "alpha": 0.3}
    if name == "region_miscall_w_a10":
        # Gentler: w = 1 / (1 + miscall_score / 1.0). NaN → w=1.
        return {"op": "region_recip", "col": "miscall_score", "alpha": 1.0}
    if name == "attn_ebi_x_region_w_a03":
        # Compound: w = (1 / (1 + miscall_score/0.3)) * attn_ebi.
        return {"op": "region_recip_x_col",
                "cols": ["attn_ebi", "miscall_score"], "alpha": 0.3}
    if name == "misalign_w_a30":
        # Continuous misalignment weight: w = 1 / (1 + non_sister/30)
        # median cell (ns≈23): w≈0.57; dense miscall (ns=50): w≈0.38
        return {"op": "misalign_recip",
                "cols": ["neighbor_count", "sister_count"],
                "alpha": 30.0}
    if name == "misalign_w_a100":
        # Gentler misalignment weight: w = 1 / (1 + non_sister/100)
        # median cell: w≈0.81; dense miscall (ns=50): w≈0.67
        return {"op": "misalign_recip",
                "cols": ["neighbor_count", "sister_count"],
                "alpha": 100.0}
    if name == "misalign_exp_s30":
        # Exponential decay: w = exp(-non_sister/30)
        # Sharper penalty than misalign_w_a30.
        return {"op": "misalign_exp",
                "cols": ["neighbor_count", "sister_count"],
                "scale": 30.0}
    if name == "misalign_w_a30_x_ebi":
        # Compound: misalignment weight × EBI attention.
        # Test whether misalignment is additive with attention (parallel to attn_ebi_x_sister).
        return {"op": "misalign_recip_x_col",
                "cols": ["attn_ebi", "neighbor_count", "sister_count"],
                "alpha": 30.0}
    if name == "attn_ebi_plus_sister":
        # Additive combination — tests whether EBI + sister are additive
        # vs multiplicative. Both signals on similar 0-1 magnitude scales.
        return {"op": "sum", "cols": ["attn_ebi", "sister_ratio"]}
    raise ValueError(f"unknown strategy: {name!r}")


STRATEGIES = [
    "ebi", "geneko", "max",
    "min", "product", "concordance_50",
    "softmax_K100", "softmax_K1k", "softmax_K10k",
    "acc_select_geneko_raw", "acc_select_geneko_weighted",
    "acc_select_chad_raw", "acc_select_chad_weighted",
    "ebi_then_geneko",
    "set_accuracy",
    "set_accuracy_ebi",
    "v5_gko", "v5_ebionly", "v5_ebifb",
    "v5_gko_cutoff_20k", "v5_ebionly_cutoff_20k", "v5_ebifb_cutoff_20k",
    "v5_ebifb_cutoff_15k",
    # v5 multi-bag SHAP variants
    "v5m_gko", "v5m_ebionly", "v5m_ebifb",
    "v5m_gko_cutoff_20k", "v5m_ebionly_cutoff_20k", "v5m_ebifb_cutoff_20k",
    "v5m_gko_cutoff_10k", "v5m_ebionly_cutoff_10k", "v5m_ebifb_cutoff_10k",
    "v5m_gko_fluor_cutoff_500",      "v5m_gko_fluor_cutoff_100",
    "v5m_ebionly_fluor_cutoff_500",  "v5m_ebionly_fluor_cutoff_100",
    # sister-coherence strategies → route to <root>/sister/<name>/ subdir
    "sister", "sister_pow2", "sister_pow4",
    "sister_floored_01", "sister_smoothed_01",
    "sister_filter_gt0_nankeep", "sister_filter_gt0",
    "sister_filter_005", "sister_filter_010", "sister_filter_015", "sister_filter_020",
    "sister_filter_03", "sister_filter_05", "sister_filter_08",
    "attn_ebi_x_sister", "attn_geneko_x_sister",
    "attn_ebi_plus_sister",
    # misalignment-based weights (continuous, derived from neighbor_count - sister_count)
    "misalign_w_a30", "misalign_w_a100", "misalign_exp_s30",
    "misalign_w_a30_x_ebi",
    # region-homogeneity-based "barcode miscall" weighting / filtering
    "region_miscall_filter_03", "region_miscall_filter_05",
    "region_miscall_w_a03", "region_miscall_w_a10",
    "attn_ebi_x_region_w_a03",
    # fluor-attention strategies (use --signal-set no_phase by default)
    "fluor_ebi", "fluor_geneko", "fluor_max", "fluor_ebi_then_geneko",
]

# Strategies that read from the FLUOR sidecar (per-(cell, channel)) instead
# of the phase sidecar. The runner enables the fluor sidecar when the
# strategy name starts with "fluor_" OR when --signal-set != phase_only.
FLUOR_STRATEGIES = {
    "fluor_ebi", "fluor_geneko", "fluor_max", "fluor_ebi_then_geneko",
}

SISTER_STRATEGIES = {
    "sister", "sister_pow2", "sister_pow4",
    "sister_floored_01", "sister_smoothed_01",
    "sister_filter_gt0_nankeep", "sister_filter_gt0",
    "sister_filter_005", "sister_filter_010", "sister_filter_015", "sister_filter_020",
    "sister_filter_03", "sister_filter_05", "sister_filter_08",
    "attn_ebi_x_sister", "attn_geneko_x_sister",
    "attn_ebi_plus_sister",
    "misalign_w_a30", "misalign_w_a100", "misalign_exp_s30",
    "misalign_w_a30_x_ebi",
    "region_miscall_filter_03", "region_miscall_filter_05",
    "region_miscall_w_a03", "region_miscall_w_a10",
    "attn_ebi_x_region_w_a03",
}


def _install_patches(strategy_name: str, use_fluor: bool = False) -> None:
    """Patch submit_parallel_jobs to wrap phase1 workers + patch local fd.

    When ``use_fluor=True``, the fluor sidecar
    (per_experiment_v4_attn_fluor.parquet, keyed on (exp, well, seg,
    channel)) is also passed to the worker. Non-phase channel loads then
    apply per-(cell, channel) attention weighting from that sidecar.
    """
    spec = _resolve_strategy(strategy_name)
    # v5m_*_fluor_* strategies read weights from the multi-bag FLUOR sidecar
    # (per_experiment_v5_multibag_fluor.parquet). Everything else keeps the
    # v4 fluor sidecar.
    is_v5m_fluor = strategy_name in {
        "v5m_gko_fluor_cutoff_500",      "v5m_gko_fluor_cutoff_100",
        "v5m_ebionly_fluor_cutoff_500",  "v5m_ebionly_fluor_cutoff_100",
    }
    if is_v5m_fluor:
        fluor_path = str(V5M_SIDECAR_FLUOR) if use_fluor else None
    else:
        fluor_path = str(FLUOR_ATTN_SIDECAR) if use_fluor else None
    fluor_dir = str(V4_PER_EXP_FLUOR) if use_fluor else None

    # Route to the combined sidecar (with set_accuracy column) only when
    # the strategy actually needs it. Everything else keeps reading the
    # base sidecar so shared consumers stay untouched.
    is_v5 = strategy_name in V5_STRATEGIES
    if strategy_name == "set_accuracy":
        phase_sidecar = ATTN_SIDECAR_WITH_SET_ACC
    elif strategy_name == "set_accuracy_ebi":
        phase_sidecar = ATTN_SIDECAR_WITH_SET_ACC_EBI
    elif strategy_name in ("v5_gko", "v5_gko_cutoff_20k"):
        phase_sidecar = V5_SIDECAR_GKO
    elif strategy_name in ("v5_ebionly", "v5_ebionly_cutoff_20k"):
        phase_sidecar = V5_SIDECAR_EBIONLY
    elif strategy_name in ("v5_ebifb", "v5_ebifb_cutoff_20k", "v5_ebifb_cutoff_15k"):
        phase_sidecar = V5_SIDECAR_EBIFB
    elif strategy_name in ("v5m_gko", "v5m_gko_cutoff_20k", "v5m_gko_cutoff_10k"):
        phase_sidecar = V5M_SIDECAR_GKO
    elif strategy_name in ("v5m_ebionly", "v5m_ebionly_cutoff_20k", "v5m_ebionly_cutoff_10k"):
        phase_sidecar = V5M_SIDECAR_EBIONLY
    elif strategy_name in ("v5m_ebifb", "v5m_ebifb_cutoff_20k", "v5m_ebifb_cutoff_10k"):
        phase_sidecar = V5M_SIDECAR_EBIFB
    elif is_v5m_fluor:
        # Fluor-only strategies still need phase_sidecar_path to be schema-
        # compatible (contains the requested weight column) even though the
        # phase channel is not processed on --signal-set no_phase. Point at
        # the fluor parquet — the column exists there.
        phase_sidecar = V5M_SIDECAR_FLUOR
    else:
        phase_sidecar = ATTN_SIDECAR

    # v5 heads use Alex's paper_v2 per-experiment h5ads instead of v4.
    feature_dir = V5_PER_EXP if is_v5 else V4_PER_EXP

    import ops_utils.hpc.slurm_batch_utils as sbu
    _orig_submit = sbu.submit_parallel_jobs

    def _patched_submit(jobs_to_submit, *args, **kwargs):
        for job in jobs_to_submit:
            orig_func = job["func"]
            if getattr(orig_func, "__name__", "") == "pca_sweep_pooled_signal":
                job["func"] = make_patched_phase1_worker(
                    orig_func, str(phase_sidecar), spec, str(feature_dir),
                    fluor_sidecar_path=fluor_path,
                    v4_fluor_dir=fluor_dir,
                )
        return _orig_submit(jobs_to_submit, *args, **kwargs)

    sbu.submit_parallel_jobs = _patched_submit

    # Submitter-side fallback for the phase1 pre-scan that uses backed h5py.
    from ops_utils.data import feature_discovery as fd
    _orig_find = fd.find_cell_h5ad_path

    def _patched_find(experiment, channel, *a, **kw):
        if "phase" in str(channel).lower():
            p = feature_dir / f"{experiment}.h5ad"
            if p.exists():
                return p
        return _orig_find(experiment, channel, *a, **kw)

    fd.find_cell_h5ad_path = _patched_find
    try:
        from ops_model.post_process.combination.pca_optimization import phase1 as p1
        p1.find_cell_h5ad_path = _patched_find
    except Exception as exc:
        print(f"  warn: failed to patch phase1.find_cell_h5ad_path ({exc})")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--attn-strategy", required=True, choices=STRATEGIES,
                   help="Which attention-weighting strategy to use")
    p.add_argument("--output-dir",
                   default="/hpc/projects/icd.fast.ops/organelle_attribution/"
                           "pca_optimized_v0.3",
                   help="Root output dir")
    p.add_argument("--chad-annotation",
                   default="/hpc/projects/icd.fast.ops/configs/gene_clusters/"
                           "chad_positive_controls_v4.yml")
    p.add_argument("--slurm", action="store_true")
    p.add_argument("--slurm-partition", default="gpu",
                   help="SLURM partition (default 'gpu' — has more capacity than "
                        "the cpu partition right now, so concurrent strategies "
                        "are more likely to be spread across separate nodes)")
    p.add_argument("--aggregate-only", action="store_true")
    p.add_argument("--fixed-threshold", type=float, default=0.80)
    p.add_argument("--slurm-memory", default=None,
                   help="SLURM memory per OP signal job (e.g. '500GB'); "
                        "passed through to pca_optimization when set.")
    p.add_argument("--phase-memory", default=None,
                   help="SLURM memory for the Phase-1 signal job (default 600GB "
                        "in pca_optimization). Needed when peak vstack allocation "
                        "exceeds the default (65M cells × 1024 float32 ≈ 247 GiB, "
                        "which doubles during np.vstack + concatenate).")
    p.add_argument("--signal-set", default="auto",
                   choices=["auto", "phase_only", "no_phase", "all_livecell"],
                   help="Which signal set to feed v3 pca_optimization. "
                        "'auto' = phase_only for non-fluor strategies, "
                        "no_phase for strategies in FLUOR_STRATEGIES. "
                        "Pass all_livecell to combine phase + fluor channels "
                        "in one run (output → all_livecell/).")
    args = p.parse_args()

    # Resolve signal_set + whether to enable the fluor sidecar.
    is_fluor_strategy = args.attn_strategy in FLUOR_STRATEGIES
    if args.signal_set == "auto":
        signal_set = "no_phase" if is_fluor_strategy else "phase_only"
    else:
        signal_set = args.signal_set
    use_fluor = signal_set != "phase_only" or is_fluor_strategy

    _install_patches(args.attn_strategy, use_fluor=use_fluor)
    print(f"✓ installed v4-attn patches (strategy={args.attn_strategy})")
    print(f"  v4 dir: {V4_PER_EXP}")
    if args.attn_strategy == "set_accuracy":
        phase_sidecar = ATTN_SIDECAR_WITH_SET_ACC
    elif args.attn_strategy == "set_accuracy_ebi":
        phase_sidecar = ATTN_SIDECAR_WITH_SET_ACC_EBI
    else:
        phase_sidecar = ATTN_SIDECAR
    print(f"  phase sidecar: {phase_sidecar}")
    if use_fluor:
        print(f"  fluor sidecar: {FLUOR_ATTN_SIDECAR}")
        print(f"  v4 fluor dir:  {V4_PER_EXP_FLUOR}")
    print(f"  signal_set: {signal_set}")

    from ops_model.post_process.combination.pca_optimization import main as pca_main

    # Sister-coherence strategies live under <root>/sister/<name>/ to keep them
    # organized as a sibling tree to <root>/attention/<name>/.
    parent_tag = "sister" if args.attn_strategy in SISTER_STRATEGIES else "attention"
    run_tag = f"{parent_tag}/{args.attn_strategy}"

    # signal_set "all_livecell" = neither --phase-only nor --no-phase (default
    # multi-channel behavior). The other two pass the matching v3 flag.
    # 2nd-pass PCA consensus is ENABLED by default (no --no-second-pca flag) —
    # the canonical headline metric comes from the 2nd-pass output at
    # <run>/second_pca_consensus/.
    # v5 strategies use Alex's paper_v2 features → --paper-v2 subdir.
    paper_flag = "--paper-v2" if args.attn_strategy in V5_STRATEGIES else "--paper-v1"
    pca_argv = [
        "--output-dir", str(args.output_dir),
        "--cell-dino",
        "--zscore-per-experiment",
        paper_flag,
        "--run-tag", run_tag,
        "--chad-annotation", str(args.chad_annotation),
        "--fixed-threshold", str(args.fixed_threshold),
        "--slurm-partition", args.slurm_partition,
        # ⚠️ LOAD-BEARING — DO NOT REMOVE ⚠️
        # This runner's whole purpose is to inject weighting via a monkey-patch
        # of ``ops_model.features.anndata_utils.load_features_corrected``.
        # In pca_optimization/phase1.py, ``load_features_corrected`` is ONLY
        # called on the ``if apply_iss_sidecar:`` branch (line ~243). The
        # ``else:`` branch reads via ``ad.read_h5ad(_cp)`` directly, which
        # bypasses every monkey-patch we install and produces silently-
        # unweighted metrics identical to the unweighted baseline.
        #
        # History: commit e8d1026 (2026-06-10) removed the direct
        # ``load_cell_h5ad(...)`` call that used to make the load_cell_h5ad
        # monkey-patch work in either branch. Since that commit, this runner
        # MUST pass --apply-iss-sidecar or weighting silently no-ops.
        # See _weighted_pca_worker._verify_weighting_applied for the runtime check
        # that fails loudly if this flag ever gets removed.
        "--apply-iss-sidecar",
    ]
    if args.slurm_memory:
        pca_argv += ["--slurm-memory", args.slurm_memory]
    if args.phase_memory:
        pca_argv += ["--phase-memory", args.phase_memory]
    if signal_set == "phase_only":
        pca_argv.append("--phase-only")
        # 2nd-pass PCA consensus is a no-op with only 1 channel; skip.
        pca_argv.append("--no-second-pca")
    elif signal_set == "no_phase":
        pca_argv.append("--no-phase")
    # signal_set == "all_livecell": no flag — default to all channels
    if args.slurm:
        pca_argv.append("--slurm")
    if args.aggregate_only:
        pca_argv.append("--aggregate-only")

    print(f"Invoking pca_optimization with:\n  {' '.join(pca_argv)}")
    sys.argv = ["pca_optimization"] + pca_argv
    return pca_main()


if __name__ == "__main__":
    sys.exit(main())
