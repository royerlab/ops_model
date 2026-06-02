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

Output run-tag: ``attention/<strategy>`` — all results land under
``<root>/cell_dino/zscore_per_exp/paper_v1/attention/<strategy>/phase_only/...``.

SLURM compatibility: monkey-patches in the submitter process do not propagate
to submitit workers. We wrap each SLURM worker function with a picklable
closure in ``_v4_attn_worker.make_patched_phase1_worker`` so the patches are
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

import _v4_attn_worker  # noqa: E402
from _v4_attn_worker import make_patched_phase1_worker  # noqa: E402

import cloudpickle  # noqa: E402
cloudpickle.register_pickle_by_value(_v4_attn_worker)

V4_PER_EXP = Path(
    "/hpc/projects/icd.fast.ops/models/alex_lin_attention/v4/expansion_v1/"
    "per_experiment_v4"
)
ATTN_SIDECAR = Path(
    "/hpc/projects/icd.fast.ops/models/alex_lin_attention/v4/expansion_v1/"
    "per_experiment_v4_attn.parquet"
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
    """Map --attn-strategy name → strategy_spec dict consumed by _v4_attn_worker."""
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
    raise ValueError(f"unknown strategy: {name!r}")


STRATEGIES = [
    "ebi", "geneko", "max",
    "min", "product", "concordance_50",
    "softmax_K100", "softmax_K1k", "softmax_K10k",
    "acc_select_geneko_raw", "acc_select_geneko_weighted",
    "acc_select_chad_raw", "acc_select_chad_weighted",
    "ebi_then_geneko",
]


def _install_patches(strategy_name: str) -> None:
    """Patch submit_parallel_jobs to wrap phase1 workers + patch local fd."""
    spec = _resolve_strategy(strategy_name)

    import ops_utils.hpc.slurm_batch_utils as sbu
    _orig_submit = sbu.submit_parallel_jobs

    def _patched_submit(jobs_to_submit, *args, **kwargs):
        for job in jobs_to_submit:
            orig_func = job["func"]
            if getattr(orig_func, "__name__", "") == "pca_sweep_pooled_signal":
                job["func"] = make_patched_phase1_worker(
                    orig_func, str(ATTN_SIDECAR), spec, str(V4_PER_EXP),
                )
        return _orig_submit(jobs_to_submit, *args, **kwargs)

    sbu.submit_parallel_jobs = _patched_submit

    # Submitter-side fallback for the phase1 pre-scan that uses backed h5py.
    from ops_utils.data import feature_discovery as fd
    _orig_find = fd.find_cell_h5ad_path

    def _patched_find(experiment, channel, *a, **kw):
        if "phase" in str(channel).lower():
            p = V4_PER_EXP / f"{experiment}.h5ad"
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
    args = p.parse_args()

    _install_patches(args.attn_strategy)
    print(f"✓ installed v4-attn patches (strategy={args.attn_strategy})")
    print(f"  v4 dir: {V4_PER_EXP}")
    print(f"  sidecar: {ATTN_SIDECAR}")

    from ops_model.post_process.combination.pca_optimization import main as pca_main

    run_tag = f"attention/{args.attn_strategy}"
    pca_argv = [
        "--output-dir", str(args.output_dir),
        "--cell-dino",
        "--zscore-per-experiment",
        "--phase-only",
        "--paper-v1",
        "--run-tag", run_tag,
        "--chad-annotation", str(args.chad_annotation),
        "--fixed-threshold", str(args.fixed_threshold),
        "--no-second-pca",
        "--slurm-partition", args.slurm_partition,
    ]
    if args.slurm:
        pca_argv.append("--slurm")
    if args.aggregate_only:
        pca_argv.append("--aggregate-only")

    print(f"Invoking pca_optimization with:\n  {' '.join(pca_argv)}")
    sys.argv = ["pca_optimization"] + pca_argv
    return pca_main()


if __name__ == "__main__":
    sys.exit(main())
