"""Worker-side patch factory for v4 attention-weighted v3 pipeline runs.

Lives in its own importable module so cloudpickle can serialize the wrapped
closure with a stable qualname when submitit ships the worker to SLURM.

``make_patched_phase1_worker(orig_func, sidecar_path, strategy_spec, v4_dir)``
returns a picklable closure that on each worker:

    1. Loads the attention sidecar parquet (~700MB).
    2. Patches ``load_cell_h5ad`` to redirect Phase channel loads → v4 h5ads
       AND pre-multiply X by per-sgRNA-normalized attention weights, where the
       weight is computed from the sidecar columns per the ``strategy_spec``.
    3. Calls the original worker function (``pca_sweep_pooled_signal``).

Math: per-cell weights are normalized within each (sgRNA, experiment) group so
``mean(w) == 1`` per group; then ``X *= w[:, None]`` makes the downstream
``mean`` aggregation equivalent to ``weighted_mean``.

Cells whose gene is not in a head's PMA panel have ``NaN`` attention for that
head. Different strategies handle NaN differently — see ``_compute_weights``.

``strategy_spec`` keys
----------------------
* ``op``           — "column", "min", "product", "concordance", "softmax",
                     "fallback", "acc_select", "power", "filter", or "sum"
* ``col``          — for "column" / "softmax" / "acc_select" / "power" / "filter":
                     which sidecar column to use (for "acc_select" this is also
                     what we rank cells by within each (sgRNA, experiment) group)
* ``cols``         — for "min" / "product" / "concordance" / "fallback" / "sum":
                     list of cols
* ``p``            — for "power": exponent (e.g. 2.0, 4.0)
* ``threshold``    — for "filter": min value of col to keep (cells below get w=0)
* ``percentile``   — for "concordance": top-N% threshold (e.g. 50.0)
* ``K``            — for "softmax": exp(K * attn) sharpness
* ``mode``         — for "acc_select": "raw" (kept cells weight=1) or
                     "weighted" (kept cells weight=attn)
* ``gene_to_K``    — for "acc_select": ``{gene_name: K}`` mapping. K is the
                     per-sgRNA top-K cap (typically ``bin_n_cells // 4`` where
                     ``bin_n_cells`` is the smallest n_cells row at which
                     top1_acc >= 0.95 in the v3 cdino eval). K=-1 means
                     "keep all cells of this gene" (fallback).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict


def _compute_weights(spec: Dict, attns: Dict[str, "np.ndarray"], thresholds: Dict[str, float]):
    """Combine per-head attention scalars into a per-cell weight per the spec.

    Returns ``w`` (float32, length = n_cells). NaN-handling:

    * "column"      — NaN → 1.0 (uniform fallback for cells outside the head's panel)
    * "min"         — both NaN → 1; one NaN → use the other; else min
    * "product"     — any NaN → 1 (treat NaN as multiplicative identity)
    * "concordance" — NaN counts as "in agreement" so non-panel cells aren't dropped
    * "softmax"     — NaN → 0 in the exponent (so exp(0) = 1, uniform fallback)
    """
    import numpy as np

    op = spec["op"]
    if op == "column":
        a = attns[spec["col"]]
        return np.where(np.isnan(a), 1.0, a).astype(np.float32)

    if op == "min":
        a = attns[spec["cols"][0]]
        b = attns[spec["cols"][1]]
        # fmin: NaN propagates only if BOTH are NaN; otherwise picks the non-NaN.
        # Then fill remaining NaN with 1.
        m = np.fmin(a, b)
        return np.where(np.isnan(m), 1.0, m).astype(np.float32)

    if op == "product":
        a = np.where(np.isnan(attns[spec["cols"][0]]), 1.0, attns[spec["cols"][0]])
        b = np.where(np.isnan(attns[spec["cols"][1]]), 1.0, attns[spec["cols"][1]])
        return (a * b).astype(np.float32)

    if op == "concordance":
        # Binary: cell is concordant iff both heads are in their top-P% (or NaN).
        # NaN treated as "concordant" so non-panel cells aren't dropped.
        a = attns[spec["cols"][0]]
        b = attns[spec["cols"][1]]
        t_a = thresholds[spec["cols"][0]]
        t_b = thresholds[spec["cols"][1]]
        concord = ((np.isnan(a)) | (a >= t_a)) & ((np.isnan(b)) | (b >= t_b))
        # Floor of 0.01 keeps a small weight for non-concordant cells so guides
        # whose ALL cells are non-concordant don't get zero-weighted (would crash
        # the per-sgRNA normalize).
        return np.where(concord, 1.0, 0.01).astype(np.float32)

    if op == "softmax":
        # w = exp(K * attn); NaN → 0 in exponent so exp(K*0) = 1.
        a = attns[spec["col"]]
        K = float(spec["K"])
        x = np.where(np.isnan(a), 0.0, a) * K
        # Clip to avoid overflow in float32 (exp(>88) overflows)
        x = np.clip(x, None, 80.0)
        return np.exp(x).astype(np.float32)

    if op == "fallback":
        # Coalesce: use cols[0] where non-NaN, else cols[1], ..., else 1.0.
        # Each per-head attention is on a similar magnitude scale (~1e-3) so
        # mixing across cells doesn't blow out the per-sgRNA mean normalization.
        cols = spec["cols"]
        a = attns[cols[0]].copy()
        for c in cols[1:]:
            mask = np.isnan(a)
            a = np.where(mask, attns[c], a)
        return np.where(np.isnan(a), 1.0, a).astype(np.float32)

    if op == "power":
        # w = col ** p; NaN → 1 (uniform fallback)
        a = attns[spec["col"]]
        p = float(spec["p"])
        return np.where(np.isnan(a), 1.0, np.power(a, p)).astype(np.float32)

    if op == "filter":
        # Hard threshold: w = 1 if col >= threshold else 0.
        # nan_keep (default False): NaN → 0 (drop). nan_keep=True: NaN → 1 (keep).
        a = attns[spec["col"]]
        t = float(spec["threshold"])
        nan_default = bool(spec.get("nan_keep", False))
        keep = np.where(np.isnan(a), nan_default, a >= t)
        return keep.astype(np.float32)

    if op == "floor":
        # w = max(col, floor); NaN → 1.0. Prevents bottom cells from going to w=0.
        a = attns[spec["col"]]
        floor = float(spec["floor"])
        return np.where(np.isnan(a), 1.0, np.maximum(a, floor)).astype(np.float32)

    if op == "shift":
        # w = col + shift; NaN → 1.0. Additive smoothing — every cell contributes.
        a = attns[spec["col"]]
        shift = float(spec["shift"])
        return np.where(np.isnan(a), 1.0, a + shift).astype(np.float32)

    if op == "misalign_recip":
        # Misalignment weight: w = 1 / (1 + non_sister_count / alpha)
        # where non_sister_count = neighbor_count - sister_count.
        # cols = ["neighbor_count", "sister_count"]; NaN → w=1 (uniform fallback).
        nc = attns[spec["cols"][0]]
        sc = attns[spec["cols"][1]]
        alpha = float(spec["alpha"])
        non_sister = nc - sc
        return np.where(
            np.isnan(non_sister), 1.0,
            1.0 / (1.0 + non_sister / alpha),
        ).astype(np.float32)

    if op == "misalign_exp":
        # Misalignment weight: w = exp(-non_sister_count / scale)
        # cols = ["neighbor_count", "sister_count"]; NaN → w=1.
        nc = attns[spec["cols"][0]]
        sc = attns[spec["cols"][1]]
        scale = float(spec["scale"])
        non_sister = nc - sc
        return np.where(
            np.isnan(non_sister), 1.0,
            np.exp(-non_sister / scale),
        ).astype(np.float32)

    if op == "filter_le":
        # Inverse-direction hard threshold: keep cells where col <= threshold.
        # Use for "high score = bad" signals (e.g. miscall_score).
        # nan_keep (default True for this op): NaN cells keep at w=1.
        a = attns[spec["col"]]
        t = float(spec["threshold"])
        nan_default = bool(spec.get("nan_keep", True))
        keep = np.where(np.isnan(a), nan_default, a <= t)
        return keep.astype(np.float32)

    if op == "region_recip":
        # w = 1 / (1 + miscall_score / alpha); high miscall (likely barcode
        # miscall in coherent region) → low w. NaN → w=1 (uniform fallback).
        # cols = ["miscall_score"].
        ms = attns[spec["col"]]
        alpha = float(spec["alpha"])
        return np.where(
            np.isnan(ms), 1.0,
            1.0 / (1.0 + ms / alpha),
        ).astype(np.float32)

    if op == "region_recip_x_col":
        # Compound: w = (1 / (1 + miscall_score/alpha)) * other_col
        # cols = [other_col, "miscall_score"]; NaN in either → 1.0.
        other = attns[spec["cols"][0]]
        ms = attns[spec["cols"][1]]
        alpha = float(spec["alpha"])
        region_w = np.where(
            np.isnan(ms), 1.0,
            1.0 / (1.0 + ms / alpha),
        )
        other_w = np.where(np.isnan(other), 1.0, other)
        return (region_w * other_w).astype(np.float32)

    if op == "misalign_recip_x_col":
        # Compound: w = (1 / (1 + non_sister/alpha)) * other_col
        # cols = [other_col, "neighbor_count", "sister_count"].
        # NaN in either factor → 1.0 (multiplicative identity).
        other = attns[spec["cols"][0]]
        nc = attns[spec["cols"][1]]
        sc = attns[spec["cols"][2]]
        alpha = float(spec["alpha"])
        non_sister = nc - sc
        misalign_w = np.where(
            np.isnan(non_sister), 1.0,
            1.0 / (1.0 + non_sister / alpha),
        )
        other_w = np.where(np.isnan(other), 1.0, other)
        return (misalign_w * other_w).astype(np.float32)

    if op == "sum":
        # Additive combination of multiple columns; NaN → 0 for that term.
        # Scales are mixed (attn ~1e-3, sister 0-1) so the per-sgRNA
        # mean(w)=1 normalization handles the rescaling.
        cols = spec["cols"]
        acc = np.zeros_like(attns[cols[0]], dtype=np.float64)
        for c in cols:
            acc = acc + np.where(np.isnan(attns[c]), 0.0, attns[c])
        # All-NaN cells get w=0; the per-sgRNA normalization handles them.
        return acc.astype(np.float32)

    raise ValueError(f"unknown strategy op: {op!r}")


def make_patched_phase1_worker(
    orig_func: Callable,
    sidecar_path: str,
    strategy_spec: Dict,
    v4_dir: str,
) -> Callable:
    """Wrap orig_func so the worker patches load_cell_h5ad before calling it.

    Closure captures small dict + paths — the heavy sidecar load happens on
    the worker, not in the serialized pickle blob.
    """

    def _patched_worker(*args, **kwargs):
        import anndata as ad
        import numpy as np
        import pandas as pd

        from ops_utils.data import feature_discovery as fd
        from ops_model.post_process.combination.pca_optimization import phase1 as p1

        # 1) Figure out which sidecar columns we need.
        if strategy_spec["op"] in ("column", "softmax", "acc_select",
                                    "power", "filter", "filter_le",
                                    "floor", "shift", "region_recip"):
            needed_cols = [strategy_spec["col"]]
        else:
            needed_cols = list(strategy_spec["cols"])

        print(f"[v4-attn worker] strategy={strategy_spec}  loading cols={needed_cols}")
        sidecar = pd.read_parquet(
            sidecar_path,
            columns=["experiment", "well", "segmentation_id"] + needed_cols,
        )
        sidecar["well"] = sidecar["well"].astype(str)

        # Pre-compute global percentile thresholds (for concordance).
        thresholds = {}
        if strategy_spec["op"] == "concordance":
            pct = float(strategy_spec["percentile"])
            for c in needed_cols:
                vals = sidecar[c].to_numpy()
                vals = vals[np.isfinite(vals)]
                thresholds[c] = float(np.percentile(vals, 100 - pct))
                print(f"  concordance threshold {c}: top-{pct:.0f}% cutoff = {thresholds[c]:.4g}")

        # Group by experiment, dedupe duplicate (well, seg) by max per column.
        sidecar_by_exp = {}
        for exp, g in sidecar.groupby("experiment"):
            agg = g.groupby(["well", "segmentation_id"])[needed_cols].max()
            sidecar_by_exp[exp] = agg

        v4_dir_p = Path(v4_dir)

        # 2) Install patches.
        _orig_find = fd.find_cell_h5ad_path
        _orig_load = fd.load_cell_h5ad

        def _patched_find(experiment, channel, *a, **kw):
            if "phase" in str(channel).lower():
                p = v4_dir_p / f"{experiment}.h5ad"
                if p.exists():
                    return p
            return _orig_find(experiment, channel, *a, **kw)

        def _patched_load(experiment, channel, *a, **kw):
            if "phase" not in str(channel).lower():
                return _orig_load(experiment, channel, *a, **kw)
            p = v4_dir_p / f"{experiment}.h5ad"
            if not p.exists():
                return _orig_load(experiment, channel, *a, **kw)

            adata = ad.read_h5ad(p)
            # Drop NaN-sgRNA + duplicate (well, seg) rows that break v3 aggregation.
            n0 = adata.n_obs
            keep = adata.obs["sgRNA"].notna().values & ~adata.obs.duplicated(
                subset=["well", "segmentation_id"], keep="first"
            ).values
            if not keep.all():
                adata = adata[keep].copy()
                print(f"  [{experiment}] dropped {n0 - adata.n_obs:,} bad rows → {adata.n_obs:,} cells")

            if experiment not in sidecar_by_exp:
                print(f"  [{experiment}] WARN no sidecar entries — uniform weights")
                return adata

            sidecar_exp = sidecar_by_exp[experiment]
            well = adata.obs["well"].astype(str).values
            seg = adata.obs["segmentation_id"].values
            idx = pd.MultiIndex.from_arrays([well, seg])

            # Reindex pulls the per-cell attention values for each requested column.
            reindexed = sidecar_exp.reindex(idx)
            attns = {c: reindexed[c].to_numpy(dtype=np.float32) for c in needed_cols}

            sgrna = adata.obs["sgRNA"].astype(str).values
            gene_names = adata.obs["gene_name"].astype(str).values

            if strategy_spec["op"] == "acc_select":
                # Per (sgRNA, exp) group, rank cells by chosen attention column,
                # keep top K where K = gene_to_K[gene] // already-divided-by-4.
                # K=-1 means "keep all cells of this gene".
                attn_col = strategy_spec["col"]
                a = attns[attn_col]
                gene_to_K = strategy_spec["gene_to_K"]
                # Per-cell K cap: -1 means no cap.
                K_per_cell = np.array(
                    [gene_to_K.get(g, -1) for g in gene_names], dtype=np.int32
                )
                # Rank per sgRNA descending; NaN attention → sent to bottom.
                a_rank_src = np.where(np.isnan(a), -np.inf, a)
                df = pd.DataFrame({"sgRNA": sgrna, "a": a_rank_src})
                ranks = df.groupby("sgRNA", observed=True)["a"].rank(
                    method="first", ascending=False, na_option="bottom"
                ).to_numpy()
                # Keep cells whose rank ≤ K, or all cells when K < 0 (no cap).
                keep = (K_per_cell < 0) | (ranks <= K_per_cell)
                if strategy_spec.get("mode", "raw") == "weighted":
                    # Use attention value for kept cells; NaN-fallback to 1
                    w = np.where(np.isnan(a), 1.0, a).astype(np.float32)
                    w = np.where(keep, w, 0.0).astype(np.float32)
                else:
                    # Raw: kept = 1, dropped = 0
                    w = keep.astype(np.float32)
                # Diagnostic
                n_kept = int(keep.sum())
                print(
                    f"  [acc_select/{strategy_spec.get('mode','raw')}] kept "
                    f"{n_kept:,}/{len(keep):,} cells ({n_kept/len(keep):.0%})"
                )
            else:
                w = _compute_weights(strategy_spec, attns, thresholds)

            df = pd.DataFrame({"sgRNA": sgrna, "w": w})
            group_mean = df.groupby("sgRNA", observed=True)["w"].transform("mean")
            group_mean = np.where(group_mean > 0, group_mean, 1.0)
            w_norm = (w / group_mean).astype(np.float32)

            X = np.asarray(adata.X, dtype=np.float32)
            X *= w_norm[:, None]
            adata.X = X

            # Diagnostics
            n_real = int(any(np.isfinite(attns[c]).sum() > 0 for c in needed_cols))
            covered = sum(np.isfinite(attns[c]) for c in needed_cols)
            n_any = int((covered > 0).sum())
            print(
                f"  [{experiment}] {adata.n_obs:,} cells, "
                f"attn-covered (any head)={n_any:,} ({n_any/adata.n_obs:.0%}), "
                f"w_norm in [{w_norm.min():.3g}, {w_norm.max():.3g}]"
            )
            return adata

        fd.find_cell_h5ad_path = _patched_find
        fd.load_cell_h5ad = _patched_load
        p1.find_cell_h5ad_path = _patched_find
        p1.load_cell_h5ad = _patched_load

        # 3) Run the original phase1 worker.
        return orig_func(*args, **kwargs)

    return _patched_worker
