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
    fluor_sidecar_path: str = None,
    v4_fluor_dir: str = None,
) -> Callable:
    """Wrap orig_func so the worker patches load_cell_h5ad before calling it.

    Closure captures small dict + paths — the heavy sidecar load happens on
    the worker, not in the serialized pickle blob.

    Parameters
    ----------
    sidecar_path
        Phase sidecar parquet — per-(experiment, well, segmentation_id).
        Always loaded.
    v4_dir
        Directory holding ``<exp>.h5ad`` v4 phase features. Phase channel
        loads are redirected here and weighted by ``sidecar_path``.
    fluor_sidecar_path
        Optional fluor sidecar parquet — per-(experiment, well,
        segmentation_id, channel). When provided AND v4_fluor_dir is also
        provided, non-phase channel loads are redirected to the v4 fluor
        h5ad (sliced by channel) and weighted by this sidecar. When None,
        non-phase channels pass through unweighted (legacy behavior).
    v4_fluor_dir
        Directory holding ``<exp>.h5ad`` v4 fluor features (one h5ad per
        experiment, with channel_name in obs; multiple rows per cell when
        the cell was imaged in multiple channels). When provided alongside
        fluor_sidecar_path, the worker uses these h5ads (mirroring how
        phase uses ``v4_dir``); else it falls back to the v3 production
        fluor h5ads via _orig_load.
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

        # Group phase sidecar by experiment, dedupe (well, seg) by max per column.
        sidecar_by_exp = {}
        for exp, g in sidecar.groupby("experiment"):
            agg = g.groupby(["well", "segmentation_id"])[needed_cols].max()
            sidecar_by_exp[exp] = agg

        def _normalize_channel(channel: str) -> str:
            """Map v3-internal channel name to the fluor sidecar key.
            v3 uses commas+spaces (e.g. 'ER/Golgi COP-II, SEC23A'); the
            sidecar uses underscores (matches Alex's PMA CSV channel column).
            """
            return channel.replace(", ", "_").replace(" ", "_")

        # Optional fluor sidecar — keyed on (experiment, channel) → DF[well,seg].
        fluor_by_exp_channel: Dict[str, Dict[str, "pd.DataFrame"]] = {}
        if fluor_sidecar_path:
            print(f"[v4-attn worker] loading fluor sidecar: {fluor_sidecar_path}")
            fluor = pd.read_parquet(
                fluor_sidecar_path,
                columns=["experiment", "well", "segmentation_id", "channel"]
                + needed_cols,
            )
            fluor["well"] = fluor["well"].astype(str)
            fluor["channel"] = fluor["channel"].astype(str)
            for (exp, ch), g in fluor.groupby(["experiment", "channel"], observed=True):
                agg = g.groupby(["well", "segmentation_id"])[needed_cols].max()
                fluor_by_exp_channel.setdefault(exp, {})[ch] = agg
            n_pairs = sum(len(v) for v in fluor_by_exp_channel.values())
            print(f"  fluor sidecar: {n_pairs} (exp, channel) slices loaded")

        v4_dir_p = Path(v4_dir)

        # 2) Install patches.
        _orig_find = fd.find_cell_h5ad_path
        _orig_load = fd.load_cell_h5ad

        v4_fluor_dir_p = Path(v4_fluor_dir) if v4_fluor_dir else None

        # Thread-local hand-off: v3's phase1 calls find_cell_h5ad_path RIGHT
        # before load_features_corrected. We stash the channel from find() so
        # the patched load_features_corrected can slice the multi-channel
        # v4 fluor h5ad correctly.
        import threading as _threading
        _fluor_ctx = _threading.local()

        def _patched_find(experiment, channel, *a, **kw):
            channel_lower = str(channel).lower()
            if "phase" in channel_lower:
                p = v4_dir_p / f"{experiment}.h5ad"
                if p.exists():
                    return p
            elif v4_fluor_dir_p is not None:
                # Non-phase: point at v4 fluor h5ad (the worker's _patched_load
                # OR _patched_load_features_corrected will slice it by channel).
                p = v4_fluor_dir_p / f"{experiment}.h5ad"
                if p.exists():
                    # Stash the channel for the subsequent
                    # load_features_corrected call.
                    _fluor_ctx.last_channel = str(channel)
                    _fluor_ctx.last_experiment = experiment
                    return p
            return _orig_find(experiment, channel, *a, **kw)

        def _apply_weights(
            adata: "ad.AnnData",
            attns: dict,
            experiment: str,
            tag: str,
        ) -> "ad.AnnData":
            """Per-sgRNA normalize the per-cell weights, pre-multiply X, return adata.

            ``attns`` maps each needed_col → length-n_cells np.float32 array
            (NaN where the cell has no attention for that column).
            """
            sgrna = adata.obs["sgRNA"].astype(str).values
            gene_names = adata.obs["gene_name"].astype(str).values

            if strategy_spec["op"] == "acc_select":
                attn_col = strategy_spec["col"]
                a = attns[attn_col]
                gene_to_K = strategy_spec["gene_to_K"]
                K_per_cell = np.array(
                    [gene_to_K.get(g, -1) for g in gene_names], dtype=np.int32
                )
                a_rank_src = np.where(np.isnan(a), -np.inf, a)
                df = pd.DataFrame({"sgRNA": sgrna, "a": a_rank_src})
                ranks = df.groupby("sgRNA", observed=True)["a"].rank(
                    method="first", ascending=False, na_option="bottom"
                ).to_numpy()
                keep = (K_per_cell < 0) | (ranks <= K_per_cell)
                if strategy_spec.get("mode", "raw") == "weighted":
                    w = np.where(np.isnan(a), 1.0, a).astype(np.float32)
                    w = np.where(keep, w, 0.0).astype(np.float32)
                else:
                    w = keep.astype(np.float32)
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

            covered = sum(np.isfinite(attns[c]) for c in needed_cols)
            n_any = int((covered > 0).sum())
            print(
                f"  [{experiment}/{tag}] {adata.n_obs:,} cells, "
                f"attn-covered (any head)={n_any:,} ({n_any/adata.n_obs:.0%}), "
                f"w_norm in [{w_norm.min():.3g}, {w_norm.max():.3g}]"
            )
            return adata

        def _lookup_attns(sidecar_slice, well_arr, seg_arr):
            """Reindex a per-(well, seg) sidecar slice to get per-cell attention."""
            idx = pd.MultiIndex.from_arrays([well_arr, seg_arr])
            reindexed = sidecar_slice.reindex(idx)
            return {c: reindexed[c].to_numpy(dtype=np.float32) for c in needed_cols}

        def _patched_load(experiment, channel, *a, **kw):
            channel_str = str(channel)
            is_phase = "phase" in channel_str.lower()

            if is_phase:
                # ---- Phase path: load v4 h5ad + apply phase sidecar weighting.
                p = v4_dir_p / f"{experiment}.h5ad"
                if not p.exists():
                    return _orig_load(experiment, channel, *a, **kw)

                adata = ad.read_h5ad(p)
                n0 = adata.n_obs
                keep = adata.obs["sgRNA"].notna().values & ~adata.obs.duplicated(
                    subset=["well", "segmentation_id"], keep="first"
                ).values
                if not keep.all():
                    adata = adata[keep].copy()
                    print(f"  [{experiment}] dropped {n0 - adata.n_obs:,} bad rows → {adata.n_obs:,} cells")

                if experiment not in sidecar_by_exp:
                    print(f"  [{experiment}] WARN no phase sidecar entries — uniform weights")
                    return adata

                well = adata.obs["well"].astype(str).values
                seg = adata.obs["segmentation_id"].values
                attns = _lookup_attns(sidecar_by_exp[experiment], well, seg)
                return _apply_weights(adata, attns, experiment, channel_str)

            # ---- Non-phase channel: prefer v4 fluor h5ad (sliced by
            # channel) if v4_fluor_dir was provided; otherwise fall back
            # to v3 production fluor.
            channel_norm = _normalize_channel(channel_str)
            v4_fluor_p = (Path(v4_fluor_dir) / f"{experiment}.h5ad"
                            if v4_fluor_dir else None)
            using_v4 = v4_fluor_p is not None and v4_fluor_p.exists()

            if using_v4:
                adata_all = ad.read_h5ad(v4_fluor_p)
                # Slice to rows for THIS channel only.
                m_chan = (adata_all.obs["channel_name"].astype(str)
                          == channel_norm).values
                if not m_chan.any():
                    print(f"  [{experiment}/{channel_str}] no v4 fluor rows "
                          f"for channel {channel_norm!r} — falling back to v3")
                    adata = _orig_load(experiment, channel, *a, **kw)
                else:
                    adata = adata_all[m_chan].copy()
            else:
                adata = _orig_load(experiment, channel, *a, **kw)

            # Always drop bad rows: NaN sgRNA breaks downstream aggregation,
            # duplicate (well, seg) breaks v3's per-cell join. Apply regardless
            # of which branch (v4 fluor / v3 fallback) produced the adata.
            if "sgRNA" in adata.obs.columns:
                n0 = adata.n_obs
                keep = adata.obs["sgRNA"].notna().values & ~adata.obs.duplicated(
                    subset=["well", "segmentation_id"], keep="first"
                ).values
                if not keep.all():
                    adata = adata[keep].copy()
                    print(f"  [{experiment}/{channel_str}] dropped "
                          f"{n0 - adata.n_obs:,} bad rows → {adata.n_obs:,} cells")

            if not fluor_by_exp_channel:
                return adata  # no fluor sidecar → unweighted

            exp_channels = fluor_by_exp_channel.get(experiment, {})
            if channel_norm not in exp_channels:
                # No fluor attention for this (exp, channel) — pass through.
                print(f"  [{experiment}/{channel_str}] no fluor sidecar entries "
                      f"(normalized to {channel_norm!r}) — uniform weights")
                return adata

            well = adata.obs["well"].astype(str).values
            seg = adata.obs["segmentation_id"].values
            attns = _lookup_attns(exp_channels[channel_norm], well, seg)
            return _apply_weights(adata, attns, experiment, channel_str)

        fd.find_cell_h5ad_path = _patched_find
        fd.load_cell_h5ad = _patched_load
        p1.find_cell_h5ad_path = _patched_find
        p1.load_cell_h5ad = _patched_load

        # ALSO patch load_features_corrected — v3's phase1 calls it instead of
        # load_cell_h5ad when --apply-iss-sidecar is set (the default). For
        # v4 fluor h5ads (multi-channel), we must slice by channel here too,
        # else v3 sees rows from ALL channels mixed together.
        if v4_fluor_dir_p is not None:
            from ops_model.features import anndata_utils as au

            _orig_lfc = au.load_features_corrected

            def _patched_load_features_corrected(cell_path, *a, **kw):
                p = Path(cell_path) if cell_path else None
                # Detect v4 fluor h5ad by parent dir match.
                is_v4_fluor = (
                    p is not None
                    and p.parent == v4_fluor_dir_p
                    and p.exists()
                )
                if not is_v4_fluor:
                    return _orig_lfc(cell_path, *a, **kw)

                # Load the full multi-channel h5ad then slice by the channel
                # that the immediately-preceding _patched_find captured.
                channel = getattr(_fluor_ctx, "last_channel", None)
                if channel is None:
                    print(f"  WARN: load_features_corrected on v4 fluor h5ad "
                          f"{p.name} without a channel context — returning full")
                    return _orig_lfc(cell_path, *a, **kw)
                channel_norm = _normalize_channel(channel)
                adata = ad.read_h5ad(p)
                m = (adata.obs["channel_name"].astype(str) == channel_norm).values
                if not m.any():
                    print(f"  WARN: no v4 fluor rows for channel {channel_norm!r} "
                          f"in {p.name} — falling back to v3")
                    return _orig_lfc(cell_path, *a, **kw)
                adata = adata[m].copy()
                # Drop NaN-sgRNA + duplicate (well, seg) rows (mirror phase).
                n0 = adata.n_obs
                keep = adata.obs["sgRNA"].notna().values & ~adata.obs.duplicated(
                    subset=["well", "segmentation_id"], keep="first"
                ).values
                if not keep.all():
                    adata = adata[keep].copy()
                print(f"  [{getattr(_fluor_ctx,'last_experiment','?')}/"
                      f"{channel}] v4 fluor sliced to {adata.n_obs:,} "
                      f"rows (from {n0:,})")
                return adata

            au.load_features_corrected = _patched_load_features_corrected
            try:
                from ops_model.post_process.combination.pca_optimization \
                    import phase1 as _p1
                # phase1 imports via `from ... import` inside the function,
                # so the name resolves at call time from au — no extra patch
                # needed. But if phase1 had a top-level import, patch it too.
            except Exception:
                pass

        # 3) Run the original phase1 worker.
        return orig_func(*args, **kwargs)

    return _patched_worker
