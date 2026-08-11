"""Full SHAP pipeline for the OPS atlas matrix: shards → merge → captions → smoke atlas.

One pipeline for all 6 atlas variants (2 grain × 3 contrast). Selects
behavior via two orthogonal flags:

  --variant {distinct, ntc}        — which SHAP step to run
  --aggregation-level {gene, complex} — gene-KO vs CHAD protein-complex

The variants:
  * `top-attention` (default): runs `ko_shap_features.py` against the
    paired phase + fluor top-attention h5ads from
    `consolidate_top_attention_cells.py`. Output:
    `attention_<contrast>_{geneKO,chad}/ko_shap_features.csv`.
  * `all-cells`: runs `ntc_shap_features.py --contrast {distinct,ntc,global}`
    against the per-channel `all_cells_*.h5ad` files from
    `consolidate_all_cells.py`. Output:
    `all_cells_<contrast>_{geneKO,chad}/ntc_shap_features.csv`.

Steps (same shape for both variants):
  1. shards     — N parallel SLURM shards rank features per gene.
  2. merge      — concat per-shard CSVs.
  3. captions   — synthesize per-gene natural-language captions (twice
                  for ntc — one per contrast).
  4. smoke_atlas — render `attention_atlas_shap.py --local` against
                   the new CSVs for `--smoke-genes` genes (login-node).

Outputs land under `--out-dir`. Defaults auto-swap based on
`--variant` and `--aggregation-level` so the four (variant × grain)
combinations route to disjoint directories with no clobbering.

Usage:
  # Distinctiveness (default, gene-level):
  python organelle_profiler/scripts/ko_shap/run_shap_pipeline.py

  # Distinctiveness CHAD:
  python organelle_profiler/scripts/ko_shap/run_shap_pipeline.py --aggregation-level complex

  # NTC + Median (gene-level):
  python organelle_profiler/scripts/ko_shap/run_shap_pipeline.py --variant ntc

  # NTC + Median CHAD:
  python organelle_profiler/scripts/ko_shap/run_shap_pipeline.py --variant ntc --aggregation-level complex
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

STEPS = {
    0: "prebuild_caches (login-node sequential cache build per channel)",
    1: "ko_shap_features (sharded SLURM SHAP run)",
    2: "merge_shards (concat per-shard CSVs)",
    3: "captions (synthesize per-gene natural-language captions)",
    4: "pick_cells (all-cells only: pick representative cells for atlas image rows)",
    5: "smoke_atlas (parallel SLURM atlas render — 1 page per worker)",
}

ATLAS_SCRIPT = Path(
    "/hpc/mydata/gav.sturm/ops_mono/ops_process/ops_analysis/napari/attention_atlas_shap.py"
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Distinctiveness defaults (variant=distinct):
#   gene-level   -> consolidated_v3/  (built from pma_top_*_v3.csv)
#   complex-level -> consolidated_v3_chad/  (built from pma_top_*_chad_v1.csv,
#                    with obs.gene relabeled to predicted_class = complex name)
# When --aggregation-level complex AND user kept distinctiveness defaults,
# all five paths auto-swap to _chad variants.
_GENE_BASE = Path("/hpc/projects/icd.fast.ops/models/alex_lin_attention/consolidated_v3")
_CHAD_BASE = Path("/hpc/projects/icd.fast.ops/models/alex_lin_attention/consolidated_v3_chad")

DEFAULT_PHASE_H5AD = _GENE_BASE / "top_attention_cells_phase.h5ad"
DEFAULT_FLUOR_H5AD = _GENE_BASE / "top_attention_cells_fluor.h5ad"
DEFAULT_CACHE_PHASE = Path(
    "/hpc/projects/icd.fast.ops/models/alex_lin_attention/shap_caches/v4/phase"
)
DEFAULT_CACHE_FLUOR = Path(
    "/hpc/projects/icd.fast.ops/models/alex_lin_attention/shap_caches/v4/fluor"
)
# `DEFAULT_OUT_DIR` / `_CHAD_OUT_DIR` are legacy aliases kept only so any
# downstream import-path consumer (e.g. atlas helpers) still resolves.
# New code should index `_TOP_ATTN_OUT_DIR_BY_CONTRAST` directly.
DEFAULT_OUT_DIR = Path(
    "/hpc/projects/icd.fast.ops/models/alex_lin_attention/attention_distinct_geneKO"
)

_CHAD_PHASE_H5AD = _CHAD_BASE / "top_attention_cells_phase.h5ad"
_CHAD_FLUOR_H5AD = _CHAD_BASE / "top_attention_cells_fluor.h5ad"
_CHAD_CACHE_PHASE = Path(
    "/hpc/projects/icd.fast.ops/models/alex_lin_attention/shap_caches/v4_chad/phase"
)
_CHAD_CACHE_FLUOR = Path(
    "/hpc/projects/icd.fast.ops/models/alex_lin_attention/shap_caches/v4_chad/fluor"
)
_CHAD_OUT_DIR = Path(
    "/hpc/projects/icd.fast.ops/models/alex_lin_attention/attention_distinct_chad"
)
# Per-(variant, contrast, grain) out-dir map. Six slots × two grains = 12
# isolated output paths so no approach can accidentally clobber another.
_ALEX_BASE = Path("/hpc/projects/icd.fast.ops/models/alex_lin_attention")
# Naming scheme: {variant}_{contrast}_{grain}.
#   variant ∈ {attention, all_cells}
#   contrast ∈ {distinct, ntc, global}
#   grain ∈ {geneKO, chad}
# Reads at a glance — no version suffixes, no opaque "top20_v4*" or
# "ntc_v2*". Same string also drives the run_all_shap.py orchestrator's
# log file naming.
_TOP_ATTN_OUT_DIR_BY_CONTRAST = {
    "gene": {
        "distinct": _ALEX_BASE / "attention_distinct_geneKO",
        "ntc":      _ALEX_BASE / "attention_ntc_geneKO",
        "global":   _ALEX_BASE / "attention_global_geneKO",
    },
    "complex": {
        "distinct": _ALEX_BASE / "attention_distinct_chad",
        "ntc":      _ALEX_BASE / "attention_ntc_chad",
        "global":   _ALEX_BASE / "attention_global_chad",
    },
}
_ALL_CELLS_OUT_DIR_BY_CONTRAST = {
    "gene": {
        "distinct": _ALEX_BASE / "all_cells_distinct_geneKO",
        "ntc":      _ALEX_BASE / "all_cells_ntc_geneKO",
        "global":   _ALEX_BASE / "all_cells_global_geneKO",
    },
    "complex": {
        "distinct": _ALEX_BASE / "all_cells_distinct_chad",
        "ntc":      _ALEX_BASE / "all_cells_ntc_chad",
        "global":   _ALEX_BASE / "all_cells_global_chad",
    },
}

# All-cells variant defaults. Same per-channel all_cells h5ads serve all
# 3 contrasts × both grain levels — only cache + out paths differ.
DEFAULT_ALL_CELLS_DIR = _ALEX_BASE / "all_cells_v2"
DEFAULT_NTC_CACHE_DIR = _ALEX_BASE / "ntc_caches" / "v2"
DEFAULT_NTC_OUT_DIR   = _ALEX_BASE / "ntc_v2"
_NTC_CHAD_CACHE_DIR   = _ALEX_BASE / "ntc_caches" / "v2_chad"
_NTC_CHAD_OUT_DIR     = _ALEX_BASE / "ntc_v2_chad"

# ---------------------------------------------------------------------------
# Variant-specific naming. Variants:
#   top-attention → ko_shap_features.py (consolidated_v3 top-attention cells)
#   all-cells     → ntc_shap_features.py (per-channel all_cells_v2 h5ads)
# ---------------------------------------------------------------------------
SHARD_PREFIX = {
    "top-attention": "ko_shap_features_shard",
    "all-cells":     "all_cells_shap_features_shard",
}
MERGED_NAME = {
    "top-attention": "ko_shap_features.csv",
    "all-cells":     "all_cells_shap_features.csv",
}


# ---------------------------------------------------------------------------
# Step 1 — shards
# ---------------------------------------------------------------------------
def _count_distinct_genes(cache_phase, cache_fluor):
    """Read the union of gene labels from the phase + fluor cache obs
    parquets so we can size the SLURM array to gene count rather than
    submitting empty shards. Returns 0 on any read failure (caller
    falls back to the user-provided --n-shards).
    """
    import pandas as pd
    from pathlib import Path
    genes = set()
    for cache_dir in (cache_phase, cache_fluor):
        obs_path = Path(cache_dir) / "obs.parquet"
        if not obs_path.exists():
            return 0
        try:
            obs = pd.read_parquet(obs_path, columns=["gene"])
            genes |= set(obs["gene"].astype(str).unique())
        except Exception:
            return 0
    return len(genes)


def _run_shard_distinct(
    phase_h5ad: str,
    fluor_h5ad: str,
    cache_phase: str,
    cache_fluor: str,
    out_dir: str,
    shard: int,
    n_shards: int,
    contrast: str = "distinct",
    no_resume: bool = False,
) -> None:
    """SLURM worker: invoke ko_shap_features.main() with shard args.

    Top-level so cloudpickle can serialize it cleanly. Sets sys.argv
    rather than refactoring ko_shap_features to expose a non-argparse
    entry point.
    """
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parent))
    import ko_shap_features

    argv = [
        "ko_shap_features.py",
        "--phase-h5ad", phase_h5ad,
        "--fluor-h5ad", fluor_h5ad,
        "--cache-phase", cache_phase,
        "--cache-fluor", cache_fluor,
        "--out-dir", out_dir,
        "--shard", str(shard),
        "--n-shards", str(n_shards),
        "--contrast", contrast,
    ]
    if no_resume:
        argv.append("--no-resume")
    _sys.argv = argv
    ko_shap_features.main()


def _run_shard_ntc(
    all_cells_dir: str,
    cache_dir: str,
    out_dir: str,
    shard: int,
    n_shards: int,
    aggregation_level: str,
    chad_config: str | None,
    contrast: str = "distinct",
    no_resume: bool = False,
) -> None:
    """SLURM worker: invoke ntc_shap_features.main() with shard args.
    `contrast` is one of distinct/ntc/global — selects negative-class
    source for the all-cells pipeline (parallel to the top-attention
    pipeline's --contrast flag)."""
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parent))
    import ntc_shap_features

    argv = [
        "ntc_shap_features.py",
        "--all-cells-dir", all_cells_dir,
        "--cache-dir", cache_dir,
        "--out-dir", out_dir,
        "--shard", str(shard),
        "--n-shards", str(n_shards),
        "--contrast", contrast,
        "--aggregation-level", aggregation_level,
    ]
    if chad_config:
        argv += ["--chad-config", chad_config]
    if no_resume:
        argv.append("--no-resume")
    _sys.argv = argv
    ntc_shap_features.main()


def _run_prebuild(h5ad: str, cache_dir: str, rank_filter: str | None) -> None:
    """SLURM worker: build a single per-channel cache.

    One worker per cache_dir means there's exactly one builder per
    cache — no flock contention, no NFS rename-busy retries, no
    cleanup races. Subsequent shap-feature shards just `np.load` the
    `.ready` cache.
    """
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent))
    import ko_shap_features as ks
    ks._build_cache(_Path(h5ad), _Path(cache_dir), rank_type_filter=rank_filter)


def _step0_prebuild_caches(args) -> None:
    """Build every per-channel cache as a SLURM array (one task per channel)
    before submitting the shap-feature shards.

    Without this, 200 shap-feature shards race into `_build_cache` for
    each of ~56 channels at once — even with `fcntl.flock`, GPFS
    rename-busy / NFS stat-caching / open-handle holds from concurrent
    `np.load`s produce sporadic shard failures. Running one SLURM task
    per cache dir means each cache has exactly one builder; later
    shards take the fast `.ready` → `np.load` path with zero contention.

    Distinct: 2 cache dirs (phase, fluor), each <5 GB → 2 short jobs.
    NTC:     ~56 per-channel cache dirs, 5–80 GB each → 56 parallel
             tasks. Wall time = max(channel build) ≈ 10–15 min instead
             of the 30–60 min sequential.
    """
    print(f"\n{'='*60}")
    print(f"STEP 0: {STEPS[0]} ({args.variant})")
    print(f"{'='*60}")

    if args.variant == "top-attention":
        targets = [
            (Path(args.phase_h5ad), Path(args.cache_phase), ""),
            (Path(args.fluor_h5ad), Path(args.cache_fluor), None),
        ]
    else:
        all_cells_dir = Path(args.all_cells_dir)
        cache_root = Path(args.cache_dir)
        targets = []
        for h5ad in sorted(all_cells_dir.glob("all_cells_*.h5ad")):
            stem = h5ad.stem
            ch_key = "phase" if stem == "all_cells_phase" else stem.replace("all_cells_fluor_", "")
            cache_dir = cache_root / ch_key
            targets.append((h5ad, cache_dir, None))

    pending = []
    for h5ad, cache_dir, rank_filter in targets:
        if not h5ad.exists():
            print(f"  [warn] {h5ad} missing — skipping", flush=True)
            continue
        ready = cache_dir / ".ready"
        obs_parquet = cache_dir / "obs.parquet"
        # Cache exists — verify it's not STALE (i.e. h5ad gained rows
        # since the cache was last built, e.g. after
        # consolidate_top_attention_cells added NTC/GLOBAL rows). If
        # stale, clear sentinels here on the login node so the SLURM
        # prebuild job below treats this cache as missing and rebuilds.
        # Without this check, 200 shards each hit the stale cache,
        # detect staleness inside _build_cache, and 199 idle while
        # one rebuilds under flock — wastes ~10 min × 199 CPU-mins.
        if ready.exists() and obs_parquet.exists():
            try:
                import anndata as _ad
                import pandas as _pd
                _src = _ad.read_h5ad(h5ad, backed="r")
                try:
                    n_src = len(_src.obs)
                finally:
                    _src.file.close()
                n_cache = len(_pd.read_parquet(obs_parquet, columns=["gene"]))
                if n_src > n_cache:
                    print(f"  [stale] {cache_dir.name}: cache has {n_cache:,} rows "
                          f"vs h5ad {n_src:,} ({n_src - n_cache:,} new). "
                          f"Clearing sentinels → will rebuild as SLURM job.",
                          flush=True)
                    for f in (ready, obs_parquet,
                              cache_dir / "X.npy",
                              cache_dir / "features.txt",
                              cache_dir / "median.npy",
                              cache_dir / "organelle.txt",
                              cache_dir / "category.txt",
                              cache_dir / "global_std.npy"):
                        try:
                            f.unlink()
                        except FileNotFoundError:
                            pass
                else:
                    print(f"  [skip] {cache_dir.name}: .ready exists "
                          f"(cache={n_cache:,} = h5ad={n_src:,})", flush=True)
                    continue
            except (FileNotFoundError, OSError, ValueError) as e:
                print(f"  [stale-check skipped] {cache_dir.name}: "
                      f"{type(e).__name__}: {e}", flush=True)
                # Fall through — if cache load fails downstream, _build_cache
                # will rebuild under the lock as a last-resort fallback.
                continue
        cache_dir.mkdir(parents=True, exist_ok=True)
        pending.append((h5ad, cache_dir, rank_filter))

    if not pending:
        print("  All caches already have .ready — nothing to build.")
        return

    print(f"  Submitting {len(pending)} SLURM cache-build task(s)...")
    if args.dry_run:
        for h5ad, cache_dir, _ in pending:
            print(f"  [DRY RUN] {cache_dir.name}  <-  {h5ad.name}")
        return

    jobs = [
        {
            "name": f"prebuild_{cache_dir.name[:40]}",
            "func": _run_prebuild,
            "kwargs": dict(
                h5ad=str(h5ad),
                cache_dir=str(cache_dir),
                rank_filter=rank_filter,
            ),
            "metadata": {"channel": cache_dir.name},
        }
        for h5ad, cache_dir, rank_filter in pending
    ]
    # Phase build materializes ~19M cells × ~5k raw features in RAM
    # before NaN-filter, so peak RSS sits around 350-400 GB. Smaller
    # fluor channels (5-15 GB cache) need <50 GB. Sized at 500 GB so
    # the worst case (phase) fits with headroom; small-channel tasks
    # waste their allocation but they finish in 1-2 min so it's fine.
    slurm_params = {
        "timeout_min": 90,
        "slurm_mem": "500GB",
        "cpus_per_task": 4,
        "slurm_partition": args.partition,
    }
    log_subdir = (f"all_cells_shap_prebuild/{args.cache_dir.name}"
                  if args.variant == "all-cells"
                  else f"shap_prebuild/{args.cache_phase.parent.name}")
    experiment_name = ("all_cells_shap_prebuild" if args.variant == "all-cells"
                       else "shap_prebuild")

    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs

    result = submit_parallel_jobs(
        jobs_to_submit=jobs,
        experiment=experiment_name,
        slurm_params=slurm_params,
        log_dir=log_subdir,
        manifest_prefix=experiment_name,
        wait_for_completion=True,
        verbose=True,
    )
    if result.get("failed"):
        raise RuntimeError(f"Cache prebuild failed: {result['failed']}")
    print(f"STEP 0 complete: {len(pending)} caches built.")


def _step1_shards(args) -> None:
    print(f"\n{'='*60}")
    print(f"STEP 1: {STEPS[1]} ({args.variant})")
    print(f"{'='*60}")
    print(f"  Out dir: {args.out_dir}")
    print(f"  Shards:  {args.n_shards}")
    print(f"  Per-shard: {args.cpus_per_task} CPUs, {args.mem}, "
          f"{args.timeout_min}min timeout, partition={args.partition}")

    if args.dry_run:
        print("  [DRY RUN] Would submit SLURM array via submit_parallel_jobs")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.variant == "top-attention":
        # Pre-build (or load) the SHAP caches on the login node BEFORE
        # submitting shards. Without this, all N shards see "no cache"
        # at startup and race on X.npy writes — which can corrupt the
        # file or produce inconsistent reads on NFS. `_build_cache` is a
        # no-op load when the cache already exists (~10s mmap), so this
        # only pays the 5-10 min build cost once per (h5ad, cache_dir)
        # pair.
        print("  Pre-building SHAP caches on login node (one-time cost)...")
        from ko_shap_features import _build_cache
        _build_cache(args.phase_h5ad, args.cache_phase, rank_type_filter="")
        _build_cache(args.fluor_h5ad, args.cache_fluor, rank_type_filter=None)
        print("  Caches ready — submitting shards.")

        # Clamp n_shards to the gene count so we don't submit empty
        # shards — under genes_sorted[shard::n_shards] partitioning,
        # shards beyond gene_count get 0 genes and crash on the empty
        # CSV at post-summary. Discover gene count from the cache obs
        # (already loaded by `_build_cache` above).
        n_genes_distinct = _count_distinct_genes(args.cache_phase, args.cache_fluor)
        effective_n_shards = max(1, min(args.n_shards, n_genes_distinct or args.n_shards))
        if effective_n_shards != args.n_shards:
            print(f"  Clamping --n-shards {args.n_shards} → {effective_n_shards} "
                  f"(matches available gene/complex count)")

        jobs = [
            {
                "name": f"shard{i:02d}",
                "func": _run_shard_distinct,
                "kwargs": dict(
                    phase_h5ad=str(args.phase_h5ad),
                    fluor_h5ad=str(args.fluor_h5ad),
                    cache_phase=str(args.cache_phase),
                    cache_fluor=str(args.cache_fluor),
                    out_dir=str(args.out_dir),
                    shard=i,
                    n_shards=effective_n_shards,
                    contrast=args.contrast,
                    no_resume=args.no_resume,
                ),
                "metadata": {"shard": i, "n_shards": effective_n_shards},
            }
            for i in range(effective_n_shards)
        ]
        experiment_name = "ko_shap_top20"
        log_subdir = f"ko_shap_top20/{args.out_dir.name}"
    else:  # variant == "ntc"
        # Unlike the distinctiveness pipeline, ntc_shap_features builds
        # its caches per-channel inside the worker (one cache subdir per
        # all_cells_<viz_channel>.h5ad). Different shards write
        # different channels concurrently — no NFS race like the
        # single-cache distinctiveness pipeline — so no login-node
        # pre-build needed.
        print(f"  All-cells dir: {args.all_cells_dir}")
        print(f"  Cache dir:     {args.cache_dir}")
        args.cache_dir.mkdir(parents=True, exist_ok=True)

        jobs = [
            {
                "name": f"shard{i:02d}",
                "func": _run_shard_ntc,
                "kwargs": dict(
                    all_cells_dir=str(args.all_cells_dir),
                    cache_dir=str(args.cache_dir),
                    out_dir=str(args.out_dir),
                    shard=i,
                    n_shards=args.n_shards,
                    aggregation_level=args.aggregation_level,
                    chad_config=str(args.chad_config) if args.chad_config else None,
                    contrast=args.contrast,
                    no_resume=args.no_resume,
                ),
                "metadata": {"shard": i, "n_shards": args.n_shards},
            }
            for i in range(args.n_shards)
        ]
        experiment_name = "all_cells_shap"
        log_subdir = f"all_cells_shap/{args.out_dir.name}"

    slurm_params = {
        "timeout_min": args.timeout_min,
        "slurm_mem": args.mem,
        "cpus_per_task": args.cpus_per_task,
        "slurm_partition": args.partition,
    }

    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs

    result = submit_parallel_jobs(
        jobs_to_submit=jobs,
        experiment=experiment_name,
        slurm_params=slurm_params,
        log_dir=log_subdir,
        manifest_prefix=experiment_name,
        wait_for_completion=True,
        verbose=True,
    )
    if result.get("failed"):
        raise RuntimeError(f"SHAP shards failed: {result['failed']}")
    print("STEP 1 complete.")


# ---------------------------------------------------------------------------
# Step 2 — merge
# ---------------------------------------------------------------------------
def _step2_merge(args) -> None:
    print(f"\n{'='*60}")
    print(f"STEP 2: {STEPS[2]} ({args.variant})")
    print(f"{'='*60}")

    shards = sorted(args.out_dir.glob(f"{SHARD_PREFIX[args.variant]}*.csv"))
    print(f"  Shards found: {len(shards)} in {args.out_dir}")
    if args.dry_run:
        print(f"  [DRY RUN] Would concat into {MERGED_NAME[args.variant]}")
        return
    if not shards:
        raise FileNotFoundError(
            f"No shard CSVs in {args.out_dir} (looking for "
            f"{SHARD_PREFIX[args.variant]}*.csv) — run step 1 first."
        )

    if args.variant == "top-attention":
        # Reuse the distinctiveness merge script — it's pinned to the
        # ko_ filename pattern and already knows about the post-merge
        # AUROC summary.
        merge_script = SCRIPT_DIR / "merge_shap_shards.py"
        subprocess.run(
            [sys.executable, str(merge_script), "--out-dir", str(args.out_dir)],
            check=True,
        )
    else:  # ntc — inline merge (5-line concat, sort by contrast first)
        import pandas as pd
        # Skip shards that are 0 bytes OR have content but no parseable
        # rows (e.g. only headers, or whitespace from a killed write).
        # Both arise legitimately when a shard's gene subset was empty
        # after the auto-clamp-n_shards-to-gene-count logic, or when a
        # shard timed out mid-write. EmptyDataError is the pandas signal
        # for "file readable but no CSV content".
        good_frames = []
        n_empty = 0
        for s in shards:
            if s.stat().st_size == 0:
                n_empty += 1
                continue
            try:
                good_frames.append(pd.read_csv(s))
            except pd.errors.EmptyDataError:
                n_empty += 1
                continue
        if n_empty:
            print(f"  [merge] skipped {n_empty} empty/unreadable shards "
                  f"(typical when shard count > gene count, or when a "
                  f"timed-out shard left a header-only file).")
        if not good_frames:
            raise RuntimeError(
                f"All {len(shards)} shards empty/unreadable — nothing "
                f"to merge. Re-run step 1 to regenerate shard CSVs."
            )
        df = pd.concat(good_frames, ignore_index=True) \
               .sort_values(["contrast", "gene", "shap_rank"])
        out = args.out_dir / MERGED_NAME[args.variant]
        df.to_csv(out, index=False)
        print(f"  Merged {len(good_frames)}/{len(shards)} shards → "
              f"{df['gene'].nunique()} genes, "
              f"contrasts: {sorted(df['contrast'].astype(str).unique())}")
        print(f"  Saved: {out}")
    print("STEP 2 complete.")


# ---------------------------------------------------------------------------
# Step 3 — captions
# ---------------------------------------------------------------------------
def _step3_captions(args) -> None:
    print(f"\n{'='*60}")
    print(f"STEP 3: {STEPS[3]} ({args.variant})")
    print(f"{'='*60}")

    features_csv = args.out_dir / MERGED_NAME[args.variant]
    print(f"  Features in:  {features_csv}")

    if args.dry_run:
        print("  [DRY RUN] Would run generate_shap_captions_combined.py")
        return
    if not features_csv.exists():
        raise FileNotFoundError(f"Missing {features_csv} — run step 2 first.")

    captions_script = SCRIPT_DIR / "generate_shap_captions_combined.py"

    # One captions CSV per run — each invocation handles exactly ONE
    # (variant × contrast × grain), so we never emit captions for a
    # contrast we didn't compute. Filenames follow the variant's
    # naming scheme but include the contrast suffix for all-cells
    # (which can run any of 3 contrasts per out_dir).
    if args.variant == "top-attention":
        # Out_dir is already per-contrast, so a fixed filename is fine.
        contrasts_to_emit = [(args.contrast, args.out_dir / "ko_shap_captions.csv")]
    else:  # all-cells — suffix with contrast for safety even though out_dir is per-contrast.
        contrasts_to_emit = [(args.contrast,
                              args.out_dir / f"all_cells_shap_captions_{args.contrast}.csv")]

    for contrast, captions_csv in contrasts_to_emit:
        print(f"  Captions out [{contrast}]: {captions_csv}")
        cmd = [
            sys.executable, str(captions_script),
            "--features", str(features_csv),
            "--captions", str(captions_csv),
            "--contrast", contrast,
        ]
        if args.include_counts and args.variant == "top-attention":
            # Counts read from the SHAP cache to compute abundance deltas.
            # NTC variant doesn't have a single merged cache yet, so
            # --include-counts is silently distinctiveness-only.
            cmd.extend([
                "--include-counts",
                "--cache-phase", str(args.cache_phase),
                "--cache-fluor", str(args.cache_fluor),
            ])
            print("  --include-counts: caption tails will use SHAP cache",
                  flush=True)
        subprocess.run(cmd, check=True)

    print("STEP 3 complete.")


# ---------------------------------------------------------------------------
# Step 4 — pick representative cells for atlas image rows (all-cells only)
# ---------------------------------------------------------------------------
def _picker_out_paths(args) -> tuple[Path, Path]:
    """Output CSV paths for the picker — same out_dir as captions + SHAP CSV
    so a variant directory is self-contained for the atlas step."""
    return (
        args.out_dir / f"picked_phase_{args.contrast}.csv",
        args.out_dir / f"picked_fluor_{args.contrast}.csv",
    )


def _step4_pick_cells(args) -> None:
    """Run `ntc_pick_cells.py` once per all-cells variant.

    Picker reads the merged SHAP features CSV + the per-channel cache
    dir and emits one phase CSV + one fluor CSV containing both KO
    rows (rank_type=='top', `target_gene=={gene}`) and KO-typical NTC
    rows (rank_type=='ntc_ko_typical', same target_gene). The atlas
    step then consumes both as `--phase-csv` / `--fluor-csv` and as
    `--ntc-pma-{phase,fluor}-csv` (single CSV serves both via the
    `target_gene` + `rank_type` filters in `_sample_ntc_rows`).

    Auto-skipped for top-attention runs — those keep the PMA top-
    attention FOVs as image rows (consistent across the 3 contrasts
    so the atlas pages line up for side-by-side comparison).
    """
    print(f"\n{'='*60}")
    print(f"STEP 4: {STEPS[4]} ({args.variant})")
    print(f"{'='*60}")

    if args.variant != "all-cells":
        print("  variant=top-attention — picker is a no-op; "
              "atlas keeps PMA top-attention FOVs. Skipping.")
        return

    features_csv = args.out_dir / MERGED_NAME[args.variant]
    out_phase, out_fluor = _picker_out_paths(args)
    if not features_csv.exists():
        raise FileNotFoundError(
            f"Missing features CSV ({features_csv}) — run step 2 first."
        )

    cmd = [
        sys.executable, str(SCRIPT_DIR / "ntc_pick_cells.py"),
        "--shap-features-csv", str(features_csv),
        "--cache-dir",         str(args.cache_dir),
        "--out-phase-csv",     str(out_phase),
        "--out-fluor-csv",     str(out_fluor),
        "--contrast",          args.contrast,
        "--aggregation-level", args.aggregation_level,
    ]
    if args.aggregation_level == "complex" and args.chad_config:
        cmd += ["--chad-config", str(args.chad_config)]

    print(f"  Features:  {features_csv}")
    print(f"  Cache:     {args.cache_dir}")
    print(f"  Out phase: {out_phase}")
    print(f"  Out fluor: {out_fluor}")
    if args.dry_run:
        print("  [DRY RUN] would run: " + " ".join(cmd))
        return
    subprocess.run(cmd, check=True)
    print("STEP 4 complete.")


# ---------------------------------------------------------------------------
# Step 5 — smoke atlas
# ---------------------------------------------------------------------------
def _step5_smoke_atlas(args) -> None:
    print(f"\n{'='*60}")
    print(f"STEP 5: {STEPS[5]} ({args.variant})")
    print(f"{'='*60}")

    if args.smoke_genes <= 0:
        print("  smoke_genes=0 — skipping atlas step.")
        return

    features_csv = args.out_dir / MERGED_NAME[args.variant]

    # Smoke atlas matches the run's actual contrast (used to be locked
    # to NTC for the all-cells variant — now it follows --contrast).
    if args.variant == "top-attention":
        captions_csv = args.out_dir / "ko_shap_captions.csv"
    else:  # all-cells
        captions_csv = args.out_dir / f"all_cells_shap_captions_{args.contrast}.csv"
    atlas_contrast = args.contrast
    output_pdf = args.out_dir / (
        f"atlas_smoke_{args.contrast}_top{args.smoke_genes}.pdf"
    )

    print(f"  Features:  {features_csv}")
    print(f"  Captions:  {captions_csv}")
    print(f"  Output:    {output_pdf}")

    if args.dry_run:
        print(f"  Pages:     {args.smoke_genes} (1 worker per page)")
        return
    if not features_csv.exists() or not captions_csv.exists():
        raise FileNotFoundError(
            "Missing features or captions CSV — run earlier steps first."
        )
    if not ATLAS_SCRIPT.exists():
        raise FileNotFoundError(f"Atlas script not found: {ATLAS_SCRIPT}")

    # Pre-select the smoke genes (top-N by best AUROC) and pass them
    # explicitly via `--genes`. Without this filter, the atlas does its
    # SHAP-data prep over ALL 1000 genes on the login node before
    # submitting SLURM jobs — minutes of wasted work for a smoke run.
    import pandas as pd
    cap_df = pd.read_csv(captions_csv)
    smoke_genes = cap_df["gene"].astype(str).head(args.smoke_genes).tolist()
    print(f"  Pages:     {len(smoke_genes)} (1 SLURM worker per page)")
    print(f"  Genes:     {', '.join(smoke_genes)}")

    cmd = [
        sys.executable, str(ATLAS_SCRIPT),
        "--genes", *smoke_genes,
        "--workers", str(len(smoke_genes)),
        "--output", str(output_pdf),
        "--shap-features-csv", str(features_csv),
        "--shap-captions-csv", str(captions_csv),
        "--contrast", atlas_contrast,
        "--aggregation-level", args.aggregation_level,
        "--no-strict",
    ]
    if args.smoke_skip_cp_4i:
        cmd.append("--skip-cp-4i")
    if args.variant == "top-attention":
        cmd.extend([
            "--shap-cache-phase", str(args.cache_phase),
            "--shap-cache-fluor", str(args.cache_fluor),
        ])
    else:
        # All-cells: use the step-4 picker outputs as image-row sources.
        # Same CSV feeds both the KO image row (`--{phase,fluor}-csv`
        # → filter to rank_type=='top') and the NTC strip pool
        # (`--ntc-pma-{phase,fluor}-csv` → filter to
        # rank_type=='ntc_ko_typical' + target_gene==current_gene).
        picker_phase, picker_fluor = _picker_out_paths(args)
        if picker_phase.exists() and picker_fluor.exists():
            cmd.extend([
                "--phase-csv",            str(picker_phase),
                "--fluor-csv",            str(picker_fluor),
                "--ntc-pma-phase-csv",    str(picker_phase),
                "--ntc-pma-fluor-csv",    str(picker_fluor),
            ])
            print(f"  Image rows: picker outputs "
                  f"({picker_phase.name}, {picker_fluor.name})")
        else:
            raise FileNotFoundError(
                f"Picker outputs missing — expected {picker_phase} + "
                f"{picker_fluor}. Run step 4 first (or `--from-step 4`)."
            )
    if args.smoke_no_ntc:
        cmd.append("--no-ntc")
    else:
        cmd.extend([
            "--ntc-experiments", str(args.smoke_ntc_experiments),
            "--ntc-per-experiment", str(args.smoke_ntc_per_experiment),
        ])
    # Pass-through bucket for any atlas flag we don't expose explicitly
    # (e.g. `--map-channel-threshold 0.2 --max-fluor-channels 10` for
    # variable-page-size). Appended last so it can override anything
    # set above.
    if getattr(args, "extra_atlas_args", None):
        cmd.extend(args.extra_atlas_args)
        # Mirror run_all_atlases.py: tag variable-page-size output PDFs.
        if "--map-channel-threshold" in args.extra_atlas_args:
            idx = args.extra_atlas_args.index("--map-channel-threshold")
            try:
                if float(args.extra_atlas_args[idx + 1]) > 0:
                    new_out = output_pdf.with_name(
                        f"{output_pdf.stem}_variable{output_pdf.suffix}"
                    )
                    # Replace --output value in cmd
                    out_idx = cmd.index("--output")
                    cmd[out_idx + 1] = str(new_out)
                    output_pdf = new_out
            except (ValueError, IndexError):
                pass
    subprocess.run(cmd, check=True)
    print(f"STEP 5 complete. PDF: {output_pdf}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full sharded SHAP pipeline: shards → merge → captions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--variant", choices=("top-attention", "all-cells"), default="top-attention",
        help="Which extraction pipeline to read positives + negatives from:\n"
             "  • top-attention (default): ko_shap_features.py against the "
             "consolidated_v3{,_chad}/top_attention_cells_{phase,fluor}.h5ad "
             "files. ~100 top-attention cells per (gene, channel).\n"
             "  • all-cells: ntc_shap_features.py against per-channel "
             "all_cells_v2/all_cells_<channel>.h5ad files. Up to "
             "~2000 cells/(sgRNA, channel) — broader population signal.",
    )
    parser.add_argument(
        "--contrast", choices=("distinct", "ntc", "global"),
        default="distinct",
        help="Negative-class source. Positives are fixed by --variant; this "
             "selects what they're compared AGAINST:\n"
             "  • distinct: other-gene cells (excluding NTCs).\n"
             "  • ntc: NTC cells.\n"
             "  • global: random / all-non-this-gene cells.\n"
             "Each (variant × contrast) writes to its own out-dir so all 6 "
             "approaches coexist on disk.",
    )
    parser.add_argument(
        "--from-step", type=int, default=0, choices=[0, 1, 2, 3, 4, 5],
        help="Start from this step (0=prebuild caches as SLURM array, "
             "1=shards, 2=merge, 3=captions, 4=pick_cells [all-cells only, "
             "auto-skipped for top-attention], 5=smoke atlas).",
    )
    parser.add_argument(
        "--smoke-genes", type=int, default=5,
        help="Number of pages for the step-5 smoke atlas (one worker per "
             "page). Set to 0 to skip the atlas step entirely.",
    )
    parser.add_argument(
        "--include-counts", action="store_true",
        help="Pass through to step 3 captions (distinct variant only): "
             "append abundance tails to each channel section using the "
             "SHAP cache. Silently a no-op for --variant ntc until a "
             "merged NTC SHAP cache is available.",
    )
    parser.add_argument(
        "--smoke-no-ntc", action="store_true", default=False,
        help="Skip NTC pool loading during the smoke atlas. Default is "
             "to include NTCs so the smoke output matches the full "
             "atlas layout (NTC strip + CP rows visible).",
    )
    parser.add_argument(
        "--smoke-with-ntc", dest="smoke_no_ntc", action="store_false",
        help="(Default) Include NTC strip in smoke atlas. Pass "
             "--smoke-no-ntc to skip if you only need the SHAP layer.",
    )
    parser.add_argument(
        "--smoke-skip-cp-4i", action="store_true", default=False,
        help="Drop CP/4i imaging rows from the smoke atlas. Off by "
             "default so the smoke matches the full atlas (CP TOMM20, "
             "CP/4i markers visible). Turn on for a faster spot-check "
             "that only renders standard fluor channels.",
    )
    parser.add_argument("--smoke-ntc-experiments", type=int, default=5)
    parser.add_argument("--smoke-ntc-per-experiment", type=int, default=200)
    parser.add_argument(
        "--extra-atlas-args", nargs=argparse.REMAINDER, default=[],
        help="Trailing flags forwarded verbatim to attention_atlas_shap.py "
             "in step 5. Use for `--map-channel-threshold 0.2 "
             "--max-fluor-channels 10` (variable page size) or any other "
             "atlas flag the pipeline doesn't expose explicitly. Must be "
             "the LAST positional bucket — everything after this tag is "
             "absorbed. The pipeline auto-tags the output PDF `_variable` "
             "when `--map-channel-threshold > 0` is present.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-resume", action="store_true",
        help=(
            "Re-process every gene/complex from scratch even if a "
            "per-shard CSV already exists in --out-dir. Default "
            "behavior is to skip genes already listed in the existing "
            "shard CSV — which silently masks bug fixes that should "
            "add NEW channels/rows to those genes (e.g. the 4i-marker "
            "slug fix). Pass this flag whenever you've changed the "
            "neg-pool resolution or feature set and need a clean rerun."
        ),
    )
    parser.add_argument("--no-qos", action="store_true",
                        help="Skip setting OPS_SLURM_QOS=icd.ops.processing.")

    # SLURM tunables. Defaults differ per variant — set after parsing.
    # Tuned for the post `--channel-rank-max=None` workload, where each
    # gene now carries cells from all ~56 fluor channels (vs the legacy
    # 3) → ~18× more SHAP fits per gene at the gene-level distinctiveness
    # scale (1000 genes × 56 channels = ~56K classifiers). Bumped from
    # the old (50, 24GB, 10min) tuning, which would burn through the
    # timeout on the very first shard. CHAD distinctiveness has fewer
    # items (90 complexes) but the same per-item compute, so the same
    # defaults work — just finishes faster.
    parser.add_argument("--n-shards",      type=int, default=200)
    parser.add_argument("--cpus-per-task", type=int, default=8)
    parser.add_argument("--mem",           default=None,
                        help="Per-shard memory (default: 64GB distinct, 64GB ntc).")
    parser.add_argument("--timeout-min",   type=int, default=None,
                        help="Per-shard timeout (default: 60 distinct, 120 ntc).")
    parser.add_argument("--partition",     default="cpu")

    # Distinctiveness-variant paths.
    parser.add_argument("--phase-h5ad",  type=Path, default=DEFAULT_PHASE_H5AD)
    parser.add_argument("--fluor-h5ad",  type=Path, default=DEFAULT_FLUOR_H5AD)
    parser.add_argument("--cache-phase", type=Path, default=DEFAULT_CACHE_PHASE)
    parser.add_argument("--cache-fluor", type=Path, default=DEFAULT_CACHE_FLUOR)

    # All-cells variant paths (only the all-cells SHAP step consumes these).
    parser.add_argument("--all-cells-dir", type=Path, default=DEFAULT_ALL_CELLS_DIR)
    parser.add_argument("--cache-dir",     type=Path, default=DEFAULT_NTC_CACHE_DIR)

    # Common output (default depends on variant + grain — set after parsing).
    parser.add_argument("--out-dir",     type=Path, default=None)

    parser.add_argument(
        "--aggregation-level", choices=("gene", "complex"), default="gene",
        help="Run SHAP at gene level (default) or CHAD protein-complex level. "
             "When 'complex' AND user kept variant defaults, paths auto-swap "
             "to the _chad variants. For --variant distinct, build inputs "
             "first via `consolidate_top_attention_cells --aggregation-level "
             "complex`.",
    )
    from organelle_profiler.feature_extraction.consolidate_top_attention_cells import (
        DEFAULT_CHAD_CONFIG,
    )
    parser.add_argument(
        "--chad-config", type=Path, default=DEFAULT_CHAD_CONFIG,
        help=f"CHAD positive_controls YAML (default: {DEFAULT_CHAD_CONFIG.name}).",
    )

    args = parser.parse_args()

    # Variant-specific resource defaults. Both variants now budget for
    # the all-channels (~56) workload:
    #   * distinct: 1000 genes × ~56 channels = ~56K SHAP fits;
    #     consolidated_v3 fluor h5ad is ~17GB on disk → 13GB float32
    #     in memory + per-shard cohort accumulators + LightGBM working
    #     set. 96GB is the comfortable headroom; 64GB OOM'd on the
    #     full v3 build at the cohort-mean step.
    #   * ntc: per-channel all_cells h5ads (~3M cells × ~1200 features
    #     each) need similar memory; longer timeout because each shard
    #     iterates ~57 h5ads sequentially.
    if args.mem is None:
        args.mem = "96GB"
    if args.timeout_min is None:
        # Bumped NTC 120 → 240. Most shards finish in 50-90 min, but the
        # tail (1-2% of shards landing on slow nodes or processing fat
        # complexes with 50k+ NTC cells) can hit the 2h cap and be SIGTERMed
        # mid-write — leaving a 0-byte output CSV that any resume attempt
        # then trips over. 4h budget covers the 99th percentile while
        # leaving the same wall-clock ceiling well below GPU job costs.
        args.timeout_min = 60 if args.variant == "top-attention" else 240

    # Auto-swap defaults per (variant, contrast, grain). Skip swap if the
    # user explicitly passed any of the path args — they get exactly what
    # they specified.
    if args.variant == "top-attention":
        if args.aggregation_level == "complex":
            if args.phase_h5ad == DEFAULT_PHASE_H5AD:
                args.phase_h5ad = _CHAD_PHASE_H5AD
            if args.fluor_h5ad == DEFAULT_FLUOR_H5AD:
                args.fluor_h5ad = _CHAD_FLUOR_H5AD
            if args.cache_phase == DEFAULT_CACHE_PHASE:
                args.cache_phase = _CHAD_CACHE_PHASE
            if args.cache_fluor == DEFAULT_CACHE_FLUOR:
                args.cache_fluor = _CHAD_CACHE_FLUOR
            print(f"[CHAD] aggregation-level=complex; "
                  f"caches={args.cache_phase.parent}")
        if args.out_dir is None:
            args.out_dir = _TOP_ATTN_OUT_DIR_BY_CONTRAST[args.aggregation_level][args.contrast]
    else:  # all-cells
        if args.aggregation_level == "complex":
            if args.cache_dir == DEFAULT_NTC_CACHE_DIR:
                args.cache_dir = _NTC_CHAD_CACHE_DIR
            print(f"[CHAD] aggregation-level=complex; cache-dir={args.cache_dir}")
        if args.out_dir is None:
            args.out_dir = _ALL_CELLS_OUT_DIR_BY_CONTRAST[args.aggregation_level][args.contrast]
    print(f"[{args.variant}/{args.contrast}/{args.aggregation_level}] "
          f"out → {args.out_dir}")

    if not args.no_qos:
        os.environ.setdefault("OPS_SLURM_QOS", "icd.ops.processing")

    steps_to_run = [s for s in sorted(STEPS) if s >= args.from_step]
    print(f"SHAP pipeline ({args.variant}, {args.aggregation_level}) → {args.out_dir}")
    print(f"Steps: {', '.join(f'{s}. {STEPS[s]}' for s in steps_to_run)}")
    if args.dry_run:
        print("[DRY RUN MODE]")

    try:
        if 0 in steps_to_run:
            _step0_prebuild_caches(args)
        if 1 in steps_to_run:
            _step1_shards(args)
        if 2 in steps_to_run:
            _step2_merge(args)
        if 3 in steps_to_run:
            _step3_captions(args)
        if 4 in steps_to_run:
            _step4_pick_cells(args)
        if 5 in steps_to_run:
            _step5_smoke_atlas(args)
    except (RuntimeError, FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"\nPipeline ABORTED: {e}", file=sys.stderr)
        sys.exit(1)

    print("\nDone.")


if __name__ == "__main__":
    main()
