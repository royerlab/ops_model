# Titration

How does phenotypic signal degrade as you spend fewer cells? These scripts take the
cell-level PCA-reduced h5ads that `pca_optimization` produced, repeatedly subsample
them at decreasing cell budgets, re-aggregate to guide level, and score mAP metrics at
each point — producing "score vs cells" curves.

Both scripts score the same five metrics: **activity**, **distinctiveness** (guide
level), and **corum**, **chad**, **ebi** (gene level, sharing a similarity cache).
The set is defined once by `METRICS` in `titration.py`; `METRIC_COLUMNS` and the CSV
schema derive from it.

## Files

| file | what it does |
|---|---|
| `titration.py` | **Source of truth.** Titrates each reporter *independently* — one curve per reporter. Owns every shared primitive (samplers, scoring, schedule ladder, caching, CSV merge, plotting, and the CLI flags that mirror `pca_optimization`'s path nesting). |
| `combined_titration.py` | Titrates *groups* of reporters **together** — subsamples N reporters at each budget, h-concatenates their NTC-normalized guide features into one matrix, and scores that. Answers "how does the whole cp panel degrade?" rather than "how does each marker degrade?". Imports its primitives from `titration.py` and adds only group resolution, the h-concat step, an optional 2nd-pass PCA, and cross-group comparison plots. |

## Running `titration.py`

```bash
uv run python -m ops_model.post_process.combination.titration.titration \
    -o <pca_optimization_root> \
    --cell-dino --paper-v1 --with-cp --with-4i --fixed-threshold 0.8 \
    --per-guide-median-titration --slurm
```

The path-resolution flags (`--cell-dino`, `--paper-v1`, `--with-cp`, `--with-4i`,
`--only-cp`, `--only-4i`, `--fixed-threshold`, `--distance`, `--run-tag`,
`--phase-only` / `--no-phase`, …) mirror `pca_optimization` exactly and must match the
run whose `per_signal/*_cells.h5ad` you want to titrate. **`-o/--output-dir` is
required** and takes the same root you passed to `pca_optimization`; the other flags
select the variant beneath it. `--paper-v1` is a bare toggle here — it only nests the
output path, so unlike `pca_optimization` it takes no experiment-list value.

**Titration mode picks the schedule and the output subdir** — pick exactly one:

| flag | schedule starts at | output subdir |
|---|---|---|
| `--per-guide-median-titration` | median non-NTC cells/guide | `titration_guide_median/` |
| `--per-guide-max-titration` | p90 non-NTC cells/guide | `titration_per_guide/` |
| `--per-guide-min-titration` | min non-NTC cells/guide | `titration_guide_min/` |
| `--per-ko-max-titration` | p90 non-NTC cells/KO | `titration_geneKO_max/` |
| `--per-ko-min-titration` | min non-NTC cells/KO | `titration_geneKO_min/` |
| `--min-exp-titration` | total cells, fewest experiments first | `titration_min_exp/` |
| *(none)* | total cells, down to `MIN_CELLS` (5,000) | `titration/` |

Per-guide and per-KO ladders step down by ×0.75 to 1; the total-cells ladder stops at
`MIN_CELLS`.

Other useful flags:

- `--slurm` — fan out on the cluster. `--per-target-slurm` is **on by default**: one
  task per (reporter, target) bin, merged into the canonical CSV afterwards. Use
  `--no-per-target-slurm` for one job per reporter. (Local runs fall back to
  one-reporter-at-a-time automatically, with a message.)
- `--bootstrap N` — N draws per point, adding `_std`/`_sem` columns and error bars.
  `--bootstrap-replace` makes it a true with-replacement bootstrap (per-guide only).
- `--replot` — regenerate every plot from existing CSVs, no re-scoring.
- `--no-cache` — re-score points already present in the CSV. By default a point is
  skipped only if every metric column is populated.

**Outputs**, under `<variant>/<titration_subdir>/`:

```
<reporter>/<reporter>_titration.csv                  one row per titration point
<reporter>/<reporter>_titration_<xaxis>_<scale>.png|svg
titration_combined.csv                               all reporters concatenated
titration_combined_<xaxis>_<scale>.png|svg           all-reporter overview
```

`<xaxis>` is one of `totalcells` / `perpert` / `perguide`, `<scale>` one of
`linear` / `log2` / `log10`.

## Running `combined_titration.py`

```bash
uv run python -m ops_model.post_process.combination.titration.combined_titration \
    -o <pca_optimization_root> \
    --cell-dino --paper-v1 --with-cp --with-4i \
    --per-guide-median-titration --slurm \
    --group pRb=<per_signal>/pRb_4i_cells.h5ad \
    --group p21=<per_signal>/p21_4i_cells.h5ad \
    --group pRb_p21=<per_signal>/pRb_4i_cells.h5ad,<per_signal>/p21_4i_cells.h5ad
```

That example answers "does combining these two markers beat either alone?" — each
group gets its own curve and all three are overlaid.

Note the split: the `--group` paths say what is **read**; `-o` plus the
path-resolution flags say where output is **written**. They're independent, so
passing the wrong channel-set flags silently relocates your results.

Inherits all of `titration.py`'s flags (same path resolution, same mode flags, same
SLURM and bootstrap options) and adds:

- `--group NAME=path1[,path2,...]` — **required, repeatable.** A named set of
  `*_cells.h5ad` files to combine. `NAME` becomes the output directory leaf, so it
  must be unique within a run and contain no path separator. One group per flag; a
  group with a single path titrates that reporter on its own.
- `--second-pca-threshold 0.4` — fit a 2nd-pass PCA on the concatenated guide matrix
  at every step, keeping components to that cumulative variance. Skipped for
  single-reporter groups. Scopes output into a `_sec40` suffix so thresholds don't
  clobber each other.
- `--median-start-policy pool|max_reporter` and `--no-shared-start` — control where
  each group's median-mode schedule starts. By default groups share the smallest
  median so curves align at the top of the x-axis.
- `--n-workers` — threads for the per-reporter prep loop inside each step (defaults to
  `--slurm-cpus`, which this script raises to 32).
- `--max-schedule-points N` — keep only the top N (largest) targets.
- `--no-compare` / `--compare-only` — skip, or only regenerate, the cross-group
  comparison plots. With ≥2 groups the comparison runs automatically. `--compare-only`
  needs the group *names* but not their input paths, so pass `--group NAME=` with any
  still-valid path.

**Outputs:**

```
<variant>/combined_titration/<mode>/<NAME>[_secNN]/combined_titration_<NAME>.csv
<variant>/combined_titration_compare/<mode>/<A_vs_B_vs_C>/compare_all_metrics_*.png|svg|csv
<variant>/combined_titration_compare/<mode>/<A_vs_B_vs_C>/groups.txt   # name -> reporters
```

## Downstream

These scripts are used to generate the raw data consumed by the paper figure notebook:

[`notebooks/figure_3/reporter_titration_plots.ipynb`](https://github.com/czbiohub-sf/ops-paper-analysis/blob/main/notebooks/figure_3/reporter_titration_plots.ipynb)
in [`czbiohub-sf/ops-paper-analysis`](https://github.com/czbiohub-sf/ops-paper-analysis)



## Notes

- Both scripts are cache-aware and idempotent — re-running fills in only the missing
  targets, so an interrupted or extended run is cheap to resume.
- Scoring is deterministic for a given `--seed`/`random_seed`, including the copairs
  null sampling. Two runs on the same inputs give bit-identical CSVs.
- `ops_utils` on `origin/main` is sufficient. (The old requirement to check out
  `feat/ebi-plus-map-score` went away when the `ebi_plus` metric was dropped;
  pre-2026-08-03 CSVs may still carry unused `ebi_plus_*` columns.)
- To add or remove a metric, edit `METRICS` in `titration.py` — the CSV schema, the
  result dict, the cache-completeness check, and the plot grids all derive from it.
