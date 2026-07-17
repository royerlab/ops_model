# `combination/` — combining experiments into multi-experiment profiles

This package combines per-experiment, cell-level feature embeddings (DINO / cell-DINO /
CellProfiler / …) from many OPS experiments into **combined guide- and gene-level AnnData
objects**, scored on phenotypic metrics. The supported pipeline is the **`pca_optimization`
subpackage**.

- **How it works (internals / data flow):** [`pca_optimization_dataflow.md`](pca_optimization_dataflow.md)
- **What every file in this dir is (core vs supplemental):** [`SCRIPT_MAP.md`](SCRIPT_MAP.md)

> The older config-driven `cli.py` / `baseline.yml` path (`run_combination`,
> `PcaOptimizationCombiner`, `ComprehensiveCombiner`, …) has been **removed**. Use the
> argparse subpackage below. (History: `SCRIPT_MAP.md` §3.)

---

## What it produces

A two-phase pipeline:

- **Phase 1** — one job per biological signal group: pool cells across experiments that share
  that signal → (optional z-score / downsample) → fit PCA → pick `n_pcs` (variance sweep or
  fixed cutoff) → aggregate to guide/gene → save a per-signal h5ad.
- **Phase 2** — one aggregation job: load the per-signal h5ads, horizontally concatenate, NTC-
  normalize, aggregate to gene level, score metrics (activity / distinctiveness / CORUM / CHAD /
  EBI), compute UMAP+PHATE, and write the canonical outputs.

Canonical Phase 2 outputs (under the resolved output dir, see [Output layout](#output-layout)):
- `guide_pca_optimized.h5ad` — combined guide-level profiles
- `gene_embedding_pca_optimized.h5ad` — combined gene-level profiles + UMAP/PHATE
- `pca_report.csv`, `metrics/`, `plots/`, and (default) `second_pca_consensus/`

---

## Entry point

```bash
python -m ops_model.post_process.combination.pca_optimization [FLAGS]
```

Full flag reference (60+ options):

```bash
python -m ops_model.post_process.combination.pca_optimization --help
```

**Exactly one feature-mode flag is required** (no implicit default):
`--cell-dino` · `--dino` · `--cell-profiler` · `--dynaclr` · `--subcell` · `--organelle-profiler`.

### Run from a config file

Instead of (or alongside) CLI flags, pass a YAML config with `--config`:

```bash
python -m ops_model.post_process.combination.pca_optimization --config my_run.yml
```

The config keys are the **same arguments**, written as their snake_case names
(`--cell-dino` → `cell_dino`, `--phase-only` → `phase_only`, `--output-dir` → `output_dir`).
Config values populate the defaults; **any flag passed explicitly on the command line still
overrides the config** (e.g. `--config my_run.yml --no-slurm`). To turn off a default-on flag
in the config, set it `false` (e.g. `second_pca: false`). Unknown keys are rejected.

A worked example (the validation-cohort run) is at
[`pca_optimization/example_config.yml`](pca_optimization/example_config.yml):

```yaml
cell_dino: true
output_dir: /hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized_v0.3
experiments: ops0146,ops0147,ops0150,ops0151
phase_only: true
fixed_threshold: 0.80
slurm: true
```

Programmatic equivalent:

```python
from ops_model.post_process.combination.pca_optimization import run_from_config
run_from_config("my_run.yml")
```

### Combine embeddings outside the experiment structure

If your cell-level embeddings don't live in the standard
`…/3-assembly/<feature_dir>/anndata_objects/` layout, add a **`signal_paths`** mapping to the
config (no separate flag). It maps each signal-group name to one h5ad path, or a list of paths
that get **pooled**:

```yaml
cell_dino: true                      # still pick a feature mode (for metadata/labels)
output_dir: /path/to/out
signal_paths:
  Phase: /data/runA/phase.h5ad       # one file = one signal
  MAP4:                              # multiple files pooled into one signal
    - /data/runA/map4.h5ad
    - /data/runB/map4.h5ad
```

Each h5ad must have the **same schema as the discovery `features_processed_*.h5ad`** (obs with
`sgRNA` / `perturbation` / `experiment`; `X` = the embedding matrix). When `signal_paths` is set,
experiment discovery is skipped, every other option (PCA threshold, normalization, downsampling,
SLURM, second-pass PCA, …) applies as usual, and output lands under `<output_dir>/external/`.

---

## Inputs

For the discovery-based feature modes, the pipeline scans the standard storage roots for each
experiment's cell-level h5ads at:

```
<storage_root>/<experiment>/3-assembly/<feature_dir>/anndata_objects/features_processed_<signal>.h5ad
```

`<feature_dir>` per mode: `cell_dino_features` (`--cell-dino`), `dino_features` (`--dino`),
`cell-profiler` (`--cell-profiler`), `dynaclr_features`, `subcell_features`. Channels are mapped
to biological-signal groups via the channel maps
(`/hpc/projects/icd.fast.ops/configs/ops_channel_maps.yaml`).
`--organelle-profiler` instead reads consolidated `all_cells_*.h5ad` files from `--op-root`.

Restrict the experiment set with `--experiments ops0100,ops0105,…`, or `--paper-v1`
(the curated `good_experiment_list_v1.yml`). **Always start with `--dry-run`** to print the
discovered signal-group manifest without processing.

---

## Quick start

**0. Dry run** — see what would be processed (no compute):
```bash
python -m ops_model.post_process.combination.pca_optimization \
    --cell-dino --phase-only \
    --experiments ops0100,ops0105,ops0117,ops0119,ops0120 \
    --dry-run
```

**1. Local run** (no SLURM) — small/quick combine, fixed PCA threshold:
```bash
python -m ops_model.post_process.combination.pca_optimization \
    --cell-dino --phase-only \
    --experiments ops0100,ops0105,ops0117,ops0119,ops0120 \
    --fixed-threshold 0.80 \
    --output-dir /hpc/projects/icd.fast.ops/experiments/<you>/combine_test \
    -y
```
Omitting `--slurm` runs Phase 1 + Phase 2 in-process.

**2. SLURM run** — production combine (one Phase-1 job per signal + one Phase-2 job):
```bash
python -m ops_model.post_process.combination.pca_optimization \
    --cell-dino --zscore-per-experiment \
    --paper-v1 \
    --slurm
```

**3. Validation cohort** (4 experiments, Phase-only, custom CHAD file) — from the module docstring:
```bash
python -m ops_model.post_process.combination.pca_optimization \
    --output-dir /hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized_v0.3 \
    --cell-dino --zscore-per-experiment \
    --run-tag paper_v1/validation_4exp_phase_only \
    --experiments ops0146,ops0147,ops0150,ops0151 \
    --phase-only \
    --chad-annotation /hpc/projects/icd.fast.ops/configs/gene_clusters/val_library_chad_positive_controls_v1.yml \
    --slurm
```

---

## Key flags

**Feature mode (pick one, required):** `--cell-dino` `--dino` `--cell-profiler` `--dynaclr`
`--subcell` `--organelle-profiler` (`--op-root <dir>`).

**Channel subset:** `--phase-only` (brightfield only) · `--no-phase` (fluorescent only) ·
default = all. Sibling layouts: `--with-cp` / `--with-4i` / `--only-cp` / `--only-4i` /
`--include-cellpainting`.

**PCA threshold:** `--fixed-threshold 0.80` (default; single cutoff) · `--fixed-threshold 0`
(run the full **consensus variance sweep** instead). CP features default to a lower sweep range.

**Downsampling:** `--downsampled` (equalize cells across signal groups, floor 750k) ·
`--target-cells N` (force exact count) · `--downsample-per-guide --cells-per-guide 250`.

**Normalization:** `--norm-method ntc|global` (default `ntc`) ·
`--zscore-per-experiment` / `--no-zscore-per-experiment` (default on).

**SLURM:** `--slurm` to dispatch; tune with `--slurm-memory` `--slurm-time` `--slurm-cpus`
`--slurm-partition` `--phase-memory` `--slurm-agg-memory` `--slurm-agg-time`. Omit `--slurm`
to run locally.

**Embeddings / reproducibility:** `--seed` (default 1 for `--umap-type max`, 42 for `gav`) ·
`--umap-type max|gav` · `--distance cosine|euclidean`.

**Second-pass PCA** (on by default): `--no-second-pca` to disable · `--second-pca-threshold`
(0 = consensus sweep) · `--second-pca-consensus-metrics activity,distinctiveness,ebi`.

**Chromosome-arm correction** (optional): `--chrom-arm-correct` (+ `--chrom-arm-method`,
`--chrom-arm-knn`, `--chrom-arm-qval`, `--chrom-arm-map-csv`).

**Experiment selection:** `--experiments ops0100,ops0105` · `--paper-v1` · `--signals "Phase,ER_SEC61B"`
(retry specific signal shards) · `--run-tag <subpath>` (organizational subfolder).

**Misc:** `-y/--yes` (skip confirmation) · `--direct` (use `--output-dir` verbatim, skip auto-nesting) ·
`--clean` (wipe prior Phase-1 outputs first) · `--exclude-dud-guides` (default on).

---

## Re-running phases cheaply (skip the expensive PCA sweep)

After a full run you can regenerate downstream artifacts from the on-disk per-signal/combined
h5ads without redoing Phase 1:

| Flag | Reuses | Recomputes |
|---|---|---|
| `--aggregate-only` | per-signal h5ads | Phase 2: concat → normalize → score → embed |
| `--second-pca-only` | `guide_pca_optimized.h5ad` | second-pass PCA only |
| `--umap-only` | combined h5ads | UMAP/PHATE + embedding plots |
| `--overlays-only` | combined h5ads | interactive HTML overlays (refits UMAP only if `--seed` differs) |
| `--chad-umap-only` | `gene_embedding_pca_optimized.h5ad` | CHAD-colored UMAP |
| `--sweep-seed` | gene h5ad | a grid PNG of UMAP layouts across seeds |

Pass the **same flags** that define the output path (feature mode, channel subset, threshold,
distance, …) so the tool resolves to the same directory — see below.

---

## Output layout

The output path is auto-nested from the flags (use `--direct` to bypass):

```
<output-dir>/<feature>/[zscore_per_exp/][paper_v1/][<run-tag>/]<channel-subset>/<mode>/<distance>/[agg_<method>/]
```

- `<feature>`: `cell_dino` · `dino` · `cellprofiler` · `dynaclr` · `subcell` · `organelle_profiler`
- `<channel-subset>`: `all_livecell` (default) · `phase_only` · `no_phase` · `*_downsampled` · `with_cp` …
- `<mode>`: `fixed_80%` (fixed threshold) · `consensus_sweep` · `batch` (`--preserve-batch`) · `no_pca`
- `<distance>`: `cosine` (default) · `euclidean`

Inside the resolved dir:
- `per_channel/` (standard) or `per_signal/` (downsampled) — Phase 1 per-signal h5ads + sweep CSVs
- `guide_pca_optimized.h5ad`, `gene_embedding_pca_optimized.h5ad` — Phase 2 combined outputs
- `pca_report.csv`, `metrics/`, `plots/`, `second_pca_consensus/`

---

## Supplemental tools (run separately, not part of a combine)

Post-hoc analysis/visualization that consume the combined outputs, in the `analysis/`
subpackage (run via `python -m ops_model.post_process.combination.analysis.<name>`) — see
`SCRIPT_MAP.md`: `analysis/compare_map_scores.py`, `analysis/compare_modalities.py`,
`analysis/pca_component_to_feature.py`, `analysis/embedding_overlays.py`, plus the
`titration/` and `hand_annotations/` analyses.
