# `combination/` — core pipeline vs supplemental scripts

Inventory of every `.py` in `ops_model/src/ops_model/post_process/combination/` (read in
full), classified as **core combination pipeline** (produces the combined multi-experiment
guide/gene h5ads), **deprecated**, or **supplemental** (optional stages, downstream
analysis/plotting, one-off tooling). 34 modules, ~27k LOC.

> **Decision (canonical going forward):** the **argparse subpackage**
> `python -m ops_model.post_process.combination.pca_optimization` is the single supported
> entry point. **Everything tied only to the config/`baseline.yml` path is deprecated** —
> see §3.

> Line numbers/sizes are a snapshot; anchor on names.

---

## 0. The two implementations — one kept, one deprecated

There were **two independent implementations** of the same two-phase pca_optimized pipeline.
We are standardizing on the subpackage and retiring the config-driven path.

| Path | Entry point | Modules | Status |
|---|---|---|---|
| **Argparse subpackage** | `python -m …combination.pca_optimization …` | `pca_optimization/` (`__init__`, `handlers`, `phase1`, `phase2`, `sweep_core`, `aggregation`, `slurm`, `parser`, …) | **CANONICAL** |
| **Config-driven CLI** | `python -m …combination.cli --config baseline.yml` → `run_combination` | `cli.py`, `config_handler.py`, `combiners.py`, `file_validator.py`, `cell_filters.py` | **DEPRECATED** |

Verified: the subpackage is **fully independent** of the config-path modules (no imports of
`config_handler` / `combiners` / `cell_filters` / `file_validator`). Both paths ultimately
call the shared aggregation primitives in `ops_model.features.anndata_utils`
(`aggregate_to_level`, `hconcat_by_perturbation`, `normalize_guide_adata`).

---

## 1. CORE — the canonical pipeline (`pca_optimization/`)

Two-phase flow: pool cells per biological signal → fit PCA → sweep n_pcs → aggregate to
guide/gene → NTC-normalize → save guide/gene h5ads → (optional) score + embed.

| File | LOC | Entry | Role |
|---|---|---|---|
| `pca_optimization/__init__.py` | 501 | `main` | Orchestration hub; parses args, discovers experiments, dispatches handlers. |
| `pca_optimization/__main__.py` | 13 | `python -m …` shim | Re-exports `main`. |
| `pca_optimization/parser.py` | 641 | lib | `_build_parser` — the 60+ CLI flags (the real config surface now). |
| `pca_optimization/slurm.py` | 240 | lib | Phase 1/2 submitit job submission + chaining. |
| `pca_optimization/handlers.py` | 1961 | lib | One handler per CLI mode (standard/downsampled, aggregate-only, second-pca, op, umap-only…). Decides which stages run. |
| `pca_optimization/phase1.py` | 571 | lib (SLURM worker) | `pca_sweep_pooled_signal` — Phase 1: pool→PCA→threshold sweep per signal. |
| `pca_optimization/phase2.py` | 1101 | lib (SLURM worker) | `aggregate_channels` — Phase 2: concat per-signal → normalize → score → embed → save; `apply_second_pass_pca` (optional Phase 3). |
| `pca_optimization/sweep_core.py` | 755 | lib | Threshold-sweep scoring + per-signal h5ad/CSV writing (incl. `--no-pca` raw save). |
| `pca_optimization/aggregation.py` | 580 | lib | Phase 2 primitives: hconcat, NTC-normalize, gene re-agg, **`_atomic_write_h5ad` (obs-sanitizing save)**, panel annotation, canonical h5ad write. |

Shared dependency (outside this dir): `ops_model.features.anndata_utils` aggregation helpers.

---

## 2. CORE-OPTIONAL — subpackage stages that run only under specific flags
| File | LOC | Role | When it runs |
|---|---|---|---|
| `pca_optimization/embeddings.py` | 478 | UMAP/PHATE + metric overlays + distinctiveness/consistency scoring. | Standard Phase 2; also standalone via `--umap-only`. |
Optional add-on stages, moved into `pipeline_add_ons/` on `alexhillsley/refactor`
(`pca_sweep_op_signal` is still re-exported through `pca_optimization`'s namespace):
| File | LOC | Role | When it runs |
|---|---|---|---|
| `pipeline_add_ons/op_signal.py` | 395 | `pca_sweep_op_signal` — Phase 1 variant reading OrganelleProfiler `all_cells_*.h5ad`. | Only `--organelle-profiler` mode. |
| `pipeline_add_ons/chromosome.py` | 308 | Chromosome-arm overlay plots (PNG/SVG/HTML) on gene embeddings. | Only with `--chromosome-csv` or active chrom-arm correction. |
| `pipeline_add_ons/guide_chrom_arm_correction.py` | 740 | Removes chromosome-arm clustering artifacts from guide PCA embeddings (3 strategies). | Optional post-processing; called from subpackage `handlers`/`embedding_overlays` when configured. |

---

## 3. DEPRECATED — the config/`baseline.yml` path

**REMOVED** on branch `alexhillsley/refactor` (verified: no non-test/non-scratch code imported
them). Deleted: `cli.py`, `combiners.py`, `config_handler.py`, `file_validator.py`,
`classifier_combiner.py`, `classifier_aggregator.py` (+ tests `test_classifier_combiner.py`,
`test_classifier_aggregator.py`, `test_combination_e2e.py`).
**Held:** `cell_filters.py` (pending the port-vs-drop decision — see §Migration).

The table below is retained as a record of what was removed and why.

| File | LOC | Role | Notes |
|---|---|---|---|
| `cli.py` | 286 | `run_combination` dispatch + `validate_and_save`. | Whole module deprecated. (The recent obs-sanitization fix here is moot — `aggregation._atomic_write_h5ad` already does it on the canonical path.) |
| `config_handler.py` | 242 | `CombinationConfig` + `load_config(yaml)`. | The `baseline.yml`/`CombinationConfig` schema retires with the CLI. Verify external `load_config` users first (§Migration). |
| `combiners.py` | 1220 | `PcaOptimizationCombiner` (duplicate Phase 1/2), the deprecated `ComprehensiveCombiner`, and `_process_signal_group`/`_sweep_pca_thresholds`/`_prepare_cells_for_scoring`. | The whole file is config-path-only. Removing it deletes the **duplicate** pca_optimized implementation and the last in-repo caller of `anndata_utils.concatenate_experiments_comprehensive`. |
| `file_validator.py` | 155 | Input-file validation for `comprehensive`/`vertical`. | Skipped on the pca path; only the deprecated methods use it. |
| `cell_filters.py` | 272 | `build_cell_filter` + `DudGuide`/`TopPhenotype`/`Composed` filters. | Config-path-only — **but the subpackage has no cell-filtering equivalent.** See §Migration: port or consciously drop. |

Also retiring with this path (alternative methods that were never the canonical flow):
| File | LOC | Role | Notes |
|---|---|---|---|
| `combiners.ComprehensiveCombiner` (in `combiners.py`) | — | `comprehensive` method → `anndata_utils.concatenate_experiments_comprehensive`. | Deprecated; see features `REFACTOR_PLAN_1`. |
| `classifier_combiner.py` | 219 | `ClassifierCombiner` — MLP-classifier aggregation. | **Never wired into `cli`**; depends on `CombinationConfig`. Dormant → retire. |
| `classifier_aggregator.py` | 1206 | MLP/CosineClassifier training machinery for the above. | Only via `ClassifierCombiner`. Dormant → retire. |

---

## 4. SUPPLEMENTAL — downstream analysis / plotting (`analysis/` subpackage; consume combined outputs, not in the core path)

Moved into `analysis/` on branch `alexhillsley/refactor`. Run as
`python -m ops_model.post_process.combination.analysis.<name>`.

| File | LOC | Entry | Role |
|---|---|---|---|
| `analysis/embedding_overlays.py` | 3080 | lib (`save_extra_overlays`) | Static/interactive UMAP overlays, Leiden + GO enrichment, super-category/CHAD/CORUM/EBI annotation. Invoked (lazily) by subpackage Phase 2 for *extra* plots; pure visualization. |
| `analysis/compare_map_scores.py` | 781 | `main` | Compare mAP metric CSVs across conditions (phase vs no-phase, DINO vs CP). |
| `analysis/compare_modalities.py` | 868 | `main` | Cross-modality distinctiveness at fixed cell budget (cp/4i/livecell set comparisons). |
| `analysis/pca_component_to_feature.py` | 680 | `main` | Interpret PCA loadings → CellProfiler feature categories. |

---

## 5. SUPPLEMENTAL — `titration/` (cell-count titration analyses; all standalone, none in the core path)
| File | LOC | Entry | Role |
|---|---|---|---|
| `titration/titration.py` | 2262 | `main` | Per-reporter cell-count titration; score 4 metrics vs cell budget. |
| `titration/combined_titration.py` | 2316 | `main` | Multi-marker group titration (hconcat panels at each budget). |
| `titration/titration_reporter_pair.py` | 1040 | `main` | Two-reporter (e.g. Phase+fluor) titration with optional dual-PCA. |
| `titration/titration_phase_paired_fluor.py` | 572 | `main` | Phase + each fluor marker across budgets. |
| `titration/titration_phase_paired_dual_fluor.py` | 571 | `main` | Phase + two channel-disjoint fluor markers. |
| `titration/titration_paired_plots.py` | 559 | lib | Shared plot helpers for the two phase-paired drivers above. |

(Note: several titration modules import `anndata_utils` aggregation helpers and the
subpackage's sweep/aggregation — they're independent of the deprecated config path.)

---

## 6. SUPPLEMENTAL — `hand_annotations/` (manual curation / presentation; standalone)
| File | LOC | Entry | Role |
|---|---|---|---|
| `hand_annotations/embedding_param_sweep.py` | 754 | `main` | UMAP/PHATE parameter grid sweep on an existing run. |
| `hand_annotations/hand_annotated_umap_animation.py` | 1362 | `main` | GIF walking through hand-annotated gene clusters on the gene UMAP. |
| `hand_annotations/resolve_cluster_picks.py` | 329 | `main` | Resolve representative gene/channel per hand-annotated cluster from attention CSVs. |

---

## 7. SUPPLEMENTAL — misc tooling (in `analysis/`)
| File | LOC | Entry | Role |
|---|---|---|---|
| `analysis/marker_norm_sweep_runner.py` | 37 | lib | Submitit wrapper subprocessing an external marker-normalization sweep script. Independent of the pipeline. Moved into `analysis/` on `alexhillsley/refactor`. |

---

## Migration impact of deprecating the config/baseline path

Status after removal on `alexhillsley/refactor`:

- **Tests** `test_classifier_combiner.py` / `test_classifier_aggregator.py` / `test_combination_e2e.py`
  — **deleted** with the modules they covered.
  - ⚠️ **Follow-up:** `test_combination_e2e.py` was our only end-to-end combination test. It
    needs **re-adding against the canonical subpackage** (drive Phase 1
    `pca_sweep_pooled_signal` + Phase 2 `aggregate_channels`, or `main` with argparse flags).
    The subpackage's `_atomic_write_h5ad` already sanitizes obs, so no `cli`-side fix is needed.
- **External `load_config` users** — false alarms, no action: `project_shared_umap.py` defines
  its **own** local `load_config`; the organelle stage uses its **own** `_load_config()`; the
  energy_distance reference is a comment. None imported the deleted `config_handler`.
- **`experiments/scratch/20260414_debugging/pca_optimized_with_filter.py`** — scratch; imports
  the now-deleted `combiners` (`PcaOptimizationCombiner`, `_sweep_pca_thresholds`) and
  `config_handler`. **Now broken** — left as-is (scratch); update or delete when convenient.

**Open decision (blocking `cell_filters.py` removal):**
- **Cell filtering** — `cell_filters.py` (dud-guide / top-phenotype filters) has no subpackage
  equivalent. **Held, not deleted.** Port it into `pca_optimization` Phase 1, or accept
  dropping the feature? It currently has zero importers (its only caller, `combiners.py`, is
  gone), so it's dead-but-harmless until decided.

## Takeaways
- **The canonical core is just `pca_optimization/`** (§1) + the `anndata_utils` aggregation
  primitives. §2 are flag-gated stages of it.
- Deprecating the config path (§3) removes ~2.4k LOC (`cli`+`config_handler`+`combiners`+
  `file_validator`+`cell_filters`) plus the dormant classifier (~1.4k LOC), and eliminates the
  **duplicate pca_optimized implementation** — the single biggest hazard in this dir.
- §4–§7 are legitimate standalone analysis tools — supplemental, low-coupling, keep as-is.
