# SHAP atlas runbook

End-to-end commands for **all 6 SHAP approaches** (2 extraction variants × 3 negative-class contrasts) × 2 grain levels (gene + CHAD complex). One unified CLI shape — no fallbacks, no path arguments needed.

## The 6 approaches

|   | distinct (vs other-KO) | ntc (vs NTC) | global (vs random) |
|---|---|---|---|
| **top-attention** (positives = top-100 attention cells) | 1. `top-attn-distinct` | 2. `top-attn-ntc` | 3. `top-attn-global` |
| **all-cells** (positives = up to 2k cells across attention range) | 4. `all-cells-distinct` | 5. `all-cells-ntc` | 6. `all-cells-global` |

**Variant** selects the extraction pipeline (which positives + which h5ads). **Contrast** selects the negative class. Each `(variant × contrast × grain)` writes to its own isolated output directory.

---

## Step 0 — build consolidated h5ads (one-time per data release)

### Top-attention extraction (gene + CHAD)

The consolidated h5ad now embeds NTC top-100 + GLOBAL random-100 cells in the same file as KO positives. Defaults wired — no path args needed:

```bash
# Gene-level. Output: consolidated_v3/. Inputs: pma_top_*_v3.csv + ntc_v3 NTC + 100 random GLOBAL/(exp, viz_channel).
python -m organelle_profiler.feature_extraction.consolidate_top_attention_cells

# CHAD-level. Output: consolidated_v3_chad/. Auto-swaps to chad_v1 + chad_ntc_v3.
python -m organelle_profiler.feature_extraction.consolidate_top_attention_cells \
    --aggregation-level complex
```

### All-cells extraction (one set serves gene + CHAD)

```bash
# Per-channel all_cells_<channel>.h5ad files. z-score now OFF by default
# (raw values, matching top-attention scale). Output: all_cells_v2/.
python -m organelle_profiler.feature_extraction.consolidate_all_cells --paper-v1
```

---

## Step 1+ — SHAP for each of the 6 approaches

Use `run_shap_pipeline.py` with `--variant {top-attention, all-cells}` × `--contrast {distinct, ntc, global}`. All path / cache args auto-route.

### One command for all 12 (recommended)

```bash
# 6 approaches × 2 grain levels = 12 SHAP runs in one shot.
# Sequential by default; each waits on its SLURM array to finish.
python organelle_profiler/scripts/ko_shap/run_all_shap.py

# Same but firing all 12 child pipelines in parallel.
python organelle_profiler/scripts/ko_shap/run_all_shap.py --parallel

# Subset (filters compose). Preview with --dry-run.
python organelle_profiler/scripts/ko_shap/run_all_shap.py \
    --variants top-attention --grains complex --dry-run
```

Or run each manually:

### CHAD complex-level (90 complexes — fast)

```bash
# Top-attention pipeline (3 contrasts)
for c in distinct ntc global; do
  python organelle_profiler/scripts/ko_shap/run_shap_pipeline.py \
      --variant top-attention --contrast $c \
      --aggregation-level complex --no-resume
done

# All-cells pipeline (3 contrasts)
for c in distinct ntc global; do
  python organelle_profiler/scripts/ko_shap/run_shap_pipeline.py \
      --variant all-cells --contrast $c \
      --aggregation-level complex --no-resume
done
```

### Gene-KO level (1000 genes — slower)

```bash
# Top-attention pipeline (3 contrasts)
for c in distinct ntc global; do
  python organelle_profiler/scripts/ko_shap/run_shap_pipeline.py \
      --variant top-attention --contrast $c --no-resume
done

# All-cells pipeline (3 contrasts)
for c in distinct ntc global; do
  python organelle_profiler/scripts/ko_shap/run_shap_pipeline.py \
      --variant all-cells --contrast $c --no-resume
done
```

Per-approach output directories (all named `{variant}_{contrast}_{grain}` — read at a glance, no clobbering):

| Variant × Contrast | Gene-level output | CHAD-level output |
|---|---|---|
| attention × distinct | `attention_distinct_geneKO/ko_shap_features.csv` | `attention_distinct_chad/...` |
| attention × ntc | `attention_ntc_geneKO/...` | `attention_ntc_chad/...` |
| attention × global | `attention_global_geneKO/...` | `attention_global_chad/...` |
| all_cells × distinct | `all_cells_distinct_geneKO/ntc_shap_features.csv` | `all_cells_distinct_chad/...` |
| all_cells × ntc | `all_cells_ntc_geneKO/...` | `all_cells_ntc_chad/...` |
| all_cells × global | `all_cells_global_geneKO/...` | `all_cells_global_chad/...` |

---

## Step Atlas — render PDFs (top-attention variant only)

`run_all_atlases.py` renders the 3 top-attention atlases (each contrast at the chosen grain).

```bash
# CHAD-level atlases (3 contrasts)
python organelle_profiler/scripts/ko_shap/run_all_atlases.py --aggregation-level complex

# Gene-level atlases (3 contrasts; default is complex so pass --aggregation-level gene)
python organelle_profiler/scripts/ko_shap/run_all_atlases.py --aggregation-level gene
```

The atlas's violin BG (for ntc / global contrasts) reads NTC / GLOBAL cells from the SAME consolidated h5ad as the positives — no all_cells_v2 detour, no pipeline-fingerprint AUROC=1.0 saturation. Hard-errors at startup if NTC / GLOBAL rows are missing from the cache → re-run Step 0.

---

## Step Compare — 6-way comparison diagnostic

```bash
# Shorthand: base path used by all 12 outputs.
BASE=/hpc/projects/icd.fast.ops/models/alex_lin_attention

# CHAD-level 6-way comparison.
python organelle_profiler/scripts/ko_shap/shap_approach_compare.py \
    --csv "attention-distinct=$BASE/attention_distinct_chad/ko_shap_features.csv" \
    --csv "attention-ntc=$BASE/attention_ntc_chad/ko_shap_features.csv" \
    --csv "attention-global=$BASE/attention_global_chad/ko_shap_features.csv" \
    --csv "all_cells-distinct=$BASE/all_cells_distinct_chad/ntc_shap_features.csv@distinct" \
    --csv "all_cells-ntc=$BASE/all_cells_ntc_chad/ntc_shap_features.csv@ntc" \
    --csv "all_cells-global=$BASE/all_cells_global_chad/ntc_shap_features.csv@global" \
    --output /hpc/mydata/gav.sturm/shap_approach_compare/chad_6way.pdf

# Gene-level 6-way + expected-hit highlight.
python organelle_profiler/scripts/ko_shap/shap_approach_compare.py \
    --csv "attention-distinct=$BASE/attention_distinct_geneKO/ko_shap_features.csv" \
    --csv "attention-ntc=$BASE/attention_ntc_geneKO/ko_shap_features.csv" \
    --csv "attention-global=$BASE/attention_global_geneKO/ko_shap_features.csv" \
    --csv "all_cells-distinct=$BASE/all_cells_distinct_geneKO/ntc_shap_features.csv@distinct" \
    --csv "all_cells-ntc=$BASE/all_cells_ntc_geneKO/ntc_shap_features.csv@ntc" \
    --csv "all_cells-global=$BASE/all_cells_global_geneKO/ntc_shap_features.csv@global" \
    --expected-genes "TIMM23,TIMM44,TOMM20,HSPA5,ERN1,EIF2AK3,RPL26,NOP56,NOP58,UTP14A,POLR1E,TUBA1A,GBF1,COPA,RAB7A,ATP5F1B,DDOST,DYNC1H1,EIF4G1,GOLGA2,HMGB1,HMGCR,MTOR,PXN,RAB11A,RAB6A,RICTOR,RPTOR,SEC61A1,UBAP2L" \
    --output /hpc/mydata/gav.sturm/shap_approach_compare/gene_6way.pdf
```

Outputs per run: 1 PDF (AUROC distribution + feature dominance + pairwise Jaccard + expected-hit heatmap), 1 summary CSV, 1 expected-hits CSV.

---

## Quick reference — what each approach asks

| # | Approach | Positives | Negatives | Biological question |
|---|---|---|---|---|
| 1 | top-attn × distinct | this complex's top-100 attention | other complexes' top-100 attention | "What attention-flagged morphology is unique to this complex?" |
| 2 | top-attn × ntc | this complex's top-100 attention | top-100 attention NTC cells | "What does the model find distinctive in KO vs NTCs at peak attention?" |
| 3 | top-attn × global | this complex's top-100 attention | 100 random cells | "How does the attention-flagged signature stand against the typical cell?" |
| 4 | all-cells × distinct | up to 2k KO cells (any attention) | up to 2k other-gene KO cells | "Population-level — what shifts this gene's cells vs the broader KO population?" |
| 5 | all-cells × ntc | up to 2k KO cells | all NTC cells | "Population-level — what's different between KO cells and NTCs overall?" |
| 6 | all-cells × global | up to 2k KO cells | all non-{this gene} cells | "Population-level — what makes this gene stand out vs the screen?" |

**Key invariant after the unified refactor:** for every approach, positives + negatives come from the SAME extraction pipeline (no scale mismatch / pipeline-fingerprint). AUROC ≈ 1.0 with no biological meaning is no longer reachable.

---

## Notes

- **No fallback paths.** If a contrast's required cohort isn't in the consolidated h5ad, the script hard-errors with a message pointing back at Step 0.
- **`--no-resume`** clears the per-shard CSVs before re-running so bug fixes can add new rows. Default behavior resumes from the existing CSV (skips already-listed genes).
- **Per-(variant × contrast) output directories** never overlap, so all 6 approaches can coexist on disk at both grain levels.
- The archived all-cells outputs at `_archive_unused_ntc_shap_features/` are from the pre-unification era. They can be deleted — the new commands produce fresh outputs in the new `ntc_v2_<contrast>` directories.
