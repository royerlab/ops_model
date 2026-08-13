# DiffEx single-cell classifier PoC (options B & C)

The classifier DiffEx will explain (see [../PLAN.md](../PLAN.md), classifier §1).
PoC = **binary HSPA5-vs-rest** on **phase** single-cell crops.

- **B** — ResNet18 on the 160×160 phase crops (pixels → class logits).
- **C** — MLP on **CellDINO** embeddings of the *same* crops (`ops_model.models.cell_dino`).

Both score **top-attention cells** (settled), differing only in the feature space.

## Locked design (Gav, 2026-06-16)
- Positives: top-1000 attention cells of HSPA5 (`pma_phase_cells_v2_all.parquet`).
- Negatives: 1000 cells sampled from the **top-5** attention cells of **other genes**
  (the "distinct" contrast — strong-vs-strong).
- Crop 160×160, phase-only (`Phase2D`), no cell mask (full crop context).
- Split: **3-way train/val/test, grouped by experiment** (confound guard — val & test
  cells come from experiments never trained on). val = model selection; **test = the
  clean reported number** (scored once, never used for selection). Stratified-random
  fallback if a class is missing from a side.
- Success: held-out **test AUROC ≫ 0.5** (generalizing across experiments ⇒ biology, not batch).

## Decision: C reuses the local CellDINO encoder on the same crops
Rather than join Alex's per-gene dumps, option C runs the local encoder
(`CellDinoModel`: channel-adaptive DINO ViT-L/16, resize 224 + per-image z-score,
`in_channels=1`) on the identical crops B uses, and caches the embeddings. One crop
pipeline; B and C see identical cells.

## Run (GPU)
```bash
# single run, interactively on a GPU node
python -m ops_model.models.attention.diffex.classifier.run --model B --gene HSPA5
python -m ops_model.models.attention.diffex.classifier.run --model C --gene HSPA5

# or submit both to SLURM (one GPU job each)
python -m ops_model.models.attention.diffex.classifier.submit --gene HSPA5

# sweep: all 98 EBI complexes + NTC control, model C
python -m ops_model.models.attention.diffex.classifier.submit --grain complex --all-classes --models C
# then rank them
python -m ops_model.models.attention.diffex.classifier.aggregate --grain complex --model C
```
`--grain {geneKO,complex}` selects the parquet + class column (`gene` vs `predicted_class`).
NTC is included as a negative-control bin (its AUROC should be near chance).
Outputs land under `<out-dir>/<gene>/` (default out-dir
`/hpc/projects/icd.fast.ops/models/diffex`): `model_{B,C}.pt`, `metrics_{B,C}.json`,
and a shared `cache/` (crops + CellDINO features). SLURM logs →
`ops_mono/slurm_logs/diffex_clf/`.

## Layout
- `config.py` — all params (the locked defaults above).
- `data.py` — cell-table query, crop materialization (`BaseDataset`), split.
- `models.py` — ResNet (B) + MLP head (C).
- `celldino_features.py` — embed crops with the local CellDINO encoder (cached).
- `train.py` — shared train/eval loop (AUROC).
- `run.py` — orchestrator + `run_poc()` entry point.
- `submit.py` — SLURM submission (`submit_parallel_jobs`).

## Status
Pipeline verified end-to-end on CPU for **B** (tiny config): cell table → crops
(non-degenerate, masked) → train → AUROC → artifacts. **C** needs a GPU (CellDINO).
Next: run B & C on HSPA5 at full scale (GPU), compare AUROC, pick the DiffEx target.
