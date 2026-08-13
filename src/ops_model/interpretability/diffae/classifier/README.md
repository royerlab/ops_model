# DiffEx single-cell classifier

The per-cell classifier that DiffEx explains. It separates one class — a gene
knockout or a protein complex — from a strong contrast set drawn from other
classes, using **phase** single-cell crops. The downstream stages traverse the
DiffAE latent to shift *this* classifier's score, so its decisions are what the
counterfactual morphs explain.

Two interchangeable feature spaces, scoring the same cells with the same split:

- **B** — ResNet18 on the 160×160 phase crops (pixels → class logits).
- **C** — MLP on **CellDINO** embeddings of the same crops. Option C runs the
  local CellDINO encoder (`ops_model.models.cell_dino`) over the identical crops
  B uses and caches the result, so B and C always see the same cells.

Defaults live in `config.py`: the top 1000 attention cells per class, 160×160
`Phase2D` crops, 30 epochs, and a 3-way train/val/test split grouped by
experiment so val and test cells come from experiments never trained on. Models
are scored by AUROC.

## Run (GPU)

```bash
# single run, interactively on a GPU node
uv run python -m ops_model.interpretability.diffae.classifier.run --model B --gene HSPA5
uv run python -m ops_model.interpretability.diffae.classifier.run --model C --gene HSPA5

# or submit both to SLURM (one GPU job each)
uv run python -m ops_model.interpretability.diffae.classifier.submit --gene HSPA5

# sweep: all EBI complexes + NTC control, model C
uv run python -m ops_model.interpretability.diffae.classifier.submit --grain complex --all-classes --models C

# then rank them
uv run python -m ops_model.interpretability.diffae.classifier.aggregate --grain complex --model C
```

`--model {B,C}` is required. `--gene` names the class value (default `HSPA5`),
and `--grain {geneKO,complex}` selects the parquet plus class column (`gene` vs
`predicted_class`, default `geneKO`). NTC is included as a negative-control bin,
so its AUROC should sit near chance. `--n-per-class`, `--crop-size`, `--epochs`
and `--split-mode` override the config defaults.

Outputs land under `<out-dir>/<gene>/`, with `--out-dir` defaulting to
`$OPS_BASE_PATH/models/diffex`: `model_{B,C}.pt`, `metrics_{B,C}.json`, and a
shared `cache/` of crops and CellDINO features. Running C after B reuses that
cache. SLURM logs go to `slurm_logs/diffex_clf_<grain>/`.

## Layout

- `config.py` — all params and their defaults.
- `data.py` — cell-table query, crop materialization (`BaseDataset`), split.
- `models.py` — ResNet (B) + MLP head (C).
- `celldino_features.py` — embed crops with the local CellDINO encoder (cached).
- `train.py` — shared train/eval loop (AUROC).
- `run.py` — orchestrator + `run_poc()` entry point.
- `submit.py` — SLURM submission (`submit_parallel_jobs`).
