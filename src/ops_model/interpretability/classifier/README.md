# Set classifier (interpretability)

Train, evaluate, and explain a permutation-invariant **set classifier** that predicts a
gene knockout — or a protein-complex / pathway label — from a **set** of single-cell
embeddings (a "bag" of `N` cells for one perturbation). The model is an inducing-point
set transformer that pools cells (optionally per marker/channel) into one prediction, so
accuracy rises with the number of cells per set.

These scripts run **standalone**, independent of the rest of `ops_model`: the only inputs
are an **embedding parquet** and (for complex/pathway classification) a **gene→label
metadata CSV**, both passed as paths.

## Install

```bash
pip install -e ".[classifier]"   # adds hydra-core, omegaconf, pyarrow, tqdm, matplotlib
```

## Inputs

**Embedding parquet** (`data.parquet_entries[].path`). One row per (cell, channel), with columns:

| column | meaning |
|---|---|
| `embeddings` | per-cell embedding vector (list/array) |
| `gene_name` | perturbation / gene knockout (empty ⇒ treated as `NTC` control) |
| `experiment` | experiment id (used for z-standardization and stratified splits) |
| `name`, `channel_type`, `channel_index` | channel identity; `biological_annotation.organelle` + `biological_annotation.marker` form the marker/channel name |
| `well`, `x_pheno`, `y_pheno`, `segmentation_id`, `index` | per-cell coordinates/id (needed only to write score dumps + rankings) |

Multiple parquet files can be listed; each entry accepts optional `exclude_experiments`,
`exclude_fluorescent_experiments`, and `column_remap`.

**Label map CSV** (optional, for EBI-complex / pathway classification). Set
`data.label_map_path` with `label_map_gene_col` (default `gene_name`) and
`label_map_label_col` (default `pathway`). Genes absent from the CSV are dropped unless
`label_map_fallback_to_gene: true`. Without a label map, the model classifies the raw
`gene_name` (the "1K gene" task).

## 1. Train

Hydra-driven (`configs/`). All keys can be overridden on the CLI.

```bash
# 1K gene classification, all fluorescent markers pooled (channels_per_set=null)
python -m ops_model.interpretability.classifier.train \
    --config-name train_set_classifier_fluor_cp_4i_paper_v2_null \
    data.parquet_entries='[{path:/path/to/embeddings.parquet}]' \
    save_path=set_classifier.pt num_epochs=200

# EBI protein-complex classification (drive with your label map)
python -m ops_model.interpretability.classifier.train \
    --config-name train_set_classifier_fluor_cp_4i_ebionly_null \
    data.label_map_path=/path/to/ebi_complexes.csv
```

Writes the best-val checkpoint to `save_path` (stores `gene_to_idx`, `channel_to_idx`, and
— for label-mapped runs — `label_to_idx` / `label_remap`).

To later compute cell scores, also set `dump_train_dir` / `dump_val_dir` so training caches per-gene
`.pt` dumps (embeddings + cell metadata) alongside the checkpoint.

## 2. Evaluate

Sweeps top-1/top-5 accuracy vs. number of cells per set (`n_cells_list`), averaged over
`n_repetitions` random bags per gene. Writes a PNG curve, a JSON summary, and a per-class CSV.

```bash
python -m ops_model.interpretability.classifier.eval \
    --config-name eval_set_classifier \
    checkpoint_path=set_classifier.pt \
    val_dump_dir=/path/to/dumps/val \
    n_cells_list='[10,20,50,100,200,500,1000,2000,5000]' \
    output_plot_path=acc_vs_ncells.png per_class_output_file=per_class.csv
```

The val set can come from a pre-dumped `val_dump_dir` (fast) or directly from
`data.parquet_entries` (per-channel mode).

## 3. Score (per-cell attribution)

Ranks each cell by its score — the leave-one-out marginal
`P(class | bag) − P(class | bag∖cell)`, averaged over random bag partitions and over bag
sizes. Consumes the per-gene `.pt` **dumps** produced by training (step 1 with `dump_*_dir`
set). Plain argparse:

```bash
python -m ops_model.interpretability.classifier.score \
    --checkpoint set_classifier.pt \
    --dump_dir /path/to/dumps/train /path/to/dumps/val \
    --channel Phase2D \
    --genes KIF23 HSPA5 AURKB \
    --bag_sizes 1 2 5 10 20 50 100 200 500 \
    --out_csv score_ranking.csv
```

Reps are tapered from `--reps` (anchor, smallest bag) down to `--min_reps` as bag size grows
(the per-cell marginal variance collapses with bag size); use `--flat_reps` for a constant
rep count or `--reps_schedule` for explicit per-bag reps. Large pools (e.g. the NTC control)
are subsampled to `--max_cells`. Output is a CSV with one row per cell: score, per-bag
marginals, and the cell's coordinates (so downstream montages can render straight from it).

## Configs

`configs/` holds representative examples (paths are placeholders — point them at your data):

| Config | Task |
|---|---|
| `train_set_classifier_phase_paper_v2.yaml` | phase, 1K gene |
| `train_set_classifier_phase_paper_ebi.yaml` | phase, EBI complex |
| `train_set_classifier_fluor_cp_4i_paper_v2_null.yaml` | fluor, all markers (cps=null), 1K gene |
| `train_set_classifier_fluor_cp_4i_ebionly_null.yaml` | fluor, all markers (cps=null), EBI complex |
| `eval_set_classifier.yaml` | eval (one config for all four) |

`channels_per_set: null` pools **all** markers per set. To also train on smaller random
marker subsets, set it to a list, e.g. `[1, 2, 5, 10, 20, null]` (each set then draws a
random number of markers from that list).
Point `CONFIG_PATH` at another directory to load configs from elsewhere.

## Module layout

- `train.py` — model architecture (`MixedChannelClassifier`, set-transformer blocks), the
  parquet/dump datasets + label-map logic, the training loop, and `build_model` (checkpoint → model).
- `eval.py` — accuracy-vs-`n_cells` sweep.
- `score.py` — per-cell score ranking.
