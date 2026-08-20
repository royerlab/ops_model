# Set classifier — predict a perturbation from a *set* of single-cell embeddings

A single perturbed cell is a weak signal; a *population* of cells is not. This
subpackage trains a **permutation-invariant set classifier** that takes an unordered
bag of per-cell embeddings drawn from one perturbation and predicts the class:
either the gene knockout itself, or a coarser label (EBI protein complex / pathway)
supplied by a gene→label CSV. Because the pooling is attention-based over the set
axis, the bag size is a free parameter — the headline measurement is **accuracy vs.
cells-per-set** (`eval.py`), and the same model gives a **per-cell contribution
score** (`score.py`) that ranks which individual cells carry the phenotype.

Runs standalone from an embedding parquet (+ the optional label CSV); nothing else in
`ops_model` is required at run time.

## Install

Needs the optional `classifier` extra (`hydra-core`, `omegaconf`, `pyarrow`, `tqdm`,
`matplotlib`) — declared in `ops_model/pyproject.toml` and *not* part of the default
monorepo env:

```bash
uv pip install -e "ops_model[classifier]"
```

## Model (what is actually implemented)

Only the **mixed-channel** path exists: `mixed_channels_mode: true` and
`MixedChannelClassifier`. All cells of a gene, across every channel/marker, are pooled
into one set; each cell keeps a channel id so the model can condition on which marker
it came from. `train.py`, `eval.py` and `score.py` all raise on
`mixed_channels_mode=false` — the older two-level per-channel `SetClassifier` was
removed, so config comments mentioning a "per-channel set transformer" or a
"cross-channel set transformer" are stale.

Forward pass (`MixedChannelClassifier`, one level):

1. `input_proj`: `Linear(emb_dim → d_model)` on every cell.
2. channel conditioning (`model.channel_conditioning`):
   `add` (add channel embedding) · `concat` (concat + project back) ·
   `adaln` (AdaLN from the mean channel embedding) ·
   `adaln-token` (AdaLN per cell, from its own channel) · `none`.
3. `ISABBlock`: `n_layers_cell` × ISAB with `n_inducing_cell` inducing points
   (O(N·m) instead of O(N²)), then pooling — `pool_type: pma` (default, learnable
   seed query) or `mean` (masked mean) — then LayerNorm.
4. head: `Dropout` + `Linear`, or `CosineClassifier` (L2-normalized logits with a
   learned temperature) when `model.cosine_classifier: true`.

Sets shorter than `n_cells_per_set` are zero-padded and masked, so genes with few
cells are still usable. Multi-GPU is automatic (`nn.DataParallel` when
`torch.cuda.device_count() > 1`).

`eval.py` / `score.py` rebuild the model from the checkpoint's saved config
(`d_model`, `n_heads`, `n_layers_cell`, `n_inducing_cell`, `d_ff`, `dropout`,
`cosine_classifier`, `channel_conditioning`); `pool_type` is **not** re-read, so
checkpoints trained with `pool_type: mean` will not reload there.

## Inputs

### Embedding parquet

Read row-group at a time (`pyarrow`), so a parquet larger than RAM is fine.
Required columns: `embeddings` (flat per-cell vector), `gene_name` (null → `NTC`),
`experiment`, `name`, `biological_annotation.organelle`,
`biological_annotation.marker`. The channel label is
`organelle_marker` when either annotation is present, else `name`; rows with
`name == "Phase2D"` are always labeled `Phase2D` (`_build_channel_label`) — these
exact strings are what `exclude_channel_names` / `include_channel_names` /
`only_channels` match against, and they are printed under `Cells per channel`.

Optional per-cell columns `well`, `x_pheno`, `y_pheno`, `segmentation_id`,
`channel_type`, `index` are read only when dumping (`dump_train_dir` /
`dump_val_dir`); missing ones are filled with defaults, except that `well` and
`segmentation_id` are mandatory under `data.cell_stratify: true`. A legacy
`channel_index` column is accepted in place of `index`, and per-entry
`column_remap: {canonical: source}` renames others.

Per-parquet-entry filters: `exclude_experiments`, `exclude_fluorescent_experiments`.
Global: `max_row_groups`, `max_genes` / `max_channels` (top-N by cell count),
`max_cells_per_group`, `min_cells_per_group` (+ `min_cells_drop_val`),
`data.max_train_cells` (subsample train cells only).

Train/val split happens as rows are read: per-row random by `val_fraction`, or —
with `data.cell_stratify: true` — a deterministic blake2b hash of
`(experiment, well, segmentation_id)` so every row of a cell (all its markers, and
the same cell across phenotyping / Cell Painting / 4i) lands on one side. Embeddings
are z-standardized per (channel × experiment); with
`z_standardize_control_only: true` the mean/std come from `NTC` cells only but are
applied to all cells. `z_standardize: false` disables it.

### Label map CSV (optional)

`configs/ebi_complexes.csv` — 311 rows, columns `gene_name`, `pathway`: 310 genes
annotated with their EBI Complex Portal complex, plus an `NTC,NTC` row, giving
**99 classes** (98 complexes + NTC). Wire it up with `data.label_map_path`,
`data.label_map_gene_col`, `data.label_map_label_col`;
`data.label_map_fallback_to_gene: false` (the shipped setting) **drops** genes absent
from the CSV, `true` keeps them as their own single-gene class. When a label map is
used the checkpoint additionally stores `label_to_idx` and `label_remap`, and
`eval.py` picks them up automatically.

Note: the shipped configs give `label_map_path` as a path *relative to the launch
directory* (`src/ops_model/interpretability/classifier/configs/ebi_complexes.csv`),
i.e. they assume you run from the `ops_model/` repo root. From anywhere else,
override it with an absolute path.

## Entry points

All three are modules; hydra's config dir is `configs/` next to the code (override
with the `CONFIG_PATH` env var). Hydra runs with `version_base="1.3.0"`, so the
working directory is **not** changed — relative `save_path` / `metrics_out` land in
the launch dir, while hydra still writes its run dir (resolved config + log) under
`outputs/<date>/<time>/`.

### `train.py` — hydra, `--config-name`

```bash
# default config is train_set_classifier_phase_1K
uv run python -m ops_model.interpretability.classifier.train

# pick a config and override any key
uv run python -m ops_model.interpretability.classifier.train \
    --config-name train_set_classifier_fluor_ebi \
    n_cells_per_set=200 \
    save_path=$OPS_BASE_PATH/models/set_classifier/fluor_ebi.pt \
    metrics_out=$OPS_BASE_PATH/models/set_classifier/fluor_ebi_metrics.json
```

Two data modes:

- **process parquet** — `data.parquet_entries` are read and, if `dump_train_dir` /
  `dump_val_dir` are set, the processed datasets are written out (one `<gene>.pt` per
  gene + `metadata.pt`). Both dump dirs must be absent or empty, else the run aborts.
- **reuse a dump** — set both `load_train_dir` and `load_val_dir` to skip parquet
  processing entirely (the fast path used by the `*_ebi` configs).

Sampling / schedule knobs: `n_cells_per_set` (cells per set), `channels_per_set`
(see below), `train_n_cell_sets_per_gene` / `val_n_cell_sets_per_gene` (draws per
gene per epoch, via `RepeatSampler`), `batch_size`, `num_workers`, `num_epochs`,
`learning_rate`, `weight_decay`, `max_grad_norm`, `warmup_epochs` (linear warmup then
cosine decay, AdamW), `eval_every`, `seed`, `device`.

Validation: the primary pass uses `n_cells_per_set`; `val_n_cells_per_set: [500, 200]`
adds extra passes logged as `val_n<N>/accuracy`; `phase2d_val: true` adds a
`Phase2D`-only val loader (`val_phase2d/...`), only available in the
parquet-processing mode.

Outputs:

| output | when | contents |
|---|---|---|
| `save_path` checkpoint | every improvement in the primary `val/accuracy` | `model_state_dict`, `gene_to_idx`, `channel_to_idx`, resolved `config`, `epoch`, `val_acc` (+ `label_to_idx`, `label_remap` with a label map) |
| `dump_{train,val}_dir/` | if set | `<gene>.pt` (`embeddings`, `channel_ids`, and `cell_metadata` when dumped from parquet) + `metadata.pt` (`gene_to_idx`, `channel_to_idx`, `emb_dim`, `perturbation_list`, `n_cells`) |
| `metrics_out` JSON | if set | `n_cells_per_set`, `val_accuracy`, `val_n<N>_accuracy` of the selected epoch |

### `eval.py` — accuracy vs. cells-per-set (hydra)

Needs a checkpoint plus a **val dump directory** (`dump_val_dir` from training).
For each `n_cells` it runs `n_repetitions` full val passes with different seeds and
reports mean top-1 and top-5 (`k = min(5, n_classes)`) with SEM error bars across
repetitions.

```bash
uv run python -m ops_model.interpretability.classifier.eval \
    checkpoint_path=$OPS_BASE_PATH/models/set_classifier/phase_1K.pt \
    val_dump_dir=$OPS_BASE_PATH/models/set_classifier/dumps/phase/val \
    output_plot_path=$OPS_BASE_PATH/models/set_classifier/phase_1K_n_cells.png \
    per_class_output_file=$OPS_BASE_PATH/models/set_classifier/phase_1K_per_gene.csv \
    n_cells_list='[10,100,1000]' n_repetitions=30
```

Other keys: `only_channels` (restrict to named channels) · `split_channels: true`
(evaluate each channel independently, one single-channel set per pass; adds a
`channel_name` column and one plotted series per channel) · `n_channel_shards` /
`channel_shard_id` (deterministic channel stride for parallel jobs) ·
`sample_without_replacement` · `label_map_path` (+ `..._gene_col`, `..._label_col`)
to score a checkpoint against a different gene set within the trained label space ·
`batch_size`, `num_workers`, `plot_title`, `plot_x_log`, `show_progress`, `seed`,
`device`. Channels present in the val dump but not in the checkpoint are dropped and
the remaining ids remapped to checkpoint space; a checkpoint-vs-dump gene mismatch is
an error unless a label map is given.

Outputs: the figure at `output_plot_path`, a sidecar JSON at the same basename
(`mean_accuracy_top1/top5`, `stderr_top1/top5`, every per-repetition accuracy, plus
`class_names` when a label map is in play), and — if `per_class_output_file` is set —
a per-gene CSV (`n_cells`, `gene_idx`, `gene_name`, optional `label_name`,
`n_repetitions`, `n_samples`, `top1_correct`, `top5_correct`, `top1_acc`,
`top5_acc`).

### `score.py` — per-cell scores (argparse, not hydra)

Ranks the cells of a `(gene, channel)` pair by their **leave-one-out marginal
contribution** to `P(true class)`: `P(class | bag) − P(class | bag \ cell)`, averaged
over random bag partitions and then uniformly over the bag-size grid. Bag size 1 is
the deterministic single-cell `P(class | cell)`. Bags are single-channel.

Reads the same dump directories as `eval.py`, but requires the `cell_metadata`
written when the dump was created from parquet (the output carries each cell's
coordinates so montages can be rendered from the ranking directly).

```bash
uv run python -m ops_model.interpretability.classifier.score \
    --checkpoint $OPS_BASE_PATH/models/set_classifier/phase_1K.pt \
    --dump_dir $OPS_BASE_PATH/models/set_classifier/dumps/phase/train \
               $OPS_BASE_PATH/models/set_classifier/dumps/phase/val \
    --channel Phase2D --genes KIF23 HSPA5 AURKB \
    --bag_sizes 1 2 5 10 20 50 100 200 500 --reps 50 \
    --out_csv score_phase.csv
```

`--reps` is the anchor for the smallest bag; larger bags are tapered toward
`--min_reps` (default 10) because the per-sample marginal variance collapses with bag
size. `--flat_reps` disables the taper, `--reps_schedule` sets reps per bag size
explicitly. `--max_cells` (default 65000) subsamples huge pools such as NTC, with its
own `--subsample_seed` kept separate from the partition `--seed`. `--device` defaults
to `cuda`, falling back to CPU.

Output: one CSV row per cell — `gene`, `channel_name`, `rank`, `score`, `bag1`, one
`marg_<B>` per bag size, then `split`, `experiment`, `well`, `y_pheno`, `x_pheno`,
`segmentation_id`, `zarr_channel_index`, `n_cells`.

## Shipped configs

| `--config-name` | channels | classes | data |
|---|---|---|---|
| `train_set_classifier_phase_1K` (train default) | `Phase2D` only (`max_channels: 1`) | gene-level (config comment: 1001 genes + NTC) | one phenotyping parquet; **creates** the phase dumps |
| `train_set_classifier_phase_ebi` | `Phase2D` only | EBI complexes via `ebi_complexes.csv` | **reuses** a pre-dumped phase dataset (`load_{train,val}_dir`) |
| `train_set_classifier_fluor_1K` | all fluorescent markers — phenotyping + Cell Painting (ops0094) + 4i (ops0144); drops `Phase2D` and `no label_no label` | gene-level | three parquets, `cell_stratify: true`; **creates** the shared fluor dump |
| `train_set_classifier_fluor_ebi` | same fluorescent set | EBI complexes, `label_map_fallback_to_gene: false` (non-EBI genes dropped) | **reuses** the fluor dump |
| `eval_set_classifier` (eval default) | — | — | `eval.py`: set `checkpoint_path`, `val_dump_dir`, `output_plot_path` |

All four training configs use `n_cells_per_set: 100`, `d_model: 512`, `n_heads: 4`,
`dropout: 0.0`, `n_layers_cell: 2`, `n_inducing_cell: 32`,
`channel_conditioning: "concat"`, `cosine_classifier: true`,
`z_standardize_control_only: true`, `val_n_cells_per_set: [500, 200]`.

### `channels_per_set`

How many markers each set is drawn from (`cps` in the code), applied to train and val
alike:

- `null` — no subsetting: sample across all of the gene's channels (the shipped
  setting for the fluorescent configs, whose comment suggests
  `[1, 2, 5, 10, 20, null]` to also train on smaller random marker subsets).
- an int — every set picks exactly that many random channels.
- a list, e.g. `[1, 2, null]` — each set independently picks one entry, `null`
  meaning "all channels".

If a gene has fewer channels than the chosen value, all available ones are used.
Irrelevant for the phase configs (single channel).
