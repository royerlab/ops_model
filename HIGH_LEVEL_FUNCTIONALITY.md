# ops_model — High Level Functionality

All commands must be run from the **monorepo root** (`ops_monorepo/`) using `uv run`.
See `RUNNING.md` for environment setup.

---

## Batch Processing Embeddings → AnnData

**File:** `ops_model/src/ops_model/features/batch_process_embeddings.py`

Converts existing embedding CSVs into AnnData objects (`.h5ad`). Skips experiments where anndata objects already exist. Submits one SLURM job per (experiment, channel) pair by default.

**Run via SLURM (default) from a config list:**
```bash
uv run python ops_model/src/ops_model/features/batch_process_embeddings.py \
    --config_list configs/cell_dino/cell_dino_config_list.txt
```
| Flag | Description |
|------|-------------|
| `--config <path>` | Process a single config file instead of a list |
| `--no-slurm` | Run sequentially on the current node instead of submitting SLURM jobs |
| `--force` | Reprocess even if anndata objects already exist |
| `--validate_only` | Check outputs exist without processing |
| `--stop_on_error` | Halt on first failure (default: continue) |
| `--output_report <path>` | Save a JSON summary of results to the given path |

---

## SubCell Inference

**File:** `ops_model/src/ops_model/models/subcell.py`

Extracts SubCell bg embeddings (1536 features per cell) from OPS image crops. Spawns one SLURM job per protein channel in `out_channels`, each paired with the DNA channel. Outputs `subcell_features_{channel}.csv` in `output_dir`, with intermediate chunks saved under `chunks_{channel}/`.

**Run from a config list:**
```bash
uv run python ops_model/src/ops_model/models/subcell.py \
    --config_list experiments/embedding/configs/subcell/subcell_config_list.txt
```

| Flag | Description |
|------|-------------|
| `--config_path <path>` | Path to a single YAML config file |
| `--config_list <path>` | Path to a `.txt` file with one config path per line |

Config must specify `data_manager.dna_channel` (e.g. `DAPI`) and `data_manager.out_channels` (list of protein channels). Each protein channel is paired with `dna_channel` as a separate SLURM job. See the module docstring in `subcell.py` for the full config schema.

---

## CellProfiler Feature Extraction

**File:** `ops_model/src/ops_model/models/cellprofiler/cp_features.py`

Extracts hand-crafted CellProfiler morphology and intensity features from OPS image crops. Submits SLURM jobs per experiment. Outputs a cell-level CSV and aggregated AnnData objects at cell, guide, and gene level.

**Run from a single config:**
```bash
uv run python ops_model/src/ops_model/models/cellprofiler/cp_features.py \
    --config_path experiments/embedding/configs/cell-profiler/ops0141_cp_20260401.yml
```

| Flag | Description |
|------|-------------|
| `--config_path <path>` | Path to a single YAML config file |
| `--config_list <path>` | Path to a `.txt` file with one config path per line |

See `experiments/embedding/configs/cell-profiler/example_unnormalized.yml` for the config format and `experiments/embedding/configs/cell-profiler/README.md` for full documentation including normalization options.

---

## DINOv3 Inference

**File:** `ops_model/src/ops_model/models/dinov3.py`

Extracts DINOv3 embeddings from OPS image crops. Spawns one SLURM job per (config, channel) pair via `submit_parallel_jobs`. SLURM parameters are read from the first config in the list.

**Run from a config list:**
```bash
uv run python ops_model/src/ops_model/models/dinov3.py \
    --config_list experiments/embedding/configs/reprocessing/batches/dino_configs_batch2c_subset_2.txt
```

| Flag | Description |
|------|-------------|
| `--config_path <path>` | Path to a single YAML config file |
| `--config_list <path>` | Path to a `.txt` file with one config path per line |

See `experiments/embedding/configs/dinov3/` for example config files.

---

## Cell-DINO Inference

**File:** `ops_model/src/ops_model/models/cell_dino.py`

Extracts Cell-DINO embeddings from OPS image crops. Spawns one SLURM job per (config, channel) pair. Does **not** skip experiments with existing outputs — all channels are reprocessed and overwritten unconditionally.

**Run from a config list:**
```bash
uv run python ops_model/src/ops_model/models/cell_dino.py \
    --config_list experiments/embedding/configs/cell_dino/cell_dino_config_list.txt
```

| Flag | Description |
|------|-------------|
| `--config_path <path>` | Path to a single YAML config file |
| `--config_list <path>` | Path to a `.txt` file with one config path per line |

See `experiments/embedding/configs/cell_dino/example.yml` for the config format.

---

## Evaluating Embeddings

**File:** `ops_model/src/ops_model/eval/run_eval.py`

Evaluates OPS embedding quality at guide and/or gene level. Outputs a summary CSV. When both embeddings are provided, the `activity_map` from guide-level evaluation is used to filter active genes for gene-level evaluation.

```bash
uv run run_eval \
    --guide_embedding /path/to/guide_embeddings.h5ad \
    --gene_embedding /path/to/gene_embeddings.h5ad
```
| Flag | Description |
|------|-------------|
| `--guide_embedding <path>` | Path to guide-level `.h5ad` file |
| `--gene_embedding <path>` | Path to gene-level `.h5ad` file |
| `--output <path>` | Output CSV path (default: `<embedding_dir>/<timestamp>_eval.csv`) |

---

## Combining AnnData Objects (`pca_optimized`)

**File:** `ops_model/src/ops_model/post_process/combination/cli.py`

Combines per-experiment, per-channel cell-level `.h5ad` files into guide- and gene-level AnnData objects across multiple experiments. The `pca_optimized` method fits a shared PCA on downsampled cells, projects all experiments into that space, then aggregates to guide/gene level. Outputs `<stem>_guide.h5ad` and `<stem>_gene.h5ad`.

```bash
uv run python -m ops_model.post_process.combination.cli \
    --config /path/to/config.yaml
```

| Flag | Description |
|------|-------------|
| `--config <path>` | Path to YAML config file (required) |
| `--output-path <path>` | Override `output_path` from the config |
| `--verbose` | Enable debug-level logging |

See `/hpc/projects/icd.fast.ops/experiments/evaluations/dynaclr/phase/subset_1/subset_1_phase_config.yml` for an example `pca_optimized` config.

---

## Combining AnnData Objects (`classifier`)

**File:** `ops_model/src/ops_model/post_process/combination/cli.py`

Combines per-experiment, per-channel cell-level `.h5ad` files into a gene-level AnnData using a learned MLP aggregator. For each `(perturbation, reporter)` pair, K averaged views are pre-computed from N sampled cells; the MLP is trained to predict perturbation identity from randomly sampled per-reporter views concatenated across reporters. The penultimate-layer representations become the gene-level embeddings. Outputs `<stem>_gene.h5ad`.

```bash
uv run python -m ops_model.post_process.combination.cli \
    --config /path/to/config.yaml
```

| Flag | Description |
|------|-------------|
| `--config <path>` | Path to YAML config file (required) |
| `--output-path <path>` | Override `output_path` from the config |
| `--verbose` | Enable debug-level logging |

See `/hpc/projects/icd.fast.ops/experiments/classifier_aggregation/testing/classifier_test_config.yml` for an example `classifier` config.