# ops_model

[![License](https://img.shields.io/pypi/l/ops_model.svg?color=green)](https://github.com/ahillsley/ops_model/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/ops_model.svg?color=green)](https://pypi.org/project/ops_model)
[![Python Version](https://img.shields.io/pypi/pyversions/ops_model.svg?color=green)](https://python.org)
[![CI](https://github.com/ahillsley/ops_model/actions/workflows/ci.yml/badge.svg)](https://github.com/ahillsley/ops_model/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/ahillsley/ops_model/branch/main/graph/badge.svg)](https://codecov.io/gh/ahillsley/ops_model)

DL models for analysis of Optical Pooled Screening (OPS) data at CZB SF.

The pipeline has three stages:

1. **Feature extraction** — OPS image crops → per-cell embeddings.
2. **Embedding post-processing** — combine per-experiment / per-marker embeddings into a final guide- and gene-level embedding.
3. **Analysis** — downstream / paper analyses.

> All commands are run from the **monorepo root** (`ops_monorepo/`) with `uv run`.
> See `RUNNING.md` for environment setup and `HIGH_LEVEL_FUNCTIONALITY.md` for the full command reference.

---

## 1. Feature extraction

Each model reads OPS image crops (per experiment + channel, defined in a YAML config) and writes per-cell embeddings. The model entry points take either a single `--config_path` or a `--config_list` (a `.txt` file with one config path per line) and submit one SLURM job per (config, channel) pair.

### Cell-DINO — `models/cell_dino.py`

Cell-DINO embeddings from OPS crops. Reprocesses and overwrites existing outputs unconditionally.

```bash
uv run python ops_model/src/ops_model/models/cell_dino.py \
    --config_list experiments/embedding/configs/cell_dino/cell_dino_config_list.txt
```

See `experiments/embedding/configs/cell_dino/example.yml` for the config format.

### DINOv3 — `models/dinov3.py`

DINOv3 embeddings from OPS crops (one SLURM job per config/channel pair).

```bash
uv run python ops_model/src/ops_model/models/dinov3.py \
    --config_list experiments/embedding/configs/dinov3/<config_list>.txt
```

See `experiments/embedding/configs/dinov3/` for example config files.

### SubCell — `models/subcell.py`

SubCell embeddings (1536 features per cell). Pairs each protein channel with the DNA channel, one SLURM job per pair. Requires `data_manager.dna_channel` and `data_manager.out_channels` in the config.

```bash
uv run python ops_model/src/ops_model/models/subcell.py \
    --config_list experiments/embedding/configs/subcell/subcell_config_list.txt
```

### CellProfiler — `features/cp_features.py`

Hand-crafted CellProfiler morphology and intensity features. Outputs a cell-level CSV plus aggregated AnnData at cell, guide, and gene level.

```bash
uv run python ops_model/src/ops_model/features/cp_features.py \
    --config_path experiments/embedding/configs/cell-profiler/ops0141_cp_20260401.yml
```

See `experiments/embedding/configs/cell-profiler/README.md` for the config format and normalization options.

> The DINO / SubCell models emit feature **CSVs**, which are converted to AnnData (`.h5ad`) with `features/batch_process_embeddings.py` (`--config_list ...`). CellProfiler writes AnnData directly.
>
> A `dynaclr` model exists in `models/dynaclr.py` but does not currently expose a standalone CLI entry point.

---

## 2. Embedding post-processing

The `pca_optimization` pipeline pools per-cell embeddings across experiments that share a biological signal, fits a shared PCA on downsampled cells, sweeps PCA variance thresholds to pick the number of PCs, aggregates to guide/gene level, and scores phenotypic metrics (activity, distinctiveness, and CORUM / CHAD / EBI consistency).

It is driven by a single YAML config whose keys are the CLI argument names in snake_case:

```bash
uv run python -m ops_model.post_process.combination.pca_optimization \
    --config ops_model/src/ops_model/post_process/combination/pca_optimization/example_config.yml
```

See `src/ops_model/post_process/combination/pca_optimization/example_config.yml` for the full key set (feature type, experiments, normalization, PCA threshold, SLURM, etc.). Any flag passed on the command line overrides the config value.

To combine embeddings that live **outside** the standard experiment layout, set `signal_paths:` in the config — a mapping of signal-group name → one `.h5ad` path (or a list of paths to pool) — instead of relying on experiment discovery.

---

## 3. Analysis

_TODO._
