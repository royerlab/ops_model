<a href="https://github.com/czbiohub-sf">
  <img src="https://avatars.githubusercontent.com/u/28747162?v=4" alt="Chan Zuckerberg Biohub San Francisco" width="90" align="right" />
</a>

# ops_model

**This repository is the result of work done at the [Chan Zuckerberg Biohub San Francisco](https://github.com/czbiohub-sf).**

DL models for analysis of Optical Pooled Screening (OPS) data at CZB SF.

## Preprint

This repository accompanies the following preprint, and should be preserved and kept public indefinitely:

> [A multimodal perturbation atlas defines the phenotypic resolution of cellular morphology](https://www.biorxiv.org/content/10.64898/2026.06.01.728087v1.abstract) — bioRxiv, 2026. doi:10.64898/2026.06.01.728087

## Data availability

The raw imaging dataset that these pipelines process is available for download through the Biohub OPS Explorer portal:

> [OPS Explorer — perturbation atlas collection](https://biohub.ai/ops-explorer?collection=6a3f8b91-1c5e-4d3a-9b4c-f7e0a2d8b6f3)


The pipeline has three stages:

1. **Feature extraction** — OPS image crops → per-cell embeddings.
2. **Embedding post-processing** — combine per-experiment / per-marker embeddings into a final guide- and gene-level embedding.
3. **Analysis** — downstream / paper analyses.

---

## Installation

`ops_model` is **not a standalone package.** It is one submodule of the
[`czbiohub-sf/ops_monorepo`](https://github.com/czbiohub-sf/ops_monorepo) uv workspace and is only
supported as a piece of that larger project. Install the monorepo, not this repo on its own:

```bash
git clone --recurse-submodules git@github.com:czbiohub-sf/ops_monorepo.git
cd ops_monorepo
uv sync
```

All commands are then run from the **monorepo root** (`ops_monorepo/`) with `uv run`, as in every
example below. `RUNNING.md` covers environment setup, including the `module load uv` and
`UV_CACHE_DIR` steps for cluster use.

The pipelines read and write under `$OPS_BASE_PATH`, which has no default and must be set:

```bash
export OPS_BASE_PATH="/path/to/ops_data"
```

### Why it only works inside the monorepo

- This package imports `ops_utils` throughout (78 import sites), and two `diffae` modules import
  `organelle_profiler`. Both are sibling submodules resolved through the uv workspace, and neither
  is published on PyPI.
- An unrelated package named `ops-utils` *does* exist on PyPI, so resolving this package's
  dependencies outside the workspace pulls in the wrong project entirely.
- The monorepo root carries dependency overrides that this package's resolution relies on — for
  example `matplotlib>=3.10.5`, which defeats a strict pin inherited from `cytoself`, and
  `iohub>=0.3.7`.

### Dependencies

Declared dependencies and the optional extras (`classifier`, `models`, `test`, `dev`) live in
[`pyproject.toml`](pyproject.toml); Python 3.12 is required. Exact resolved versions for the whole
workspace are pinned in the monorepo's `uv.lock`, which is the authoritative environment
specification.

---

## Paper figures

The pipelines in this repo produce the underlying data — CSVs and AnnData objects — plus diagnostic plots. The **figures in the paper are not generated here.** To regenerate a specific figure, see [`czbiohub-sf/ops-paper-analysis`](https://github.com/czbiohub-sf/ops-paper-analysis), which holds one notebook per figure and reads the outputs described below.

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

### CellProfiler — `models/cellprofiler/cp_features.py`

Hand-crafted CellProfiler morphology and intensity features. Outputs a cell-level CSV plus aggregated AnnData at cell, guide, and gene level.

```bash
uv run python ops_model/src/ops_model/models/cellprofiler/cp_features.py \
    --config_path experiments/embedding/configs/cell-profiler/ops0141_cp_20260401.yml
```

See `experiments/embedding/configs/cell-profiler/README.md` for the config format and normalization options.

> The DINO / SubCell models emit feature **CSVs**, which are converted to AnnData (`.h5ad`) with `features/batch_process_embeddings.py` (`--config_list ...`). CellProfiler writes AnnData directly.
>
> A `dynaclr` model exists in `models/dynaclr.py` but does not currently expose a standalone CLI entry point.

---

## 2. Embedding post-processing

The `pca_optimization` pipeline pools per-cell embeddings across experiments that share a biological signal, fits a shared PCA on downsampled cells, sweeps PCA variance thresholds to pick the number of PCs, aggregates to guide/gene level, and scores phenotypic metrics (activity, distinctiveness, and CORUM / CHAD / EBI consistency).

Inputs are always explicit: each `--signal` names one signal group and lists the per-cell `.h5ad` files to pool into it. There is no experiment discovery and no channel/experiment filtering.

```bash
uv run python -m ops_model.post_process.combination.pca_optimization \
    -o <output_root> --cell-dino --zscore-per-experiment \
    --chad-annotation <chad.yml> \
    --ebi-annotation <ebi_complexes.yaml> \
    --gene-panel <annotated_gene_panel.csv> \
    --signal Phase=<...>/ops0146_Phase.h5ad,<...>/ops0147_Phase.h5ad \
    --signal ER_SEC61B=<...>/ops0146_ER.h5ad \
    --run-tag my_run --slurm
```

`-o`, the three annotation inputs, and at least one `--signal` are required — there are no default paths. Everything else (feature type, normalization, PCA threshold, sampling budget, SLURM) can come from a YAML config whose keys are the CLI argument names in snake_case, passed with `--config`; any flag given on the command line overrides the config value. `--signal` is repeatable and so is CLI-only. See `src/ops_model/post_process/combination/pca_optimization/example_config.yml` for the key set.

The feature-mode flag, `--zscore-per-experiment`, `--run-tag`, the threshold mode, the distance metric and a non-mean `--agg-method` each add a path segment beneath `-o`, so independent runs never overwrite each other.

---

## 3. Analysis

### Cell-count titration — `post_process/combination/titration/`

How much of the phenotypic signal survives when you spend fewer cells? Both scripts repeatedly subsample the cell-level h5ads that `pca_optimization` produced, re-aggregate to guide level, and re-score the same five metrics at each cell budget — producing "score vs cells" curves. Signals are always named explicitly; there is no discovery.

`titration.py` titrates each reporter **independently**, one curve per reporter:

```bash
uv run python -m ops_model.post_process.combination.titration.titration \
    -o <pca_optimization_root> \
    --cell-dino --paper-v1 --per-guide-median-titration --slurm
```

`combined_titration.py` titrates **groups** of reporters together, h-concatenating their NTC-normalized guide features before scoring — so you can ask whether combining markers beats either alone:

```bash
uv run python -m ops_model.post_process.combination.titration.combined_titration \
    -o <pca_optimization_root> --cell-dino --paper-v1 \
    --per-guide-median-titration --slurm \
    --group pRb=<per_signal>/pRb_4i_cells.h5ad \
    --group p21=<per_signal>/p21_4i_cells.h5ad \
    --group pRb_p21=<per_signal>/pRb_4i_cells.h5ad,<per_signal>/p21_4i_cells.h5ad
```

Both write one CSV per curve plus diagnostic plots, and are cache-aware — re-running fills in only the missing cell budgets. The titration mode flag selects both the sampling schedule and the output subdirectory; see [`titration/README.md`](src/ops_model/post_process/combination/titration/README.md) for the mode table, the full flag set, and the output layout.

### Interpretability — `interpretability/`

What did a perturbation actually *do* phenotypically? Two independent approaches, one working in embedding space and one in image space.

**Set classifier** — `interpretability/classifier/`

A permutation-invariant classifier that predicts a gene knockout (or an EBI protein-complex label) from a *set* of single-cell embeddings, which turns "is this phenotype detectable?" into a measurable accuracy. `train.py` and `eval.py` are Hydra entry points reading the YAMLs in `configs/`; `eval.py` sweeps accuracy against the per-class cell-count threshold.

```bash
uv run python -m ops_model.interpretability.classifier.train \
    --config-name=train_set_classifier_phase_ebi

uv run python -m ops_model.interpretability.classifier.eval \
    --config-name=eval_set_classifier
```


**DiffAE counterfactuals** — `interpretability/diffae/`

EExplores changing phenotypes *in image space*: generating counterfactual single-cells that shift a classifier's decision, so a phenotype can be inspected visually rather than only scored. Three sequential stages — a per-cell classifier to explain, a conditional diffusion autoencoder, then contrastive direction discovery plus DDIM traversal — each with a `run.py` (local) and a `submit.py` (SLURM).

```bash
uv run python -m ops_model.interpretability.diffae.classifier.run --model C --gene HSPA5
uv run python -m ops_model.interpretability.diffae.generator.run
uv run python -m ops_model.interpretability.diffae.directions.run --target HSPA5
```

Stage 3 also accepts `--grain complex` to explain a protein complex rather than a single gene. `traversal/` holds shared crop and morphometric precompute; `figures/` holds the figure scripts built on these outputs.

See [`interpretability/README.md`](src/ops_model/interpretability/README.md) for the layout, and the per-subgroup READMEs for the full flag sets and output layouts.

---

## Ownership and maintenance

This repository is owned by the [Leonetti group](https://biohub.org/leonetti/) at the [Chan Zuckerberg Biohub San Francisco](https://github.com/czbiohub-sf).

Maintainers (see also [`.github/CODEOWNERS`](.github/CODEOWNERS)):

- Alexander Hillsley ([@ahillsley](https://github.com/ahillsley))
- Gav Sturm ([@gav-sturm](https://github.com/gav-sturm))

Please open an issue or pull request for questions, bugs, or contributions.
