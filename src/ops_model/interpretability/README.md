# `interpretability/` — explaining OPS phenotypes

Tooling that asks *what* a perturbation did, rather than producing embeddings. Two
independent subgroups: a **set classifier** over single-cell embeddings, and the
**DiffAE / DiffEx** counterfactual image pipeline.

These scripts are used to generate figure 3 of the paper

```
interpretability/
├── classifier/            set classifier (train / eval / score)
└── diffae/                DiffAE counterfactual pipeline
    ├── classifier/          stage 1 — single-cell classifier to explain
    ├── generator/           stage 2 — conditional DiffAE
    ├── directions/          stage 3 — direction discovery + traversal
    ├── traversal/           shared asset/morphometric precompute
    └── figures/             paper figure scripts
```

## `classifier/` — set classifier

Permutation-invariant classifier that predicts a gene knockout (or a protein-complex
/ pathway label) from a *set* of single-cell embeddings. Runs standalone from an
embedding parquet plus an optional gene→label metadata CSV; entry points are
`train.py` (hydra, configs in `configs/`), `eval.py` (accuracy vs cells per set),
and `score.py` (per-cell scores). Needs the optional `ops_model[classifier]` extra.
See [`classifier/README.md`](classifier/README.md).

## `diffae/` — counterfactual interpretability (DiffEx)

Explains geneKO / protein-complex phenotypes *in image space*: counterfactual
single-cell morphs and per-pixel change maps, in the CellDINO embedding space.
Three core stages — a single-cell classifier to explain, a conditional diffusion
autoencoder, then contrastive direction discovery plus CFG/DDIM traversal — each
with `run.py` (local) and `submit.py` (SLURM); `traversal/` holds shared asset
precompute and `figures/` the paper figures. See [`diffae/README.md`](diffae/README.md).

## Downstream

`classifier/eval.py` generates the raw accuracy-vs-cell-count data consumed by the
paper figure notebook:

[`notebooks/figure_4/eval_accuracy_curves.ipynb`](https://github.com/czbiohub-sf/ops-paper-analysis/tree/main/notebooks/figure_4)
in [`czbiohub-sf/ops-paper-analysis`](https://github.com/czbiohub-sf/ops-paper-analysis)

That notebook plots top-1 accuracy vs the per-class cell-count threshold, for the
phase and fluorescence classifiers at both gene and EBI-complex level.

The diffae model is illustrated in figure 3K and was used to generate the artificial images shown in 3L, 3M and 3N.

