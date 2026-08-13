# DiffEx — counterfactual interpretability for the attention atlas

Explain geneKO / protein-complex phenotypes **in image space**: generate
counterfactual single-cell morphs ("if this control cell were a KD, what would it
look like?") and the per-pixel change map, instead of relying on Organelle-Profiler/Cell-Profiler features.
Adapted from DiffEx (arXiv:2502.09663) in conjunction with the set-classifier,
working in the **CellDINO embedding space**.


## Pipeline (three stages, each a subpackage)

| stage | package | what it does |
|---|---|---|
| 1 | [`classifier/`](classifier/) | per-class single-cell classifier on **top-attention cells** — the model whose decision DiffEx explains / that ranks directions. B = ResNet on phase crops; **C = MLP on CellDINO features** (chosen). |
| 2 | [`generator/`](generator/) | **conditional diffusion** generator (the DiffAE): UNet that generates a cell image conditioned on its CellDINO embedding (conditioning dropout + EMA + CFG). |
| 3 | [`directions/`](directions/) | **contrastive direction discovery** (InfoNCE + decorrelation, unsupervised) → rank directions by a control-vs-target classifier → **CFG traversal** α∈[−,+] → DDIM-sample a counterfactual strip + Δ-pixel heatmap, verified by re-encoded score. |

## Run order (each stage has `run.py` for local + `submit.py` for SLURM)

```bash
# Stage 1 — classifier (per gene/complex, or sweep --all-classes)
python -m ops_model.interpretability.diffae.classifier.submit --grain complex --all-classes --models C
python -m ops_model.interpretability.diffae.classifier.aggregate --grain complex --model C

# Stage 2 — train the conditional DiffAE (resume-able; gate = embedding/noise ratio)
python -m ops_model.interpretability.diffae.generator.submit --epochs 120 --batch-size 48
python -m ops_model.interpretability.diffae.generator.diagnose_conditioning   # conditioning-strength check

# Stage 3 — directions + counterfactual traversal for a target
python -m ops_model.interpretability.diffae.directions.submit --grain geneKO --target HSPA5
```
