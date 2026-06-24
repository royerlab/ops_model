# DiffEx — counterfactual interpretability for the attention atlas

Explain geneKO / protein-complex phenotypes **in image space**: generate
counterfactual single-cell morphs ("if this control cell were a KD, what would it
look like?") and the per-pixel change map, instead of relying on OP/CP features.
Adapted from DiffEx (arXiv:2502.09663) and Alex Lin's EvolutionaryScale pipeline,
working in the **CellDINO embedding space** that the SetTransformer already uses.

See [PLAN.md](PLAN.md) for the design rationale and the full running log.

## Pipeline (three stages, each a subpackage)

| stage | package | what it does |
|---|---|---|
| 1 | [`classifier/`](classifier/) | per-class single-cell classifier on **top-attention cells** — the model whose decision DiffEx explains / that ranks directions. B = ResNet on phase crops; **C = MLP on CellDINO features** (chosen). |
| 2 | [`diffae/`](diffae/) | **conditional diffusion** generator: UNet that generates a cell image conditioned on its CellDINO embedding (conditioning dropout + EMA + CFG). |
| 3 | [`directions/`](directions/) | **contrastive direction discovery** (InfoNCE + decorrelation, unsupervised) → rank directions by a control-vs-target classifier → **CFG traversal** α∈[−,+] → DDIM-sample a counterfactual strip + Δ-pixel heatmap, verified by re-encoded score. |

## Run order (each stage has `run.py` for local + `submit.py` for SLURM)

```bash
# Stage 1 — classifier (per gene/complex, or sweep --all-classes)
python -m ops_model.models.attention.diffex.classifier.submit --grain complex --all-classes --models C
python -m ops_model.models.attention.diffex.classifier.aggregate --grain complex --model C

# Stage 2 — train the conditional DiffAE (resume-able; gate = embedding/noise ratio)
python -m ops_model.models.attention.diffex.diffae.submit --epochs 120 --batch-size 48
python -m ops_model.models.attention.diffex.diffae.diagnose_conditioning   # conditioning-strength check

# Stage 3 — directions + counterfactual traversal for a target
python -m ops_model.models.attention.diffex.directions.submit --grain geneKO --target HSPA5
```

Outputs: `/hpc/projects/icd.fast.ops/models/diffex/{<grain>,diffae,directions}/…`.

## Status
Stages 1 & 3 built and validated end-to-end; Stage-2 DiffAE conditioning was the
hard part — see PLAN.md (the v1 generator ignored the embedding; the rebuild with
conditioning dropout + EMA fixes it). Current focus: training the DiffAE to a
conditioning ratio high enough for visible morphs, then scaling across targets.
