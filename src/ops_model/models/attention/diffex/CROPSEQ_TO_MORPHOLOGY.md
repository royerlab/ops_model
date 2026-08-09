# Idea: transcriptome-controlled morphology generation (CROP-seq → DiffAE)

**Goal:** use paired CROP-seq on the same geneKO library to let the DiffAE generate *how a cell's phenotype
changes as its transcriptome moves toward a KO state* — i.e. drive the traversal by a **transcriptional
direction** (CROP-seq) instead of (or in addition to) the CellDINO morphological direction.

## Reframing vs today
Current DiffEx traversal = real cell → DDIM-inverted `xT` (identity/nuisance) + **CellDINO gene-direction**
(morphology conditioning), morph α NTC→KO, decode image. This idea keeps the entire image decoder (DDIM +
inversion + guidance `w`) and swaps the *driver* to a transcriptional signature.

## Hard constraint (shapes everything): no single-cell pairing
CROP-seq is destructive scRNA-seq; OPS is imaging — **no cell is in both modalities**. So supervision is only
**per-perturbation** (gene KO → mean transcriptional shift Δt_g AND a morphological distribution), never
cell-level `(transcriptome → image)`.
- Model learns **transcriptional-signature → morphological-distribution**; within-gene image variation comes
  from the stochastic `xT` (same as today).
- Conditioning vectors are **per-gene pseudobulk** (or per-guide if guide calls are clean).

## Two paths

### Path B — reuse the trained DiffAE via a transcriptome→CellDINO map (POC first)
Fit a perturbation-level regressor `Δt_g → ΔCellDINO_g` (linear → small MLP) over the shared KOs. A
transcriptional vector → predicted CellDINO shift → **existing morpho DiffAE renders it**.
- Pros: reuses the whole trained pipeline + viewer; days not weeks. **The map's R² is itself a headline
  result** ("fraction of KO morphology predictable from KO transcriptome").
- Cons: bottlenecked through CellDINO.

### Path A — condition the DiffAE directly on transcriptome (full, CPA-flavored)
Project t through `cond_proj` into the FiLM/cross-attn slot the CellDINO emb uses now; train on
`(image_i, t_{gene(i)})`. Cleanest is a **CPA-style shared perturbation embedding** `e_g`: a transcriptome
decoder reconstructs CROP-seq (`NTC + e_g`), the DiffAE decodes the image (`anchor xT + e_g`), `e_g` shared →
ties the modalities through one latent. Any transcriptional state → `e_g` → image.
- Pros: end-to-end; supports unseen/combined signatures + continuous "dial a pathway, watch morphology".
- Cons: real training effort; guard against collapse to gene-means (xT + guidance mitigate, as today).

**Plan:** B as a weekend POC (also tests whether transcriptome predicts morphology at all) → A if promising.

## Transcriptional vector options (cheapest first)
1. pseudobulk logFC vs NTC (per gene); 2. learned scRNA latent (scVI/PCA) mean per gene; 3. pathway/program
module scores (most interpretable "dials"). Start with (1)/(2) for the direction, expose (3) as the control.

## The novel payoff: transcriptome↔morphology divergence map
Plot every gene by (transcriptional effect size, morphological effect size). The DiffAE then lets you *see*:
- **transcriptionally loud, morphologically silent** → counterfactual "what it would look like if it manifested"
- **morphologically loud, transcriptionally quiet** → morphology carrying signal transcriptome misses
- cross-modal interpolation between two genes' transcriptomes; agreement w/ CellDINO morph = validation,
  divergence = the interesting biology.

## Viewer tab concept
"Transcriptome → Morphology" tab: α-slider drives the **transcriptional** traversal; side-by-side vs the
existing CellDINO-driven morph (agreement = validation, divergence = biology). Reuses traversal/montage render.

## To scope
1. CROP-seq path/format (h5ad? per-cell w/ guide calls or pseudobulked); gene-overlap with the imaging 1000-lib.
2. per-gene vs per-guide signatures.
3. matched NTC/control in CROP-seq to define Δ.
4. payoff emphasis: generator vs divergence-map.
