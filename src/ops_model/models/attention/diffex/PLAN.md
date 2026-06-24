# DiffEx interpretability — plan

Living design doc. Goal: interpret geneKO / protein-complex phenotypes **into image space**
using a DiffEx-style diffusion counterfactual (arXiv:2502.09663), since OP/CP classical
features are judged too weak to describe the phenotypes.

## What is DiffEx?
DiffEx (*Explaining a Classifier with Diffusion Models to Identify Microscopic Cellular
Variations*, arXiv:2502.09663) explains any image classifier by generating visually interpretable
**counterfactuals** — showing, in pixel space, what about an image drives the classifier.
- **Architecture:** a **diffusion autoencoder (DiffAE)** — semantic encoder → low-dim latent
  `z_sem`; conditional diffusion decoder reconstructs the image from `z_sem`. The classifier score
  is concatenated onto `z_sem`; a bank of MLP **direction models** is trained in that latent with a
  **contrastive loss** → distinct, disentangled directions. Explain class k = shift `z_sem` along a
  direction and decode.
- **Interpretability features we exploit:** counterfactual morphs; a *global, reusable* attribute
  vocabulary (directions shared across all classes — an image-grounded OP/CP replacement);
  per-class attribute ranking; classifier-agnostic & forward-only (no retraining); continuous edit
  strength α (dose-like morphs); quantitative faithfulness via re-encoding.

Status: **designing the per-cell classifier** (the immediate task). Diffusion model + DiffEx
guidance are later phases, sketched here for context.

### Scope (locked)
- **4 classifiers** = 2 modalities × 2 grains:
  1. phase-only · geneKO  (1001 KO + NTC-way)
  2. phase-only · complex (98-way, EBI)
  3. all-fluorescence · geneKO
  4. all-fluorescence · complex
- **2 diffusion models** (phase-only, all-fluorescence) — unconditional, shared across grains.
- **Negative contrast = `distinct`** (vs all other geneKOs / complexes), to isolate exactly
  what is unique to each perturbation. Falls out of the multi-class softmax for free (§1.3).

---

## 0. Why this shape (decisions already made)

- **Drop OP/CP.** DiffEx discovers its explanatory vocabulary directly in image space, so we
  don't need to translate CellDINO → OP/CP → language. The diffusion model *is* the vocabulary.
- **One unconditional diffusion model, not one per gene.** It only learns to generate realistic
  single-cell crops. DiffEx's *classifier guidance* does all per-gene steering at inference.
- **Diffusion autoencoder (DiffAE), faithful to the DiffEx paper.** Semantic encoder → latent
  `z_sem` + conditional diffusion decoder, trained jointly. NOT latent diffusion (no separate VAE)
  and NOT a bare guided DDPM. The contrastive direction discovery (§3) requires this semantic
  latent — it's the core of the method, not an add-on.
- **A small per-cell classifier, NOT the SetTransformer.** The SetTransformer never sees pixels
  (input = bag of CellDINO embeddings), so it gives DiffEx no `image→logit` gradient. Rather than
  bolt CellDINO on the front and wrestle the per-bag→per-image mismatch, we train a clean
  per-image classifier. The SetTransformer still does what it's proven at — **selecting the cells**
  the classifier trains on. (Full-stack `image→CellDINO→SetTransformer` faithfulness check is a
  later reviewer-defense nicety, not on the critical path.)

## Final product (per gene-KO / per complex atlas page)

1. **Evidence cells** — top-X attention cells passing the mAP/accuracy threshold. X is variable
   per gene; the value itself is the **penetrance readout** (HSPA5 ≈ 10, RPL10 ≈ 800). Printed.
2. **Counterfactual morph** — NTC → KO (and optionally KO → NTC, often cleaner). Averaged across
   the top-X cells, not one cherry-picked cell.
3. **Per-channel difference heatmap** — always show **phase + the top highest-mAP channel(s) for
   that class** (NOT every channel); localizes the phenotype to the relevant marker.
4. **Faithfulness number** — % of counterfactuals whose re-encoded logit actually flipped to KO.

---

## 1. THE CLASSIFIER (current focus)

**3 candidate classifiers (DiffEx target) — per Alex, 2026-06-16.** Cell selection is settled
(top-X attention cells, §1.7); the options differ in WHAT classifier scores them — the source of
DiffEx's differentiable `image → class-k logit`. All train/score on top-attention cells (which
classify accurately).

- **A. SetTransformer native** — score a generated cell by inserting it into a real NTC reference
  bag and reading the bag logit. Most faithful (actual deployed model, no new training).
  **Risk (Alex):** the SetTransformer may not treat one synthetic cell as real, and/or +1 cell
  won't move the bag logit → weak faithfulness signal. Cross-check, don't depend on it.
- **B. ResNet CNN on single cells**, trained only on top-attention cells. DiffEx-standard.
  **Only option needing NO CellDINO encoder** for generated images (scores pixels directly).
  Simplest, cleanest gradients; not SOTA representation. Detail in §1.1–1.6 below.
- **C. CellDINO features + MLP**, trained only on top-attention cells. More SOTA / accurate, light
  to train, reuses the already-provided CellDINO features. Scoring *generated* images runs the
  **local** CellDINO encoder (`ops_model/models/cell_dino.py` → `CellDinoModel`: channel-adaptive
  DINO ViT-L/16, resize 224 + per-image z-score, `in_channels=1`); heavier per step than B, NOT a blocker.

**Recommendation:** prototype **B and C on HSPA5 (phase)** — **fully unblocked locally** (we have the
phase attention rankings, the crops, and the CellDINO encoder). Pick by per-cell classification
accuracy. B = lightest (small CNN on pixels, no CellDINO in the loop); C = stronger classifier but
runs ViT-L per generated image. A = faithfulness cross-check only.

DiffEx needs a differentiable `image → class-k logit`. Option B (CNN) design detail:

### 1.1 Input / modality  — **DECISION NEEDED**
- SetTransformer regime is **phase-only** (paper-v1). To stay faithful + keep the diffusion model
  to 1 channel, **recommend starting phase-only**. Extend to phase+fluor later (fluor carries the
  ER/mito visual signal the reader wants, but multiplies diffusion difficulty).
- Crop: reuse `ops_utils.data.bbox_utils.BaseDataset` — 128×128, multi-channel, cell mask
  available. Same loader the atlas uses. **No new data infra.**
- Open: feed the cell mask as an extra channel (focus model on the cell, suppress neighbours)?
  Likely yes — cheap confound reduction.

### 1.2 Architecture  — DECISION: backbone (start CNN, escalate if needed)
DiffEx is **classifier-agnostic** (it explains a plain supervised classifier; doesn't need
internal layers). So this slot is a free choice. What DiffEx needs is NOT top accuracy but:
clean image→logit gradient, confound-robustness (keys on biology, not plate/intensity), and
faithfulness to the real phenotype.

- **v1 (recommended): small from-scratch CNN** (ResNet18-ish, N-channel stem). Clean gradient,
  no giant frozen encoder in the path, easy to keep confound-robust, fail-fast for the PoC.
  Honest caveat: a from-scratch ResNet is **standard, not SOTA** for cell-image representation.
- **Escalation path if CNN counterfactuals are too weak/insensitive:** fine-tuned **CellDINO**
  (your near-SOTA self-supervised ViT) or a channel-aware ViT (ChannelViT / DINO4Cells family).
  SOTA representation → more sensitive classifier → more sensitive counterfactuals, at the cost
  of a heavier gradient path and higher confound risk (frozen SSL features can encode batch).
- Tradeoff is real because phenotypes are subtle. Decide empirically on HSPA5: if the CNN can't
  separate HSPA5 cleanly cross-experiment, escalate the backbone before blaming the generator.

### 1.3 Task framing  — multi-class, distinct contrast for free
- **One N-way softmax classifier** per (modality × grain): 1001-KO+NTC-way for geneKO,
  98-way for complex. Shared backbone, per-class linear heads.
- DiffEx guides toward `logit_k`. Because softmax is normalized against all other classes,
  **guiding toward class k IS the `distinct` contrast** (what's unique to k vs everything else) —
  no separate binary models needed.
- PoC = train the geneKO N-way model, then run DiffEx toward the HSPA5 logit.
- Caveat (logged, not a blocker): distinct **suppresses shared phenotypes** by construction
  (two ER-stress genes won't show their shared ER signature, only their difference). Keep
  **vs-NTC as a complementary second pass** for genes where the absolute phenotype is wanted.

### 1.4 Training set (a query, not new infra)
- Positives for class k: top-X attention cells for k, ranked by `attn_geneko` / `pma_attention`,
  passing the mAP/accuracy threshold. X variable per gene (→ penetrance).
- Negatives = the **top-attention cells of the OTHER classes** (like-with-like), NOT random/weak
  cells of them — else the model relearns "has *any* phenotype" instead of "has *this* one".
  With a softmax over top-attention cells per class this is automatic.
- Sources:
  - attention sidecar: `/hpc/projects/icd.fast.ops/models/alex_lin_attention/v4/expansion_v1/per_experiment_v4_attn.parquet`
    (cols: experiment, well, segmentation_id, attn_ebi, attn_geneko, attn_chad, ...)
  - PMA parquets: `.../v4/pma_*` (cols: gene, experiment, well, segmentation, pma_attention)
  - cell-set builder already exists: `organelle_profiler.feature_extraction.consolidate_top_attention_cells`
- Join `(experiment, well, segmentation_id)` → bbox → `BaseDataset` crops.

### 1.5 Confound guardrail (critical — or DiffEx faithfully explains the batch effect)
- Balance / stratify NTC negatives across the same experiments+wells as the positives.
- **Validate cross-experiment** (train on subset of experiments, test on held-out) — biological
  signal generalizes, plate/intensity artifacts don't.
- Sanity bar: per-cell classifier AUROC should track the SetTransformer's per-gene mAP ordering
  (sharp genes like HSPA5 easy, diffuse genes like RPL10 hard).

### 1.6 Success criteria
- HSPA5-vs-NTC held-out AUROC clearly > 0.5 and > a same-data **all-cells** classifier
  (proves attention selection removes confounds).
- Cross-experiment generalization holds.

---

## 2. Diffusion autoencoder (DiffAE) — the real cost/risk
Faithful to DiffEx. Two jointly-trained parts on single-cell crops (one DiffAE per modality:
phase-only, all-fluor):
- **Semantic encoder** → `z_sem` (a low-dim semantic latent capturing cell appearance).
- **Conditional diffusion decoder** (`UNet2D`, N input channels) that reconstructs the crop
  conditioned on `z_sem` (+ stochastic DDIM latent for detail).
- Data is NOT the constraint (millions of cells); compute/engineering is.
- De-risk on HSPA5 PoC before scaling. First sanity check: round-trip reconstruction quality
  (encode→decode) on held-out cells — if it can't reconstruct, directions are meaningless.

## 3. Contrastive direction discovery (the interpretability core)
Faithful to DiffEx — this is where the explanation comes from, NOT per-image classifier guidance:
- Concatenate the §1 classifier score onto `z_sem` → semantic code.
- Train a bank of MLP direction models `D_1…D_N` that each shift the code by `α·Δz_k`, with a
  **contrastive loss**: edits from the same direction stay similar, edits across directions stay
  dissimilar → distinct, disentangled, reusable attributes.
- **Distinct contrast (per §1.3):** select the direction(s) that move the classifier toward
  class k's logit → "what is distinct about geneKO/complex k". The discovered `D_1…D_N` form a
  **shared attribute vocabulary across all 1001 genes / 98 complexes** — the real payoff vs a
  one-off morph, and an image-grounded replacement for OP/CP.
- Faithfulness check, baked in day one: re-encode the edited image through §1 classifier, confirm
  the logit actually moved toward k; report % success on each atlas page.

---

## Milestones / de-risk order
1. **Classifier on HSPA5** (this task) — fail-fast signal that the cell sets are learnable.
2. **DiffAE** on the same crop set; gate on round-trip reconstruction quality (§2).
3. **Contrastive direction discovery** (§3); pick HSPA5's distinct direction; eyeball morph +
   per-channel heatmap + flip-rate.
4. If HSPA5 works → scale to atlas (reuse shared directions). If not → it won't work anywhere; stop.

## Open questions
- Phase-only vs phase+fluor for v1? (recommend phase-only)
- Mask as extra input channel? (recommend yes)
- Threshold definition for "X cells that pass": fixed mAP cutoff vs per-gene accuracy knee?
- Negative contrast for v1: NTC only, or also distinct/global?

---

## §1.7 Cell selection (settled, all options) + Option-A bag scoring
**Settled — cell selection (all 3 classifier options):** cells fed to DiffEx / used to train the
classifier = the **top-X attention cells** (PMA attention rank), exactly as used for the atlas.

**Option A only — scoring a *generated* image with the bag model:** insert the generated cell into
a fixed real NTC reference bag, read logit_k. Alex's concern: the SetTransformer may not treat the
synthetic cell as real and +1 cell may not move the bag logit. → why A is a cross-check, and why
B/C (self-contained single-cell classifiers) are the primary path.

## Assets inventory (checked 2026-06-16)
**Have locally:**
- **CellDINO encoder IS local**: `ops_model/models/cell_dino.py` → `CellDinoModel` (channel-adaptive
  DINO ViT-L/16, ckpt `channel_adaptive_dino_vitl16_pretrain_cells-…pth`, resize 224 + per-image
  z-score, `in_channels=1`). Plus the precomputed CellDINO feature dumps (below). → image→embedding
  for *generated* cells is available locally; no encoder request needed.
- **MixedChannelClassifier** code (`train_set_classifier.py`) + inference (`export_pma_attention.py`).
- Per-gene **embedding dumps WITH cell_metadata**: `v4/{train,val}_ops_zstdcontrol_cdino_v2/`
  (+ `metadata.pt` w/ gene_to_idx, channel_to_idx). Metadata → zarr crop mapping works
  (`_load_cell_crop`: experiment/well/x_pheno/y_pheno/segmentation_id/zarr_channel_index).
- Attention outputs (`per_experiment_v4_attn.parquet`, `pma_phase_cells_*`).
- katamari clone on branch `main` (esmc_paper commit) — likely NOT Alex's attn-classifier branch.

## Requests for Alex L. (to scale beyond the phase·geneKO PoC)
**Note:** the B/C HSPA5 phase prototype needs NOTHING from Alex — phase attention rankings, crops,
and the CellDINO encoder are all local.
1. **Trained MixedChannelClassifier checkpoints** for the 4 models (phase·geneKO, phase·complex/EBI,
   fluor·geneKO, fluor·complex) — the actual `.pt` files or the **wandb artifact IDs**
   (`alex-lin/cellstate-set-classifier/model-…`). Needed for option A, and to generate attention
   rankings for the models we don't already have rankings for.
2. **Fluorescent + complex (EBI/CHAD) embedding dumps + label maps** — only phase·geneKO dumps
   (`*_cdino_v2`) confirmed local. Need the fluor dumps and the complex `label_map_path`.
3. **If pursuing option A:** how to score a single generated cell inside a real reference bag (his
   concern: one synthetic cell may not move the bag logit).
4. **The right git branch** to check out for the latest attn-classifier code (clone is on `main`).

---

## Build log

### 2026-06-16 — classifier B/C package built
Package: `ops_model/models/attention/diffex/classifier/` (config, data, models,
celldino_features, train, run, submit, README). Locked params: binary HSPA5-vs-rest,
negatives = other genes' top-5 (distinct), 1000/class, 160×160 phase crops (no mask),
3-way train/val/test split grouped by experiment (val=selection, test=clean reported AUROC;
train+val AUROC logged per epoch to watch over/under-fit). Outputs under
`/hpc/projects/icd.fast.ops/models/diffex/<gene>/`.
- **Decision:** option C reuses the **local** CellDINO encoder (`cell_dino.py`) on the SAME
  crops B uses (cached) — no dump-join; B and C see identical cells.
- **Verified:** full B pipeline end-to-end on CPU (tiny config) — filtered parquet read (no OOM),
  store resolution, crops materialized non-degenerate, train→AUROC→artifacts. SLURM submitter
  dry-run OK (2 GPU jobs). (Crop cache key includes mask state so masked/unmasked don't collide.)
- **Next (GPU):** `python -m ops_model.models.attention.diffex.classifier.submit --gene HSPA5`
  → compare B vs C held-out AUROC, pick the DiffEx target. C needs GPU (CellDINO ViT-L).

### 2026-06-16 — HSPA5 PoC results (job 34280479, experiment-grouped split, 1000/class)
| model | test AUROC | val | train@best |
|---|---|---|---|
| B (ResNet on crops) | 0.80 | 0.85 | 1.00 (overfits) |
| **C (CellDINO+MLP)** | **0.96** | 0.95 | 0.999 |
- PoC validated: HSPA5 top-attention cells are cleanly + cross-experiment classifiable → real DiffEx target.
- **C is the chosen DiffEx target** (far more sensitive/generalizing; B memorizes). C scores generated
  counterfactuals via the local `cell_dino.py` encoder.
- Outputs: `/hpc/projects/icd.fast.ops/models/diffex/HSPA5/`.
- **Next:** (a) sanity-check a diffuse gene (e.g. RPL10, top-800 penetrance) to confirm the approach
  holds across penetrance; (b) begin the DiffAE stage (§2) with C as the classifier.

### 2026-06-16 — 98 EBI complexes + NTC sweep (model C, jobs 34285052 + 34285381)
All 99 bins ran. Per-class test AUROC (distinct vs pooled top cells of other perturbations),
experiment-grouped split. Range 0.748–0.958, median 0.860.
- Most distinct: Chaperonin-containing T-complex 0.96, DNA pol α:primase 0.95, eIF4F 0.95,
  replication fork protection 0.94, COPI 0.93.
- Least distinct: RNA Pol II 0.75, U1 snRNP 0.75, SRP 0.76, NSL HAT 0.78.
- **NTC = 0.86 (mid-pack) is EXPECTED, not a confound** — NTC lacks the CRISPR cut, so it's a real
  distinct (no-DSB) state. (Earlier "red flag" retracted.)
- Artifacts: `…/diffex/complex/auroc_ranking_C.csv`, `auroc_hist_C.png`.
- Takeaway: every complex's top-attention cells carry a separable phenotype → strong DiffEx targets
  across the board; clear biologically-sensible ranking.

### 2026-06-16 — DiffAE stage (§2) built + PoC launched (job 34292127)
Package: `diffex/diffae/` (config, data, model, train, recon, run, submit). Faithful DiffAE:
ResNet18 semantic encoder → `z_sem` (512), conditional `diffusers.UNet2DModel` decoder with
`z_sem` injected via `class_embed_type="identity"` (→ time embedding), trained jointly with DDPM
denoising loss. Gate = DDIM-invert→reverse reconstruction (PSNR + montage; uses DDIMInverseScheduler).
- Decisions (locked): broad training set (all classes incl NTC, all ranks, ~50k crops), 160×160
  phase, per-image z-score/3 normalization, PoC-first.
- Reuses the classifier crop pipeline (`materialize_crops`). Verified end-to-end on CPU (synthetic):
  conditioning, DDPM step, DDIM recon, checkpoints all work.
- PoC launched: 50k crops, 80 epochs, batch 32, 1 GPU, 720min. Outputs → `…/diffex/diffae/phase_v1/`.
- **Gate to watch:** reconstruction PSNR (recon montages every 10 epochs). If it reconstructs cells
  faithfully → proceed to §3 (contrastive direction discovery). If not → fix before directions.
- **Next stage (§3):** contrastive direction discovery on `z_sem` + the option-C classifier score.

### 2026-06-17 — DiffAE v1 result + ARCHITECTURE SWITCH to Alex's design
- v1 run (job 34298429, jointly-trained encoder): trained healthily, **reconstruction PSNR ~33–34 dB,
  visually faithful** (gate passed), converged ~ep9, but hit the 12h wall at ep37/80 (too slow/many
  epochs). First quota failure (34292127) fixed by freeing disk + wrapping recon writes in try/except.
- **Alex (EvolutionaryScale) already has a working DiffEx** (Notion: Imaging AI). Key design: condition
  the image-DDPM on the **FROZEN backbone embedding** (not a learned encoder); discover K direction
  MLPs **unsupervised** (InfoNCE + VICReg); rank directions **post-hoc** with a logistic-regression
  classifier (control vs KD); traverse α∈[−3,+3], DDIM-sample per step; verify by monotonic score.
- **SWITCHED to Alex's design** (job 34312003): DiffAE now conditions on the **frozen CellDINO
  embedding** (dropped the learned encoder; `cond_proj` injects it into the UNet time-embedding).
  Generator, option-C classifier, and SetTransformer now all live in the SAME CellDINO space →
  Stage-3 directions are discovered there and ranked by option-C directly. epochs 20, reuses crop
  cache + new CellDINO-embedding cache.
- **§3 plan (Alex's recipe):** K direction MLPs (InfoNCE+VICReg, unsupervised) on CellDINO embeddings
  → rank by option-C classifier score shift → α-traversal → DDIM-sample images → verify monotonicity.

### 2026-06-18 — Stage 3 built + first HSPA5 traversal (job 34385092)
Package `diffex/directions/` (config, model=DirectionBank, losses=InfoNCE+decorrelation,
train_directions, rank=LR score-shift, data=gather target+NTC, traverse=DDIM invert→reverse +
re-encode verify, run, submit). Verified 2a+2b on synthetic; full pipeline ran on GPU in 5 min.
- HSPA5: LR acc 0.999, selected direction #6 (shift 1.31), **6/8 traversals monotonic**, mean
  score Δ +0.71 (correct sign). **Full DiffEx machine works end-to-end.**
- **BUT effect is weak**: re-encoded scores stay on NTC side (−6..−14), visual change subtle.
  Cause: unit direction × α≤3 ≪ the real control→KD gap (clusters far apart, LR acc 0.999); plus
  x_T anchoring from DDIM inversion.
- **Fix (next): scale α to ‖mean(KD)−mean(NTC)‖** (likely ~10–30, not 3); optionally reduce x_T
  anchoring; train DiffAE longer. Outputs: `…/diffex/directions/geneKO/HSPA5/`.

### 2026-06-18 — Stage 3 retry: gap-scaled α + Δ-pixel heatmap overlay (job 34386543)
- Added: heatmap overlay (Δpixels vs α=0, red/blue) on the traversal montage; α now in units of
  the control→KD gap (Δ=9.64). 6/8→1/8 monotonic, score Δ 0.71→0.28 — **gap-scaled α OVERSHOT**.
- **Diagnosis from montage:** at large α the edits land on crop BORDERS/background, not the cell
  → embedding goes off-manifold, DiffAE renders boundary artifacts. Two root issues:
  (a) crops are **unmasked** → direction may exploit context (confluency/neighbors), not the cell;
  (b) DiffAE **edit-fidelity** limited (reconstructs well but doesn't render edits onto the cell).
- **Next options:** α-magnitude sweep for the on-manifold regime (~0.25–0.5×gap); try masked crops;
  train DiffAE longer / stronger conditioning; reduce x_T anchoring. Pipeline is correct; counterfactual
  QUALITY needs iteration (the hard part of DiffEx).

### 2026-06-18 — BUG FOUND & FIXED: generate-from-noise (job 34388244)
- **Bug:** traversal DDIM-INVERTED the real cell → x_T, which encodes the image and overrides the
  embedding → editing the embedding barely moved the picture (and recon was a too-good 34 dB). Alex's
  spec: α=0 is the DDPM *reconstruction*, i.e. **generate from noise conditioned on the embedding** (no
  inversion). Switched to fixed random noise per cell (constant across α), conditioned on z0+α·d.
- **Result:** mean re-encoded score Δ **0.28 → 5.3**, 6/8 monotonic, correct sign — embedding now drives
  generation, direction validated. BUT the visual morph is still **subtle** (CellDINO registers texture
  the eye misses; HSPA5 phase phenotype may be genuinely subtle). Δ-pixel heatmap localizes the
  changing cell regions = a useful interpretability output on its own.
- **Next levers:** push α to 2–3×gap; strengthen DiffAE (classifier-free guidance / longer); test a
  gross-morphology target (complex) to tell if subtlety is biology vs method.

### 2026-06-18 — obvious targets (TOMM20, TIMM23, Arp2/3): diagnosis = METHOD-limited
- TOMM20 (job 34392814): lr 0.96, gap 6.2, score Δ −3.1, 7/8 monotonic. TIMM23 (34392821): lr 0.99,
  Δ −3.1, 7/8. (sign arbitrary per Alex.) Arp2/3 complex: filename bug (target "2/3" has a slash →
  unslugified PNG name) — FIXED (slugify filenames in traverse._plot); re-run 34392960.
- **Key finding:** TOMM20 (obvious mito phenotype) morphs just as SUBTLY as HSPA5, edge-concentrated
  Δ. → the visual subtlety is **METHOD-limited, not biology**.
- **Root cause:** DiffAE under-utilizes the embedding — the noise latent dominates DDIM generation
  (inverted OR random), so embedding edits weakly change pixels even though CellDINO/classifier
  register them (score moves, monotonic).
- **Fix: classifier-free / edit guidance** at sampling: ε̃ = ε(c0) + w·(ε(c_α) − ε(c0)), w≈3–5 (no
  retrain). If insufficient → retrain DiffAE with conditioning dropout for proper CFG.

### 2026-06-18 — edit-guidance w-sweep (TOMM20, job 34393392): INSUFFICIENT → must retrain DiffAE
- w=1/3/5 score Δ = 0.55/1.83/2.23 (guidance amplifies) BUT monotonic 0.38/0.25/0.12 (degrades),
  and the **cell still does not transform by eye even at w=5** (edge-concentrated Δ only).
- **Conclusion:** sampling-time guidance cannot fix an under-conditioned model. The DiffAE generates
  from the NOISE latent and only weakly uses the embedding (why recon hit a too-good 34 dB). 
- **SOLID FIX (next): retrain DiffAE with conditioning dropout** (~10–20%, learned null embedding) →
  forces embedding use + enables true CFG ε̃=ε(∅)+w(ε(c)−ε(∅)). If still weak → cross-attention
  conditioning (spatial) instead of global FiLM.
- Also fix: (a) direction discovery is run-to-run unstable (fix seed + more epochs); (b) replace the
  recon gate with an UNCONDITIONAL-generation check (null-embedding samples should look generic;
  conditional should match target) — recon PSNR was misleading.

### 2026-06-18 — conditioning diagnostic = DEAD (0.008), then proper DiffAE rebuild (job 34394595)
- **Diagnostic** (`diffae/diagnose_conditioning.py`, job 34393969): same fixed noise under
  null/ctrl-centroid/KD-centroid. MSE(ctrl-vs-kd)=0.0010, MSE(noise-vs-noise)=0.133 →
  **emb/noise ratio = 0.008**. The embedding has <1% control; the DiffAE generates from noise and
  ignores the embedding. (null-vs-ctrl 0.042 ≫ ctrl-vs-kd 0.001 → reacts to embedding *presence*,
  not *content*.) Confirms: cells don't change because conditioning is ~dead.
- **Rebuild** (`diffae/train.py` rewritten): conditioning dropout (0.15, learned null_emb) + EMA
  (0.9995) + resume-across-jobs + deeper cond MLP. **Gate = conditioning ratio** (not recon PSNR),
  logged every 5 epochs; EMA-best saved by ratio. Target: ratio ≫ 0.008 (→ ~0.3+).
- Retrain 34394595 (phase_v1, reuses caches, 120 epochs, batch 48, resume). Watching cond_ratio
  trajectory. **After it works:** switch directions/traverse to true CFG ε(∅)+w(ε(c)−ε(∅)).
- If time-embedding conditioning still can't climb → escalate to cross-attention (UNet2DConditionModel).
