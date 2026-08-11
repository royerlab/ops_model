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

Status (historical): *designing the per-cell classifier* — that phase is long done; see
ACTIVE EFFORTS below for the current state.

---

## ACTIVE EFFORTS (dashboard — updated 2026-07-08)

### LATEST (2026-07-08) — phenotype-cell handoff, v2 mAP, EBI matrix, viewer embedding tab
- **Phenotype-cell CSV for Ritvik** (`viewer/phenotype_cells.py`) → `viewer_assets/phenotype_cells_for_attention.csv`.
  20 cells × (geneKO + EBI complex) × marker, for SetTransformer attention pixel-patches on the REAL
  phenotype cells. **160,420 cells / 53 markers** (phase + 52 fluor). Cols incl `map_score`, `geneKO`,
  `ebi_complex`, `rank_source`, `segmentation_id` (=pma `segmentation`), `x/y_pheno`, `rank`, `pma_attention`.
  - **Per-marker top-20, NOT the model's global top-20** — the pma `rank` is GLOBAL per geneKO (across all
    56 channels), so `_csv_top` re-ranks WITHIN each (channel, perturbation) and takes the 20 highest-attention
    cells present in that channel (two-pass chunked `head` keeps memory bounded).
  - **Fluor filtered by mAP ≥ 0.2** (phase = ALL perturbations): geneKO by distinctiveness, complex by EBI mAP.
  - **`rank_source` col:** `"model"` (all current cells). Reserved `"fallback"` for markers not in the model.
- **v2 distinctiveness switch:** `catalog.dist_matrix` now reads `paper_v2/with_cp/with_4i/all_livecell`
  (single 56-reporter matrix: 43 live + 7 CP + 6 4i), replacing the paper_v1 3-way split. Added 4i
  `FIXED_REP` mappings (p53/pRb/pS6/p21/b-catenin/c-Myc). **52/56 pma channels now map**; the 4
  excluded (NFkB, RSP6, Rb, gH2AX) are EXPECTED — genuinely absent from the v2 matrix.
- **EBI complex mAP matrix** (`viewer/build_complex_ebi_map.py`) → `complex_reporter_ebi_map.csv` (98×56).
  Runs copairs `phenotypic_consistency_ebi` per-marker on the v2 `with_cp/with_4i` per_signal gene
  embeddings, over ALL perturbations (activity_map=None), NOT the wrong `complex_reporter_chad_consistency`.
  `catalog.complex_dist()` reads it. Also wired into the aggregation pipeline
  (`post_process/combination/pca_optimization/aggregation.py` → `complex_reporter_ebi_consistency.csv`).
- **3 new live-cell markers (cisGolgi, VIM, LMNB1):** HAVE v2 distinctiveness + EBI mAP, but are NOT in
  Alex's pma cell CSVs yet (his attention output predates them) → no attention-ranked cells with crop
  metadata. **FALLBACK (per user, TODO):** select cells around the CENTROID of the existing CellDINO
  embeddings. Source found: `{exp}/3-assembly/cell_dino_features_v2/anndata_objects/features_processed_<reporter>.h5ad`
  (e.g. `mStayGold-CENPRaltORF`, `VIM`, `LMNB1`) — 1024-d embedding + crop metadata (`label_int`=segmentation,
  `x/y_position`, `well`, `experiment`, `perturbation`). Per (marker, pert): centroid → 20 closest →
  `rank_source="fallback"`, `pma_attention`/`rank` null. (Or wait for Alex's reprocessed pma CSV.)
- **SetTransformer accuracy scoring (`viewer/set_classifier.py` + `viewer/mimic_alex_embed.py`) — PARKED,
  waiting on SetTransformer v2 (no-mask classifier).** Full journey + why:
  - Reconstructed Alex's cellstate-set-classifier (ISAB/PMA/cosine head); 5 ckpts in `v4/wandb/cellstate_set_classifier/`.
    Real-bag CEILING validated: feeding Alex's own `.pt` embeddings → P(target) 0.90–0.999 (HSPA5/KIF11/POLR1B/TIMM23 all hit).
  - Raw `embed_crops` → classifier FAILS (OOD, cos 0.47, constant argmax). Built `mimic_alex_embed` to reproduce Alex's
    exact pipeline: **128 Phase2D crop → seg-mask (`cell_seg`) → percentile-norm → CellDINO → z-std(control)**. Findings:
    the **segmentation mask is the load-bearing step** (cos 0.47→0.91); percentile-norm is canceled by CellDINO's z-score.
    At realistic bag sizes (100 cells, per-experiment z-std) the mimic matches the ceiling: **95–100% hit-rate** on real cells.
  - Generated-cell per-α curve works end-to-end for **POLR1B** (P 0.01→0.98, flips to target at α≥1.5) — proof the pipeline
    is correct — but **the segmentation of GENERATED cells is the blocker**: cellpose on fake crops unreliably captures the
    cell (latches onto the bright nucleolus, not the whole body), so only nucleolar-phenotype genes (POLR1B) score; HSPA5/
    KIF11/TIMM23 stay at P≈0 despite healthy embeddings. Diameter/centroid tuning didn't fix it robustly.
  - **DECISION:** masking generated cells is too fragile to rely on. **Wait for SetTransformer v2 — a classifier trained
    WITHOUT masks** → then our unmasked `embed_crops` is in-distribution, no cellpose needed, and the per-α bag score works
    for all genes. The mimic (mask path) + POLR1B validation are kept as a reference/cross-check.
- **Model-metrics curves** (`diffae/plot_metrics.py`): loss + cond_ratio over epochs, one line per DiffAE
  → `model_metrics_curves.png/.svg`.
- **Viewer embedding tab** (`build_umap_montage.py` + `webapp/`): OSD montage, UMAP↔PHATE, points/images
  toggle, 44 anndata color-by fields, opacity/zoom sliders, click→perturbation sidebar; gene descriptions
  from the gene-embedding h5ad (`gene_desc.json`, fixes VAMP2-style blanks).

### STATUS SUMMARY (historical detail condensed 2026-07-08)
- **Generators:** phase `phase_v1` = PRODUCTION (0.468); 500k warm retrain PARKED (peaked 0.542 then
  declined). Fluor **50/50 markers trained** (ep≥98). v2/v3 aug did NOT beat v1. Directions default =
  deterministic **mean_diff** α (see build log for the full DiffAE saga).
- **Viewer** (`viewer/` — `submit.py`, `catalog.py`, `precompute.py`, `build_umap_montage.py`, `webapp/`):
  static precompute → dependency-free web app; per-marker driver shares the NTC gather + dedups real
  cells; embedding tab (see LATEST). Live demo `login-01:8765`.
- **Score:** authoritative = Alex's **SetTransformer** bag `P(target)` (§7 ckpts downloaded) — supersedes
  the per-cell N-way MLP (`nway_clf.py`) and the old binary LR badge. Generated-cell bag scoring PARKED: the
  mask-mimic works on real cells (95–100%) + POLR1B generated, but segmenting fake cells is too fragile →
  waiting for SetTransformer v2 (no-mask classifier). See LATEST.
- **Infra PR (#51): OPENED + ACCEPTED/MERGED** — `diffex-viewer-dev.tf` in `sfbiohub-infra` (S3 bucket
  `diffex-viewer-dev` + nonprod read-only IRSA role `biohub-nonprod-diffex-viewer` for SA `diffex-viewer` in
  ns `argus-diffex-viewer-rdev` + read-write uploader role; mirrors `proteohub-argus-s3-reader-dev.tf`, 1 TB
  ceiling). `terraform apply` provisions the bucket/roles → then `aws s3 sync viewer_assets/ s3://diffex-viewer-dev/`
  → Argus boot-download. Next: create the app repo + `argus register` (see App-staging build-log entry).

### OPEN BUILDOUT
1. **Full NTC drain — LAUNCHED 2026-07-11** (master job `34826112`): all ~1000 geneKO genes/marker for the
   46 valid-`rep` fluor markers (was top-8 seed; 42 were partial ~100–194, 4 hub markers already ~complete).
   Command = `submit seed --map-thr 0 --timeout 720` (**not** `--all-genes`; that flag never existed — the
   `--map-thr 0` = every gene with distinctiveness ≥ 0 = all ~1000). Resume is automatic (skips built targets).
   The 4 rep=None markers (NFkB/RSP6/Rb/gH2AX) are intentionally excluded. ~500 GB. See build-log 2026-07-11.
2. **Full A→B anchors** — `submit anchors --k 10` across all markers + complexes.
3. **Fluor complex traversals — DONE (resume 2026-07-11, master `34826156`)**: 98 EBI complexes × 50 markers
   were already ~complete; only 5 markers partial (peroxisome_Peroxi, pS6, pRb, NPM3, SRRM2) → `submit
   fluor-complex` resume finishes them. (phase complex = 190 = 98 NTC-anchored + 92 complex→complex anchor pairs.)
4. **Wire SetTransformer bag score** into `precompute` + a per-α curve panel — BLOCKED on **SetTransformer v2
   (no-mask classifier)**; the mask path is too fragile on generated cells (see LATEST). Once v2 lands, unmasked
   `embed_crops` bags score directly (no cellpose).
5. **S3 hosting** — infra PR #51 MERGED. Argus app scaffold built (`/hpc/mydata/gav.sturm/diffex-viewer`, forked
   from `czbiohub-sf/mops-viewer`). Remaining: `terraform apply` → create `czbiohub-sf/diffex-viewer` repo →
   `argus register`/bootstrap (needs argus CLI) → upload `viewer_assets/` (or hand to Kyle) → PR+`stack` label. See build-log.
6. **Centroid fallback** for cisGolgi/VIM/LMNB1 (see LATEST).
7. **Multi-α montage** — `--alphas` flag looping per-α decodes + an α switch in the explorer.
8. **Image-UMAP montage** (`czi-ai/latent-lens`) — idea track; needs full-gene coverage.
9. **Attention-head tab + viewer reorientation** (DONE) — new tab overlaying CellDINO attention-head
   pixel weights (inferno) on the real phenotype cells; Browse (marker+perturbation) drives ALL views.
   See `### 2026-07-08 — Attention-head tab` build-log entry.
10. **Per-marker embedding montages** (FUTURE) — the Embedding tab now switches its IMAGES per marker but
    reuses the SHARED phase gene-UMAP LAYOUT for every marker (`submit montage` loops modalities, all with
    the phase `UMAP_H5AD`; only tiles swap). Fluor markers currently place only their ~8 seed geneKO frames
    (sparse). FUTURE: give each marker its OWN gene embedding (per-reporter CellDINO gene UMAP/PHATE), so
    genes sit at that marker's coordinates. Needs a per-marker gene-embedding h5ad (`obsm X_umap/X_phate` per
    gene) — does NOT exist yet (`pca_optimized_v0.3/.../paper_v2/` only has aggregate combos: phase_only,
    all_livecell, with_cp, no_phase, only_cp — no per-single-marker layout). Build = aggregate per-marker
    CellDINO gene embeddings → PCA → UMAP/PHATE per reporter, then `build_montage_web(modality=<slug>,
    h5ad=<per-marker>)`. Also depends on fuller fluor geneKO traversals (item 1) to be non-sparse.

---

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

### 2026-06-29 — Plan C implemented (deterministic direction) + v2_aug retrain
- **Reproducibility root cause:** unsupervised InfoNCE direction bank is GPU/seed-nondeterministic
  and `best_k = argmax|shift|` flips run-to-run → same cell highlighted different regions.
- **Fix (plan C, implemented):** `directions/config.py` `direction_method` — default **`mean_diff`**
  (deterministic control→KD centroid vector; also `lr_weight`) as PRIMARY; the paper's unsupervised
  bank kept as `direction_method="unsupervised"` secondary track. `traverse(fixed_dir=…)` uses the
  global deterministic direction; `deterministic=True` sets seeds + cuDNN-deterministic. `+α = toward
  KO` by construction (no more sign flip). `rank.supervised_direction()` computes it; LR kept for the
  re-encode score check only.
- Reproducibility proof in progress: TIMM23 run twice (jobs 34654037/34654040) → pixel-diff strips.
- **Next model (v2_aug)**: orientation-aug DiffAE retraining (job 34651296), cond_ratio climbing
  0.04→0.14 @ep19/120 (aug ramps slower); resume to convergence, then switch directions default to it.

### Active experiments (2026-07-03)
- **v1 remains the best model.** v2 (dihedral) and v3 (continuous rot+scale) augmentation did NOT
  beat v1 — not more orientation-stable, weaker/less-convincing phenotypes; cond_ratio ceiling falls
  with aug (v1 0.47 → v2 0.25 → v3 0.20; curves in `coding_exps/diffex/diffae_training_curves.png`).
  Flow-matching transport (CellFlow-style, `directions/flow.py`) also explored → smoother but less
  clean phenotypes, noisy negative extreme → NOT adopted. Reverted default to v1 + mean-diff α.
- **Generator data-scaling test (RUNNING):** does 50k→**500k** crops help? Two no-aug chains, 24 ep:
  - scratch `phase_v1_500k` — jobs `34667092→34667174→34667175`
  - warm-start from v1 `phase_v1_500k_warm` — jobs `34667176→34667177→34667178`
  - Compare cond_ratio/loss vs v1 (0.47) + visual morphs. mem_gb=400 (500k float32 crops ≈ 51GB each).
- **Direction depth test (pending):** gather 1k→**~12k**/class (the distinctiveness peak) for a tighter
  mean-diff centroid — cheap, per-target (~30-40 min CellDINO/target), no retraining.

### Future direction — per-fluorescent-marker models (2026-07-03)
Reproduce the best phase pipeline **per fluorescent marker** (~60 live-cell markers) — a per-marker
counterfactual view of each gene-KO / complex phenotype in the channel where attention is most
informative. **~60 models** (one DiffAE + direction set per marker).
- **Attention source EXISTS:** `…/alex_lin_attention/v4/pma_fluorescent_cells_all.csv`
  (+ `pma_fluorescent_cells_ebi_all.csv` for complexes) — the fluor analog of `pma_phase_cells_v2_all.parquet`.
  CellDINO fluor train/val sets also present (`train/val_ops_zstdcontrol_cdino_fluorescent`).
- **Scope:** `good_experiment_list_v2.yml` (87 exps; fluor channels GFP×74, mCherry×23, Cy5×2). Each
  experiment's channel→biological-marker label is in `ops_process/ops_analysis/configs/ops_channel_maps.yaml`.
  Marker/experiment enumeration tooling: `ops_utils/data/feature_discovery.py`,
  `ops_utils/analysis/embedding_discovery.py`.
- **Per-marker pieces:** gather top-attention fluor cells (control + KD) → CellDINO embed the MARKER
  channel → mean-diff direction → **per-marker DiffAE** (phase generator can't decode fluor) → traverse.
  Direction/traverse code unchanged; needs a per-marker DiffAE + a marker→(experiments, channel) map.
- **Complication (deferred):** the v2 list is **live-cell fluor only** — 4i / Cell-Painting (fixed-cell)
  channels are excluded. If added later they need their **own per-round link CSVs** (`link_csv_dir` in
  `ops_model/data/data_loader.py`), not the default live 3-assembly link.
- **NEEDS DESIGN CONFIRM before building** (60 DiffAE trainings is a large program).

### Future direction — attention-informed cell selection (2026-06-29)
Currently we take a flat top-1000 attention-ranked cells per class and pick traversal/feature
cells by index. To exploit attention ranking more (only touches `directions/data.py gather` +
`rank.supervised_direction`, not the DiffAE):
- **Pick highest-attention cells for the traversal/featured strips** (most representative morph),
  not an arbitrary cell index.
- **Per-target penetrance depth** — use the attention-accuracy knee (HSPA5≈top10, RPL10≈top800)
  instead of a flat top-1000, so sharp phenotypes aren't diluted by the diffuse tail.
- **Attention-weight** the mean_diff / classifier so the most-phenotypic cells dominate the axis.

### Future direction — orientation-invariance via augmentation (2026-06-29)
Observation: along a traversal the cell often spuriously rotates/transposes (orientation is
encoded in the CellDINO embedding, so the discovered direction carries an orientation component the
DiffAE renders). Fix: during DiffAE training, augment the TARGET image with the dihedral group
(4 rotations × flip = 8 views, incl. transpose) while conditioning on the embedding of the CANONICAL
(un-augmented) cell. Teaches the model orientation ≠ embedding-determined → orientation absorbed by
the (fixed) noise latent, phenotype carried by the embedding → traversals stop rotating. Do NOT
recompute CellDINO on the augmented crop (defeats the decoupling). Also serves as general aug to
sharpen conditioning.

### 2026-07-08 — Attention-head tab + viewer reorientation (BUILT)
**Status: built + deployed to `viewer_assets/`.** `viewer/build_attention_heads.py` rendered **984/1000**
phase geneKO genes (16 npz still corrupt/mid-write by Kevin — `BTF3L4, BUB1B, DAD1, DHRS9, FECH,
FOXD4L1, GTPBP4, INO80D, MTOR, NCBP2, NRAS, POLR2F, RPS19BP1, TWF1, TYK2, YIPF5` — builder is idempotent,
skips-loud, re-run picks them up). `global_max=2.44`. Webapp reoriented (`webapp/{index.html,app.js,
style.css}` v36, copied to `viewer_assets/`): persistent `#browse` block (marker+grain+perturbation+
cells/page) drives 3 view tabs — Traversal / Embedding / **Attention heads**. Attn view = inferno LUT +
live per-map/per-gene/fixed normalization + opacity, head dropdown from `heads.json`; greys out for
non-phase / complex / missing-gene. Embedding now rings + pans to the selection. **Availability decoupled
from manifest** — app fetches `attention_heads/phase/index.json` (no `precompute.build_manifest` change).
- **ALL 4 modality×grain combos now IN (2026-07-08 pm).** Kevin dumped pixel_attribution (maps+crops+
  patch_masks) for fluor geneKO (`fluorescence_pixel_attribution/<marker>/`), phase complex + fluor complex
  (`complex_pixel_attribution/{phase,fluorescence/<marker>}/`) — same npz schema as phase. Builder rewritten
  **SLURM-parallel** (`build_attention_heads.py render` → `submit_parallel_jobs`, 40 shards, ~1 min) into a
  uniform `attention_heads/<modality>/<grain>/<key>/` layout + single `attention_heads/index.json`
  ({global_max, assets:{modality:{grain:[keys]}}}). **23 modalities, 1455 keys** (phase geneKO 984 + phase
  complex 93 + 16 fluor markers). Webapp resolves assets by (marker→`jsSlug(marker_channel)`|"phase", grain,
  target→gene|slug); non-phase-geneKO lack ranking metrics (auroc/spec) but render fine from the npz `heads`.
  (`fluorescence_attention/<MARKER>.npz` = the older ranking-features-only dump, superseded.)
- **16 corrupt phase genes: left as-is.** Full-size but bad-zip at SOURCE — needs Kevin to regenerate; 984/1000.
- **Attn viewer UX (2026-07-08 pm):** top-crossbar selection (marker+perturbation comboboxes w/ search, mAP|A–Z
  sort) drives Traversal/Embedding/Attention-heads tabs; attn view = per-perturbation color-coded blocks
  (rows=heads, cols=cells), pin/reset controls, per-cell/gene/fixed norm + dual clim + opacity sliders,
  Ritvik-faithful overlay (σ=2 smooth + cell mask + inferno@α0.6); embedding click → selects in search box.

### 2026-07-08 — Attention-head tab + viewer reorientation (design)
New viewer view: overlay CellDINO **attention-head pixel weights** (inferno) on the **real phenotype
cells** (`viewer/phenotype_cells.py` output), so you can see WHERE in each cell each top attention head
looks — the classifier's spatial evidence, alongside the generative counterfactual morph.

**Data (Kevin L., already under `viewer_assets/attention_heads/`, verified 2026-07-08):**
- `phase/pixel_attribution_cache/<GENE>.npz` (1000 geneKO genes): `maps (20,6,128,128) f16` ∈ [0,~0.34]
  (20 cells × top-6 heads × pixel attribution), `crops (20,128,128) f32` (z-scored cell crops),
  `heads (6,2) int32` = the ranked (layer,head) pairs, `patch_masks (20,196) bool` (14×14 ViT patches).
- `phase/head_rankings_per_gene.json`: per-gene ranked heads + metrics (`layer,head,feature,spec_p10,
  spec_min,auroc_vs_ntc`); order matches the npz `heads` array. **The 20 cells ARE the phase·geneKO
  top-20 phenotype cells** from `phenotype_cells.py` (same selection).
- `celldino_attention_head_analysis/fluorescence_attention/<MARKER>.npz` — per-marker fluor analog
  (structure TBD) → follow-on. **v1 scope = phase · geneKO only** (no complexes: head_rankings is gene-keyed).

**Precompute (`viewer/build_attention_heads.py`, to write):** ship **raw** data so normalization + inferno
+ opacity are LIVE display options (user-selectable, per decision). Per gene → per cell: write
`cell<c>/crop.webp` (grayscale, per-crop robust min-max) + per head `cell<c>/head<h>.webp` (grayscale
attribution scaled by a FIXED global max so absolute intensity is preserved). Per-gene `heads.json` =
ranked-head metrics (`layer,head,feature,spec_p10,spec_min,auroc_vs_ntc`) + `n_cells` + `gene_max` +
`global_max`. The webapp applies a 256-entry **inferno LUT** in a canvas and composites over the crop —
so a Display dropdown offers **per-map / per-gene / fixed** normalization live (per-map = rescale by the
loaded tile's own max; per-gene = by `gene_max`; fixed = by `global_max`), plus an opacity slider, without
re-fetching. Count ≈ 1000×20×(6+1) ≈ 140k grayscale WebP (traversal already ~370k). Keeps the app
dependency-free; no 3×-image blowup from baking each norm.

**Viewer reorientation (`webapp/index.html` + `app.js`):** today the left panel is 3 *control* tabs
(Browse / Anchor / Embedding), each with its own selectors, and the Embedding montage ignores the
Browse (marker,perturbation) selection. **Reorient:** make Browse (marker + grain + perturbation) a
PERSISTENT selector block = single source of truth (`state.marker`, `state.target`); below it a **View
switcher** — Traversal | Embedding | Attention heads — each rendering the main stage for the CURRENT
selection. Fold today's Anchor/display controls under Traversal; α/cell/embedding-mode under Embedding;
head selector + overlay-opacity under Attention heads. Embedding also gains browse→highlight/pan and
keeps montage-click→browse select (closes the selection loop).

**Attention-head view UI:** grid of the 20 phenotype crops with the selected head's inferno overlay;
head dropdown lists the 6 ranked heads with `(L,H) · AUROC·NTC / spec`; overlay-opacity slider; raw-crop
toggle. Reuses Browse's cells-per-page paging.

**Manifest:** add per-(marker,target) `attn` availability + `n_heads` so the View switcher greys out
Attention heads where absent (v1: present only for phase geneKO genes with an npz).

**Decisions (locked 2026-07-08):** (a) normalization = **live user option** in Display settings
(per-map / per-gene / fixed) via the grayscale+LUT approach above; (b) full reorientation approved
(persistent Browse + view switcher); (c) **phase-only v1**, fluor (Kevin's per-marker npz) is a follow-on.

### 2026-07-08 — phenotype-cell handoff CSV + v2 mAP + EBI matrix (see dashboard LATEST)
- **`viewer/phenotype_cells.py`** — the Ritvik handoff: top-20 REAL phenotype cells per (marker × perturbation)
  → `viewer_assets/phenotype_cells_for_attention.csv` (160,420 cells / 53 markers). Key correctness fix:
  the pma `rank` is GLOBAL per geneKO (across all channels), so `_csv_top` re-ranks WITHIN each
  (channel, perturbation) and takes each marker's own top-20 by attention (two-pass chunked `head`).
  Fluor filtered by mAP ≥ 0.2; phase = ALL. Added `map_score`, `geneKO`, `ebi_complex`, `rank_source` cols.
- **v2 mAP:** `catalog.dist_matrix` → `paper_v2/with_cp/with_4i/all_livecell` (56 reporters, live+CP+4i);
  added 4i `FIXED_REP`; 52/56 pma channels map (NFkB/RSP6/Rb/gH2AX expected-excluded).
- **`viewer/build_complex_ebi_map.py`** — complex×reporter EBI mAP (98×56) via copairs
  `phenotypic_consistency_ebi`, all-perturbation, on v2 per_signal; also emitted by the aggregation pipeline.
- **Pending:** centroid fallback for cisGolgi/VIM/LMNB1 (mAP present, no pma cells) from
  `cell_dino_features_v2/features_processed_<reporter>.h5ad` (embedding + crop metadata); SetTransformer
  bag scoring parked on Alex's v2 CellDINO extraction.

### 2026-07-09 — Argus app staging (S3 hosting)
- **Infra PR #51 OPENED + ACCEPTED/MERGED** (`sfbiohub-infra`, branch `diffex-viewer-dev`,
  `terraform/accounts/biohub-nonprod/diffex-viewer-dev.tf`): S3 bucket `diffex-viewer-dev`, read-only IRSA
  role `biohub-nonprod-diffex-viewer` (trusts SA `diffex-viewer` in ns `argus-diffex-viewer-rdev`), read-write
  uploader role `diffex-viewer-dev-readwrite`. 1 TB ceiling. → `terraform apply` provisions bucket+roles.
- **App scaffold built** at `/hpc/mydata/gav.sturm/diffex-viewer` — forked from `czbiohub-sf/mops-viewer`
  (same Argus+S3 pattern; names already align with the `.tf`). Serving layer swapped Gradio → **static nginx**:
  - `Dockerfile` (nginx, bakes `webapp/` shell) + `nginx.conf` (port 8080, `/healthz`) + `docker-entrypoint-diffex.sh`
    (drops shell into the S3-populated webroot, starts nginx).
  - `.infra/common.yaml`: aws-cli `fetch-assets` init container (`aws s3 cp s3://diffex-viewer-dev/ → /usr/share/nginx/html/`),
    `web` emptyDir, serviceAccount `diffex-viewer`, OIDC proxy; `.infra/rdev/values.yaml` carries the read-only role ARN.
  - `.argus-ci.yaml` app=`diffex-viewer`; `scripts/uploader/*` sync `viewer_assets/` (manifest.json at bucket ROOT) via the readwrite role; `DEPLOY.md` runbook.
  - Static because the webapp reads assets by relative path from `manifest.json`; the S3 sync drops assets as siblings of the baked shell.
- **Remaining to deploy:** `terraform apply` → create `czbiohub-sf/diffex-viewer` repo + push → `argus register app`
  (team-sci-biohub) + bootstrap/reconcile `.github` (needs argus CLI, interactive) → upload `viewer_assets/` (our
  role, or **Kyle from HPC** — he offered) → PR + `stack` label → Argus builds + deploys rdev behind Okta. Confirm
  the registered namespace/SA matches `argus-diffex-viewer-rdev`/`diffex-viewer` before relying on the IRSA trust.

### 2026-07-11 — full 1k-geneKO + complex buildout LAUNCHED
- **geneKO (master `34826112`, 49 jobs):** `submit seed --map-thr 0 --timeout 720` — all ~1000 geneKO
  genes/marker for the 46 valid-`rep` fluor markers (49,000 targets). Resume skips the ~100–194 already
  seeded per marker; the 4 hub markers (ChromaLIVE_561 957, LysoTracker 932, NPM3 930, NucleoLIVE 780)
  were already near-complete. Runs with **no concurrency cap**.
- **complex (master `34826156`, 50 jobs):** `submit fluor-complex` resume — nearly all 98 EBI complexes
  were already built for ~45 markers; only 5 partial (peroxisome_Peroxi 35, pS6 35, pRb 59, NPM3 86, SRRM2 88).
- **`submit.py` fixes made for this run:**
  - **BUG:** `cmd_seed` used `if args.map_thr:` — `0` is falsy, so `--map-thr 0` silently fell back to top-8
    (a wrong launch, 34826071, was cancelled). Changed all gates to `args.map_thr is not None` → `--map-thr 0`
    now correctly means all ~1000 genes.
  - **`--parallel` now defaults to `None` = no concurrency cap** (only sets `slurm_array_parallelism` when given),
    so full buildouts don't need `--parallel 100`. Added `--timeout` (default 180) to override the per-marker wall.
- **After the builds land:** `submit sync` to refresh manifest + attention + montages from the new cache.
- The 4 rep=None markers (NFkB/RSP6/Rb/gH2AX) remain intentionally excluded (no v2 distinctiveness reporter column).

### 2026-07-12 — DiffAE cond_ratio PEAKS EARLY then declines; check before extending/rebuilding
Resumed the 6 under-trained (ep55) fluor DiffAEs to ep120 (dihedral). Key lesson: **most peaked their
conditioning ratio around ep39–55 and then DECLINED** — extending to 120 did NOT improve them:
| marker | best cond_ratio | peaked @ | verdict |
|---|---|---|---|
| CLTA | 0.390 | ep54 | plateaued/declined → no gain |
| ATP1B3 | 0.171 | ep54 | plateaued/declined → no gain |
| PSMB7 | 0.830 | ep54 | declined to ~0.49 → **stopped** |
| TFRC | 0.320 | ep39 | declined to ~0.21 → **stopped** |
| VAMP3 | 0.231 | ep54 | declined to ~0.17 → **stopped** |
| **SLC3A2** | **0.277** | **ep109** | still climbing (>pre55 0.259) → **kept running** |

**RULES (to not repeat the wasted compute):**
1. `diffae_best.pt` is saved on BEST cond_ratio, so it already captures the peak regardless of final epoch —
   a longer run does NOT give a better checkpoint unless best_ratio actually advanced.
2. **Before extending training** past its current point, check the cond_ratio trajectory
   (`torch.load(train_state)['history']`). If it peaked early and is declining, stop — the best is already banked.
3. **Before clearing + rebuilding a marker's traversals** (1k geneKO / 98 complex / anchors), confirm the model
   IMPROVED: compare `diffae_best.pt` mtime + best_ratio vs the value the existing traversals were built on.
   Only rebuild if the best genuinely advanced PAST what the current traversals used.
- **Rebuild status:** CLTA/ATP1B3/PSMB7/TFRC/VAMP3 = NO rebuild (best@≤ep54 already used by the Jul-11 traversals).
  **SLC3A2 = the one rebuild candidate** — its best advanced to 0.277@ep109 (Jul 12) vs the Jul-11 traversal
  checkpoint (~0.259); once it finishes, clear + rebuild ONLY SLC3A2's traversals with the improved model.
### 2026-07-12 — accuracy-selected cell variant (Kevin's accuracy_ranking CSVs)
Cell selection can now use **classifier-accuracy rank** instead of **attention rank**. Source =
`…/alex_lin_attention/v4/accuracy_ranking/`: `pergene_phase_cell_rankings.csv` (geneKO·phase),
`ebi_pergene_phase_cell_rankings.csv` (complex·phase), `ebi_class_channel_cell_rankings.csv` (complex·fluor,
55 marker channels). NTC has NO accuracy data → NTC anchor always stays attention-sourced (shared/cached).
- **phase** = side-by-side A/B: modality `phase` (attention, "phase_attention") vs `phase_topacc`
  ("phase_accuracy"). Full 1k geneKO + 98 complex + 182 anchors built for phase_topacc. Hooks:
  `_gather_class(parquet=…)`, `precompute_marker(accuracy_parquet=, variant=)`, and the anchor path
  `_gather_df`/`_setup`/`precompute_target(accuracy_parquet=, variant=)`. Accuracy dirs have LARGER
  control→KD gaps (cleaner separation) than attention.
- **fluor** = REPLACED IN PLACE (no `_acc` duplicate), via `precompute_marker(accuracy_fluor_csv=, force=True)`.
  **⚠️ COVERAGE SPLIT:** the fluor accuracy CSV only covers the complexes each marker actually distinguishes
  (a SUBSET of the 98 — **median 13/marker, range 1–32, 53 distinct total**), so each fluor marker's `complex/` set is now MIXED:
  accuracy-selected for its covered complexes, still attention-selected for the rest. This is intentional/interim
  — we will get full 98-complex + 1k-geneKO accuracy coverage for every marker later and **rebuild the entire
  cache anyway**, at which point the split disappears. geneKO fluor has NO accuracy CSV yet (phase-only + complex-fluor).
- **fluor complex→complex ANCHORS: NET-NEW with accuracy** (didn't exist in attention). Per marker: top-5
  accuracy-covered complexes (by `class_channel_acc`) → 20 A→B pairs, accuracy cells for both, via per-channel
  parquets in `accuracy_ranking/fluor_complex_by_channel/`. 54 markers × ~20 ≈ 982 pairs → `<marker>/complex/<A__to__B>`.
- **PHASE SWAP DONE (2026-07-12): accuracy is now the canonical `phase`.** `viewer_assets/phase` (attention) +
  its `_directions/phase` archived to `viewer_assets_backup/{phase_attention,_directions_phase_attention}` (same FS,
  reversible); `phase_topacc` → `phase`. Manifest label reverted to plain "Phase". NOTE: the separate build-cache
  `…/diffex/directions/{phase,phase_topacc}/` (anchor gather cache, OUTSIDE viewer_assets) was NOT renamed — only the
  served `viewer_assets` traversals + `_directions` were swapped; a future full rebuild regenerates it anyway.

- **min-ep gate REMOVED as default** (`submit seed --min-ep` default 98→**0**; `catalog.complete_markers` default
  98→**0**). Epoch count is NOT a quality signal — a marker peaking at ep54 is as usable as one at ep120, and
  `diffae_best.pt` banks the peak. Inclusion now gates on **checkpoint presence** (diffae_best.pt + train_state),
  not epoch. Pass `--min-ep N` only to re-impose a floor. (Right now all 56 markers pass either way since the
  Jul-11 resume pushed the 6 past ep98, but the default now won't silently drop a future under-trained marker.)
