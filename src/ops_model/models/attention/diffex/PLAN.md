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

## ACTIVE EFFORTS (dashboard — updated 2026-07-07)

Eight parallel workstreams. Each line: what · where · how to check · next step.
Root for everything: `/hpc/projects/icd.fast.ops/models/diffex/`.

### CURRENT STATE (2026-07-07)
- **Viewer cache:** 29 markers / **303 targets** (241 NTC-anchored + 62 A→B), 1.2 GB; 29 shared
  `_anchors/<anchor>/` real-cell dirs (dedup working); **scores.json for all 303** (binary-LR score —
  to be replaced by SetTransformer, #7). Live demo: `login-01:8765`.
- **Canonical build code is now IN-REPO** (scratchpad drivers retired): `viewer/submit.py`
  (`seed|anchors|manifest|montage` subcommands) + `viewer/catalog.py` (dist matrices, per-marker
  top-gene ranking, complete-marker catalog, dist/desc maps) + `precompute.py` (`precompute_marker`
  per-marker driver, `precompute_target`, `build_manifest`) + `build_umap_montage.py` + `webapp/`.
- **Fluor generators:** **38/59 complete** (ep≥98); 21 still training (array `34670327`).
- **500k warm retrain — CONCLUDED, PARK IT.** cond_ratio peaked 0.542 then *declined*
  (0.36 → 0.34 → 0.33) → not beating v1; stay on `phase_v1` (0.468). Effort #1 closed.
- **UMAP montage (#8):** cell 35 / α=2 built (1000/1052 genes) — all precomputed CellDINO
  (`gene_bulked_Phase` centroids + `features_processed_Phase` z0, NO re-embed). **Array-parallelized:
  8 GPU chunks ~2.5 min + assemble 49 s** (vs 16 min single-job).
- **Infra PR (#5):** **SUBMITTED** — sfbiohub-infra PR #51 (open). Write access granted; no storage limit.
  Argus is stateless (restart drops local data) → assets live in S3, deployment downloads on boot.
  Draft `.tf` mirrors `proteohub-argus-s3-reader-dev.tf` (bucket + nonprod-cluster read-only). Ready to
  push (pending user OK). Requesting **1 TB** ceiling. Separate app-side track: make viewer Argus-ready
  per `czbiohub-sf/biohub-argus-example-app` + Argus MCP.

### TO COMPLETE THE FULL CACHE BUILDOUT
1. **Finish 21 remaining fluor generators** (array `34670327`) → add their slugs to
   `catalog.COMPLETE_LAUNCH` → `submit seed`.
2. **Full NTC drain:** seed is currently **top-8 genes/marker**. For full coverage expand per-marker
   targets to ALL genes (~1000) → ~500 GB. Add a `submit seed --all-genes` mode (the per-marker
   driver already amortizes the shared NTC gather, so cost scales with KD gathers).
3. **Full A→B anchors:** `submit anchors --k 10` across ALL markers + complexes (currently only a
   few markers' top-5).
4. **Fluor complex traversals** (currently phase-complex only): per-marker `grain=complex` with the
   EBI fluor CSV.
5. **Real scores:** wire SetTransformer (#7) per-α bag `P(target)`, supersede the binary LR; add
   `has_scores` to the manifest target entries (viewer already fetches `scores.json` directly).
6. **S3 hosting** (#5) — access granted. Push the S3-bucket PR (bucket + nonprod-cluster
   read-only, 1 TB ceiling) → on merge `aws s3 sync viewer_assets/ s3://diffex-viewer-dev/`; Argus
   downloads from S3 on boot. Then make the app Argus-ready (`biohub-argus-example-app` + Argus MCP).

### MULTI-ALPHA MONTAGE — what's needed
- Now: one montage zarr per (cell, α) — `submit montage --cell 35 --alpha 2`.
- Multi-α: fan out `submit montage --alpha {1,2,3,4,5}` (each reuses the same z0 + gene_bulked
  centroids; per-α decode array + a zarr per α), then an **α switch in the explorer** (latent-lens
  napari layer per α, or a small selector). Next: add a `--alphas` flag that loops the per-α decode
  arrays sharing one prep, and check whether latent-lens supports an α-indexed/stacked montage.

### 1. Phase generator: 500k warm-start retrain (does more data beat v1?)
- **Goal:** see if cond_ratio climbs past v1's 0.468 with 10× data, warm-started from v1.
- **Checkpoints** (`diffae/`): `phase_v1/`=PRODUCTION (50k, ep120, 0.468); `phase_v1_500k_warm/`
  =warm-from-v1 (500k, resuming from ep15/0.542, the promising one); `phase_v1_500k/`=scratch
  (500k, ep12/0.416, PARKED).
- **Running:** resume chain `34673256 → 34673257 → 34673258` (afterany, 720min each, **mem_gb=200**
  — 96 OOMs on the 51GB crop cache; see `diffae/submit.py` RUNBOOK).
- **Check:** `python -c "import torch;h=torch.load('.../phase_v1_500k_warm/diffae_train_state.pt',map_location='cpu')['history'];print([round(e.get('cond_ratio',-1),3) for e in h if e.get('cond_ratio',-1)>0])"`
- **Next:** if cond_ratio trends >0.55 → finish to ep120 + re-render a v1-vs-warm compare; if it
  plateaus ~0.54 → stay on v1. Do NOT invest more in the scratch run.

### 2. Fluorescent per-marker generators (train all ~60 markers)
- **Goal:** a DiffAE generator per fluorescent marker; then direction/traversal per marker.
- **Done (ep≥98, 31 markers):** all 23 launch-covered live+CP+4i + 5 early (NucleoLive, NPM3,
  FastAct, LysoTracker, ChromaLIVE-mito) + 3 no-PMA (LMNB1, VIM, cisGolgi via anndata).
- **Still training / undertrained:** H2BC21 (ep17), VPS35, EEA1, RAB7A, pHrodo, BODIPY, HSPA1B,
  CLTA, c-Myc, Rb, RSP6, MAP1LC3B, b-catenin, caspase, gH2AX, and remaining launch markers
  in array `34670327` (throttle 8).
- **Channels/config** in `directions/_ranking/fluor_marker_launch.json` (+ recovered from training pickles).
- **Next:** let the array drain; re-render grids/viewer assets for markers as they finish.

### 3. cells×α traversal grids (review artifacts)
- **Where:** `directions/_grids/`. Format: rows=cells × cols=α (**±2/±3/±4/±5**, w=2), via
  `directions/batch.py::marker_grid` (now supports phase, fluor, complex, and A→B anchor).
- **Done:** 28 complete-marker grids w/ top marker-specific geneKO (from `gene_best_marker.csv`
  distinctiveness assignment, margin-broken); custom KIF23@NucleoLive/@H2BC21, TSEN2@MAP4,
  phase KIF23; **40S→60S anchor-switch demo** (`34676955`).
- **Top-gene source:** `organelle_attribution/pca_optimized_v0.3/cell_dino/zscore_per_exp/paper_v1/
  .../plots/marker_overlay/gene_best_marker.csv` (+ distinctiveness_raw matrices for live/cp/4i).

### 4. Shareable traversal viewer (MOPS-style)  ← main build
- **Design:** static precompute → dependency-free web viewer (no live GPU). α = scrub timeline
  (gif-timed play, speed control, clickable pause ticks; default pauses = ends+middle), w=2 fixed.
- **Code:** `viewer/precompute.py` (batched decode + threaded WebP + per-frame classifier score +
  crop-cache cleanup) · `viewer/webapp/` (index/app/style, cache-busted `?v=N`).
- **UI now (v14):** left tabs **Browse / Anchor & display**; grouped grid (rows=perturbation ×
  cols=cells-per-page, one header/row, colour bar on lead cell only); **anchor menu** (default NTC,
  swap to any class with precomputed A→B assets); heatmapped **classifier % badge** (being corrected
  to N-way — see #6); wiki **right sidebar** (gene function + GO/Reactome, **OpenCell + GeneCards**
  links, complex members); colorbar **α=1 "true centroid"** marker.
- **Assets/cache:** `viewer_assets/<modality>/<grain>/<slug>/cell<c>/frame_<i>.webp` (+ meta.json,
  scores.json); A→B at `<anchor>__to__<target>/`; `manifest.json` + webapp copied into that dir.
  ~0.6GB now, growing. Rebuild manifest: `scratchpad/build_manifest.py`.
- **LIVE DEMO on Bruno:** `http://login-01:8765` (`python -m http.server 8765 --bind 0.0.0.0
  --directory .../viewer_assets`); VS Code port-forward 8765, or `http://login-01:8765` inside noVNC.
- **Speed:** `num_workers=12` on the crop DataLoader → materialize **56s→5s (11×)**; per-target
  ~1.5min. `precompute_target(load_workers=…)`; SLURM `cpus_per_task=12`.
- **PER-MARKER DRIVER** `precompute_marker(grain, targets, …)`: all a marker's geneKOs share the
  SAME ~20 NTC/anchor base cells + seeds, so gather the control ONCE + reuse across targets; save
  20 real cells ONCE under `<modality>/_anchors/<anchor>/` (meta carries `real_dir`). Dedups real
  cells + amortizes ckpt load & control gather. `real.webp` toggle shows real-vs-generated row.
- **Cache builds:** current seed = **per-marker** array `34680796` (30 jobs: 28 fluor + phase
  geneKO + phase complex, top-8 each). A→B pairs `34678847`; 40S↔60S anchors done (viewer anchor
  menu works); demo phase geneKOs `34679534` + complexes `34679137`.
- **Next:** full NTC drain + full A→B (K=10 → ~5,400 traversals ~46GB ~150 GPU-hr) — gated on go.

### 5. Infra / hosting PR (S3 bucket + Argus read-only) — ACCESS GRANTED
- **Infra guidance:** write access to `sfbiohub-infra` granted; **no storage limit**. Argus is
  **stateless** — a restart drops local data — so the pattern is: assets in an S3 bucket in
  `biohub-nonprod`, deployment **downloads from S3 on boot** (same-AWS copy is fast). Two tracks:
  1. **Infra PR (this):** create the S3 bucket + permission for the **nonprod cluster to read only**.
     Mirror existing repo practice — `proteohub-argus-s3-reader-dev.tf` is the canonical Argus+S3
     reader (IRSA role, `s3:GetObject`/`ListBucket`).
  2. **App-readiness (separate, app repo):** make the viewer Argus-ready per
     `czbiohub-sf/biohub-argus-example-app`; the **Argus MCP server** helps. On boot the app pulls
     from `s3://diffex-viewer-dev/` via the read-only role.
- **Draft:** `scratchpad/infra_pr/terraform/accounts/biohub-nonprod/diffex-viewer-dev.tf` — bucket
  (`cztack//aws-s3-private-bucket`) + read-only IRSA role (`argus-diffex-viewer-rdev/diffex-viewer`)
  + `diffex-viewer-dev-readwrite` uploader role (our CLI uploads only, NOT the cluster). Already
  matches the proteohub reader pattern. PR body requests a **1 TB ceiling** (actual footprint small).
- **Next:** user OK → push branch `diffex-viewer-dev` → PR → merge →
  `aws s3 sync viewer_assets/ s3://diffex-viewer-dev/` (readwrite role) → wire Argus app boot-download.

### 6. N-way single-cell CellDINO classifier MLPs (viewer score fix)  ← in build
- **Why:** the viewer badge must be **1-of-N distinctiveness** `P(target class)` (out of all
  classes), NOT the binary target-vs-NTC LR the traversal code used — that LR is trivially
  separable (acc 1.0) → saturated 0%/100% (the SAMM50 constant-cell artifact). The real trained
  classifier is the **bag-level SetTransformer** (attention over a set of cells → perturbation);
  the correct per-CELL proxy is **model C = `MLPHead` on CellDINO features**, class-vs-rest-distinct.
- **Plan:** one **N-way softmax MLP per (marker, grain)** — geneKO AND EBI-complex head per marker.
  Trained on `embed_crops` CellDINO features (SAME space the viewer re-encodes generated cells into)
  of top-attention cells over ALL classes incl NTC. Score: generated image → CellDINO (embed_crops,
  already computed in precompute) → MLP → softmax → `P(target)`.
- **Code:** `viewer/nway_clf.py::train_nway(grain, out_root, marker_channel, channel, fluor_csv,…)`
  → `<root>/_clf/<modality>/<grain>/{mlp.pt, classes.json, metrics.json}` (val top1/top5).
- **Scale:** ~60 markers × 2 grains + phase ≈ ~120 MLP training jobs (n_per_class≈100).
- **Next:** validate on one (marker,grain); launch training; **swap the score in `precompute`**
  (drop the binary LR, load the marker+grain MLP, `softmax(MLP(gemb))[target]`); re-run precompute
  to re-score. Only HSPA5 has an old per-class binary model C on disk (`extra/HSPA5/model_C.pt`).

### 7. REAL SetTransformer bag classifier (Option B — Alex's checkpoints, 2026-07-06)
- **Unblock:** Alex Lin shared the trained **`cellstate-set-classifier`** (SetTransformer) checkpoints
  on W&B (`czi.wandb.io/ai_imaging/cellstate-set-classifier`). It's the actual reported 1-of-N model
  and is a **bag/MIL** model (set of cells → perturbation class), so the honest viewer score is a
  **per-α `P(target)` curve** over an N(~100)-cell generated bag — NOT per-image.
- **Runs:** 1K phase geneKO `miwkg1cy` · 1K fluorescent (pooled markers) `hx6q8byj` ·
  EBI phase `epzvv0m1` · EBI fluorescent (pooled) `ggdfggsn` · EBI fluorescent (single-marker) `ciw91el9`.
- **Plan:** pull ckpts via `wandb` artifacts → load a forward (set of CellDINO features → class
  logits) → per α, feed the ~100 generated cells (embed_crops features) → `softmax → P(target)` →
  plot the curve under the strip (shows when the traversal "convinces" the real classifier; matches
  distinctiveness which is population-level). Reuse the `gemb` we already compute.
- **Relationship to #6:** this SUPERSEDES the per-cell MLP as the authoritative score (real model,
  no ~120 trainings); keep `nway_clf` MLP only as an optional per-image proxy.
- **Next:** confirm the checkpoints load here (arch + weights via wandb), build the set→logits
  forward, wire the per-α bag score into `precompute` / a curve panel in the viewer.

### 8. Image-UMAP montage viewer (latent-lens) — idea/track
- **Idea:** place each geneKO's generated cell (fixed source cell morphed toward that gene at α) at
  the gene's CellDINO-UMAP coordinate → zoom the embedding to see one cell take on each
  neighborhood's phenotype. Use **`czi-ai/latent-lens`** (`fit_umap` + `build_montage` → multiscale
  montage zarr + napari/live viewer) — purpose-built for this; don't hand-roll the pyramid/viewer.
- **Inputs we supply:** per-gene CellDINO embeddings (UMAP coords) + generated crops (callable) +
  gene categories/colors. **Next:** prototype flavor-B image-UMAP on the phase genes we have, after
  the per-marker relaunch; needs full-gene coverage for the complete embedding.

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
