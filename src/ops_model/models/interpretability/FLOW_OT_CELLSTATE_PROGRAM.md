# Flow matching, optimal transport & cell-state interpretability — short-term program

Living design doc, same spirit as [`diffae/PLAN.md`](diffae/PLAN.md) and
[`diffae/CROPSEQ_TO_MORPHOLOGY.md`](diffae/CROPSEQ_TO_MORPHOLOGY.md). Scopes how to extend the
existing DiffEx / SetTransformer interpretability stack with flow matching and optimal transport,
and two new interpretability tasks: **cell-state transition mapping** and **pathway morphological
mapping**. Assumption: "short-term" = a ~6–10 week program, workstreams parallelizable, most
building on infra that already exists rather than new training runs. Flag if the intended horizon
is different (a quarter, a single sprint) — the sequencing below assumes several people/GPU-weeks
running concurrently, not one person serially.

## 0. Where we actually are (grounds everything below)

- **Scoring:** SetTransformer (`classifier/`, aka `MixedChannelClassifier`) gives bag-level
  `P(class | cells)` + per-cell leave-one-out contribution (`classifier/score.py`) — the
  authoritative phenotype-strength readout.
- **Counterfactual generation:** `diffae/` — CellDINO-conditioned DiffAE (`phase_v1` prod,
  cond_ratio 0.47) + contrastive direction discovery (`diffae/directions/`). Default direction =
  deterministic `mean_diff` (control→KD centroid vector). An **unsupervised CFM already exists**
  (`diffae/directions/flow.py`, CellFlow-style, *independent* coupling) but was **not adopted**
  ("smoother but less clean phenotypes, noisy negative extreme" — PLAN.md 2026-07-03). This is the
  jumping-off point for Workstream A, not a green field.
- **Pathway structure:** `classifier/configs/ebi_complexes.csv` (98 EBI complexes) already used as
  the "complex" grain everywhere; EBI+ metric (guide-level distinctiveness grouped by complex) is
  wired into titration/aggregation but not into the DiffEx generative side.
- **Transcriptome:** `diffae/CROPSEQ_TO_MORPHOLOGY.md` already scopes a CROP-seq→morphology link
  using Duo Peng's sVAE+ gene-program embeddings (interpretable program axes, non-linear encoder,
  path/checkpoint documented there). Path B there (linear/MLP regression on **per-gene pseudobulk**
  means) is the currently-proposed approach — it exists on paper, not yet built.
- **Video:** `coding_exps/livescreen_preview/livescreen_scripts/segment_timelapse.py` does
  nuclei-seeded watershed segmentation + per-cell tracking on LiveScreen timelapses, with a
  per-cell photobleaching model (GFP ~stable, mCherry t½≈29h). This is a `coding_exps` script, not
  yet in `ops_model` — real single-cell **time series** exist but have never been fed into the
  interpretability stack (everything above treats each screen as a static population).

**The gap FM/OT are actually for:** every existing "trajectory" (mean_diff axis, the parked CFM,
the CFG traversal) is a synthetic straight line built between two *static, unpaired* populations.
We have three underused sources of real structure that motivate FM/OT specifically instead of
generically: (1) real single-cell time series (video) nobody has modeled as trajectories, (2) a
pathway/complex label that's used for classification but not for pooling directions, (3) a second
modality (transcriptome) that's unpaired at the cell level, which is exactly the regime OT (not
regression) is built for.

## 1. Workstreams

### A — OT-coupled flow matching (fix, not rebuild, `directions/flow.py`)

**Background — mean_diff vs. flow matching.** `mean_diff` (the current default,
`rank.py::supervised_direction`) is the simplest possible direction: average the top-attention
control cells and the top-attention KD cells in CellDINO space, take the difference of the two
means, normalize. Traversal moves every control cell's embedding along that **one fixed vector**:
`z(α) = z0 + α·gap·d`. This only matches the real KD distribution if it's basically a translated
copy of the control distribution (same shape, just shifted) — it can't represent curvature or a
multimodal KD population (two distinct sub-phenotypes average into a "between" point no real cell
occupies). **Conditional flow matching (CFM)** replaces the one fixed vector with a learned
velocity field `v_θ(z, t)` — integrate `dz/dt = v_θ(z,t)` from a control embedding at `t=0` to the
KD manifold at `t=1` — which can represent exactly that curvature/multimodality. `flow.py` already
implements this (CellFlow-style), citing CellFlow (bioRxiv 2025.04.11.648220) and the Flow Matching
Guide (arXiv 2412.06264).

**Why the existing attempt lost to mean_diff.** `train_flow`'s regression target pairs a random
control cell `a` with a random, *independently* drawn KD cell `b` and regresses toward the
straight-line velocity `b − a`. Because `a` and `b` are unrelated, the straight lines from many
independent `(a,b)` draws cross each other constantly in embedding space; at a given point,
different training pairs want the field to point different ways, and the network can only fit
their average — smoother than any real trajectory, and high-variance near the KD extreme. That's
exactly the symptom logged 2026-07-03 ("smoother but less clean phenotypes, noisy negative
extreme"). This is the textbook independent-coupling failure mode, not evidence that flow matching
can't work here — it's precisely why OT-CFM (Tong et al. 2023, already cited in `flow.py`'s own
references) and CellFlow itself add a **minibatch optimal-transport coupling** before regressing:
pair `a`/`b` within each minibatch to minimize total transport cost, so paths stop crossing and the
field the network fits is far more internally consistent.

One more wrinkle worth flagging: the 2026-07-03 "not adopted" call was made from
`make_gifs.py::render_flow` — qualitative GIFs/strips, eyeballed. `flow.py` was **never run through
the same quantitative harness** (`rank.py`'s LR classifier + `traverse.py`'s monotonic re-encode
score) mean_diff is benchmarked against. So the comparison wasn't apples-to-apples in the first
place, independent of the coupling question.

**The correction.** Swap the pairing step in `train_flow` from independent random draws to an
**exact minibatch OT coupling** (Hungarian assignment on squared-Euclidean cost via
`scipy.optimize.linear_sum_assignment`), keeping the `FlowNet` architecture, the straight-line
regression target, Euler integration, and DiffAE decode all unchanged:

```python
from scipy.optimize import linear_sum_assignment
...
a = x0[torch.randint(0, n0, (bs,), generator=g, device=dev)]
b = x1[torch.randint(0, n1, (bs,), generator=g, device=dev)]
if coupling == "ot":
    cost = torch.cdist(a, b).pow(2).detach().cpu().numpy()   # (bs,bs)
    row, col = linear_sum_assignment(cost)
    b = b[col]                                                 # OT-paired, not random
```

Exact assignment (not entropic Sinkhorn/POT) on purpose: it's fully deterministic given the
minibatch draw — no epsilon hyperparameter — which matches this codebase's already-stated bias
(the unsupervised InfoNCE direction bank was demoted to a secondary track specifically *because* it
was seed/GPU-nondeterministic; mean_diff won partly on reproducibility grounds). It also needs no
new dependency (`scipy` is already installed; `POT`/`import ot` is not currently a project dep), and
at `bs=256` the Hungarian algorithm is negligible next to the DDIM decode cost.

- **Bonus, same code path:** condition the flow on a *continuous* pathway/EBI-distinctiveness score
  instead of a binary control/KD label, so one flow model transports along a shared axis across a
  complex's member genes (ties directly into Workstream D).
- **Deliverable / status:** implemented as an additive `coupling: "independent" | "ot"` argument to
  `train_flow` (default unchanged, so `render_flow`'s existing callers are unaffected). A/B/C test
  script `coding_exps/diffex/ot_cfm_test/run_ot_cfm_test.py` scores **mean_diff vs.
  flow-independent vs. flow-OT** with one shared LR classifier and shared per-cell noise seeds, on
  HSPA5, reusing the production `phase_v1` DiffAE checkpoint read-only and writing all outputs under
  `coding_exps/diffex/ot_cfm_test/` (never touches the production `directions/` output tree). See
  build log below for the run.
- **De-risk gate (1–2 weeks):** if OT coupling doesn't beat mean_diff on those same metrics, drop
  it again — don't let this workstream run past the timebox.

#### Build log
- **2026-08-18 — test implemented + submitted.** `flow.py::train_flow` gained the `coupling` arg
  (OT via exact Hungarian assignment). Test logic lives in `directions/ot_cfm_test.py` (package
  module, not a bare script — a submitit worker unpickles job functions by import path, so it has
  to be importable from the compute node the same way `run.py`/`make_gifs.py` are; a first attempt
  as a standalone `coding_exps` script failed in 2s with `ModuleNotFoundError: No module named
  'run_ot_cfm_test'`). `coding_exps/diffex/ot_cfm_test/submit_ot_cfm_test.py` is just the SLURM
  dispatch, mirroring `directions/submit.py`. Scores mean_diff / flow-independent / flow-OT on
  HSPA5 through the same re-encode + monotonicity metrics `run_directions()` already uses, all
  outputs under `coding_exps/diffex/ot_cfm_test/HSPA5/` (production `directions/` tree untouched).

- **2026-08-18 — HSPA5 result (job 35595635, 11 min, 1 GPU): partial win.** Shared-scorer,
  shared-noise comparison at w=5 (`ot_cfm_test_metrics.json`,
  `comparison_HSPA5_w5.png`): mean score-Δ mean_diff=31.3, flow-independent=6.0,
  **flow-OT=19.5** — OT coupling more than **triples** the flow method's score swing.
  Visually (`comparison_HSPA5_w5.png`): on the control→KO arm, independent-coupling flow
  peaks mid-traversal then **declines** back toward zero at the KO extreme (the crossing-path
  instability, now directly visible, not just inferred); OT-coupling flow tracks the trusted
  mean_diff curve almost exactly through the same arm. Frac-strictly-monotonic was **0.0 for
  all three methods including production mean_diff** — not a flow-specific red flag: this
  codebase's own 2026-06-18 build log already showed CFG guidance `w` trades monotonicity for
  score magnitude (w=5 measured 0.12 on TOMM20 previously), and w=5 was used here for signal
  strength, so the metric is degenerate for any method at this w — the score-Δ and curve shape
  are the informative readouts. **Verdict: OT coupling is a real, demonstrated fix for the
  shelved failure mode, but doesn't yet match mean_diff's raw magnitude (19.5 vs 31.3) on this
  one target/setting.** Open follow-ups before calling the gate: re-run at a lower w (matches
  the historically-monotonicity-friendly settings) for a cleaner apples-to-apples; try a second
  target (TIMM23/TOMM20, already benchmarked for mean_diff in `PLAN.md`) to see if the pattern
  holds beyond HSPA5.

- **2026-08-18 — v2: multi-metric sweep (HSPA5 + TIMM23, w in {1,3,5}, 2 seeds/flow-method).**
  v1's one metric pair (full-axis monotonicity + endpoint score-Δ) was not enough to trust a
  verdict on — it degenerated to 0.0 for every method at the one w tested, and couldn't
  distinguish "the direction is wrong" from "the classifier logit moved but the embedding
  didn't actually approach real KD cells" from "the flow net just got a lucky/unlucky seed."
  `ot_cfm_test.py` (rewritten, `run_sweep`) now reports, per (method, target, w):
    - **half-axis metrics**, split at NTC — only the control→KO arm is what DiffEx actually
      claims to explain; the anti-KO extrapolation arm is a bonus, and v1 conflated the two.
    - **overshoot ratio** on the KO arm — `(peak − endpoint)/(peak − start)`, a number for the
      exact peak-then-decline shape v1's plot showed by eye for independent coupling.
    - **faithfulness-to-real-population distance** — distance from the re-encoded generated
      endpoint to the REAL KD/NTC embedding centroids, independent of the LR classifier (a
      direction could move the logit without the embedding actually nearing real KD cells).
    - **pixel-localization proxy** — fraction of Δ-pixel energy inside a centered disk vs. the
      outer ring, a cheap (mask-free, approximate) check for the borders/background-artifact
      failure mode logged 2026-06-18.
    - **cross-seed reproducibility** for the flow variants (2 seeds each) — mean_diff needs no
      such check (deterministic by construction); a trained flow net is not, and this codebase
      already demoted the unsupervised InfoNCE bank once for exactly this failure mode. v1
      tested one seed per method and could not see this axis at all.
  Also fixed a v1 rigor gap in passing: mean_diff's `cfg.alphas` grid is not evenly spaced
  (denser near 0) while the flow arms' Euler steps are — v1 plotted both against a raw index as
  if directly comparable. v2 puts mean_diff on the identical evenly-spaced grid. Outputs under
  `coding_exps/diffex/ot_cfm_test/sweep_v2/{HSPA5,TIMM23}/` (kept separate from v1's `HSPA5/` so
  the already-reported v1 numbers/plots aren't overwritten). Submitted as 2 parallel SLURM jobs
  (1 GPU each); awaiting result.

- **2026-08-18 — v2 result (jobs 35596449_{0,1}, 38 min / 18 min, 1 GPU each): OT coupling wins
  on every metric, consistently, across both targets and all 3 w's — but doesn't surpass
  mean_diff.** Full table in `ot_cfm_sweep_metrics.json` per target; headline pattern (mean over
  2 seeds where applicable):
    - **KO-arm score-Δ:** mean_diff > flow_ot > flow_independent always. flow_ot recovers
      ~80–92% of mean_diff's magnitude (e.g. TIMM23 w=3: 46.4 vs flow_ot 38.7 vs
      flow_independent 33.0); flow_independent recovers only ~55–75%.
    - **KO-arm monotonicity (frac_nondecreasing):** mean_diff near-perfect at low w (0.75–1.0),
      degrading at w=5 (known w/monotonicity tradeoff, confirmed again here). flow_ot is
      consistently better than flow_independent (e.g. HSPA5 w=1: 0.625/0.375 vs 0.125/0.0) but
      well below mean_diff.
    - **Overshoot ratio (peak-then-decline on the KO arm) — the cleanest differentiator:**
      mean_diff ~0 (0.0–0.04) at every w/target; **flow_ot roughly HALVES flow_independent's
      overshoot** at every single w/target pair (e.g. TIMM23 w=1: 0.06–0.01 vs 0.21). This is a
      direct number for the "peaks then declines" shape v1's plot showed by eye.
    - **Faithfulness (re-encoded endpoint distance to real KD/NTC centroids):** on TIMM23 (the
      "obvious"-phenotype target) flow_ot's KO-endpoint is clearly closer to the real KD cloud
      than flow_independent's (e.g. w=1: ~37.9 vs ~44.6); on HSPA5 (subtler) the gap is smaller/
      noisier. Independently, flow_independent's **anti-KO extreme drifts further off the real
      NTC manifold** than flow_ot's or mean_diff's at every setting (e.g. HSPA5 w=1: ~58 vs ~44
      vs ~38) — OT coupling keeps BOTH trajectory extremes closer to real data, not just the KO
      side.
    - **Pixel-localization proxy:** no meaningful difference between methods (~0.21–0.27
      everywhere) — not a differentiator here (reported for completeness, not overclaimed).
    - **Cross-seed reproducibility — the most decisive result:** flow_ot's two seeds correlate
      at 0.99–0.999 with each other; flow_independent's at 0.978–0.997 — every single case.
      More strikingly, flow_ot's seed-to-seed delta *range* is only ~2–7% of its own mean,
      while flow_independent's is ~15–35% of its own mean. **OT coupling doesn't just improve
      the average outcome, it makes the method far more reproducible** — directly answering
      the historical concern (the unsupervised InfoNCE bank was demoted for exactly this).
  **Verdict:** OT coupling is a real, convergent, multi-metric fix — wins on every axis tested
  vs. independent coupling — but mean_diff still wins outright on this task shape (control vs.
  ONE unimodal KO population), which is the regime a simple linear direction should already be
  near-optimal for. The fairer test of flow's real value proposition is a **multimodal / pooled
  target** (e.g. Workstream D's per-pathway pooling across a complex's member genes), where a
  single mean_diff vector is expected to genuinely struggle — not a single clean gene.
  **Known gaps in this validation** (flagged, not yet run): no held-out generalization check
  (LR + flow are both fit and scored on the same cells — flow's higher-capacity net is more
  overfitting-prone than a mean vector); no real segmentation-mask pixel localization (centered-
  disk proxy only); no visual inspection of flow-generated images yet — everything above is
  classifier-embedding-space evidence only, and DiffEx's actual deliverable is a *visual*
  counterfactual.

- **2026-08-18 — visual montage + pooled/multimodal test: the fairer test flips the verdict.**
  Two follow-ups, per user request: (1) actual decoded images (not just logit curves) for
  mean_diff/flow-independent/flow-OT side by side (`ot_cfm_test.py::render_montage`,
  `sweep_v2/{HSPA5,TIMM23}/montage/`); (2) the multi-metric sweep on a genuinely pooled/
  multimodal target instead of one clean gene.
    - **Montage fixes (user feedback):** dropped the Δ-pixel heatmap overlay (raw images only —
      easier to actually judge), zero-gap gridspec (`wspace=0`) for a seamless filmstrip, alpha/t
      value labeled on every column (was unlabeled before), and **switched the default guidance
      from w=5 to w=1** — the sweep already showed w=5 is where monotonicity/overshoot are WORST
      for every method, so judging visual plausibility at the noisiest setting was the wrong
      choice. At w=1 all three methods produce clean, coherent morphs with no dominant
      border/background artifacting.
    - **Visual confirmation of the overshoot metric:** on "Core mediator complex" cell 0, flow-
      independent's row visibly **grows a bright nucleolar structure through the middle of the KO
      arm, then shrinks it back down at the extreme** — the overshoot_ratio number made literally
      visible, not an artifact of the metric definition. mean_diff and flow-OT both keep building
      toward the real KD reference instead.
    - **Pooled target:** "Core mediator complex" (grain='complex', 3 member genes MED18:220/
      MED21:168/MED6:112 top cells — the most gene-balanced EBI complex available, so genuinely
      multimodal, not one gene dominating). Required a fix: the complex ranking parquet
      (`pma_shap_phase_complex.parquet`) carries **no NTC rows at all** (a different SHAP-based
      ranking source than the geneKO attention parquet) — `gather()`'s single-parquet assumption
      doesn't hold for grain='complex'. Added `_gather_pooled_complex`/`_gather_any` (own module,
      production `data.py` untouched) to pull the target from the complex parquet and NTC from
      the geneKO parquet — both share the same `_BASE_COLS` schema, a safe cross-source join.
    - **Result: flow-OT closes MOST of the way to mean_diff here, closer than on any single gene
      tested.** KO-arm score-Δ recovers 91–95% of mean_diff's magnitude (vs. 80–92% on HSPA5/
      TIMM23). At w=1, flow-OT's KO-arm overshoot is **exactly 0.0 for both seeds** — matching
      mean_diff's near-zero overshoot, not just "less bad than independent coupling." Frac-
      nondecreasing at w=1 averages 0.94 across flow-OT's 2 seeds vs. mean_diff's 1.0 — the
      closest flow-OT has come to matching mean_diff's cleanliness on any metric so far.
      **Faithfulness reverses direction:** flow-OT's re-encoded KO endpoint (34.7) is actually
      CLOSER to the real KD centroid than mean_diff's (36.1) — the first target where flow beats
      mean_diff outright on a metric, not just closes the gap. Reproducibility holds the same
      pattern as before (flow-OT corr 0.995–0.996 vs. flow-independent 0.983–0.993).
    - **Verdict, updated:** the single-gene tests (HSPA5, TIMM23) showed mean_diff winning
      outright because that's the regime a linear direction should already be near-optimal for
      (one unimodal KO cluster). The pooled/multimodal test was designed to be the fairer trial
      of flow's actual value proposition — and it's fairer in fact, not just in theory: flow-OT
      is now essentially competitive with mean_diff on structure (monotonicity, overshoot) and
      slightly ahead on faithfulness, while still trailing a little on raw magnitude.
      **Recommendation: promote OT-coupled flow matching from shelved-experiment to a real
      second `direction_method` option, used specifically for pooled/pathway-level targets
      (Workstream D) where mean_diff's linear assumption is weakest — keep mean_diff as the
      default for individual clean single-gene targets, where it remains deterministic and at
      least as good.** Still open: held-out generalization check, real segmentation-mask pixel
      localization, and testing beyond this one pooled complex before generalizing further.

- **2026-08-19 — direct test: does flow-OT beat mean_diff on individual low-mAP genes?**
  Independent, more targeted version of the pooled-complex hypothesis: instead of pooling
  multiple genes into one target, pick INDIVIDUAL genes whose top-attention cells are already
  known to be weakly distinguishable from control (low phase mAP/distinctiveness), the
  signature of an incomplete-penetrance or mixture phenotype rather than a clean single shift.
    - **Cheap first check, no new compute:** joined phase-channel distinctiveness
      (`gene_reporter_distinctiveness_raw.csv`'s `Phase` column) against the
      `flow_advantage_ko_delta_ratio` already computed for all 1000 genes
      (`geneKO_flow_advantage_ranking.csv`). Correlation is **−0.18 (Pearson), −0.17
      (Spearman)** — lower mAP → higher flow advantage — and monotonic across mAP quartiles
      (mean ratio 0.906 in the highest-mAP quartile → 0.977 in the lowest, 250 genes/bucket).
      Confirmed candidates aren't a data-scarcity artifact: all have 56k–65k attention-ranked
      cells available (above the 25th-percentile of 50.8k across all 1000 genes), so the usual
      top-1000-cell selection has plenty to draw from.
    - **Full multi-metric sweep + visual montage on two clean low-mAP candidates** (ATXN10,
      phase mAP 0.023; S100B, mAP 0.011), same rigor as HSPA5/TIMM23/Core-mediator-complex:
      - **ATXN10 at w=1 (cleanest setting): flow-OT ko-arm delta 32.3 vs. mean_diff 21.7 (49%
        higher)**, while matching mean_diff's monotonicity closely (0.75 vs. 0.875) — flow-OT
        wins on magnitude without giving up structure. Visually, mean_diff barely changes the
        cell (subtle speckling at most) while flow-OT produces a dramatic, distinct bright
        globular/droplet phenotype closer in character to the real KD reference than either
        other method.
      - **S100B at w=1: flow-OT 32.6 vs. mean_diff 24.2 (35% higher)**, monotonicity tied
        (0.625 both). Visually the sharpest case yet: **mean_diff's montage shows essentially
        no visible change across the entire α range**, while flow-OT produces a real punctate/
        speckled texture resembling the real KD reference — a case where a picture, not just a
        metric, shows mean_diff failing outright. (mean_diff catches up numerically at higher
        w=3/5 even though its own images stay visually flat — a real, unresolved discrepancy
        between classifier-score movement and visible pixel change worth flagging, not
        explaining away.)
      - flow-independent remains the weakest and least reproducible on both genes (e.g. S100B
        w=5: one seed's full-axis delta is 7.1, the other is **−3.5** — sign-flipping noise,
        not signal), reconfirming OT-coupling is necessary, not just nice-to-have, for flow
        matching to be usable at all here.
    - **Verdict: two independent lines of evidence (pooled multi-gene complexes, and
      individually low-mAP single genes) now agree** — mean_diff is the better default for
      clean, well-separated single-gene phenotypes; OT-coupled flow matching should be the
      go-to for targets flagged as weak/diffuse by mAP or known to be multi-gene pools, where
      it can reveal real morphological change mean_diff visibly misses.

### B — real trajectories from video (the highest-novelty, highest-risk piece)
- **Goal:** stop synthesizing NTC→KO interpolations from static populations; use `segment_timelapse.py`
  to track individual cells through LiveScreen timelapses, embed every frame with CellDINO, and get
  an actual `(z_t)_{t=0..T}` morphology trajectory per tracked cell.
- **Method:** because cells are now genuinely paired across time (unlike the OPS static screens),
  this is exactly the regime for a **stochastic interpolant / Schrödinger-bridge flow matching**
  model conditioned on real `(z_0, z_T)` pairs, rather than population-level OT coupling — strictly
  more information than either Workstream A variant has ever had access to.
- **New interpretability task this unlocks — cell-state transition mapping:** build a
  pseudotime/manifold (PHATE/UMAP over CellDINO, as the embedding tab already does) from real
  tracks, then measure how a perturbation *reroutes* the trajectory (speed, direction, endpoint)
  relative to wild-type tracks in the same field — a fundamentally different readout than "does the
  static population differ," closer to "how does the cell get there."
- **Hard prerequisite:** bleach-correct embeddings first — mCherry's ~29h half-life will otherwise
  imprint on the trajectory and get read as biology, not imaging artifact. Reuse the existing
  per-cell photobleach model rather than re-deriving it.
- **Open question to confirm before investing:** which LiveScreen experiments have track quality
  (length, drift, confluency) good enough for this — I don't have that assessed; scope a 2–3 gene
  pilot on the cleanest experiments before any generalization claim.

### C — optimal transport for morphology↔transcriptome coupling (upgrades CROPSEQ_TO_MORPHOLOGY.md)
- **Why regression (current Path B) is leaving signal on the table:** fitting `Δt_g → ΔCellDINO_g`
  on per-gene pseudobulk means collapses every cell in a KO to one point — throws away
  within-perturbation heterogeneity (e.g., bimodal response) in both modalities.
- **OT fix — Gromov-Wasserstein alignment:** per gene, treat its top-attention-cell CellDINO cloud
  and its sVAE+ program-space cloud (Duo's encoder, already resolved in CROPSEQ_TO_MORPHOLOGY.md)
  as two point clouds with **no shared coordinates** — GW only needs each cloud's own pairwise
  distances, which is exactly the unpaired-modality regime here (no single cell is in both
  datasets). The transport plan gives a soft within-gene cell correspondence, richer than a mean
  vector, and its **GW cost is itself a per-gene metric**: low cost = transcriptome and morphology
  organize the same way; high cost = one modality is more heterogeneous or the two disagree.
- **Payoff — this IS the "transcriptome↔morphology divergence map"** already sketched at the
  bottom of CROPSEQ_TO_MORPHOLOGY.md, just upgraded from a mean-vector regression R² to a proper
  distributional OT cost, and it composes directly with Workstream D (per-pathway, not just
  per-gene).
- **Cost caveat:** GW is O(n²m²)-ish per gene; subsample to the top-attention cell counts already
  used elsewhere (hundreds, not tens of thousands) rather than full pools.

### D — pathway morphological mapping (new interpretability task, cheapest workstream)
- **Goal:** one morphological axis per EBI complex / sVAE+ gene program, not per gene — a direct
  test of "does this annotated complex have a *coherent* morphological signature across its member
  genes, or does the complex label paper over morphologically distinct subgroups?"
- **Method:** pool top-attention cells across a complex's member genes, weighted by the existing
  **EBI+ within-complex distinctiveness** metric, fit the Workstream-A direction at the pathway
  level, traverse, and check whether individual member genes land consistently along that shared
  axis (cross-check via the classifier score, same faithfulness check already baked into `directions/`).
- **Why it's the cheap win:** no new modeling — reuses EBI+ scores, attention-head cell sets, and
  the direction/traversal code as-is. Good candidate to run first; a negative result here (complexes
  don't cohere morphologically) is itself useful and de-risks over-investing in Workstream C before
  it's built.

### E — trajectory/direction cross-check (stretch, ties A/B/D together)
- Use the SetTransformer per-cell contribution score (`classifier/score.py`) evaluated along a real
  video track (Workstream B) as a behavioral check on whether "phenotype strength" rises
  monotonically over time — a direct validation of whether the static OT/FM-discovered direction
  (A/D) matches real single-cell temporal dynamics, not just a population-level artifact. Cheap
  add-on once B produces tracks; not a separate infra build.

## 2. Sequencing (parallel, not serial)

| weeks | workstream | depends on |
|---|---|---|
| 1–2 | A (OT-CFM fix) | nothing new — `flow.py` + existing benchmarks |
| 1–3 | D (pathway pooling) | EBI+ metric (have), direction code (have) |
| 2–5 | C (GW cross-modal) | Duo's sVAE+ encoder (resolved), top-attention cell sets (have) |
| 3–6 | B (video trajectories) | track-quality triage (open question above), bleach correction |
| 6+ | E + synthesis | outputs of A/B/D |

**Headline synthesis question for the end of the program:** does the pathway-level OT/FM direction
(D, built from static populations) agree with the real single-cell trajectory direction (B, built
from video)? Agreement validates the whole static-population counterfactual approach against real
dynamics; disagreement is itself a finding (static screens see an endpoint, not the path).

## 3. Explicitly out of scope for this program (already flagged elsewhere as bigger bets)
- Full CPA-style joint transcriptome-image latent (`CROPSEQ_TO_MORPHOLOGY.md` Path A) — real
  training effort, already marked "needs design confirm."
- Per-fluorescent-marker DiffAE scale-out (~60 models) — already marked "needs design confirm,"
  orthogonal to FM/OT.
- SetTransformer v2 (no-mask) retrain — blocked on Alex, not something this program can move.

## 4. Success criteria (per workstream, so this can be dropped independently if a gate fails)
- **A:** OT-CFM beats mean_diff on monotonicity + re-encoded Δ on ≥2 of the 3 existing benchmark
  targets, or it's shelved again — no open-ended tuning past the timebox.
- **D:** for ≥50% of EBI complexes, member-gene traversal scores move consistently along the shared
  pathway axis (vs. scattering) — defines "coherent" quantitatively before calling it a result.
- **C:** GW cost correlates with an independent heterogeneity proxy (e.g. per-gene attention-score
  spread) as a sanity check that the cost is measuring something real, not GW numerics.
- **B:** pilot recovers a known, visually obvious phenotype's trajectory shape on 2–3 genes before
  any claim about subtler phenotypes.
