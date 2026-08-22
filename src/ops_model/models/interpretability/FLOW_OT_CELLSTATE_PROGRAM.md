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

- **2026-08-20 — "does OT recover real bimodal structure that mean_diff can't?" — per-cell
  routing was the wrong test; distribution-shape recovery is the right one.**
  Follow-up to the mAP-based finding above: instead of weak/diffuse phenotypes, tested genes with
  a genuinely **multimodal** real KD population — scanned all 1000 genes' real embeddings
  (KMeans k=2 separation on KD minus the same stat on NTC as a null baseline, `sep_excess`) and
  picked 4 candidates with real bimodal splits: TUBGCP6, KIF11 (balance 0.46 — biologically
  plausible cell-cycle-dependent mitotic arrest), SEC61A1 (balance 0.98 — near-perfect 50/50),
  S100B (balance 0.36, weakest of the four).
    - **First attempt (rejected):** push 40 real NTC probe cells through each method, check which
      real KD cluster (A/B) each endpoint lands nearest to, compare the routed fraction to the
      true cluster balance. Numerically and visually (per-gene PCA plots, `bimodal_{gene}.png`)
      this was mixed/inconclusive — no clean "mean_diff collapses, OT splits" pattern in 2 of 4
      genes. Root cause: NTC probes have no ground-truth fate label, so there's nothing to check
      per-cell routing *against* — mean_diff doesn't literally collapse spatially either, since
      it's a per-cell translation and different probes start at different points.
    - **Reframed as a distribution-shape test (the actual mathematical claim):** mean_diff is a
      pure translation (`x + c`, same `c` for every cell) — translating a distribution cannot
      change its shape, so the KMeans separation stat on mean_diff's *output* population must be
      identical to the stat on the input NTC population, by construction, regardless of what the
      real KD looks like. Flow-OT is a nonlinear map and has no such constraint. Tested this
      directly: pushed **all** real NTC cells (not 40 probes) through both methods and measured
      output-population `sep` (`bimodal_shape_test.py`, CPU-only, reuses the already-trained
      `flow_ot.pt` checkpoints, no GPU/decode needed):

      | gene | real KD truth (sep / balance) | NTC null | mean_diff output | flow-OT output |
      |---|---|---|---|---|
      | KIF11 | 27.6 / 0.46 | 18.6 / 0.66 | 18.6 / 0.66 (= null) | 29.6 / 0.48 |
      | SEC61A1 | 27.5 / 0.98 | 18.6 / 0.66 | 18.6 / 0.66 (= null) | 33.2 / 0.73 |
      | TUBGCP6 | 32.3 / 0.46 | 18.6 / 0.66 | 18.6 / 0.66 (= null) | 40.0 / 0.50 |
      | S100B | 26.0 / 0.36 | 18.6 / 0.66 | 18.6 / 0.66 (= null) | 27.5 / 0.53 |

      mean_diff's output `sep`/`balance` matches the NTC null to the decimal in all 4 genes —
      the translation-invariance proof confirmed numerically, not just asserted: it gains **zero**
      bimodality over the unperturbed control population no matter how bimodal the real KD is.
      flow-OT's output matches or slightly *exceeds* the real KD's true separation in all 4 genes,
      and its balance tracks the true cluster proportion much more closely than mean_diff's (which
      is pinned at the NTC's own 0.66 every time) in 3/4 genes.
    - **Verdict: this is the clean, decisive demonstration** — not "does OT route individual cells
      correctly" (no ground truth exists to check that against), but "can the method manufacture
      distributional structure a linear shift provably cannot." mean_diff structurally cannot turn
      a unimodal control population into a bimodal one; flow-OT does, recovering real bimodal
      separation from the control distribution alone, across every multimodal candidate tested.

- **2026-08-20 (cont.) — does the shape-recovery finding survive actual DiffAE decode +
  re-embedding, on a real population (not 2 hand-picked cells)?** The result above was raw
  pre-decode embedding-space math only. Two visual follow-ups were tried first and were weaker:
  a 2-probe decoded-image montage (pick the 2 real NTC cells whose flow-OT endpoints land
  nearest each real KD sub-cluster, decode both methods' trajectories) showed a real but
  partial/gene-dependent pattern — informative but noisy at n=2 (`bimodal_probe_montage.py`,
  KIF11 then TUBGCP6, KO-arm-only alphas for bigger panels). The decisive follow-up, per the
  user's own proposed design: **push a real POPULATION (n=100 real NTC cells, not 2 extremes)
  through both methods to α=+3×gap, decode every endpoint through the frozen DiffAE, re-embed
  via CellDINO, and re-run the exact same KMeans-separation stat on the re-embedded population**
  (`pool_shape_recovery.py`). One bug caught before it produced silently-wrong output:
  `flow.integrate_flow` concatenates trajectory steps along `dim=0`, which only gives the
  documented `(n_record+1, dim)` shape for a batch-size-1 call — feeding it the whole 100-cell
  pool at once silently flattened batch and time together, and `[-1]` grabbed one stray row
  instead of 100 endpoints (caught via a shape-mismatch crash inside DiffAE's cond projection,
  not a silent bad result — fixed by integrating per-cell, the same convention used everywhere
  else in this codebase). Result on TUBGCP6 (n=100, α=+3):

  | | sep | balance |
  |---|---|---|
  | real KD (truth) | 32.3 | 0.46 |
  | decoded NTC (α=0 null — decode-pipeline's own noise floor) | 22.1 | 0.92 |
  | decoded mean_diff | 24.8 | 0.92 |
  | decoded flow-OT | 33.3 | 0.37 |

  **Initial (WRONG) read:** decoded flow-OT's sep/balance (33.3/0.37) looked like it matched
  real truth (32.3/0.46) while decoded mean_diff's (24.8/0.92) looked stuck at the decode-only
  null (22.1/0.92) — read at the time as "the finding survives contact with real image
  generation." **This was wrong, caught by the user visually inspecting the PCA scatter**
  ("mean_diff captures real cluster B somewhat, flow-OT is closer to NTC and doesn't cover
  either cluster well" — exactly right). The bug: `_split_score` re-clusters each method's
  OWN output against itself (is this population internally bimodal at all?) — it never checks
  whether the two sub-groups it finds actually sit where the REAL clusters are. A population
  that's half stuck-at-NTC and half moved-toward-one-real-cluster is ALSO internally bimodal, so
  it passes that stat without meaning what the sep/balance numbers implied.
  **Corrected test: 3-way nearest-real-centroid classification** (real NTC centroid vs. real
  cluster-A centroid vs. real cluster-B centroid, in the same embedding space) instead of blind
  self-reclustering. Sanity-checked first (real cluster-A members self-classify as A 86.5% of
  the time, cluster-B as B 99.4%, real NTC pool as NTC 88.8% — the centroids are discriminative).
  Actual result on the same 100 decoded cells:

  | | classifies as NTC | classifies as A | classifies as B |
  |---|---|---|---|
  | decoded mean_diff | 0% | 28% | **72%** |
  | decoded flow-OT | **18%** | **56%** | 26% |

  **Reversed conclusion:** mean_diff actually captures real cluster B fairly well post-decode
  (72%) and never stalls at NTC. flow-OT does worse two ways: 18% of its cells never really left
  NTC (a real stall/failure mode the self-clustering stat couldn't see at all), and among the
  ones that moved, it's skewed toward A (56%) over B (26%) — the opposite of an even, truth-
  matching split. **The decode-validated population test does NOT support the hypothesis once
  measured correctly.** This does not touch the separate, provable raw-embedding-space result
  above (mean_diff's PRE-decode output is mathematically identical to the NTC null via
  translation-invariance, independent of any clustering-metric choice) — only this decode+re-
  embed extension, and the visual 2-probe montages before it, are affected. **Open question,
  not yet resolved:** whether flow-OT's 18% NTC-stall + A/B skew is itself informative (e.g. an
  artifact of α=+3 overshoot, or a real asymmetry in the trained field) or just noise from the
  documented generated-vs-real domain gap — untested at this point.

- **2026-08-20 (cont.) — resolving that open question: audit the ORIGINAL raw-embedding bar
  chart (all 4 genes) with the same nearest-real-centroid test, since it has the identical
  self-clustering blind spot and was never checked this way either.** Same 3-way classification
  (nearest of: real NTC centroid / real cluster-A centroid / real cluster-B centroid), same
  parameters as the original bar chart (ALL real NTC cells, t_max=1.0, no decode):

  | gene | true balance | mean_diff (NTC / A / B, %) | flow-OT (NTC / A / B, %) |
  |---|---|---|---|
  | KIF11 | 0.46 | 5.3 / 24.0 / 70.7 | 2.9 / 21.3 / 75.8 |
  | SEC61A1 | 0.98 | 0.1 / 46.9 / 53.0 | 0.1 / 38.2 / 61.7 |
  | TUBGCP6 | 0.46 | 13.6 / 67.9 / 18.5 | 3.3 / 65.2 / 31.5 |
  | S100B | 0.36 | 21.2 / 20.3 / 58.5 | 15.4 / 25.8 / 58.8 |

  **Not a clean win either direction.** KIF11 is a near-tie. SEC61A1: mean_diff's split
  (46.9/53.0) tracks the true near-50/50 balance better than flow-OT's (38.2/61.7) — mean_diff
  wins. TUBGCP6 and S100B: flow-OT stalls at NTC less (3.3% vs 13.6%; 15.4% vs 21.2%) and reaches
  the minority cluster more — flow-OT wins. So the raw pre-decode field itself does NOT
  universally recover true bimodal structure better than mean_diff; it's gene-dependent, roughly
  a wash across these 4.
  **This also answers the open question above:** on TUBGCP6, the raw/t_max=1 field stalls at
  NTC only 3.3% of the time, vs. 18% for the decoded/α=+3 pool test, while the A:B ratio among
  non-stalled cells stays similar between the two. The NTC-stall failure mode is small in the
  field itself and gets substantially worse specifically after decode + the larger α — points at
  decode/overshoot as the amplifier of that failure mode, not the trained field's core behavior.
  **Where this leaves Workstream A's headline claim:** the only fully intact result is mean_diff's
  translation-invariance (mathematically forced, gene- and metric-independent — it cannot create
  bimodal structure it doesn't already have). Whether OT-coupled flow matching actually recovers
  real multimodal structure *better* than mean_diff is NOT settled — gene-dependent at the raw
  level, and worse under decode+large-α. Not yet tested: whether this pattern holds at smaller α
  (e.g. α=1, the canonical KD-gap point) instead of the α=+3 overshoot regime used in the pool
  decode test, on more than one gene.

- **2026-08-20 (cont.) — tested that directly: (gene x alpha) sweep, TUBGCP6 and KIF11 at
  α=1 and α=3, using the corrected nearest-real-centroid classification** (now wired into
  `pool_shape_recovery.py` itself — `_classify_nearest` replaces `_split_score` as the trusted
  metric; the old stat is kept in the JSON for continuity only, docstring flags it as unreliable).
  Surfaced a bigger confound first: **`decoded_ntc_null` (α=0, literally re-decoding the SAME
  real NTC cell with zero shift) only classifies as NTC 52-53% of the time in both genes** — the
  rest drifts to a real KD sub-cluster from decode noise ALONE, and the drift direction is
  gene-specific (TUBGCP6 drifts toward cluster B: 36%; KIF11 drifts toward cluster A: 44%). Any
  decode-level percentage below has to be read against this noisy null, not against zero — this
  is itself a new, previously-uncharacterized weakness of the generation pipeline (consistent
  with the documented generated-vs-real domain gap, now precisely quantified at the population
  level for the first time).

  | | TUBGCP6 α=1 | TUBGCP6 α=3 | KIF11 α=1 | KIF11 α=3 |
  |---|---|---|---|---|
  | decoded_ntc_null (NTC/A/B) | 53/11/36 | 53/11/36 | 52/44/4 | 52/44/4 |
  | decoded mean_diff (NTC/A/B) | 10/39/51 | 0/28/72 | 4/59/37 | 0/17/83 |
  | decoded flow-OT (NTC/A/B) | 2/47/51 | **18/56/26** | 0/39/61 | **5/27/68** |

  **Two things are now clear and reproduce across both genes:**
  1. mean_diff's NTC-fraction only ever drops or holds at 0 as α increases (10%→0%; 4%→0%) —
     expected, a straight-line extrapolation moves monotonically further from NTC by construction.
  2. **flow-OT's NTC-fraction goes UP with larger α, not down** (2%→18%; 0%→5%) — a real,
     gene-general overshoot/fold-back-toward-NTC behavior mean_diff cannot exhibit by
     construction (it's linear; there's no mechanism for a larger shift to look MORE like the
     starting point). This confirms and generalizes the single-gene overshoot observation from
     the previous entry: it's α-dependent and specific to the learned flow field, not a decode
     fluke on one gene.
  The A-vs-B balance question (does either method's split better match true cluster
  proportions) does NOT show a consistent winner across the two genes — TUBGCP6 α=1 has flow-OT
  reaching slightly more A while matching mean_diff's B; KIF11 α=1 has flow-OT reaching notably
  more B than mean_diff. No stable pattern.
  **Overall verdict for Workstream A, stated plainly:** the one fully robust, gene-general
  finding across this entire investigation is mean_diff's translation-invariance (provable) and,
  now, flow-OT's α-dependent NTC-overshoot (empirical, 2/2 genes). Whether flow-OT recovers real
  multimodal structure better than mean_diff is NOT a settled win for either method — it's
  gene-dependent, entangled with a large decode-pipeline domain-gap bias that exists even with
  zero intended perturbation, and gets worse (more NTC fold-back) at larger α for flow-OT
  specifically. This is a materially weaker conclusion than where this workstream started the
  day; recommend NOT promoting flow-OT over mean_diff for the bimodal/multimodal use case on the
  strength of anything found today — the pooled-complex result (2026-08-18 entry above, which
  used the classifier score directly rather than this centroid-classification approach) remains
  the strongest actual evidence for flow-OT's value and hasn't been revisited with this scrutiny.

- **2026-08-21 — literature review, then testing its top-priority fix directly.** Ran a
  literature review (Tong et al. 2302.00482, CellFlow bioRxiv 2025.04.11.648220, Rectified Flow
  2209.03003, Flow Matching Guide 2412.06264, Cheng & Schwing 2503.10636, Boïté/Delon/Nadjahi
  2605.12174, CellOT/Bunne et al., moscot/Klein et al. — PDFs in `Optical_Flow/papers/`) against
  the two open problems above. Findings, condensed: (1) the α>1 extrapolation dial has NO basis
  in this literature at all — Rectified Flow's marginal-preserving guarantee only holds for
  t∈[0,1], and CellFlow's actual mechanism for "stronger effect" is a conditioning covariate
  baked into training, never longer integration; our `FlowNet` makes this worse than the generic
  case since raw scalar `t` has no positional encoding and training only samples t~Uniform(0,1),
  so t=3 is doubly out-of-distribution. (2) our batch=256 exact-Hungarian coupling diverges from
  CellFlow's own recipe (entropic/Sinkhorn, batch≈512+, OT-pairing batch decoupled from the SGD
  batch per Cheng & Schwing's "oversampling" fix) on every axis; Tong et al.'s own ablation shows
  batch=1 is IDENTICAL to independent coupling and harder multimodal cases need larger batches.
  **Implemented fix #2 directly** (`flow.py::train_flow(coupling="ot_sinkhorn")` +
  `_sinkhorn_col_sample`): every 50 steps, resample an OT-pairing batch as large as the dataset
  allows (both genes only have 1000 cells/class, so this is close to population-level OT, not
  just "a bit bigger than 256"), solve an entropic Sinkhorn plan, cache it, and have SGD
  re-subsample 256-sized minibatches from that cached pairing in between. Two real bugs caught
  before getting a result: an absolute `eps=0.05` overflowed the log-domain updates (CellDINO
  squared-distances run into the hundreds/thousands — fixed by scaling eps relative to the cost
  matrix's own mean, standard Sinkhorn stability practice), then a wrong update rule (additive
  `f = ... + f` instead of the correct closed-form replacement each iteration, since f/h only
  enter the marginal conditions as outer multiplicative factors) that made both potentials grow
  unboundedly until they overflowed.
  **Controlled A/B result (same run, same device, t_max=1 only, all real NTC, nearest-real-
  centroid classification), against the true real-KD cluster-A/B proportions:**

  | gene | true A/B | ot (NTC/A/B) | ot_sinkhorn (NTC/A/B) |
  |---|---|---|---|
  | KIF11 | 31.4% / 68.6% | 1.8/40.9/57.3 | 3.2/21.9/74.9 |
  | TUBGCP6 | 68.4% / 31.6% | 6.9/67.3/25.8 | 5.7/55.1/39.2 |

  Normalizing out NTC-stall and comparing the A-fraction to true A: KIF11 error drops slightly
  (ot 10.2pp → ot_sinkhorn 8.8pp) but TUBGCP6 error nearly triples (ot 3.9pp → ot_sinkhorn
  10.0pp) — Sinkhorn overcorrects PAST the true ratio on TUBGCP6. NTC-stall moves in opposite
  directions too (KIF11 worse: 1.8%→3.2%; TUBGCP6 better: 6.9%→5.7%).
  **Verdict: this specific, literature-grounded fix does NOT decisively resolve the multimodal-
  recovery problem** — small, mixed, gene-dependent effects in both directions, not a closed
  gap. This actually matches the literature review's own caveat, stated before this test ran:
  "I could not find any paper that guarantees minibatch OT-CFM recovers an a-priori-unknown
  multimodal split correctly... it should reduce the bias, not provably close the gap." Fix #1
  (the α>1 extrapolation dial having no theoretical basis) remains untested/unimplemented — it
  requires a real graded strength covariate in the data (partial knockdown, sgRNA efficacy
  tiers, timepoints), which hasn't been confirmed to exist yet, and is a separate, likely more
  consequential problem from the batch/solver axis tested here.

- **2026-08-21 (cont.) — does restricting the OT pairing to only the highest-rank (purest-
  phenotype) KD cells help?** User's hypothesis: the existing pipeline already selects only
  `rank_type=="top"` cells per class (`data.py::_top_cells`, `n_per_class=1000`), but within
  that top-1000 there's still a range of confidence — pairing against only the very top of that
  range should give a cleaner, less ambiguous target distribution.
  **Checked the premise before testing it:** does rank-restriction disproportionately exclude
  whichever mode is subtler/less confidently different from NTC? Computed self-clustering
  balance directly on kd[:100]/kd[:200]/kd[:500]/kd[:1000] (rank-ordered, row 0 = best rank) for
  both genes — the minority-mode proportion stayed close to constant across every cutoff (KIF11
  ~29-31%, TUBGCP6 ~29-32%), so purification isn't quietly cutting into the subtler mode. Safe
  to test.
  **Result** (`sinkhorn_ablation.py::run_sinkhorn_ablation(kd_top_k=...)` — restricts only the
  KD side used for training/pairing; evaluation centroids always come from the FULL population,
  so the ground truth being tested against never changes), reported as %-points off the true
  A-fraction (KIF11 true A=31.4%, TUBGCP6 true A=68.4%), excluding NTC-stalled cells:

  | gene | coupling | k=100 | k=200 | k=500 | k=1000 (baseline) |
  |---|---|---|---|---|---|
  | KIF11 | ot | 0.7pp | 5.3pp | 5.0pp | 10.2pp |
  | KIF11 | ot_sinkhorn | 8.0pp | 8.3pp | 7.5pp | 8.8pp |
  | TUBGCP6 | ot | 7.9pp | 3.8pp | 5.2pp | 3.9pp |
  | TUBGCP6 | ot_sinkhorn | 11.8pp | 4.3pp | 10.7pp | 10.0pp |

  **A real effect, but not a general rule.** KIF11 + exact-Hungarian ('ot') improves sharply and
  close to monotonically as k shrinks — 10.2pp error at k=1000 down to 0.7pp at k=100,
  essentially nailing the true split. That doesn't generalize: TUBGCP6 does NOT improve
  monotonically with purity (best around k=200, worse again at the most extreme k=100 for BOTH
  couplings), and `ot_sinkhorn` barely benefits from purification on either gene. One consistent
  side benefit regardless of the balance result: NTC-stall drops as TUBGCP6's target purifies
  (6.9%→2.8%→1.8%→2.5% from k=1000→100) — purification does reduce that specific failure mode
  even where it doesn't fix balance.
  **Verdict:** purity helps, but there's a per-gene/per-coupling sweet spot rather than "more
  purity is always better" — a real, usable lever (especially for exact-Hungarian coupling),
  not a universal fix. Combined with the batch/solver result above, three semi-independent
  levers (coupling solver, batch size, target purity) each move the outcome a little, in
  gene-dependent and sometimes opposite directions, with no single lever closing the gap to
  mean_diff's simplicity/determinism on its own.

- **2026-08-21 (cont.) — two more ideas, one clean null (with a provable reason) and one clean
  win.** User asked for other directions beyond levers 1-3 above; picked rectified-flow reflow
  and a mixture-of-flows design (real single-cell trajectory tracking, the third idea, stays
  out of scope — a separate workstream, not a pipeline tweak).
    - **Reflow (`flow.py::reflow_train`, Liu et al. 2209.03003): retrain a fresh FlowNet on an
      already-trained field's own (x0, endpoint) pairs — exact, non-crossing by construction,
      which provably straightens the ODE.** Tested before-vs-after on t=1 balance recovery,
      same nearest-centroid methodology, both genes/couplings: **essentially no change**
      (TUBGCP6/ot: 67.3/25.8 → 67.3/25.8, bit-for-bit identical; every other cell ≤1.1pp).
      Not a bug — reflow is *marginal-preserving by design* (that's the point: straighten the
      path, keep the same t=1 distribution), so testing it at exactly t=1 balance was always
      going to show zero effect. The place reflow could actually matter is the α>1
      overshoot/NTC-stall problem (a straighter path should extrapolate more predictably past
      its training range than a curved one) — untested here, and the correctly-targeted retest
      if this is worth revisiting.
    - **Mixture of flows: split the real KD population into its two discovered sub-clusters
      FIRST, train ONE separate flow per sub-cluster** (`mixture_flow_test.py` — NTC→cluster-A-
      only and NTC→cluster-B-only, each a clean unimodal sub-problem by construction) **— a
      clean win.** Each sub-flow reliably routes the full real NTC population toward its OWN
      designated mode:

      | gene | coupling | net_A → A | net_B → B |
      |---|---|---|---|
      | KIF11 | ot | 96.0% | 99.7% |
      | KIF11 | ot_sinkhorn | 98.4% | 96.6% |
      | TUBGCP6 | ot | 90.6% | 99.6% |
      | TUBGCP6 | ot_sinkhorn | 92.4% | 98.4% |

      90-99.7% purity across every combination — far more reliable than any unified-flow result
      today. Confirms the premise from every result above: OT-CFM is already good at clean,
      unimodal targets (matches mean_diff tying/winning there too); the problem all day was
      always the multimodal *routing* decision, never the transport itself. **Design implication
      for the actual interpretability deliverable:** don't ask one field to guess which mode an
      NTC cell belongs to (shown twice today — nearest-centroid audit + this — not reliably
      answerable from static data). Instead, for a flagged bimodal target, decode BOTH
      "traverse toward mode A" and "toward mode B" as two explicit, separately-trained,
      high-purity counterfactuals, and let the two decoded outcomes speak for themselves rather
      than forcing a single predicted route.

- **2026-08-21 (cont.) — user's hypothesis: does OT do better on genuinely MORE complex
  phenotypes, not just the clean-bimodal cases tested so far?** Correct instinct, and it
  surfaced a real problem with every gene tested today: `bimodality_scan.py`'s original
  selection (fixed KMeans k=2 separation) can only ever find genes that already fit a clean
  2-cluster model — exactly the easy regime a 2-vector piecewise mean_diff already handles, and
  the WRONG regime to look for OT's advantage in.
  **Built `complexity_scan.py`: proper model selection (PCA→20 dims, Gaussian-mixture BIC,
  k=1..5) across all 1000 genes** (caches already existed for all of them from the original
  buildout), replacing the fixed-k=2 assumption. Result: only 83/1000 genes are genuinely
  clean-bimodal (best_k=2) by BIC; 616 are unimodal; **301 (30%) are best fit by 3-5 real
  modes.** Bigger finding: **TUBGCP6 and KIF11 — today's own test genes — are themselves
  best_k=5 and best_k=3, not 2.** Every mixture-of-flows/per-cluster-mean_diff result earlier
  today forced an artificial 2-way split onto genes with more real structure than that, which
  may explain some of the day's noisiness in both methods (cells belonging to an unmodeled 3rd
  mode had to be arbitrarily absorbed into whichever of the 2 forced clusters was nearest).
  **Reran mixture-of-flows vs. per-cluster mean_diff at each gene's TRUE best_k**
  (`mixture_flow_multik.py`; clusters assigned via the same GMM used for best_k, not KMeans),
  on TUBGCP6 (k=5), KIF11 (k=3), and SSX2IP (k=3, sizes 197/434/369 — picked fresh and
  unbiased directly from the complexity scan, never touched before today). Own-cluster routing
  accuracy (does each method's push for cluster i actually land nearest cluster i, not a
  neighboring mode or NTC):

  | gene | cluster (size) | mean_diff | OT | OT margin |
  |---|---|---|---|---|
  | KIF11 | 0 (n=90) | 100.0% | 100.0% | tie |
  | KIF11 | 1 (n=539) | 84.4% | 86.2% | +1.8pp |
  | KIF11 | 2 (n=371) | 89.5% | 99.3% | +9.8pp |
  | SSX2IP | 0 (n=197) | 100.0% | 99.9% | tie |
  | SSX2IP | 1 (n=434) | 78.1% | 96.1% | +18.0pp |
  | SSX2IP | 2 (n=369) | 70.7% | 85.7% | +15.0pp |
  | TUBGCP6 | 0 (n=270) | 57.5% | 91.7% | +34.2pp |
  | TUBGCP6 | 1 (n=99) | 100.0% | 100.0% | tie |
  | TUBGCP6 | 2 (n=206) | 81.1% | 93.8% | +12.7pp |
  | TUBGCP6 | 3 (n=158) | 98.5% | 95.4% | -3.1pp |
  | TUBGCP6 | 4 (n=267) | 73.5% | 92.0% | +18.5pp |

  **The clearest, most decisive result of the day: OT wins clearly in 6/11 clusters (average
  +13.4pp margin among non-tie clusters, up to +34.2pp), ties in 3 (the smallest/most-distinct
  clusters — both methods trivially hit ~100%), and mean_diff wins exactly once, by 3.1pp.**
  SSX2IP — the one gene selected with zero prior bias, straight from this scan — shows OT ahead
  on every non-trivial cluster. **Verdict: the hypothesis was right, and the mechanism makes
  sense** — mean_diff's per-cluster vector is a rigid straight-line push from the NTC centroid,
  fine when a target cluster is small/distinct/easy (where both methods already tie at ~100%),
  but once there are several real, less-separated sub-populations pulling in different
  directions, a fixed linear push increasingly can't discriminate "close enough to my target"
  from "actually closer to a neighboring mode." A learned nonlinear field handles that
  non-convex geometry better. This is the opposite regime from every earlier test today (few,
  cleanly-separated modes) — which is exactly where mean_diff was shown to already excel, so
  the two results aren't in tension, they're describing different regimes. **Net updated
  picture for Workstream A: mean_diff remains the right default for unimodal and cleanly
  bimodal targets (861/1000 genes by this scan); OT-CFM (via mixture-of-flows on GMM-discovered
  sub-clusters) is the better choice specifically for the 301 genuinely-multi-modal (k>=3)
  targets, with a real, sizeable, reproducible-across-3-genes advantage there.** Not yet tested:
  whether this holds up through actual DiffAE decode (everything in this entry is raw
  pre-decode embedding space, and today's earlier decode-validation work showed that step can
  reverse conclusions — see the pool_shape_recovery correction above), and only 3 of the 301
  candidate genes have been checked.

- **2026-08-21 (cont.) — decode-validated on 10 more genes: the multi-K result survives real
  image generation.** Ran the full pipeline (mean_diff vector + trained OT flow per GMM-
  discovered cluster, decode every pushed cell through the frozen DiffAE, re-embed via
  CellDINO, classify against the real (K+1)-way centroids — `mixture_decode_validation.py`,
  n_pool=20/cluster) on the next 10 highest-bic_gain candidates from the complexity scan (an
  unbiased continuation of the same ranked list, not hand-picked): HAUS6(k=4), EFR3A(k=4),
  RNF11(k=5), ZNF131(k=4), CYP4V2(k=5), B4GALT3(k=5), CENPH(k=3), SACM1L(k=5), ASCC3(k=5),
  FLII(k=5) — 45 clusters total.

  **OT wins 31/45 clusters, mean_diff wins 10/45, ties 4/45. Mean margin (OT − mean_diff) =
  +7.3pp, median +10pp.** Combined with the earlier 3-gene raw-embedding result (TUBGCP6/
  KIF11/SSX2IP: 6 OT wins, 1 mean_diff win, 3 ties, avg +13.4pp), that's 13 genes / 56 clusters
  total, same direction, now confirmed through actual generation — not a raw-embedding
  artifact (the exact failure mode the pool_shape_recovery correction earlier today warned
  about). Two caveats on the absolute percentages (not on the comparison itself, which stays
  fair since both methods hit the same confound equally): (1) decode noise ALONE (zero
  intended shift) already lands 45-55% of untouched control cells nearest a gene's largest
  cluster in a few cases (CYP4V2 cluster_1: 55%; CENPH cluster_2: 50%; ASCC3 cluster_3: 45%) —
  those clusters' raw purity numbers are inflated by proximity to the decode-noise floor, not
  real signal, for BOTH methods equally; (2) n_pool=20/cluster gives only 5%-point resolution,
  so any single cluster's margin carries real sampling noise — the trustworthy signal is the
  aggregate over all 45, not any one comparison.
  Checked whether mean_diff's wins cluster on the largest sub-population within a gene (a
  plausible mechanism: OT's nonlinear advantage might matter less where there's already dense
  natural support for a straight-line shift) — weak effect only (corr(cluster size, OT margin)
  = −0.20), not strong enough to call a rule.
  **This closes the loop the corrected pool_shape_recovery entry left open. Updated,
  decode-validated verdict for Workstream A: mean_diff remains the right default for unimodal
  and cleanly-bimodal targets (699/1000 genes by the complexity scan); OT-CFM via mixture-of-
  flows on GMM-discovered sub-clusters is the better choice for genuinely multi-modal (k>=3)
  targets (301/1000 genes), with a real, moderate, now decode-confirmed advantage (~13 of 56
  tested clusters go the other way, so not universal within that regime either — but the
  aggregate direction is clear and reproduced across 13 genes).** Remaining gaps: only 13/301
  candidate genes checked at this point; complexes (98) never scanned for multi-modality at
  all; the α>1 extrapolation/dose problem (literature-review finding #1) is completely separate
  and still unaddressed.

- **2026-08-21 (cont.) — user: "no point being stingy," so scaled both remaining gaps: full
  decode-validated run on all 301 geneKO candidates (launched, running), and the complexity
  scan extended to all 98 EBI complexes (`complexity_scan.py`/`mixture_decode_validation.py`
  generalized to take a `grain` parameter instead of a hardcoded geneKO path — trivial change,
  every complex already had a full flow-field cache from the original buildout).
  **Complex-grain complexity scan result (all 98, complete): 92/98 (94%) are unimodal, 4 are
  clean bimodal (k=2), and only 2 — Kinetochore_CCAN_complex (k=3, sizes 305/28/167) and
  Signal_peptidase_complex__SEC11A_variant (k=3, sizes 279/112/109) — show genuine multi-modal
  structure.** A real, informative result on its own, not just a smaller version of the geneKO
  scan: pooling cells across a complex's member genes apparently washes multimodality OUT
  rather than revealing it (94% unimodal vs. 61.6% for individual geneKOs), consistent with
  Workstream D's earlier finding that complex-level phenotypes tend to read as a coherent
  consensus signature rather than a mixture of member-gene-specific outcomes.

- **2026-08-21 (cont.) — FINAL, full-scale, decode-validated result, and it overturns the
  "OT wins specifically on complex multimodal targets" framing from earlier in the day.** Ran
  three complete decode-validated batches (all `mixture_decode_validation.py`, n_pool=20/
  cluster, generalized to take a `grain` param — trivial change, every target already had a
  cached embedding from the original buildout):
  1. All 301 genuinely multi-modal (best_k>=3) genes — 1107 clusters. 301/301 jobs completed
     clean, ~110 min wall-clock on 8 concurrent GPU nodes.
  2. The 2 multi-modal (best_k>=3) EBI complexes — 6 clusters.
  3. **The missing control, per the user's own prompt ("why not test bimodal as well?"): all
     87 genuinely CLEAN bimodal (best_k==2) genes+complexes** — 174 clusters. This is the
     rigorous version of the bimodal test; the earlier same-day bimodal work (KIF11/TUBGCP6/
     SEC61A1/S100B) is superseded by this, since KIF11 and TUBGCP6 turned out to be best_k=3/5
     all along, contaminating that earlier "bimodal" sample.

  **Combined: 390 targets, 1287 clusters. OT wins 76.4% (983), mean_diff wins 11.0% (141),
  ties 12.7% (163).** The bimodal control result (77.0% OT win rate, +10.55pp mean margin) is
  statistically indistinguishable from the k=3/4/5 result (76.3%, +11.30pp) — k=2: +10.55pp,
  k=3: +10.85pp, k=4: +12.19pp, k=5: +10.86pp. **The win rate does not depend on how many real
  modes a target has, at all.**
  **Corrected mechanism:** the driver isn't "multimodality" — it's decomposition itself. Once
  a target is split into per-cluster sub-problems (whether 2 pieces or 5), each sub-target is
  a narrower, tighter-covariance sub-population than the full undifferentiated KD population.
  mean_diff's per-cluster vector is still just a rigid TRANSLATION (same shift applied to
  every cell) — it can recenter a population but can never reshape/contract it to match a
  tighter target's own covariance. A learned flow can. That also resolves the apparent tension
  with earlier same-day results: the ORIGINAL (non-decomposed) comparisons — one mean_diff
  vector vs. one unified OT field, both targeting the FULL undifferentiated KD population —
  were much closer to a tie (SEC61A1 favoring mean_diff, KIF11 near-even), because there the
  target's spread roughly matches the source's. The decomposed, per-cluster comparison is a
  fundamentally different, and apparently much more favorable, regime for OT — not because of
  mode count, but because decomposition itself creates narrow targets translation can't fit.
  **Final verdict for Workstream A, empirically settled at scale (390 targets, 1287 clusters,
  decode-validated, not raw-embedding-only):** OT-CFM via mixture-of-flows on GMM-discovered
  sub-clusters beats per-cluster mean_diff by a consistent, real, moderate margin (~+11pp,
  ~76% cluster win rate) whenever a target is DECOMPOSED into 2 or more sub-populations —
  independent of how many real modes exist. mean_diff remains the right, simpler,
  training-free choice only when a target is treated as ONE undifferentiated population
  (unimodal genes, or any target where decomposition isn't done at all). Still open: the α>1
  extrapolation/dose problem (literature review finding #1, completely separate, unaddressed);
  whether this generalizes beyond the phase-channel geneKO/complex screens tested here.

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
