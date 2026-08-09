// "How it works" tab — a slide deck of the ML methods behind the viewer, for a general audience but rigorous.
// Each slide: an animated SVG, a short body, a "why it matters" line, explicit paper links, and a "Key terms" glossary.
// Animations are CSS keyframes (style.css .mth-*) driven by group transforms + opacity so they loop while shown.

const MTH_C = { acc: "#26c6ff", ntc: "#8b949e", ko: "#f0a020", grn: "#3fb950", pur: "#bc8cff", yel: "#d29922", fg: "#e6e8ec" };

// ---- SVG builders ----
const _cellBlob = (cx, cy, r, fill, cls = "") =>
  `<circle cx="${cx}" cy="${cy}" r="${r}" fill="${fill}" class="${cls}"/>` +
  `<circle cx="${cx - r * 0.3}" cy="${cy - r * 0.2}" r="${r * 0.28}" fill="rgba(0,0,0,.35)"/>` +
  `<circle cx="${cx + r * 0.35}" cy="${cy + r * 0.25}" r="${r * 0.16}" fill="rgba(255,255,255,.35)"/>`;
const _bars = (x, y, vals, w, gap, col, cls = "mth-rise") => vals.map((v, i) =>
  `<rect x="${x + i * (w + gap)}" y="${y - v}" width="${w}" height="${v}" rx="1.5" fill="${col}" class="${cls}" style="animation-delay:${i * 0.12}s"/>`).join("");
const _arrow = (x1, x2, y, col = "#26c6ff") =>
  `<line x1="${x1}" y1="${y}" x2="${x2 - 8}" y2="${y}" stroke="${col}" stroke-width="2.5"/>` +
  `<path d="M${x2 - 10},${y - 6} L${x2},${y} L${x2 - 10},${y + 6} Z" fill="${col}"/>`;
const _box = (x, y, w, h, l1, l2 = "") =>
  `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="8" fill="rgba(38,198,255,.1)" stroke="${MTH_C.acc}"/>` +
  `<text x="${x + w / 2}" y="${y + h / 2 + (l2 ? -2 : 4)}" fill="${MTH_C.fg}" font-size="11" text-anchor="middle">${l1}</text>` +
  (l2 ? `<text x="${x + w / 2}" y="${y + h / 2 + 12}" fill="#8b949e" font-size="9" text-anchor="middle">${l2}</text>` : "");
const _lbl = (x, y, t, col = "#8b949e", sz = 11) => `<text x="${x}" y="${y}" fill="${col}" font-size="${sz}" text-anchor="middle">${t}</text>`;
const _patchGrid = (x, y, s, n, hot) => Array.from({ length: n * n }, (_, i) => {
  const gx = x + (i % n) * s, gy = y + Math.floor(i / n) * s, on = hot.includes(i);
  return `<rect x="${gx}" y="${gy}" width="${s - 1}" height="${s - 1}" fill="${on ? MTH_C.ko : "rgba(255,255,255,.04)"}" stroke="rgba(255,255,255,.08)" ${on ? 'class="mth-glow" style="animation-delay:' + (i % 5) * 0.2 + 's"' : ""}/>`;
}).join("");

const MTH_REFS = {
  "The screen": [["Optical pooled screens · Feldman 2019", "https://doi.org/10.1016/j.cell.2019.09.016"]],
  "Fingerprint": [["DINO · Caron 2021", "https://arxiv.org/abs/2104.14294"], ["DINOv2 · Oquab 2023", "https://arxiv.org/abs/2304.07193"]],
  "Classifier": [["Set Transformer · Lee 2019", "https://arxiv.org/abs/1810.00825"], ["Attention-based multiple-instance learning · Ilse 2018", "https://arxiv.org/abs/1802.04712"]],
  "Top cells": [["Explaining by removing · Covert 2021", "https://arxiv.org/abs/2011.14878"], ["SHAP · Lundberg & Lee 2017", "https://arxiv.org/abs/1705.07874"]],
  "Diffusion": [["DDPM · Ho 2020", "https://arxiv.org/abs/2006.11239"], ["Diffusion autoencoders · Preechakul 2022", "https://arxiv.org/abs/2111.15640"]],
  "Traversal": [["Diffusion autoencoders · Preechakul 2022", "https://arxiv.org/abs/2111.15640"], ["Classifier-free guidance · Ho & Salimans 2022", "https://arxiv.org/abs/2207.12598"]],
  "DDIM": [["DDIM · Song 2020", "https://arxiv.org/abs/2010.02502"]],
  "Attention heads": [["DINO attention · Caron 2021", "https://arxiv.org/abs/2104.14294"], ["Attention is all you need · Vaswani 2017", "https://arxiv.org/abs/1706.03762"]],
  "Embedding": [["UMAP · McInnes 2018", "https://arxiv.org/abs/1802.03426"], ["PHATE · Moon 2019", "https://doi.org/10.1038/s41587-019-0336-3"]],
  "Virtual staining": [["In silico labeling · Christiansen 2018", "https://doi.org/10.1016/j.cell.2018.03.040"], ["Diffusion autoencoders · Preechakul 2022", "https://arxiv.org/abs/2111.15640"]],
  "What's next": [["CROP-seq · Datlinger 2017", "https://doi.org/10.1038/nmeth.4177"], ["Perturbation autoencoder (CPA) · Lotfollahi 2023", "https://doi.org/10.15252/msb.202211517"]],
};

// longer "Learn more" paragraph per slide (keyed by nav) — unpacks the concept for readers who want depth
const MTH_MORE = {
  "The screen": "A traditional CRISPR screen gives one number per gene — did cells grow, did a reporter switch on. An optical pooled screen instead keeps the cells in place and photographs them, so the readout is the cell's full appearance. The trick: each cell also carries a short DNA barcode unique to its CRISPR guide, and a round of in-situ sequencing lights that barcode up letter-by-letter right in the microscope — so we can match every cell's image to the exact gene it lost. One imaging run yields millions of (gene, picture) pairs. The hard part is what comes next: those cells are a mix of thousands of perturbations, each present in many copies that vary enormously for reasons unrelated to the knockout — cell cycle, size, local density, position on the plate — and most knockouts nudge the phenotype only slightly. So the central problem is signal-in-noise: for every perturbation, which cells and which features actually capture its distinctive effect, rather than the technique's inherent variability? The classifier, rankings, and generative traversals in this viewer are complementary tools for answering exactly that.",
  "Fingerprint": "A raw microscope image is hundreds of thousands of pixels — too many, and too noisy, to compare directly. We need a compact summary that keeps what's biologically meaningful (shape, texture, organelle layout) and drops what isn't (exact position, lighting). CellDINO is a vision transformer trained by self-supervision: shown only images, never gene labels, it learns embeddings where two crops of the same cell agree and different cells differ. The result is a 1,024-number fingerprint per cell whose distances track real morphological similarity — which is what lets us cluster phenotypes, rank genes, and steer the generative model later.",
  "Classifier": "A single knocked-out cell is often ambiguous — cells vary a lot even with no perturbation — so the signal lives in the distribution of a gene's cells. We therefore classify a whole bag at once (multiple-instance learning). The SetTransformer tags each cell's fingerprint with its imaging channel, uses attention so the cells in a bag can inform one another, then pools them into a single bag-vector with attention (PMA) rather than a plain average, letting it weight the informative cells. Because attention and pooling ignore order, shuffling the bag can't change the answer; because it trained across bag sizes it can score anywhere from 10 to thousands of cells. The output is a probability over 1,000 genes (+NTC), or over 99 protein complexes when we ask about pathways instead of single genes.",
  "Top cells": "Once the classifier works, we ask which individual cells it actually relies on, borrowing an idea from model explanation called \"explaining by removing\": a cell is important if taking it out of a bag makes the classifier less sure of the right answer. Concretely we take the probability of the correct class with the cell in the bag minus the probability without it, and average that marginal contribution over many random bags and bag sizes (a Monte-Carlo estimate). The highest-scoring cells are the clearest examples of a perturbation's phenotype — these top-predictive cells are exactly what the Top Cells tab shows and what the diffusion traversals are anchored to. Up-weighting them also sharpens the gene-level distinctiveness score (mAP).",
  "Diffusion": "A diffusion model learns to reverse a corruption process. In training we take a real cell and add a little Gaussian noise over and over until it's indistinguishable from static, and a network learns to look at a noisy image and predict the noise that was added. To generate, we start from pure static and repeatedly subtract the predicted noise until a realistic cell condenses out. In a diffusion autoencoder we split a cell's description in two: a compact semantic code — the cell's identity, its shape/texture/phenotype, the \"what\" — and a noise seed holding the leftover pixel-level randomness. Keeping them separate is the key: fix the noise and change the identity code, and you change what the cell is without disturbing its incidental details.",
  "Traversal": "The semantic code lives in a space where nearby points are similar-looking cells and directions correspond to consistent visual changes. To build a knockout's \"movie\" we average the codes of control (NTC) cells and of that knockout's cells; the vector between them is the direction that turns control-looking into knockout-looking. Starting from one cell's code we step along it — α = 0 is the cell itself, α = 1 applies the full average shift, larger α exaggerates subtle effects — decoding an image at each step. Because the noise seed is held fixed throughout, only the phenotype moves: you're watching the same cell change, not a slideshow of different cells.",
  "DDIM": "By default the noise seed is random, so decoding a code gives a representative cell, not any particular real one — and a traversal from a generic cell is hard to trust. DDIM fixes this two ways. First it makes generation deterministic: the same seed always yields the same image (a diffusion \"ODE\", not a random walk). Second, being deterministic, it can run in reverse — starting from a real cell's pixels and integrating backwards to recover the exact noise seed that regenerates it. We invert under the same guidance used to generate (\"guided inversion\"), so decoding at α = 0 reproduces the original cell almost perfectly (pixel correlation ≈ 0.99). Every traversal here therefore begins anchored to a genuine cell, and the changes you see are real counterfactuals, not artifacts of a random start.",
  "Attention heads": "The vision transformer doesn't read pixels one at a time — it breaks the image into a grid of patches and, in each attention \"head\", decides how much every patch should influence its summary of the cell. Reading those weights back out gives a heat-map showing where the model concentrated. Different heads specialize on different structures, so we rank heads by how well their maps separate a knockout from control and surface the informative ones. The payoff is interpretability: instead of only telling you that a gene has a distinctive phenotype, the map shows you where — e.g. a head that consistently lights up mitochondria for a mitochondrial gene — which you can check against the biology.",
  "Embedding": "A single traversal shows one gene's effect; the Embedding tab shows all of them at once — and controls for cell-to-cell variation by using the same anchor cell throughout. We morph that one cell toward each of the ~1,000 knockouts, giving 1,000 counterfactual images of the same starting cell. Each image is then positioned by its gene's coordinates on a UMAP or PHATE embedding of the gene-level phenotype space, so knockouts that produce similar morphologies land near one another. LatentLens stitches the crops into a continuous, zoomable atlas — pan and zoom to compare neighborhoods, spot phenotype clusters, and see where a gene of interest falls relative to the whole library.",
  "Virtual staining": "Fluorescent markers reveal specific structures but cost extra dyes, channels, and imaging, and you can only stain a few at once. Label-free phase imaging is cheap and gentle but hard to read. Virtual staining bridges them: we reuse the diffusion autoencoder, now conditioned on a phase image, and train it to output the matching fluorescent-marker image. The conditioning is spatial — the phase image guides generation pixel-by-pixel, so predicted structures line up with the real ones instead of being a generic \"mitochondria-style\" texture. Applied to a traversal, one phase morph becomes a full multi-channel readout: all 42 markers, for every perturbation, on the same anchored cell — a multiplexed phenotype from a single grayscale image.",
  "What's next": "So far every direction has been morphological — defined in the image-fingerprint space. But the same knockout library was also measured by CROP-seq, which reads each cell's transcriptome (its gene-expression profile) instead of its picture. That opens a new axis of control: define the traversal direction from the transcriptional change a knockout causes, and let the diffusion model render how the cell's appearance should follow. Comparing the transcriptome-driven morph with the morphology-driven one reveals which genes couple gene-expression change to visible phenotype and which decouple them — a gene that reshapes the transcriptome but barely changes the image, or vice versa. That coupling is the central question of the transcriptional project this viewer is being extended toward.",
};

const METHODS_SLIDES = [
  {
    nav: "The screen", kicker: "THE QUESTION", title: "Thousands of knockouts, mixed in one noisy dish",
    svg: () => `<svg viewBox="0 0 420 200" class="mth">
      <g>${Array.from({ length: 24 }, (_, i) => { const x = 40 + (i % 8) * 42, y = 45 + Math.floor(i / 8) * 52;
        const c = [MTH_C.ntc, MTH_C.acc, MTH_C.ko, MTH_C.grn, MTH_C.pur][i % 5];
        return `<g class="mth-pulse" style="animation-delay:${(i % 8) * 0.15}s">${_cellBlob(x, y, 13, c)}
          <rect x="${x - 10}" y="${y + 16}" width="20" height="4" fill="${MTH_C.acc}" opacity=".7"/></g>`; }).join("")}</g>
      <circle cx="166" cy="97" r="26" fill="none" stroke="#fff" stroke-width="2.5" class="mth-pulse"/>
    </svg>`,
    body: "In a <b>pooled optical CRISPR screen</b>, thousands of gene knockouts are mixed in one dish and imaged together; each cell's DNA <b>barcode</b>, sequenced in place, names the gene knocked out inside it. This yields millions of (gene, image) pairs — but each knockout's real effect is subtle and buried in enormous cell-to-cell variation.",
    why: "The core question this whole viewer answers: for each perturbation, <b>which change in the cell captures its true phenotype</b> — separated from the noise of the technique's scale and heterogeneity? Everything that follows is one answer.",
    defs: [["Pooled optical screen", "imaging a mixed population where every cell has a different gene knocked out, all together."],
      ["Barcode", "a short DNA tag, read out in situ, that identifies which CRISPR guide (gene) is in each cell."],
      ["Perturbation", "the genetic change applied to a cell — here a CRISPR knockout; NTC = non-targeting control (no gene cut)."]]
  },
  {
    nav: "Fingerprint", kicker: "REPRESENT", title: "Turning a cell image into numbers",
    svg: () => `<svg viewBox="0 0 420 200" class="mth">
      <g class="mth-breathe">${_cellBlob(85, 100, 46, MTH_C.acc)}
        ${_patchGrid(50, 65, 14, 5, [])}</g>
      ${_arrow(150, 232, 100)}
      <g>${_bars(252, 150, [70, 30, 95, 45, 60, 20, 80, 40], 15, 6, MTH_C.pur)}</g>
      ${_lbl(305, 172, "embedding vector")}
    </svg>`,
    body: "A <b>self-supervised</b> vision transformer (<b>CellDINO</b>) splits each image into patches and, learning from images alone with no labels, distils it into a numeric <b>embedding</b> — a point in a high-dimensional \"morphology space\".",
    why: "Similar-looking cells land nearby, so morphology becomes math we can compare, cluster, and rank.",
    defs: [["Embedding", "a list of numbers (a vector) that summarizes an image."],
      ["Self-supervised", "the model learns structure from raw images with no human labels."],
      ["Vision transformer (ViT)", "a network that cuts an image into patches and relates them with attention."],
      ["CellDINO", "our DINO-style self-supervised ViT trained on these cell images."]]
  },
  {
    nav: "Classifier", kicker: "THE MODEL", title: "A classifier that reads a whole group of cells",
    svg: () => { const X = [50, 160, 264, 372], L0 = [42, 80, 118, 156], LH = [60, 99, 138], L3 = [42, 80, 118, 156];
      const edges = (xa, ya, xb, yb) => ya.map((a, i) => yb.map((b, j) => `<line x1="${xa}" y1="${a}" x2="${xb}" y2="${b}" stroke="${MTH_C.acc}" stroke-width=".8" opacity=".45" class="mth-pulse" style="animation-delay:${((i + j) % 5) * 0.25}s"/>`).join("")).join("");
      return `<svg viewBox="0 0 440 200" class="mth">
        ${edges(X[0], L0, X[1], LH)}${edges(X[1], LH, X[2], LH)}${edges(X[2], LH, X[3], L3)}
        ${L0.map((y, i) => _cellBlob(X[0], y, 9, [MTH_C.grn, MTH_C.acc, MTH_C.pur, MTH_C.ko][i])).join("")}
        ${LH.map(y => `<circle cx="${X[1]}" cy="${y}" r="7" fill="rgba(38,198,255,.2)" stroke="${MTH_C.acc}"/>`).join("")}
        ${LH.map(y => `<circle cx="${X[2]}" cy="${y}" r="7" fill="rgba(38,198,255,.2)" stroke="${MTH_C.acc}"/>`).join("")}
        ${L3.map((y, i) => `<circle cx="${X[3]}" cy="${y}" r="8" fill="${i === 1 ? MTH_C.ko : "rgba(255,255,255,.07)"}" stroke="${i === 1 ? MTH_C.ko : "#30363d"}" ${i === 1 ? 'class="mth-pulse"' : ""}/>`).join("")}
        ${_lbl(50, 184, "set of cells")}${_lbl(160, 168, "attention")}${_lbl(264, 168, "pool (PMA)")}${_lbl(398, 84, "class", MTH_C.ko, 11)}
      </svg>`; },
    body: "The <b>SetTransformer</b> is a <b>multiple-instance-learning</b> classifier: it reads a <i>bag</i> of a perturbation's cells and predicts the class (1,000 genes + NTC, or 99 protein complexes). Each cell's 1,024-d CellDINO embedding is tagged with a <b>channel embedding</b>, projected to 512-d, then passed through two <b>inducing-point attention layers</b> (ISAB, 32 inducing points, 4 heads); a <b>PMA</b> layer pools the bag into one vector and a cosine classifier scores every class.",
    why: "Trained on random bags of 100 cells yet able to score any bag size (10–5,000), it reads the population phenotype and is invariant to how the cells are ordered.",
    defs: [["Multiple-instance learning", "classify a whole bag from one shared label, without labeling individual cells."],
      ["Bag / permutation-invariant", "an unordered set of a perturbation's cells; shuffling them can't change the prediction."],
      ["ISAB (inducing-point attention)", "self-attention routed through 32 learned reference points, so cost grows linearly and scales to tens of thousands of cells."],
      ["Channel embedding", "a learned tag marking each cell's imaging channel (phase, or a given fluorescent marker)."],
      ["PMA + cosine classifier", "pooling-by-attention collapses the bag to one vector; a cosine classifier turns it into class probabilities."]]
  },
  {
    nav: "Top cells", kicker: "EXPLAINING BY REMOVING", title: "Which cells carry the phenotype?",
    svg: () => `<svg viewBox="0 0 440 200" class="mth">
      <rect x="22" y="58" width="92" height="84" rx="10" fill="rgba(255,255,255,.04)" stroke="#30363d"/>
      ${_cellBlob(48, 84, 9, MTH_C.acc)}${_cellBlob(88, 88, 9, MTH_C.grn)}${_cellBlob(58, 120, 9, MTH_C.pur)}
      <g class="mth-pulse">${_cellBlob(96, 120, 11, MTH_C.ko)}</g>
      ${_lbl(68, 158, "bag + cell x")}
      <rect x="146" y="66" width="18" height="64" rx="2" fill="${MTH_C.acc}"/>
      <rect x="178" y="92" width="18" height="38" rx="2" fill="${MTH_C.ntc}"/>
      ${_lbl(155, 146, "with x")}${_lbl(187, 146, "without x")}
      ${_lbl(174, 58, "Δ = score(x)", MTH_C.ko, 10)}
      ${_arrow(210, 250, 98)}
      ${[0, 1, 2, 3, 4].map(i => `<g class="mth-emerge" style="animation-delay:${i * 0.15}s">${_cellBlob(282 + i * 32, 98, 13, i < 2 ? MTH_C.ko : MTH_C.acc)}</g>`).join("")}
      ${_lbl(346, 140, "top-predictive cells")}
    </svg>`,
    body: "To find a perturbation's most telling cells, we score each cell by how much it <b>helps the classifier</b>: the drop in predicted probability when the cell is <b>removed</b> from a bag (\"explaining by removing\"), averaged over many bag sizes and random partners. The top-scoring <b>top-predictive cells</b> carry the phenotypic signature — the cells the viewer anchors its traversals to and shows in Top Cells. Re-weighting the gene-level <b>mAP</b> by these cells sharpens the distinctiveness ranking.",
    why: "It picks, per perturbation, the handful of cells that most define its phenotype — the exemplars every traversal starts from.",
    defs: [["Explaining by removing", "gauge a cell's importance by how much the prediction drops when you take it out of the bag."],
      ["Marginal contribution", "score(x) = p(class | bag with x) − p(class | bag without x), averaged over many bags (sizes 1–500)."],
      ["Top-predictive cells", "the highest-scoring cells for a class; used as traversal anchors and in the Top Cells tab."],
      ["Distinctiveness (mAP)", "the gene-level separability score (Fig. 2), recomputed with these top cells up-weighted."]]
  },
  {
    nav: "Diffusion", kicker: "SIMULATE", title: "Building a cell by removing noise",
    svg: () => `<svg viewBox="0 0 420 200" class="mth">
      ${[0, 1, 2, 3, 4].map(i => { const x = 34 + i * 76, op = i / 4;
        const noise = Array.from({ length: (4 - i) * 6 + 2 }, (_, j) => `<circle cx="${x + 8 + (j * 13) % 44}" cy="${63 + (j * 17) % 44}" r="1.6" fill="#8b949e" opacity="${1 - op * 0.85}"/>`).join("");
        return `<rect x="${x}" y="55" width="60" height="60" rx="6" fill="rgba(0,0,0,.25)" stroke="#30363d"/>${noise}<g opacity="${op}">${_cellBlob(x + 30, 85, 19, MTH_C.acc)}</g>`; }).join("")}
      <rect x="34" y="55" width="60" height="60" rx="6" fill="none" stroke="#fff" stroke-width="2" class="mth-sweep"/>
      ${_arrow(360, 396, 30, MTH_C.acc)}${_lbl(210, 26, "reverse process — denoise →", MTH_C.acc, 11)}
      ${_lbl(64, 132, "x_T (pure noise)")}${_lbl(364, 132, "x_0 (a cell)")}
    </svg>`,
    body: "A <b>diffusion model</b> is trained by adding noise to real cells step by step until nothing remains (x_T), then learning to <b>reverse</b> it — predicting and removing the noise at each step until a realistic cell (x_0) emerges from pure static.",
    why: "Once a model can <i>paint</i> cells from noise, we can generate counterfactuals — not just read cells, but ask what-if.",
    defs: [["Diffusion model", "turns random noise into a realistic image by removing a little noise at a time."],
      ["Forward process", "gradually corrupts a real image with Gaussian noise until it is pure noise (x_T)."],
      ["Reverse process", "a network (ε-theta) predicts the noise and subtracts it, step by step, to recover an image (x_0)."],
      ["Semantic code (\"identity\")", "a compact vector capturing <i>what a cell looks like</i> — its shape, texture, phenotype — kept separate from the random noise seed. \"Semantic\" just means it carries meaning, not raw pixels."]]
  },
  {
    nav: "Traversal", kicker: "COUNTERFACTUAL", title: "\"What if this gene were knocked out?\"",
    svg: () => `<svg viewBox="0 0 420 200" class="mth">
      <g class="mth-morph">${_cellBlob(210, 86, 42, MTH_C.acc)}</g>
      <line x1="70" y1="158" x2="350" y2="158" stroke="#30363d" stroke-width="4"/>
      <circle cx="70" cy="158" r="5" fill="${MTH_C.ntc}"/><circle cx="350" cy="158" r="5" fill="${MTH_C.ko}"/>
      <circle cx="70" cy="158" r="8" fill="#fff" class="mth-slide"/>
      ${_lbl(70, 180, "NTC · α0")}${_lbl(350, 180, "knockout · α+", MTH_C.ko)}
      <rect x="188" y="14" width="44" height="16" rx="4" fill="rgba(188,140,255,.2)" stroke="${MTH_C.pur}"/>${_lbl(210, 26, "code z", MTH_C.pur, 10)}
    </svg>`,
    body: "We take the cell's <b>semantic code</b> and slide it along the <b>NTC → knockout direction</b> (the average difference between control and knockout codes), decoding each step. α = 0 is the start; α = 1 applies the full knockout shift; beyond exaggerates it.",
    why: "It renders the phenotype a perturbation induces as a smooth, watchable transformation of one cell.",
    defs: [["Semantic direction", "in the identity-code space (the \"semantic code\", see Diffusion), the vector pointing from control (NTC) toward a knockout — a mean difference."],
      ["α (alpha)", "how far we push along that direction — 0 = start, 1 = full shift, |α|>1 = extrapolation."],
      ["Counterfactual", "a generated \"what this cell would look like if…\" image — not an observed one."],
      ["Classifier-free guidance (w)", "a strength knob controlling how firmly the code steers the generated image."]]
  },
  {
    nav: "DDIM", kicker: "ANCHOR TO A REAL CELL", title: "DDIM — running the model backwards",
    svg: () => `<svg viewBox="0 0 420 200" class="mth">
      ${_cellBlob(70, 92, 34, MTH_C.grn)}${_lbl(70, 150, "real cell")}
      <g class="mth-collapse">${Array.from({ length: 22 }, (_, i) => `<circle cx="${185 + (i * 37) % 60 - 30}" cy="${92 + (i * 53) % 60 - 30}" r="2" fill="#8b949e"/>`).join("")}</g>
      ${_lbl(210, 150, "its exact seed x_T")}
      ${_cellBlob(350, 92, 34, MTH_C.grn)}${_lbl(350, 150, "same cell (r≈0.99)")}
      <g class="mth-cycA">${_arrow(110, 178, 78, MTH_C.pur)}${_lbl(144, 68, "invert ↩", MTH_C.pur, 10)}</g>
      <g class="mth-cycB">${_arrow(242, 312, 78, MTH_C.acc)}${_lbl(277, 68, "generate →", MTH_C.acc, 10)}</g>
    </svg>`,
    body: "<b>DDIM</b> (Denoising Diffusion Implicit Models) makes generation <b>deterministic</b> — same seed, same cell, every time. Because it's deterministic it can also run <b>backwards</b>: given a real cell it recovers the exact noise seed that regenerates it. So the traversal starts from a true cell (α = 0 reconstructs it, pixel r ≈ 0.99).",
    why: "The morph is anchored to a real cell's identity — its size, texture, and context are preserved while only the phenotype moves.",
    defs: [["DDIM", "Denoising Diffusion Implicit Models — a deterministic way to sample a diffusion model (same training, no randomness at generation)."],
      ["Deterministic", "same input always gives the same output — no dice-rolling — which is what makes it reversible."],
      ["Inversion (encoding)", "running the model backwards to find the exact noise seed of a specific real image."],
      ["Guided inversion", "inverting under the same guidance used for generation, so α = 0 reconstructs the cell faithfully."]]
  },
  {
    nav: "Attention heads", kicker: "INTERPRET", title: "Which pixels drove the decision?",
    svg: () => `<svg viewBox="0 0 420 200" class="mth">
      <g opacity=".5">${_cellBlob(115, 100, 62, MTH_C.acc)}</g>
      ${_patchGrid(70, 55, 15, 6, [8, 9, 15, 16, 21])}
      ${_lbl(115, 190, "head A")}
      <g opacity=".5">${_cellBlob(305, 100, 62, MTH_C.acc)}</g>
      ${_patchGrid(260, 55, 15, 6, [20, 26, 27, 33])}
      ${_lbl(305, 190, "head B")}
    </svg>`,
    body: "The vision transformer splits the cell into <b>patches</b>; each <b>attention head</b> is a spotlight that weights which patches most inform its read of the cell. Rendered as a heat-map, a head reveals <i>which structures</i> the model keys on for a perturbation — e.g. one fixated on mitochondria.",
    why: "It turns a black-box score into a visible, checkable claim: \"the model calls this gene by looking <i>here</i>.\"",
    defs: [["Patch / token", "the small square pieces a vision transformer breaks the image into."],
      ["Attention head", "one of several parallel spotlights that weight which patches matter; different heads specialize."],
      ["Saliency", "which pixels most influence the model's representation of the cell."],
      ["Ranked by distinctiveness", "we surface the heads whose maps best separate a knockout from control."]]
  },
  {
    nav: "Embedding", kicker: "THE MAP", title: "Every knockout, from one cell, on one map",
    svg: () => `<svg viewBox="0 0 440 200" class="mth">
      ${_cellBlob(46, 100, 28, MTH_C.grn, "mth-breathe")}${_lbl(46, 148, "one anchor cell")}
      ${_arrow(84, 250, 100)}
      <rect x="168" y="26" width="256" height="150" rx="10" fill="rgba(255,255,255,.03)" stroke="#30363d"/>
      ${[[205, 58, MTH_C.grn], [230, 50, MTH_C.grn], [216, 78, MTH_C.grn], [248, 66, MTH_C.grn],
        [330, 52, MTH_C.ko], [356, 68, MTH_C.ko], [338, 84, MTH_C.ko], [368, 56, MTH_C.ko],
        [246, 132, MTH_C.pur], [272, 146, MTH_C.pur], [236, 150, MTH_C.pur], [262, 120, MTH_C.pur],
        [352, 134, MTH_C.acc], [380, 148, MTH_C.acc], [366, 120, MTH_C.acc], [342, 152, MTH_C.acc]]
        .map((p, i) => `<g class="mth-emerge" style="animation-delay:${(i % 8) * 0.12}s">${_cellBlob(p[0], p[1], 8, p[2])}</g>`).join("")}
      ${_lbl(296, 194, "gene embedding — tiled by LatentLens")}
    </svg>`,
    body: "The <b>Embedding</b> tab takes a single anchor cell, traverses it toward <i>every</i> one of the ~1,000 perturbations, and drops each morphed cell at that gene's spot on a <b>gene-similarity map</b> (UMAP/PHATE). <b>LatentLens</b> tiles thousands of these crops into one zoomable montage.",
    why: "It turns 1,000 separate what-ifs into a single navigable landscape — genes with similar phenotypes cluster together, visible at a glance.",
    defs: [["Gene embedding (UMAP / PHATE)", "a 2-D map where each point is a gene, placed so phenotypically similar knockouts sit close together."],
      ["Anchor cell", "the one real cell (see DDIM) whose counterfactual we render for every perturbation, so the comparison is apples-to-apples."],
      ["LatentLens", "the tiling engine that lays thousands of image crops onto the map as a smooth, zoomable montage."]]
  },
  {
    nav: "Virtual staining", kicker: "CROSS-CHANNEL", title: "One phase image → 42 fluorescent stains",
    svg: () => `<svg viewBox="0 0 420 200" class="mth">
      ${_cellBlob(78, 100, 40, MTH_C.ntc, "mth-breathe")}${_lbl(78, 162, "phase (label-free)")}
      ${_arrow(140, 238, 100)}
      <g>${[MTH_C.grn, MTH_C.ko, MTH_C.pur, MTH_C.yel, MTH_C.acc].map((col, i) =>
        `<g class="mth-emerge" style="animation-delay:${i * 0.22}s">${_cellBlob(302, 42 + i * 30, 14, col)}</g>`).join("")}</g>
      ${_lbl(330, 184, "42 fluorescent channels")}
    </svg>`,
    body: "We retrained the <b>same diffusion autoencoder</b> as a <b>virtual-staining</b> model: <b>conditioned</b> on a label-free phase image, it predicts how that exact cell would look in each of 42 fluorescent markers.",
    why: "Run on a traversal, it renders every channel for every perturbation of one cell — a full multi-channel phenotype from a single grayscale image, no extra stains.",
    defs: [["Virtual staining", "predicting fluorescent-marker images from a label-free phase image."],
      ["Conditioning", "extra input that steers generation — here the phase image plus which marker to render."],
      ["Spatial conditioning", "the phase image guides generation pixel-by-pixel (a spatial map), so a predicted marker lines up with the real cell's structures — not just a global \"marker-style\" texture."],
      ["Channel / marker", "one fluorescent stain (e.g. a mitochondrial or nuclear dye)."]]
  },
  {
    nav: "What's next", kicker: "THE NEXT DIRECTION", title: "From transcription to morphology",
    svg: () => `<svg viewBox="0 0 420 200" class="mth">
      <g>${_bars(46, 150, [40, 78, 25, 62, 90, 35], 16, 7, MTH_C.grn)}</g>${_lbl(95, 172, "CROP-seq (mRNA)")}
      ${_arrow(190, 262, 100)}
      <g class="mth-morph">${_cellBlob(320, 95, 44, MTH_C.acc)}</g>${_lbl(320, 160, "morphology")}
    </svg>`,
    body: "The same library was also profiled by <b>CROP-seq</b> (single-cell RNA). Next we drive the traversal by <b>transcriptional</b> change instead of morphology — asking how a cell's shape follows its gene-expression state.",
    why: "Bridging transcriptome and image lets us see which genes decouple the two — the core question of the transcriptional project.",
    defs: [["CROP-seq", "a pooled CRISPR screen read out by single-cell RNA sequencing (the transcriptome)."],
      ["Transcriptome", "the set of mRNA levels in a cell — its gene-expression state."],
      ["Perturbation autoencoder (CPA)", "a model that represents a perturbation's effect as an additive vector in latent space."]]
  },
];

// display order (narrative arc): data → represent → classify+interpret → generate+arrange → cross-channel → next
const MTH_ORDER = ["The screen", "Fingerprint", "Classifier", "Top cells", "Attention heads",
  "Diffusion", "Traversal", "DDIM", "Embedding", "Virtual staining", "What's next"];
const MTH_DECK = MTH_ORDER.map(n => METHODS_SLIDES.find(s => s.nav === n)).filter(Boolean);

let _mthIdx = 0;
function renderMethods() {
  const rail = document.getElementById("tab-methods");
  if (rail && !rail.querySelector(".mth-rail")) {
    rail.innerHTML = `<div class="hint">A visual tour of the methods behind this viewer — click through, ← → to navigate.</div>
      <div class="mth-rail">${MTH_DECK.map((s, i) =>
        `<button class="mth-railitem" onclick="methodsGo(${i})"><span class="mth-num">${i + 1}</span>${s.nav}</button>`).join("")}</div>`;
  }
  const s = MTH_DECK[_mthIdx], view = document.getElementById("methods-view");
  if (!view) return;
  const refs = MTH_REFS[s.nav] || [];
  view.innerHTML = `<div class="mth-card">
    <div class="mth-kicker">${s.kicker} · ${_mthIdx + 1} / ${MTH_DECK.length}</div>
    <h2 class="mth-title">${s.title}</h2>
    <div class="mth-stage">${s.svg()}</div>
    <div class="mth-body">${s.body}</div>
    ${MTH_MORE[s.nav] ? `<details class="mth-more"><summary>Learn more</summary><p>${MTH_MORE[s.nav]}</p></details>` : ""}
    <div class="mth-why"><b>Why it matters —</b> ${s.why}</div>
    ${refs.length ? `<div class="mth-refs">📄 ${refs.map(r => `<a href="${r[1]}" target="_blank" rel="noopener">${r[0]}</a>`).join(" &nbsp;·&nbsp; ")}</div>` : ""}
    ${(s.defs || []).length ? `<details class="mth-defs" open><summary>Key terms</summary><dl>${s.defs.map(d => `<div><dt>${d[0]}</dt><dd>${d[1]}</dd></div>`).join("")}</dl></details>` : ""}
    <div class="mth-navbar">
      <button onclick="methodsStep(-1)" ${_mthIdx === 0 ? "disabled" : ""}>← back</button>
      <div class="mth-dots">${MTH_DECK.map((_, i) => `<span class="mth-dot${i === _mthIdx ? " on" : ""}" onclick="methodsGo(${i})"></span>`).join("")}</div>
      <button onclick="methodsStep(1)" ${_mthIdx === MTH_DECK.length - 1 ? "disabled" : ""}>next →</button>
    </div>
  </div>`;
  document.querySelectorAll(".mth-railitem").forEach((b, i) => b.classList.toggle("on", i === _mthIdx));
}
function methodsGo(i) { _mthIdx = Math.max(0, Math.min(MTH_DECK.length - 1, i)); renderMethods(); }
function methodsStep(d) { methodsGo(_mthIdx + d); }
document.addEventListener("keydown", (e) => {
  const active = document.querySelector(".tab.active");
  if (active && active.dataset.tab === "methods") { if (e.key === "ArrowRight") methodsStep(1); if (e.key === "ArrowLeft") methodsStep(-1); }
});
