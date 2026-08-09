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
  "Virtual staining": [["In silico labeling · Christiansen 2018", "https://doi.org/10.1016/j.cell.2018.03.040"], ["Diffusion autoencoders · Preechakul 2022", "https://arxiv.org/abs/2111.15640"]],
  "What's next": [["CROP-seq · Datlinger 2017", "https://doi.org/10.1038/nmeth.4177"], ["Perturbation autoencoder (CPA) · Lotfollahi 2023", "https://doi.org/10.15252/msb.202211517"]],
};

const METHODS_SLIDES = [
  {
    nav: "The screen", kicker: "THE DATA", title: "One dish, thousands of gene knockouts",
    svg: () => `<svg viewBox="0 0 420 200" class="mth">
      <g>${Array.from({ length: 24 }, (_, i) => { const x = 40 + (i % 8) * 42, y = 45 + Math.floor(i / 8) * 52;
        const c = [MTH_C.ntc, MTH_C.acc, MTH_C.ko, MTH_C.grn, MTH_C.pur][i % 5];
        return `<g class="mth-pulse" style="animation-delay:${(i % 8) * 0.15}s">${_cellBlob(x, y, 13, c)}
          <rect x="${x - 10}" y="${y + 16}" width="20" height="4" fill="${MTH_C.acc}" opacity=".7"/></g>`; }).join("")}</g>
      <circle cx="166" cy="97" r="26" fill="none" stroke="#fff" stroke-width="2.5" class="mth-pulse"/>
    </svg>`,
    body: "In a <b>pooled optical CRISPR screen</b>, thousands of gene knockouts share one dish. Each cell carries a DNA <b>barcode</b> — sequenced in place, in the same image — that names the single gene knocked out inside it.",
    why: "Every imaged cell arrives with a known genetic perturbation, for free, at the scale of millions of cells.",
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
    svg: () => `<svg viewBox="0 0 440 200" class="mth">
      <g>${[0, 1, 2, 3].map(i => `<g class="mth-pulse" style="animation-delay:${i * 0.2}s">${_cellBlob(40, 42 + i * 38, 12, [MTH_C.grn, MTH_C.acc, MTH_C.pur, MTH_C.ko][i])}</g>`).join("")}
        ${[0, 1, 2, 3].map(i => [0, 1, 2, 3].filter(j => j > i).map(j =>
          `<line x1="52" y1="${42 + i * 38}" x2="52" y2="${42 + j * 38}" stroke="${MTH_C.acc}" stroke-width="1" class="mth-pulse" style="animation-delay:${(i + j) * 0.15}s"/>`).join("")).join("")}</g>
      ${_lbl(40, 190, "bag of cells")}
      ${_box(96, 66, 96, 50, "ISAB ×2", "cells attend")}
      ${_arrow(196, 232, 91)}
      ${_box(234, 66, 92, 50, "PMA", "attention pool")}
      ${_arrow(330, 362, 91)}
      <g>${_bars(366, 130, [26, 62, 18, 40], 12, 6, MTH_C.acc)}</g>
      <rect x="384" y="60" width="12" height="8" fill="none" stroke="${MTH_C.ko}" stroke-width="2" class="mth-pulse"/>
      ${_lbl(388, 150, "class", MTH_C.ko, 10)}
    </svg>`,
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
      ["Semantic code (z)", "a compact vector saying <i>what</i> the cell is — kept separate from the noise seed (a diffusion autoencoder)."]]
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
    defs: [["Semantic direction", "the vector in code-space pointing from control toward a knockout (a mean difference)."],
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

let _mthIdx = 0;
function renderMethods() {
  const rail = document.getElementById("tab-methods");
  if (rail && !rail.querySelector(".mth-rail")) {
    rail.innerHTML = `<div class="hint">A visual tour of the methods behind this viewer — click through, ← → to navigate.</div>
      <div class="mth-rail">${METHODS_SLIDES.map((s, i) =>
        `<button class="mth-railitem" onclick="methodsGo(${i})"><span class="mth-num">${i + 1}</span>${s.nav}</button>`).join("")}</div>`;
  }
  const s = METHODS_SLIDES[_mthIdx], view = document.getElementById("methods-view");
  if (!view) return;
  const refs = MTH_REFS[s.nav] || [];
  view.innerHTML = `<div class="mth-card">
    <div class="mth-kicker">${s.kicker} · ${_mthIdx + 1} / ${METHODS_SLIDES.length}</div>
    <h2 class="mth-title">${s.title}</h2>
    <div class="mth-stage">${s.svg()}</div>
    <div class="mth-body">${s.body}</div>
    <div class="mth-why"><b>Why it matters —</b> ${s.why}</div>
    ${refs.length ? `<div class="mth-refs">📄 ${refs.map(r => `<a href="${r[1]}" target="_blank" rel="noopener">${r[0]}</a>`).join(" &nbsp;·&nbsp; ")}</div>` : ""}
    ${(s.defs || []).length ? `<details class="mth-defs"><summary>Key terms</summary><dl>${s.defs.map(d => `<div><dt>${d[0]}</dt><dd>${d[1]}</dd></div>`).join("")}</dl></details>` : ""}
    <div class="mth-navbar">
      <button onclick="methodsStep(-1)" ${_mthIdx === 0 ? "disabled" : ""}>← back</button>
      <div class="mth-dots">${METHODS_SLIDES.map((_, i) => `<span class="mth-dot${i === _mthIdx ? " on" : ""}" onclick="methodsGo(${i})"></span>`).join("")}</div>
      <button onclick="methodsStep(1)" ${_mthIdx === METHODS_SLIDES.length - 1 ? "disabled" : ""}>next →</button>
    </div>
  </div>`;
  document.querySelectorAll(".mth-railitem").forEach((b, i) => b.classList.toggle("on", i === _mthIdx));
}
function methodsGo(i) { _mthIdx = Math.max(0, Math.min(METHODS_SLIDES.length - 1, i)); renderMethods(); }
function methodsStep(d) { methodsGo(_mthIdx + d); }
document.addEventListener("keydown", (e) => {
  const active = document.querySelector(".tab.active");
  if (active && active.dataset.tab === "methods") { if (e.key === "ArrowRight") methodsStep(1); if (e.key === "ArrowLeft") methodsStep(-1); }
});
