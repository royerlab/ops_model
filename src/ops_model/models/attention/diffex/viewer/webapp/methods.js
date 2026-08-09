// "How it works" tab — a slide deck of the ML methods behind the viewer, for non-ML readers.
// Self-contained: renderMethods() builds the concept rail (#tab-methods) + the animated slide (#methods-view).
// Animations are CSS keyframes (see style.css .mth-*) driven by group transforms + opacity so they loop while shown.

// --- little SVG builders (kept simple on purpose) ---
const _cellBlob = (cx, cy, r, fill, cls = "") =>   // a rounded organelle-ish cell
  `<circle cx="${cx}" cy="${cy}" r="${r}" fill="${fill}" class="${cls}"/>` +
  `<circle cx="${cx - r * 0.3}" cy="${cy - r * 0.2}" r="${r * 0.28}" fill="rgba(0,0,0,.35)"/>` +
  `<circle cx="${cx + r * 0.35}" cy="${cy + r * 0.25}" r="${r * 0.16}" fill="rgba(255,255,255,.35)"/>`;
const _bars = (x, y, vals, w, gap, col) => vals.map((v, i) =>
  `<rect x="${x + i * (w + gap)}" y="${y - v}" width="${w}" height="${v}" rx="1.5" fill="${col}" class="mth-rise" style="animation-delay:${i * 0.12}s"/>`).join("");
const _arrow = (x1, x2, y, col = "#26c6ff") =>
  `<line x1="${x1}" y1="${y}" x2="${x2 - 8}" y2="${y}" stroke="${col}" stroke-width="2.5"/>` +
  `<path d="M${x2 - 10},${y - 6} L${x2},${y} L${x2 - 10},${y + 6} Z" fill="${col}"/>`;

const MTH_C = { acc: "#26c6ff", ntc: "#8b949e", ko: "#f0a020", grn: "#3fb950", pur: "#bc8cff", yel: "#d29922" };

// core papers per slide (keyed by slide nav label) — rendered as explicit links
const MTH_REFS = {
  "The screen": [["Optical pooled screens · Feldman 2019", "https://doi.org/10.1016/j.cell.2019.09.016"]],
  "Fingerprint": [["DINOv2 · Oquab 2023", "https://arxiv.org/abs/2304.07193"], ["DINO · Caron 2021", "https://arxiv.org/abs/2104.14294"]],
  "Distinctiveness": [["Set Transformer · Lee 2019", "https://arxiv.org/abs/1810.00825"]],
  "Generate": [["Diffusion autoencoders · Preechakul 2022", "https://arxiv.org/abs/2111.15640"], ["DDPM · Ho 2020", "https://arxiv.org/abs/2006.11239"]],
  "Traversal": [["Diffusion autoencoders · Preechakul 2022", "https://arxiv.org/abs/2111.15640"], ["Classifier-free guidance · Ho & Salimans 2022", "https://arxiv.org/abs/2207.12598"]],
  "DDIM inversion": [["DDIM · Song 2020", "https://arxiv.org/abs/2010.02502"]],
  "Attention": [["Attention is all you need · Vaswani 2017", "https://arxiv.org/abs/1706.03762"], ["DINO attention · Caron 2021", "https://arxiv.org/abs/2104.14294"]],
  "Virtual staining": [["Diffusion autoencoders · Preechakul 2022", "https://arxiv.org/abs/2111.15640"], ["In silico labeling · Christiansen 2018", "https://doi.org/10.1016/j.cell.2018.03.040"]],
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
    body: "In a pooled optical CRISPR screen, every cell carries a DNA <b>barcode</b> naming the single gene knocked out inside it — read from the same image as the cell's shape.",
    why: "So each imaged cell comes with a known genetic perturbation, for free, at massive scale."
  },
  {
    nav: "Fingerprint", kicker: "REPRESENT", title: "Turning a cell image into numbers",
    svg: () => `<svg viewBox="0 0 420 200" class="mth">
      ${_cellBlob(90, 100, 46, MTH_C.acc, "mth-breathe")}
      ${_arrow(150, 230, 100)}
      <g>${_bars(250, 150, [70, 30, 95, 45, 60, 20, 80, 40], 15, 6, MTH_C.pur)}</g>
      <text x="300" y="172" fill="#8b949e" font-size="12" text-anchor="middle">embedding vector</text>
    </svg>`,
    body: "A self-supervised vision model (<b>CellDINO</b>) reads each image and outputs a numeric <b>fingerprint</b> — a point in a high-dimensional \"morphology space\" — with no hand-picked features.",
    why: "Similar-looking cells land near each other, so morphology becomes math we can compare and cluster."
  },
  {
    nav: "Distinctiveness", kicker: "CLASSIFY & RANK", title: "Which knockouts actually look different?",
    svg: () => `<svg viewBox="0 0 420 200" class="mth">
      <g class="mth-collapse">${Array.from({ length: 9 }, (_, i) => { const a = i / 9 * 6.28;
        return _cellBlob(110 + Math.cos(a) * 46, 100 + Math.sin(a) * 46, 9, MTH_C.grn); }).join("")}</g>
      ${_arrow(165, 232, 100)}
      <rect x="238" y="78" width="150" height="44" rx="8" fill="rgba(88,166,255,.15)" stroke="${MTH_C.acc}" class="mth-emerge"/>
      <text x="313" y="98" fill="#fff" font-size="13" text-anchor="middle" class="mth-emerge">gene X ?</text>
      <text x="313" y="114" fill="${MTH_C.acc}" font-size="12" text-anchor="middle" class="mth-emerge">83% confident</text>
    </svg>`,
    body: "A <b>SetTransformer</b> looks at a whole <i>group</i> of a gene's cells at once (order doesn't matter) and guesses which gene it is. How reliably it succeeds becomes a <b>distinctiveness ranking</b> over all perturbations.",
    why: "It scores each gene by how recognizable its phenotype is — the backbone metric of the viewer."
  },
  {
    nav: "Generate", kicker: "SIMULATE", title: "A model that paints cells",
    svg: () => `<svg viewBox="0 0 420 200" class="mth">
      <g class="mth-collapse">${Array.from({ length: 40 }, (_, i) =>
        `<circle cx="${60 + (i * 53) % 300}" cy="${40 + (i * 71) % 130}" r="2.2" fill="#8b949e"/>`).join("")}</g>
      <rect x="150" y="16" width="30" height="16" rx="4" fill="${MTH_C.yel}" class="mth-pulse"/>
      <text x="165" y="28" fill="#111" font-size="11" text-anchor="middle">z</text>
      <g class="mth-emerge">${_cellBlob(300, 105, 50, MTH_C.acc)}</g>
    </svg>`,
    body: "A <b>diffusion autoencoder</b> learns to build a realistic cell from a <b>semantic code</b> (z, what the cell is) plus a <b>noise seed</b> (the rest). Start from noise, denoise step by step, and a cell appears.",
    why: "We can now <i>generate</i> cells — not just read them — which is what makes counterfactuals possible."
  },
  {
    nav: "Traversal", kicker: "COUNTERFACTUAL", title: "\"What if this gene were knocked out?\"",
    svg: () => `<svg viewBox="0 0 420 200" class="mth">
      <g class="mth-morph">${_cellBlob(210, 92, 44, MTH_C.acc)}</g>
      <line x1="70" y1="168" x2="350" y2="168" stroke="#30363d" stroke-width="4"/>
      <circle cx="70" cy="168" r="5" fill="${MTH_C.ntc}"/><circle cx="350" cy="168" r="5" fill="${MTH_C.ko}"/>
      <circle cx="70" cy="168" r="8" fill="#fff" class="mth-slide"/>
      <text x="70" y="190" fill="#8b949e" font-size="11" text-anchor="middle">NTC (α0)</text>
      <text x="350" y="190" fill="${MTH_C.ko}" font-size="11" text-anchor="middle">knockout (α+)</text>
    </svg>`,
    body: "Slide the semantic code along the <b>NTC → knockout direction</b> and the generated cell morphs smoothly. α = 0 is the starting cell; α = 1 applies the full knockout shift; larger α exaggerates it.",
    why: "It shows the phenotype a perturbation induces as a continuous, watchable transformation."
  },
  {
    nav: "DDIM inversion", kicker: "ANCHOR", title: "Starting the morph from a real cell",
    svg: () => `<svg viewBox="0 0 420 200" class="mth">
      ${_cellBlob(70, 92, 34, MTH_C.grn)}<text x="70" y="150" fill="#8b949e" font-size="11" text-anchor="middle">real cell</text>
      <g class="mth-collapse">${Array.from({ length: 22 }, (_, i) =>
        `<circle cx="${185 + (i * 37) % 60 - 30}" cy="${92 + (i * 53) % 60 - 30}" r="2" fill="#8b949e"/>`).join("")}</g>
      <text x="210" y="150" fill="#8b949e" font-size="11" text-anchor="middle">its exact seed</text>
      ${_cellBlob(350, 92, 34, MTH_C.grn)}<text x="350" y="150" fill="#8b949e" font-size="11" text-anchor="middle">same cell</text>
      <g class="mth-cycA">${_arrow(110, 178, 78, MTH_C.pur)}<text x="144" y="70" fill="${MTH_C.pur}" font-size="10" text-anchor="middle">invert ↩</text></g>
      <g class="mth-cycB">${_arrow(242, 312, 78, MTH_C.acc)}<text x="277" y="70" fill="${MTH_C.acc}" font-size="10" text-anchor="middle">generate →</text></g>
    </svg>`,
    body: "A random seed makes a <i>generic</i> cell. <b>DDIM inversion</b> runs the diffusion <b>backwards</b> to find the exact seed that rebuilds one specific real cell — so α = 0 reconstructs it faithfully (pixel r ≈ 0.99).",
    why: "The morph now starts from a true cell, preserving its identity while only the phenotype changes."
  },
  {
    nav: "Attention", kicker: "INTERPRET", title: "Where is the model looking?",
    svg: () => `<svg viewBox="0 0 420 200" class="mth">
      ${_cellBlob(210, 100, 62, MTH_C.acc)}
      <circle cx="238" cy="86" r="16" fill="${MTH_C.ko}" class="mth-glow"/>
      <circle cx="188" cy="118" r="10" fill="${MTH_C.yel}" class="mth-glow" style="animation-delay:.8s"/>
      <text x="210" y="185" fill="#8b949e" font-size="12" text-anchor="middle">attention heat-map</text>
    </svg>`,
    body: "Each <b>attention head</b> highlights the pixels that drove its decision. Ranked by distinctiveness, they reveal <i>which structures</i> a knockout changes — e.g. a head that fixates on mitochondria.",
    why: "It turns a black-box score into a visible, checkable claim about cell biology."
  },
  {
    nav: "Virtual staining", kicker: "CROSS-CHANNEL", title: "One phase image → 42 fluorescent stains",
    svg: () => `<svg viewBox="0 0 420 200" class="mth">
      ${_cellBlob(78, 100, 40, MTH_C.ntc, "mth-breathe")}
      <text x="78" y="162" fill="#8b949e" font-size="11" text-anchor="middle">phase (label-free)</text>
      ${_arrow(140, 238, 100)}
      <g>${[MTH_C.grn, MTH_C.ko, MTH_C.pur, MTH_C.yel, MTH_C.acc].map((col, i) =>
        `<g class="mth-emerge" style="animation-delay:${i * 0.22}s">${_cellBlob(302, 42 + i * 30, 14, col)}</g>`).join("")}</g>
      <text x="330" y="184" fill="#8b949e" font-size="11" text-anchor="middle">42 fluorescent channels</text>
    </svg>`,
    body: "We retrained the <b>same diffusion autoencoder</b> as a <b>virtual-staining</b> model: given only a label-free <b>phase</b> image, it predicts how that exact cell would look in each of 42 fluorescent markers.",
    why: "Run on a traversal, it renders every channel for every perturbation of one cell — a full multi-channel phenotype from a single grayscale image, no extra stains."
  },
  {
    nav: "What's next", kicker: "THE NEXT DIRECTION", title: "From transcription to morphology",
    svg: () => `<svg viewBox="0 0 420 200" class="mth">
      <g>${_bars(46, 150, [40, 78, 25, 62, 90, 35], 16, 7, MTH_C.grn)}</g>
      <text x="95" y="172" fill="#8b949e" font-size="11" text-anchor="middle">CROP-seq (mRNA)</text>
      ${_arrow(190, 262, 100)}
      <g class="mth-morph">${_cellBlob(320, 95, 44, MTH_C.acc)}</g>
      <text x="320" y="160" fill="#8b949e" font-size="11" text-anchor="middle">morphology</text>
    </svg>`,
    body: "The same library was profiled by <b>CROP-seq</b>. Next we drive the traversal by <b>transcriptional</b> change instead of morphology — asking how a cell's shape follows its gene-expression state.",
    why: "Bridging transcriptome and image lets us see which genes decouple the two — the project's core question."
  },
];

let _mthIdx = 0;
function renderMethods() {
  const rail = document.getElementById("tab-methods");
  if (rail && !rail.querySelector(".mth-rail")) {   // build the concept rail once
    rail.innerHTML = `<div class="hint">A visual tour of the methods behind this viewer — click through, ← → to navigate.</div>
      <div class="mth-rail">${METHODS_SLIDES.map((s, i) =>
        `<button class="mth-railitem" onclick="methodsGo(${i})"><span class="mth-num">${i + 1}</span>${s.nav}</button>`).join("")}</div>`;
  }
  const s = METHODS_SLIDES[_mthIdx], view = document.getElementById("methods-view");
  if (!view) return;
  view.innerHTML = `<div class="mth-card">
    <div class="mth-kicker">${s.kicker} · ${_mthIdx + 1} / ${METHODS_SLIDES.length}</div>
    <h2 class="mth-title">${s.title}</h2>
    <div class="mth-stage">${s.svg()}</div>
    <div class="mth-body">${s.body}</div>
    <div class="mth-why"><b>Why it matters —</b> ${s.why}</div>
    ${(MTH_REFS[s.nav] || []).length ? `<div class="mth-refs">📄 ${MTH_REFS[s.nav].map(r => `<a href="${r[1]}" target="_blank" rel="noopener">${r[0]}</a>`).join(" &nbsp;·&nbsp; ")}</div>` : ""}
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
document.addEventListener("keydown", (e) => {   // arrow-key nav only while the Methods tab is showing
  const active = document.querySelector(".tab.active");
  if (active && active.dataset.tab === "methods") { if (e.key === "ArrowRight") methodsStep(1); if (e.key === "ArrowLeft") methodsStep(-1); }
});
