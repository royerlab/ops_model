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
const _mix = (a, b, t) => { const p = h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16)); const A = p(a), B = p(b); return `rgb(${A.map((v, i) => Math.round(v + (B[i] - v) * t)).join(",")})`; };   // hex→hex lerp
const _patchGrid = (x, y, s, n, hot) => Array.from({ length: n * n }, (_, i) => {
  const gx = x + (i % n) * s, gy = y + Math.floor(i / n) * s, on = hot.includes(i);
  return `<rect x="${gx}" y="${gy}" width="${s - 1}" height="${s - 1}" fill="${on ? MTH_C.acc : "rgba(255,255,255,.04)"}" stroke="rgba(255,255,255,.08)" ${on ? 'class="mth-glow" style="animation-delay:' + (i % 5) * 0.2 + 's"' : ""}/>`;
}).join("");
// shared phenotype-cell presets (used by Embedding + The Screen): membrane aspect (ar), nucleus fraction (nf), organelle offsets (frac of s)
const MTH_PHENO = [
  { ar: 1.0, nf: 0.24, org: [[-0.27, -0.33]] },
  { ar: 1.27, nf: 0.38, org: [[0.27, 0.33]] },
  { ar: 0.82, nf: 0.5, org: [[0.27, -0.4], [-0.27, 0.4]] },
  { ar: 0.72, nf: 0.3, org: [[-0.3, -0.18], [0.32, 0.26]] },
];
const _pheno = (cx, cy, s, p, fill) => `<ellipse cx="${cx}" cy="${cy}" rx="${(s * Math.sqrt(p.ar)).toFixed(1)}" ry="${(s / Math.sqrt(p.ar)).toFixed(1)}" fill="${fill}"/><circle cx="${cx}" cy="${cy}" r="${(s * p.nf).toFixed(1)}" fill="rgba(0,0,0,.45)"/>`
  + p.org.map(o => { const ox = cx + o[0] * s, oy = cy + o[1] * s; return `<ellipse cx="${ox.toFixed(1)}" cy="${oy.toFixed(1)}" rx="${(s * 0.16).toFixed(1)}" ry="${(s * 0.07).toFixed(1)}" fill="rgba(0,0,0,.4)" transform="rotate(30 ${ox.toFixed(1)} ${oy.toFixed(1)})"/>`; }).join("");
// low-res black&white "static" — a grid of grayscale pixels (deterministic per seed), like the paper schematic
const _pixnoise = (x, y, w, h, cols, rows, seed, alpha) => { const cw = w / cols, ch = h / rows; let s = "";
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) { const k = r * cols + c, v = Math.abs(Math.sin((k + 1) * 12.9898 + seed * 78.233)) * 43758.5453, g = Math.floor((v - Math.floor(v)) * 255);
    s += `<rect x="${(x + c * cw).toFixed(1)}" y="${(y + r * ch).toFixed(1)}" width="${cw.toFixed(2)}" height="${ch.toFixed(2)}" fill="rgb(${g},${g},${g})" opacity="${alpha}"/>`; }
  return s; };

const MTH_REFS = {
  "The screen": [["Optical pooled screens · Feldman 2019", "https://doi.org/10.1016/j.cell.2019.09.016"], ["Funk et al. · Cell 2022", "https://www.sciencedirect.com/science/article/pii/S0092867422013599"], ["Liu et al. · bioRxiv 2026", "https://www.biorxiv.org/content/10.64898/2026.06.01.728087v1"]],
  "Embedding": [["Cell-DINO · Moutakanni · PLOS Comput Biol 2025", "https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1013828"], ["DINO · Caron 2021", "https://arxiv.org/abs/2104.14294"], ["DINOv2 · Oquab 2023", "https://arxiv.org/abs/2304.07193"]],
  "Classifier": [["Set Transformer · Lee 2019", "https://arxiv.org/abs/1810.00825"], ["Attention-based multiple-instance learning · Ilse 2018", "https://arxiv.org/abs/1802.04712"]],
  "Top cells": [["Explaining by removing · Covert 2021", "https://arxiv.org/abs/2011.14878"], ["SHAP · Lundberg & Lee 2017", "https://arxiv.org/abs/1705.07874"]],
  "Diffusion": [["DDPM · Ho 2020", "https://arxiv.org/abs/2006.11239"], ["Diffusion autoencoders · Preechakul 2022", "https://arxiv.org/abs/2111.15640"], ["DDIM · Song 2020", "https://arxiv.org/abs/2010.02502"]],
  "Traversal": [["DiffEx · Bourou 2025", "https://arxiv.org/abs/2502.09663"], ["Diffusion autoencoders · Preechakul 2022", "https://arxiv.org/abs/2111.15640"], ["Classifier-free guidance · Ho & Salimans 2022", "https://arxiv.org/abs/2207.12598"]],
  "DDIM": [["DDIM · Song 2020", "https://arxiv.org/abs/2010.02502"]],
  "Attention heads": [["DINO attention · Caron 2021", "https://arxiv.org/abs/2104.14294"], ["Attention is all you need · Vaswani 2017", "https://arxiv.org/abs/1706.03762"]],
  "Montage": [["UMAP · McInnes 2018", "https://arxiv.org/abs/1802.03426"], ["PHATE · Moon 2019", "https://doi.org/10.1038/s41587-019-0336-3"]],
  "Virtual staining": [["In silico labeling · Christiansen 2018", "https://doi.org/10.1016/j.cell.2018.03.040"], ["Diffusion autoencoders · Preechakul 2022", "https://arxiv.org/abs/2111.15640"]],
  "mRNA phenotypes": [["CROP-seq · Datlinger 2017", "https://doi.org/10.1038/nmeth.4177"], ["Perturbation autoencoder (CPA) · Lotfollahi 2023", "https://doi.org/10.15252/msb.202211517"]],
};

// longer "Learn more" paragraph per slide (keyed by nav) — unpacks the concept for readers who want depth
const MTH_MORE = {
  "The screen": "A traditional CRISPR screen gives one number per perturbation — did cells grow, did a reporter switch on. An optical pooled screen instead pools cells carrying many different perturbations into one dish and images them together in place, so a single experiment reads out thousands of knockouts side by side. The trick: each cell also carries a short DNA barcode unique to its CRISPR guide, and a round of in-situ sequencing lights that barcode up letter-by-letter right in the microscope — so we can match every cell's image to the exact geneKO. One imaging run yields millions of (perturbation, picture) pairs. The hard part is what comes next: those cells are a mix of thousands of perturbations, each present in many copies that vary enormously for reasons unrelated to the knockout — cell cycle, size, local density, position on the plate — and most knockouts nudge the phenotype only slightly. So the central problem is signal-in-noise: for every perturbation, which cells and which features actually capture its distinctive effect, rather than the technique's inherent variability? The classifier, rankings, and generative traversals in this viewer are complementary tools for answering exactly that. The specific dataset here is a multimodal perturbation atlas of 1,000 pooled CRISPR knockouts in A549 cells — profiled by fluorescence microscopy (39 live, 13 fixed markers), label-free phase imaging of the same live cells, and scRNA-seq, totaling ~57 million single-cell profiles. A central finding: label-free phase imaging matches — and, with sufficient cell coverage, exceeds — the phenotypic resolution of fluorescence and scRNA-seq, while capturing higher-order pathway organization that scRNA-seq does not resolve, establishing intrinsic morphology as a high-precision readout of cellular state.",
  "Embedding": "A raw microscope image is hundreds of thousands of pixels — too many, and too noisy, to compare directly. We need a compact summary that keeps what's biologically meaningful (shape, texture, organelle layout) and drops what isn't (exact position, lighting). CellDINO is a vision transformer trained by self-supervision: shown only images, never perturbation labels, it learns embeddings where two crops of the same cell agree and different cells differ. The result is a 1,024-number fingerprint per cell whose distances track real morphological similarity — which is what lets us cluster phenotypes, rank perturbations, and steer the generative model later.",
  "Classifier": "A single knocked-out cell is often ambiguous — cells vary a lot even with no perturbation — so the signal lives in the distribution of a perturbation's cells. We therefore classify a whole bag at once (multiple-instance learning). The SetTransformer tags each cell's fingerprint with its imaging channel, uses attention so the cells in a bag can inform one another, then pools them into a single bag-vector with attention (PMA) rather than a plain average, letting it weight the informative cells. Because attention and pooling ignore order, shuffling the bag can't change the answer; because it trained across bag sizes it can score anywhere from 10 to thousands of cells. The output is a probability over 1,000 perturbations (+NTC), or over 99 protein complexes when we ask about pathways instead of single geneKOs.",
  "Top cells": "Once the classifier works, we ask which individual cells it actually relies on, borrowing an idea from model explanation called \"explaining by removing\": a cell is important if taking it out of a bag makes the classifier less sure of the right answer. Concretely we take the probability of the correct class with the cell in the bag minus the probability without it, and average that marginal contribution over many random bags and bag sizes (a Monte-Carlo estimate). The highest-scoring cells are the clearest examples of a perturbation's phenotype — these top-predictive cells are exactly what the Top Cells tab shows and what the diffusion traversals are anchored to. Up-weighting them also sharpens the perturbation-level distinctiveness score (mAP).",
  "Diffusion": "The purpose of the diffusion model here is to be a <b>decoder for CellDINO space</b>: given a cell's CellDINO vector, produce the image that vector describes. That is what makes the fingerprint <i>editable</i> — nudge the vector and the decoded cell changes. It learns this by reversing a corruption process: during training we add Gaussian noise to a real cell over and over until it is pure static, and a network learns to predict — from a noisy image plus the cell's CellDINO vector — the noise that was added. To generate, we start from static and repeatedly subtract the predicted noise (steered by the vector) until a realistic cell condenses out. By default the starting noise is random, so a given vector decodes to <i>a</i> representative cell, not one particular real cell — and a morph from a generic cell is hard to trust. <b>DDIM (Denoising Diffusion Implicit Models)</b> fixes this. The \"implicit\" part means it denoises along a single fixed trajectory — a deterministic ODE rather than a random Markov walk — so the same seed always yields the same cell. Because it is deterministic it can be integrated <i>in reverse</i>: from a real cell's pixels, run the trajectory backwards to recover the exact noise seed that regenerates it. We invert under the same guidance used for generation (\"guided inversion\"), so decoding at α = 0 reproduces the original cell almost perfectly (pixel correlation ≈ 0.99). Every traversal therefore starts anchored to a genuine cell.",
  "Traversal": "The semantic code lives in a space where nearby points are similar-looking cells and directions correspond to consistent visual changes. To build a knockout's \"movie\" we take its <b>destination</b> — the centroid (average semantic code) of that knockout's <b>top cells</b>, the same ones surfaced in the Top Cells tab — and the control (NTC) centroid; the vector between them is the direction that turns control-looking into knockout-looking. Starting from one cell's code we step along it — α = 0 is the cell itself, α = 1 applies the full average shift, larger α exaggerates subtle effects — decoding an image at each step. Because the cell's identity latent — its size, texture, and local context — is held fixed throughout and only its phenotype code moves, you're watching the very same cell change, not a slideshow of different cells: every difference you see is the phenotype, with everything else about the cell kept constant.",
  "DDIM": "By default the noise seed is random, so decoding a code gives a representative cell, not any particular real one — and a traversal from a generic cell is hard to trust. DDIM fixes this two ways. First it makes generation deterministic: the same seed always yields the same image (a diffusion \"ODE\", not a random walk). Second, being deterministic, it can run in reverse — starting from a real cell's pixels and integrating backwards to recover the exact noise seed that regenerates it. We invert under the same guidance used to generate (\"guided inversion\"), so decoding at α = 0 reproduces the original cell almost perfectly (pixel correlation ≈ 0.99). Every traversal here therefore begins anchored to a genuine cell, and the changes you see are real counterfactuals, not artifacts of a random start.",
  "Attention heads": "The vision transformer doesn't read pixels one at a time — it breaks the image into a grid of patches and, in each attention \"head\", decides how much every patch should influence its summary of the cell. Reading those weights back out gives a heat-map showing where the model concentrated. Different heads specialize on different structures, and the viewer lets you step through them and compare a perturbation against its control. The payoff is interpretability: instead of only telling you that a perturbation has a distinctive phenotype, the map shows you where the model is looking — e.g. a head that consistently lights up mitochondria for a mitochondrial geneKO — which you can check against the biology.",
  "Montage": "A single traversal shows one perturbation's effect; the Montage tab shows all of them at once — and controls for cell-to-cell variation by using the same anchor cell throughout. We morph that one cell toward each of the ~1,000 knockouts, giving 1,000 counterfactual images of the same starting cell. Each image is then positioned by its perturbation's coordinates on a UMAP or PHATE embedding of the perturbation-level phenotype space, so knockouts that produce similar morphologies land near one another. LatentLens stitches the crops into a continuous, zoomable atlas — pan and zoom to compare neighborhoods, spot phenotype clusters, and see where a perturbation of interest falls relative to the whole library.",
  "Virtual staining": "Fluorescent markers reveal specific structures but cost extra dyes, channels, and imaging, and you can only stain a few at once. Label-free phase imaging is cheap and gentle but hard to read. Virtual staining bridges them: we reuse the diffusion autoencoder and condition it on a phase image in two complementary ways. The semantic path sends the phase image through a frozen Cell-DINO ViT to a pooled code that, together with a learned marker id, globally modulates the U-Net (FiLM) — telling it what to draw. The spatial path concatenates the raw phase image as an extra U-Net input channel, so generation keeps the input's pixel layout and the predicted marker stays registered to the real cell (this is what lifts fidelity from ~0.13 to ~0.78 Pearson). One model, trained on paired (phase, marker) crops, covers all 42 live markers; switching the marker id renders any channel, and applied to a traversal it gives a full multi-channel phenotype for every perturbation from a single grayscale image.",
  "mRNA phenotypes": "So far every direction has been morphological — defined in the CellDINO image-fingerprint space. But the same knockout library was also profiled by CROP-seq, which reads each cell's transcriptome (its gene-expression profile) instead of its picture, giving every perturbation a transcriptional signature (its pseudobulk expression shift versus NTC). How would we drive the diffusion decoder from that? Two designs. The <b>light one</b> reuses everything: fit a map — linear first, then a small MLP — from a perturbation's transcriptional shift to its shift in CellDINO space, then feed that predicted CellDINO direction into the existing decoder. No retraining, and the map's R² directly measures how much of morphology is even predictable from transcriptome. The <b>full one</b> conditions the diffusion model on the transcriptome directly, in the style of a compositional perturbation autoencoder (CPA): a single per-perturbation embedding drives both a transcriptome decoder (reconstructing the CROP-seq profile) and the image decoder, so the two modalities are tied through one shared latent and any transcriptional state — even an unseen combination — can be rendered. Either way the payoff is the same: a transcriptome↔morphology divergence map that flags the perturbations that reshape the transcriptome but barely change the image, or vice versa — decoupling that is invisible to either assay alone.",
};

const METHODS_SLIDES = [
  {
    nav: "The screen", kicker: "THE QUESTION", title: "Thousands of knockouts, mixed in one noisy dish",
    svg: () => { const COL = [MTH_C.grn, MTH_C.acc, MTH_C.ko, MTH_C.pur, MTH_C.yel]; const cells = [];
      for (let r = 0; r < 6; r++) for (let c = 0; c < 5; c++) { const x = 30 + c * 30 + ((r * 37) % 9 - 4), y = 40 + r * 24 + ((c * 53) % 9 - 4);
        if (((x - 92) / 74) ** 2 + ((y - 100) / 80) ** 2 <= 0.92) cells.push([x, y, (r * 2 + c) % 5]); }
      return `<svg viewBox="0 0 484 200" class="mth">
        <ellipse cx="92" cy="100" rx="80" ry="86" fill="rgba(255,255,255,.03)" stroke="#30363d" stroke-width="2"/>
        ${cells.map(([x, y, ci]) => { const r = ci === 1 ? 8.75 : 7;
          return `<g class="${ci === 1 ? "mth-hi" : "mth-soft"}" style="animation-delay:${((x + y) / 90).toFixed(2)}s">${_cellBlob(x, y, r, COL[ci])}</g>`; }).join("")}
        ${_lbl(92, 194, "well — pooled knockouts, all mixed")}
        ${_arrow(176, 200, 100)}
        ${[0, 1, 2, 3, 4, 5, 6].map(i => `<rect x="${208 + i * 4}" y="88" width="${1.5 + (i % 2) * 2}" height="24" fill="#e6e8ec"/>`).join("")}
        <text x="224" y="80" fill="#8b949e" font-size="8" text-anchor="middle">debarcode</text>
        ${_arrow(242, 270, 100)}
        <rect x="278" y="42" width="120" height="132" rx="8" fill="rgba(38,198,255,.06)" stroke="${MTH_C.acc}"/>
        <text x="338" y="59" fill="${MTH_C.acc}" font-size="9.5" text-anchor="middle">one knockout's cells</text>
        ${[[308, 88], [370, 88], [308, 126], [370, 126]].map((p, i) => `<g class="mth-hi" style="animation-delay:${i * 0.7}s">${_pheno(p[0], p[1], 15, MTH_PHENO[i], MTH_C.acc)}</g>`).join("")}
        <text x="338" y="156" fill="#8b949e" font-size="10.5" text-anchor="middle">many phenotypes —</text><text x="338" y="169" fill="#8b949e" font-size="10.5" text-anchor="middle">which is real?</text>
      </svg>`; },
    body: "In a <b>pooled optical CRISPR screen</b>, thousands of geneKOs are mixed in one dish and imaged together; each cell's DNA <b>barcode</b>, sequenced in place, names the geneKO inside it. This yields millions of (perturbation, image) pairs — but each knockout's real effect is subtle and buried in enormous cell-to-cell variation. This viewer explores one such atlas: <b>1,000 gene knockouts in A549 cells</b>, profiled by fluorescence microscopy (39 live + 13 fixed markers), label-free <b>phase</b> imaging of the same live cells, and scRNA-seq — <b>~57 million single-cell profiles</b> in all.",
    why: "The core question this whole viewer answers: for each perturbation, which change in the cell captures its true phenotype — separated from the noise of the technique's scale and heterogeneity? Everything that follows is one answer.",
    defs: [["Pooled optical screen", "imaging a mixed population where every cell has a different geneKO, all together."],
      ["Barcode", "a short DNA tag, read out in situ, that identifies which CRISPR guide (perturbation) is in each cell."],
      ["Perturbation", "the genetic change applied to a cell — here a CRISPR knockout; NTC = non-targeting control (no geneKO)."]]
  },
  {
    nav: "Embedding", kicker: "REPRESENT", title: "Turning a cell image into numbers (a ViT)",
    svg: () => { const A = MTH_C.acc, P = MTH_C.pur, gx = 14, gy = 72, gs = 15, gn = 4;
      // three discrete phenotype states — cell (shared MTH_PHENO presets), active patches, tokens, and the 1024-d vector all switch together
      const HOTC = [[2, 5, 9], [1, 6, 10, 13], [0, 4, 7, 11, 14]];
      const HOTT = [[0, 2, 4], [1, 4, 5], [0, 3, 4]];
      const VEC = [[58, 26, 72, 40, 54, 20, 66, 34], [30, 64, 20, 80, 44, 60, 28, 72], [70, 18, 50, 34, 78, 26, 42, 60]];
      const lattice = Array.from({ length: gn + 1 }, (_, k) => `<line x1="${gx + k * gs}" y1="${gy}" x2="${gx + k * gs}" y2="${gy + gn * gs}" stroke="rgba(255,255,255,.28)" stroke-width=".6"/><line x1="${gx}" y1="${gy + k * gs}" x2="${gx + gn * gs}" y2="${gy + k * gs}" stroke="rgba(255,255,255,.28)" stroke-width=".6"/>`).join("");
      const hotC = hot => hot.map(i => `<rect x="${gx + (i % gn) * gs}" y="${gy + Math.floor(i / gn) * gs}" width="${gs}" height="${gs}" fill="${A}" opacity=".45"/>`).join("");
      const tokens = hot => [0, 1, 2, 3, 4, 5].map(i => `<rect x="134" y="${42 + i * 20}" width="18" height="16" rx="3" fill="${hot.includes(i) ? "rgba(38,198,255,.6)" : "rgba(38,198,255,.12)"}" stroke="${A}"/>`).join("");
      const stateG = s => `<g class="mth-st${s}">${_pheno(44, 100, 30, MTH_PHENO[s], A)}${hotC(HOTC[s])}${lattice}${tokens(HOTT[s])}${_bars(352, 136, VEC[s], 11, 4, P, "")}</g>`;
      return `<svg viewBox="0 0 468 200" class="mth">
      ${[0, 1, 2].map(stateG).join("")}
      ${_lbl(44, 150, "image crop")}
      <text x="106" y="80" fill="#8b949e" font-size="8" text-anchor="middle">patchify</text>
      ${_arrow(82, 128, 100)}
      ${_lbl(143, 178, "patch tokens")}
      ${_arrow(158, 200, 100)}
      ${_box(204, 74, 98, 52, "Transformer", "self-attention")}
      ${_arrow(306, 346, 100)}
      ${_lbl(400, 156, "1024-d feature")}
    </svg>`; },
    body: "A <b>vision transformer</b> (CellDINO) cuts the image into a grid of <b>patches</b>, turns each into a token, and lets them <b>attend</b> to one another; the tokens are pooled into one <b>1,024-d feature vector</b> — a compact fingerprint of the cell's morphology. It's trained <b>self-supervised</b> (no labels), so similar cells get similar vectors.",
    why: "Turning each cell into a comparable vector is what makes morphology <i>measurable</i> — the basis for clustering, ranking, and steering the generative model.",
    defs: [["Patch / token", "the small square pieces the image is cut into; each becomes one input token to the transformer."],
      ["Vision transformer (ViT)", "a network that relates all patch-tokens with attention, rather than scanning with convolutions."],
      ["Attention", "a weighted lookup between tokens: each patch emits a query and a key, their match sets a weight, and the patch is updated as a weighted sum of the others' values. High weight = \"this patch is relevant to me,\" so the update pulls in information from wherever in the cell matters most."],
      ["Self-attention", "attention run within one image — queries, keys, and values all come from the same set of patch-tokens, so every patch is refined by the whole cell's context at once. That lets the ViT link distant structures (e.g. a nucleus and a far-off organelle) in a single step, which convolutions can't do locally."],
      ["Self-supervised (DINO)", "trained on images alone — no perturbation labels — so it learns general-purpose morphology features."],
      ["Feature vector / embedding", "the pooled 1,024 numbers summarizing the cell; distances between vectors track visual similarity."]]
  },
  {
    nav: "Classifier", kicker: "THE MODEL", title: "A classifier that reads a whole group of cells",
    svg: () => { const X = [118, 205, 290, 392], L0 = [46, 82, 118, 154], LH = [64, 100, 136], OY = [38, 72, 106, 140, 174], OC = [MTH_C.grn, MTH_C.acc, MTH_C.ko, MTH_C.pur, MTH_C.yel];
      const edges = (xa, ya, xb, yb, d) => ya.map(a => yb.map(b => `<line x1="${xa}" y1="${a}" x2="${xb}" y2="${b}" stroke="${MTH_C.acc}" stroke-width=".7" opacity=".28" class="mth-flow-edge" style="animation-delay:${d}s"/>`).join("")).join("");
      const set = (col, by, hl) => `<g class="${hl ? "mth-hi" : ""}" ${hl ? 'style="animation-delay:.2s"' : 'opacity=".5"'}>${[[16, by], [30, by], [16, by + 14], [30, by + 14]].map(c => _cellBlob(c[0], c[1], 5.5, col)).join("")}</g>`;
      return `<svg viewBox="0 0 470 200" class="mth">
        ${[MTH_C.grn, MTH_C.acc, MTH_C.ko, MTH_C.pur].map((col, s) => set(col, 30 + s * 40, col === MTH_C.acc)).join("")}
        <text x="6" y="182" fill="#8b949e" font-size="10">CellDINO vectors</text><text x="6" y="194" fill="#8b949e" font-size="10">(one per cell; a set = one perturbation)</text>
        ${_arrow(46, 108, 100, MTH_C.acc)}
        ${edges(X[0], L0, X[1], LH, 0.25)}${edges(X[1], LH, X[2], LH, 0.75)}${LH.map(a => OY.map((b, j) => `<line x1="${X[2]}" y1="${a}" x2="${X[3]}" y2="${b}" stroke="${MTH_C.acc}" stroke-width=".7" opacity=".28" ${j === 1 ? 'class="mth-flow-edge" style="animation-delay:1.25s"' : ""}/>`).join("")).join("")}
        ${L0.map(y => `<circle cx="${X[0]}" cy="${y}" r="7" fill="${MTH_C.acc}" class="mth-flow-node"/>`).join("")}
        ${LH.map(y => `<circle cx="${X[1]}" cy="${y}" r="7" fill="rgba(38,198,255,.2)" stroke="${MTH_C.acc}" class="mth-flow-node" style="animation-delay:.5s"/>`).join("")}
        ${LH.map(y => `<circle cx="${X[2]}" cy="${y}" r="7" fill="rgba(38,198,255,.2)" stroke="${MTH_C.acc}" class="mth-flow-node" style="animation-delay:1s"/>`).join("")}
        ${OY.map((y, i) => `<circle cx="${X[3]}" cy="${y}" r="8" fill="${i === 1 ? OC[i] : "rgba(255,255,255,.06)"}" stroke="${OC[i]}" ${i === 1 ? 'class="mth-predict" style="animation-delay:1.25s"' : ""}/>`).join("")}
        ${_lbl(118, 18, "encode + channel-embed", "#8b949e", 7.5)}
        ${_lbl(247, 18, "ISAB ×2 · inducing-point attention", "#8b949e", 7.5)}
        ${_lbl(392, 18, "PMA pool → cosine", "#8b949e", 7.5)}
        ${_lbl(392, 196, "class scores (perturbations)")}
        <text x="406" y="76" fill="${MTH_C.acc}" font-size="9">← predicted</text>
      </svg>`; },
    body: "Because single cells are noisy, we show the model a whole <b>group</b> of a perturbation's cells at once — each as its <b>CellDINO feature vector</b> (from the Embedding tab), not the raw image — and ask it to name the perturbation. It weighs the cells against each other, pools them into one verdict, and outputs a probability over the <b>1,000 perturbations</b> (or 99 protein complexes). The architecture is a <b>SetTransformer</b> — the how is below.",
    why: "A single cell is too noisy to pin down a subtle perturbation. Scoring a whole group lets the shared signal add up while random cell-to-cell variation cancels — so an effect invisible in any one cell becomes unmistakable in the population.",
    defs: [["Multiple-instance learning", "classify a whole bag from one shared label, without labeling individual cells."],
      ["Bag / permutation-invariant", "an unordered set of a perturbation's cells; shuffling them can't change the prediction."],
      ["ISAB (inducing-point attention)", "self-attention routed through 32 learned reference points, so cost grows linearly and scales to tens of thousands of cells."],
      ["Channel embedding", "a learned tag marking each cell's imaging channel (phase, or a given fluorescent marker)."],
      ["PMA + cosine classifier", "pooling-by-attention collapses the bag to one vector; a cosine classifier turns it into class probabilities."]]
  },
  {
    nav: "Top cells", kicker: "EXPLAINING BY REMOVING", title: "Which cells carry the phenotype?",
    svg: () => `<svg viewBox="0 0 468 200" class="mth">
      <rect x="14" y="46" width="66" height="108" rx="8" fill="rgba(255,255,255,.04)" stroke="#30363d"/>
      ${_cellBlob(34, 82, 9, MTH_C.acc)}${_cellBlob(58, 82, 9, MTH_C.acc)}${_cellBlob(34, 116, 9, MTH_C.acc)}
      <g class="mth-remove">${_cellBlob(58, 116, 12.5, MTH_C.ko)}</g>
      ${_lbl(47, 170, "bag · cell x")}
      ${_arrow(84, 140, 100)}
      ${[64, 100, 136].flatMap(a => [64, 100, 136].map(b => `<line x1="150" y1="${a}" x2="192" y2="${b}" stroke="${MTH_C.acc}" stroke-width=".6" opacity=".3"/>`)).join("")}
      ${[64, 100, 136].map(y => `<circle cx="150" cy="${y}" r="6" fill="rgba(38,198,255,.2)" stroke="${MTH_C.acc}"/>`).join("")}
      ${[64, 100, 136].map(y => `<circle cx="192" cy="${y}" r="6" fill="rgba(38,198,255,.2)" stroke="${MTH_C.acc}"/>`).join("")}
      ${_lbl(171, 28, "classifier", "#8b949e", 8)}
      ${_arrow(204, 240, 100)}
      <text x="288" y="30" fill="#e6e8ec" font-size="9" text-anchor="middle">accuracy score = P(perturbation X)</text>
      <rect x="270" y="54" width="30" height="88" rx="3" fill="rgba(255,255,255,.05)" stroke="#30363d"/>
      <rect x="270" y="98" width="30" height="44" rx="3" fill="${MTH_C.acc}" opacity=".8"/>
      <rect x="270" y="54" width="30" height="44" fill="${MTH_C.ko}" opacity=".92" class="mth-drop"/>
      <line x1="264" y1="55" x2="264" y2="97" stroke="${MTH_C.ko}" stroke-width="1"/>
      <text x="260" y="79" fill="${MTH_C.ko}" font-size="9" text-anchor="end">Δ = score(x)</text>
      ${_lbl(285, 48, "with x", MTH_C.ko, 9)}${_lbl(285, 156, "without x", "#8b949e", 9)}
      ${_arrow(334, 368, 100)}
      ${(() => { let cx = 372; return [0, 1, 2, 3, 4].map(i => { const top = i === 4; const r = 4.5 + i * 1.1; if (i > 0) cx += (4.5 + (i - 1) * 1.1) + r + 5; return `<g ${top ? 'class="mth-shrink"' : ""}><circle cx="${cx.toFixed(1)}" cy="100" r="${r}" fill="${top ? MTH_C.ko : MTH_C.acc}" opacity="${(0.55 + i * 0.11).toFixed(2)}"/></g>`; }).join(""); })()}
      ${_lbl(404, 128, "higher rank →")}
    </svg>`,
    body: "To find a perturbation's most telling cells, we score each cell by how much it <b>helps the classifier</b>: the drop in predicted probability when the cell is <b>removed</b> from a bag (\"explaining by removing\"), averaged over many bag sizes and random partners. The top-scoring <b>top-predictive cells</b> carry the phenotypic signature — the cells the viewer anchors its traversals to and shows in Top Cells. Re-weighting the perturbation-level <b>mAP</b> by these cells sharpens the distinctiveness ranking.",
    why: "It surfaces, per perturbation, the handful of real cells that most clearly show its phenotype — the clearest examples of what the knockout actually looks like, and the target the traversal morphs toward.",
    defs: [["Explaining by removing", "gauge a cell's importance by how much the prediction drops when you take it out of the bag."],
      ["Marginal contribution", "score(x) = p(class | bag with x) − p(class | bag without x), averaged over many bags (sizes 1–500)."],
      ["Top-predictive cells", "the highest-scoring cells for a class; used as traversal anchors and in the Top Cells tab."],
      ["Distinctiveness (mAP)", "the perturbation-level separability score (Fig. 2), recomputed with these top cells up-weighted."]]
  },
  {
    nav: "Diffusion", kicker: "DECODE & INVERT", title: "Turning a CellDINO vector back into a cell",
    sections: [
      { text: "We <b>condition</b> a diffusion model on a cell's <b>CellDINO vector</b> (the same fingerprint from the Embedding tab). Starting from noise it removes a little at each step, steered by that vector, until it produces the exact cell the vector describes — so the model becomes a <b>decoder</b> for CellDINO space.",
        svg: () => `<svg viewBox="0 0 470 150" class="mth">
        <g>${_bars(10, 80, [22, 11, 30, 15, 24, 13], 7, 3, MTH_C.pur, "mth-jit")}</g>
        <text x="40" y="98" fill="${MTH_C.pur}" font-size="11" text-anchor="middle">CellDINO z</text>
        ${_arrow(76, 96, 56, MTH_C.pur)}
        ${[0, 1, 2, 3].map(i => { const x = 98 + i * 90, op = i / 3;
          return `<rect x="${x}" y="26" width="66" height="66" rx="6" fill="#0d0f13" stroke="#30363d"/>${_pixnoise(x + 4, 30, 58, 58, 12, 12, i + 1, (1 - op * 0.92).toFixed(2))}<g opacity="${op}">${_cellBlob(x + 33, 59, 21, MTH_C.acc)}</g>`; }).join("")}
        <rect x="98" y="26" width="66" height="66" rx="6" fill="none" stroke="#fff" stroke-width="2" class="mth-sweep" style="--sw:270px"/>
        ${_lbl(268, 16, "z conditions the denoising — z paints its cell →", MTH_C.acc, 10)}
        ${_lbl(131, 108, "noise")}${_lbl(401, 108, "the cell z describes")}
      </svg>` },
      { text: "<b>DDIM = Denoising Diffusion Implicit Models.</b> \"Implicit\" means it denoises along one <b>fixed, deterministic path</b> (an ODE) instead of a random walk — so the same seed always gives the same cell. Being deterministic, it can also run in <b>reverse</b>: from a real cell's pixels it recovers the exact noise seed that regenerates it, so decoding at α = 0 reproduces that cell (r ≈ 0.99).",
        svg: () => `<svg viewBox="0 0 420 165" class="mth">
        ${_cellBlob(70, 90, 32, MTH_C.acc)}${_lbl(70, 146, "real cell")}
        <rect x="182" y="66" width="56" height="48" rx="5" fill="#0d0f13" stroke="#30363d"/>
        ${_pixnoise(185, 69, 50, 42, 12, 10, 7, 0.95)}
        ${_lbl(210, 146, "its exact seed x_T")}
        ${_cellBlob(350, 90, 32, MTH_C.acc)}${_lbl(350, 146, "generated cell (r≈0.99)")}
        <g class="mth-cycA">${_arrow(112, 176, 78, MTH_C.pur)}${_lbl(144, 60, "invert ↩", MTH_C.pur, 10)}</g>
        <g class="mth-cycB">${_arrow(244, 310, 78, MTH_C.acc)}${_lbl(277, 60, "generate →", MTH_C.acc, 10)}</g>
      </svg>` }
    ],
    body: "<b>Why a diffusion model?</b> We want to turn a <b>CellDINO vector</b> back into an image — a generative <b>decoder</b> conditioned on the fingerprint — so we can then <b>nudge the vector</b> and watch the cell change (that's the Traversal). It learns this by reversing noise: corrupt a real cell to static, then learn to undo it step by step, steered by that cell's CellDINO vector. <b>DDIM</b> makes the reverse deterministic, so it also runs backwards to recover a real cell's exact seed and anchor the edit.",
    why: "It makes CellDINO space <i>visual and editable</i>: any fingerprint — real or shifted — becomes a cell you can see, which is what turns a number into a watchable phenotype.",
    defs: [["Decoder for CellDINO", "the diffusion model is trained to reconstruct a cell from its CellDINO vector, so it maps the fingerprint space back to images — the inverse of the Embedding step."],
      ["Conditioning on z", "the vector <i>steers</i> every denoising step; change the vector (e.g. along a knockout direction) and the decoded cell changes to match."],
      ["Forward / reverse process", "forward adds Gaussian noise to a real cell until it is pure noise (x_T); the reverse network predicts and removes that noise, guided by z, to recover a cell (x_0)."],
      ["DDIM", "<b>Denoising</b> (removes noise) · <b>Diffusion</b> (the noise process) · <b>Implicit</b> (it follows one fixed, non-random path — an ODE — rather than a random walk) · <b>Models</b>. Upshot: deterministic sampling (same seed → same cell), which makes it both invertible and much faster (fewer steps)."],
      ["Inversion (encoding)", "running DDIM backwards to recover the exact noise seed of a specific real cell, so a traversal can begin from it (guided inversion → α = 0 reconstructs it, r ≈ 0.99)."]]
  },
  {
    nav: "Traversal", kicker: "COUNTERFACTUAL", title: "\"How would this cell look if we applied a given perturbation?\"",
    svg: () => `<svg viewBox="0 0 420 200" class="mth">
      <g class="mth-morph">${_cellBlob(210, 86, 42, MTH_C.acc)}</g>
      <line x1="70" y1="158" x2="350" y2="158" stroke="#30363d" stroke-width="4"/>
      <circle cx="70" cy="158" r="5" fill="${MTH_C.ntc}"/><circle cx="350" cy="158" r="5" fill="${MTH_C.ko}"/>
      <circle cx="70" cy="158" r="8" fill="#fff" class="mth-slide"/>
      ${_lbl(70, 180, "NTC · α0")}${_lbl(350, 180, "knockout · α+", MTH_C.ko)}
      ${(() => { const zA = [8, 4, 12, 6, 10, 5], zB = [11, 7, 8, 9, 7, 8]; return "<g>" + zA.map((a, i) => `<rect x="${190 + i * 7}" y="${26 - a}" width="5" height="${a}" rx="1.5" fill="${MTH_C.pur}" class="mth-zmorph" style="--sy:${(zB[i] / a).toFixed(2)}"/>`).join("") + "</g>"; })()}${_lbl(211, 40, "z (CellDINO vector)", MTH_C.pur, 9)}
    </svg>`,
    body: "Each cell is stored as two separate parts — its <b>identity</b> (size, texture, local context) and its <b>phenotype</b> (the CellDINO semantic code). The <b>destination</b> we morph toward is the <b>centroid</b> — the average semantic code — of the perturbation's <b>top cells</b> (the very cells in the Top Cells tab). We hold identity <b>fixed</b> and slide only the phenotype along the <b>NTC → knockout direction</b> (from the control centroid to that top-cell centroid), decoding each step: α = 0 is the real cell, α = 1 lands on the knockout centroid, and beyond exaggerates it. Because identity stays locked, the same cell's size, texture, and surroundings are preserved — only the phenotype changes.",
    why: "You watch one real cell change — its size, texture, and context held constant while only the perturbation's phenotype is dialed in — so what you see is the effect of the knockout alone, not the differences between cells.",
    defs: [["Semantic direction", "in semantic-code space (see Diffusion), the vector from the NTC centroid to the centroid of the perturbation's <b>top cells</b> — the destination the traversal moves toward."],
      ["α (alpha)", "how far we push along that direction — 0 = start, 1 = full shift, |α|>1 = extrapolation."],
      ["Counterfactual", "a generated \"what this cell would look like if…\" image — not an observed one."],
      ["Classifier-free guidance (w)", "a strength knob controlling how firmly the code steers the generated image."]]
  },
  {
    nav: "DDIM", kicker: "ANCHOR TO A REAL CELL", title: "DDIM — running the model backwards",
    svg: () => `<svg viewBox="0 0 420 200" class="mth">
      ${_cellBlob(70, 92, 34, MTH_C.acc)}${_lbl(70, 150, "real cell")}
      <g class="mth-collapse">${Array.from({ length: 22 }, (_, i) => `<circle cx="${185 + (i * 37) % 60 - 30}" cy="${92 + (i * 53) % 60 - 30}" r="2" fill="#8b949e"/>`).join("")}</g>
      ${_lbl(210, 150, "its exact seed x_T")}
      ${_cellBlob(350, 92, 34, MTH_C.acc)}${_lbl(350, 150, "generated cell (r≈0.99)")}
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
    nav: "Attention heads", kicker: "INTERPRET", title: "Where does the model look?",
    svg: () => { const cell = (cx, hot, lbl) => `
      <circle cx="${cx}" cy="98" r="56" fill="rgba(38,198,255,.08)" stroke="${MTH_C.acc}" stroke-width="1.5"/>
      <circle cx="${cx - 12}" cy="92" r="21" fill="rgba(188,140,255,.4)"/>
      <circle cx="${cx - 17}" cy="88" r="4" fill="#ffffff"/><circle cx="${cx - 8}" cy="96" r="3" fill="#ffffff"/>
      ${[[cx + 16, 80, 25, "M -11 0 q 5 -8 10 -1 q 6 7 12 -1"], [cx + 24, 100, -40, "M -10 1 q 6 -7 11 0 q 4 6 10 -3"], [cx + 16, 116, 60, "M -12 -1 q 4 7 9 1 q 6 -6 12 1"], [cx + 30, 120, -12, "M -9 0 q 7 -6 12 1 q 3 6 9 -2"]].map(([x, y, rot, d]) => `<path d="${d}" fill="none" stroke="${MTH_C.ko}" stroke-width="3.2" stroke-linecap="round" opacity=".82" transform="translate(${x} ${y}) rotate(${rot})"/>`).join("")}
      ${_patchGrid(cx - 42, 56, 14, 6, hot)}
      ${_lbl(cx, 186, lbl)}`;
      return `<svg viewBox="0 0 440 200" class="mth">
        ${cell(112, [16, 17, 22, 23], "head A — tubules")}
        ${cell(330, [7, 8, 13, 14], "head B — nucleoli")}
      </svg>`; },
    body: "The vision transformer splits the cell into <b>patches</b>, and each <b>attention head</b> weights which patches it focuses on. Drawn as a heat-map, a head shows <i>which structures</i> the model attends to for a given cell — for example one that concentrates on mitochondria.",
    why: "It makes the model's focus visible and checkable against known biology, rather than leaving the phenotype call as a black box.",
    defs: [["Patch / token", "the small square pieces a vision transformer breaks the image into."],
      ["Attention head", "one of several parallel attention 'spotlights'; each weights which patches to focus on, and different heads specialize."],
      ["Attention map", "a head's per-patch weights, drawn as a heat-map — where the model concentrates (an attention pattern, suggestive of but not a formal attribution of the decision)."]]
  },
  {
    nav: "Montage", kicker: "THE MAP", title: "Every knockout, from one cell, on one map",
    svg: () => { const cx = 150, cy = 98, B = [[MTH_C.acc, -2.35], [MTH_C.ko, -0.75], [MTH_C.grn, 0.55], [MTH_C.pur, 2.0]];
      // each arm = a different phenotype axis: 0 acc grows · 1 ko elongates · 2 grn (bottom) shrinks · 3 pur nucleus grows
      // tt = 0 at the first cell (≈ the original NTC cell) → 1 at the arm tip (full phenotype); gradual divergence
      const mcell = (x, y, b, i, col, ang) => { const tt = (i - 1) / 4, lp = (a, z) => a + (z - a) * tt, hue = _mix(MTH_C.ntc, col, tt), dark = _mix(col, "#000000", .3);
        let rx, ry, nr;
        if (b === 0) { rx = ry = 5 + i * 1.2; nr = rx * 0.3; }              // acc — grows (tuned to arm spacing so cells don't overlap)
        else if (b === 1) { rx = lp(11, 15.5); ry = lp(11, 7); nr = ry * 0.34; }  // ko — elongates (never smaller)
        else if (b === 3) { rx = ry = 11; nr = lp(3.2, 8.6); }            // pur — nucleus grows
        else { rx = ry = lp(11, 5); nr = rx * 0.3; }                       // grn (bottom) — shrinks
        const ox = x + rx * 0.32, oy = y + ry * 0.3;
        return `<ellipse cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" rx="${rx.toFixed(1)}" ry="${ry.toFixed(1)}" fill="${hue}" transform="rotate(${ang} ${x.toFixed(1)} ${y.toFixed(1)})"/><circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${Math.min(nr, Math.min(rx, ry) * 0.85).toFixed(1)}" fill="rgba(0,0,0,.52)"/><ellipse cx="${ox.toFixed(1)}" cy="${oy.toFixed(1)}" rx="${(rx * 0.22).toFixed(1)}" ry="${(rx * 0.1).toFixed(1)}" fill="${dark}" opacity="${(0.15 + 0.3 * tt).toFixed(2)}" transform="rotate(${ang} ${ox.toFixed(1)} ${oy.toFixed(1)})"/>`; };
      let out = "";
      B.forEach(([col, a0], b) => { for (let i = 1; i < 6; i++) { const r = 14 + i * 26, a = a0 + i * 0.15, x = cx + r * Math.cos(a), y = cy + r * Math.sin(a) * 0.66; out += `<g class="mth-branch" style="animation-delay:${(b * 1.1 + i * 0.28).toFixed(2)}s">${mcell(x, y, b, i, col, (a * 57).toFixed(0))}</g>`; } });
      return `<svg viewBox="0 -16 460 216" class="mth">
        ${out}
        <g class="mth-branch">${_cellBlob(cx, cy, 11, MTH_C.ntc)}</g>
        ${_lbl(cx, cy + 26, "Single")}${_lbl(cx, cy + 38, "Control")}${_lbl(cx, cy + 50, "Cell")}
        ${_lbl(96, 14, "Cluster A", MTH_C.acc, 10)}${_lbl(332, 100, "Cluster B", MTH_C.ko, 10)}
        <line x1="358" y1="44" x2="274" y2="85" stroke="#8b949e" stroke-width="1" opacity=".6"/>
        ${_lbl(392, 36, "each tile = a gene-KO", "#8b949e", 9)}
        ${_lbl(230, 194, "each arm = a different phenotype axis (size · shape · nucleus); distance = how different", "#8b949e", 9.5)}
      </svg>`; },
    body: "The <b>Montage</b> tab takes a single anchor cell, traverses it toward <i>every</i> one of the ~1,000 perturbations, and drops each morphed cell at that perturbation's spot on a <b>perturbation-similarity map</b> (UMAP/PHATE). <b>LatentLens</b> tiles thousands of these crops into one zoomable montage.",
    why: "It turns 1,000 separate what-ifs into a single navigable landscape — perturbations with similar phenotypes cluster together, visible at a glance.",
    defs: [["Perturbation embedding (UMAP / PHATE)", "a 2-D map where each point is a perturbation, placed so phenotypically similar knockouts sit close together."],
      ["Anchor cell", "the one real cell (see DDIM) whose counterfactual we render for every perturbation, so the comparison is apples-to-apples."],
      ["LatentLens", "the tiling engine that lays thousands of image crops onto the map as a smooth, zoomable montage."]]
  },
  {
    nav: "Virtual staining", kicker: "CROSS-CHANNEL", title: "Predicting fluorescent stains from phase",
    svg: () => { const M = [["mitochondria", MTH_C.ko], ["ER", "#2ca089"], ["nucleus", MTH_C.acc], ["actin", MTH_C.pur], ["lysosome", MTH_C.yel]];
      const comp = (cx, cy, col, kind, s) => kind === "mitochondria" ? [["M -7 0 q 3 -5 6 -1 q 4 5 8 -1", -3, -3, 25], ["M -6 1 q 4 -5 7 0 q 3 4 7 -2", 2, 1, -40], ["M -8 -1 q 3 5 6 1 q 4 -4 8 1", -1, 4, 60]].map(([d, dx, dy, rot]) => `<path d="${d}" fill="none" stroke="${col}" stroke-width="2.4" stroke-linecap="round" opacity=".85" transform="translate(${(cx + dx * s).toFixed(1)} ${(cy + dy * s).toFixed(1)}) rotate(${rot}) scale(${s})"/>`).join("")
        : kind === "ER" ? [0, 1, 2, 3].map(k => `<circle cx="${cx - 6 * s + k * 4 * s}" cy="${cy + (k % 2) * 5 * s - 2 * s}" r="${3.4 * s}" fill="none" stroke="${col}" stroke-width="${(1.2 * s).toFixed(1)}"/>`).join("")
        : kind === "nucleus" ? `<circle cx="${cx}" cy="${cy}" r="${8 * s}" fill="${col}"/>`
        : kind === "actin" ? [0, 1, 2].map(k => `<line x1="${cx - 9 * s}" y1="${cy - 6 * s + k * 6 * s}" x2="${cx + 9 * s}" y2="${cy - 4 * s + k * 6 * s}" stroke="${col}" stroke-width="${(1.4 * s).toFixed(1)}"/>`).join("")
        : [0, 1, 2, 3, 4].map(k => `<circle cx="${cx + ((k * 5) % 16 - 8) * s}" cy="${cy + ((k * 7) % 14 - 6) * s}" r="${1.8 * s}" fill="${col}"/>`).join("");
      return `<svg viewBox="0 0 470 200" class="mth">
        ${_cellBlob(42, 100, 28, MTH_C.ntc)}
        ${M.map((m, i) => `<g class="mth-cycin" style="animation-delay:${(i * 1.0).toFixed(1)}s">${comp(42, 100, m[1], m[0], 1.5)}</g>`).join("")}
        ${_lbl(42, 146, "phase cell")}
        ${_arrow(78, 130, 100)}
        <rect x="140" y="58" width="120" height="84" rx="10" fill="rgba(38,198,255,.06)" stroke="${MTH_C.acc}" stroke-width="1.5"/>
        <text x="200" y="84" fill="#e6e8ec" font-size="11" text-anchor="middle">diffusion staining</text>
        <text x="200" y="98" fill="#e6e8ec" font-size="11" text-anchor="middle">model</text>
        <text x="200" y="118" fill="${MTH_C.pur}" font-size="8.5" text-anchor="middle">semantic (what)</text>
        <text x="200" y="131" fill="#2ca089" font-size="8.5" text-anchor="middle">+ spatial (where)</text>
        ${_arrow(264, 300, 100)}
        ${M.map((m, i) => { const y = 34 + i * 33; return `<g class="mth-cyc" style="animation-delay:${(i * 1.0).toFixed(1)}s"><circle cx="328" cy="${y}" r="15" fill="rgba(255,255,255,.05)" stroke="#30363d"/>${comp(328, y, m[1], m[0], 1)}</g><text x="354" y="${y + 4}" fill="${m[1]}" font-size="9" text-anchor="start">${m[0]}</text>`; }).join("")}
        ${_lbl(356, 197, "same cell, different stain \u2014 pick any of 42")}
      </svg>`; },
    body: "We reuse the diffusion autoencoder as a <b>virtual-staining</b> model, conditioning it on a phase image in <b>two ways</b>: a <b>semantic</b> path (phase → frozen Cell-DINO ViT → a pooled code that FiLM-conditions the U-Net, with a marker id — the <i>what</i>) and a <b>spatial</b> path (the raw phase pixels concatenated into the U-Net input — keeping the <i>layout</i>, so the output stays pixel-registered). Switching the marker id renders any of 42 fluorescent channels from the same phase cell.",
    why: "The spatial conditioning is what makes it faithful — predicted markers line up with the real cell's structures (Pearson lifts ~0.13 → ~0.78). One model covers all 42 live markers from a single label-free image; run on a traversal it yields a full multi-channel phenotype per perturbation.",
    defs: [["Virtual staining", "predicting fluorescent-marker images from a label-free phase image."],
      ["Semantic (FiLM) conditioning", "phase → frozen Cell-DINO ViT → a pooled code that globally steers the U-Net (with a marker id) — the \"what\" to render, carrying no spatial layout."],
      ["Spatial conditioning", "the raw phase image concatenated as an extra U-Net input channel, so the prediction keeps the input's layout and stays pixel-registered (fidelity lift ~0.13 → ~0.78)."],
      ["FiLM", "feature-wise modulation — how the semantic code + marker id scale the U-Net's features to select what to generate."],
      ["Marker id", "a learned token selecting which of the 42 fluorescent channels to render from the same phase cell."]]
  },
  {
    nav: "mRNA phenotypes", kicker: "THE NEXT DIRECTION", title: "From transcription to morphology",
    svg: () => { const bases = ["A", "U", "G", "C", "A", "G", "U", "C"], bcol = { A: MTH_C.grn, U: MTH_C.yel, G: MTH_C.acc, C: MTH_C.pur };
      const sx = 12, sw = 84, n = bases.length, step = sw / (n - 1), sy = 52;
      const pos = bases.map((b, i) => [sx + i * step, sy + (i % 2 ? -8 : 8)]);
      const backbone = "M " + pos.map(p => `${p[0].toFixed(1)} ${p[1].toFixed(1)}`).join(" L ");
      const beads = bases.map((b, i) => `<g><circle cx="${pos[i][0].toFixed(1)}" cy="${pos[i][1].toFixed(1)}" r="5" fill="${bcol[b]}"/><text x="${pos[i][0].toFixed(1)}" y="${(pos[i][1] + 2.8).toFixed(1)}" font-size="6" fill="#0d0f13" text-anchor="middle" font-weight="700">${b}</text></g>`).join("");
      return `<svg viewBox="0 0 460 200" class="mth">
      <text x="54" y="26" fill="${MTH_C.grn}" font-size="10" text-anchor="middle">interleukin mRNA \u2191</text>
      <g class="mth-express"><path d="${backbone}" fill="none" stroke="#8b949e" stroke-width="1.6" opacity=".7"/>${beads}</g>
      <rect x="42" y="78" width="24" height="76" rx="3" fill="rgba(255,255,255,.05)" stroke="#30363d"/>
      <rect x="42" y="78" width="24" height="76" rx="3" fill="${MTH_C.grn}" class="mth-grow"/>
      ${_lbl(54, 170, "expression")}
      ${_arrow(98, 250, 100)}
      ${_lbl(176, 86, "predict", "#8b949e", 9)}
      <g class="mth-morph">${_cellBlob(348, 100, 42, MTH_C.acc)}</g>
      ${_lbl(348, 164, "morphology follows")}
    </svg>`; },
    body: "The same library was also profiled by <b>CROP-seq</b> (single-cell RNA), so each perturbation has a <b>transcriptional signature</b> too. Next we condition the diffusion decoder on <b>transcriptional</b> change instead of a morphological direction — either by mapping a perturbation's expression shift onto a CellDINO direction (reusing this decoder), or by feeding the transcriptome straight into the model as a <b>learned perturbation vector</b> added in its latent space. Then we ask how a cell's shape should follow its gene-expression state \u2014 e.g. as <b>interleukin</b> expression rises, watch the morphology change.",
    why: "Bridging transcriptome and image lets us see which perturbations <i>decouple</i> the two (loud in RNA, silent in shape — or the reverse) — the core question of the transcriptional project.",
    defs: [["CROP-seq", "a pooled CRISPR screen read out by single-cell RNA sequencing (the transcriptome), on the same knockout library."],
      ["Transcriptional signature", "a perturbation's pseudobulk mRNA shift vs NTC (or a learned scRNA latent, or pathway module scores) — the vector that stands in for the CellDINO direction."],
      ["Reuse map (light design)", "fit transcriptional-shift → CellDINO-shift (linear, then a small MLP), then feed that predicted direction into the existing diffusion decoder — no retraining; its R² measures how predictable morphology is from transcriptome."],
      ["Learned perturbation vector", "represent each perturbation\u2019s effect as a vector added in the model\u2019s latent space; a transcriptome decoder and the image decoder share it, so any transcriptional state can be rendered. (This is the idea behind a \u201ccompositional perturbation autoencoder,\u201d CPA.)"]]
  },
];

// display order (narrative arc): data → represent → classify+interpret → generate+arrange → cross-channel → next
// core methods (numbered 1..N, count stops at Montage) then "Extra" add-on slides (unnumbered)
const MTH_ORDER = ["The screen", "Embedding", "Classifier", "Top cells", "Diffusion", "Traversal", "Montage",
  "Attention heads", "Virtual staining", "mRNA phenotypes"];   // last 3 are Extras; DDIM folded into Diffusion
const MTH_EXTRA = new Set(["Attention heads", "Virtual staining", "mRNA phenotypes"]);
const MTH_DECK = [{ nav: "About", about: true }, ...MTH_ORDER.map(n => METHODS_SLIDES.find(s => s.nav === n)).filter(Boolean)];
// public deck = About + steps 1–6 (Screen…Traversal); internal = full deck. isPublic() from app.js (loaded first).
const _mthDeck = () => (typeof isPublic === "function" && isPublic()) ? MTH_DECK.slice(0, 7) : MTH_DECK;

let _mthIdx = 0;
function renderMethods() {
  const deck = _mthDeck(), coreN = deck.filter(s => !s.about && !MTH_EXTRA.has(s.nav)).length;
  if (_mthIdx >= deck.length) _mthIdx = deck.length - 1;
  const rail = document.getElementById("tab-methods");
  if (rail && rail.querySelectorAll(".mth-railitem").length !== deck.length) {   // (re)build when the deck size changes (e.g. public toggle)
    if (!rail.dataset.inited) { const sv = +localStorage.getItem("opsin.mth"); if (sv >= 0 && sv < deck.length) _mthIdx = sv; rail.dataset.inited = "1"; }   // restore last-viewed once
    rail.innerHTML = `<div class="hint">A visual tour of the methods behind this viewer — click through, ← → to navigate.</div>
      <div class="mth-rail">${deck.map((s, i) => { const x = MTH_EXTRA.has(s.nav);
        return `<button class="mth-railitem${x ? " mth-railitem-x" : ""}" onclick="methodsGo(${i})"><span class="mth-num">${x ? "+" : i + 1}</span>${x ? "Extra · " + s.nav : s.nav}</button>`; }).join("")}</div>
      <div class="mth-rail-foot">
        <a class="mth-paper" href="https://www.biorxiv.org/content/10.64898/2026.06.01.728087v1" target="_blank" rel="noopener">📄 Liu et al., bioRxiv 2026 ↗</a>
        <div class="mth-rail-brand"><a href="https://biohub.org" target="_blank" rel="noopener" title="Chan Zuckerberg Biohub SF"><img src="biohub-mark.png?v=1" alt="Biohub"/></a><span>Biohub | CellXState</span></div>
      </div>`;
  }
  const s = deck[_mthIdx], view = document.getElementById("methods-view");
  if (!view) return;
  const nav = `<div class="mth-navbar">
      <button onclick="methodsStep(-1)" ${_mthIdx === 0 ? "disabled" : ""}>← back</button>
      <div class="mth-dots">${deck.map((_, i) => `<span class="mth-dot${i === _mthIdx ? " on" : ""}" onclick="methodsGo(${i})"></span>`).join("")}</div>
      <button onclick="methodsStep(1)" ${_mthIdx === deck.length - 1 ? "disabled" : ""}>next →</button>
    </div>`;
  if (s.about) {   // About = first section: reuse the About panel content (source kept hidden in the DOM)
    const src = document.getElementById("side-about-view");
    view.innerHTML = `<div class="mth-card mth-about">${src ? src.innerHTML : ""}${nav}</div>`;
    const pub = typeof isPublic === "function" && isPublic();
    view.querySelectorAll(".about-tabdesc").forEach(el => el.classList.toggle("feat-hidden", pub && typeof PUBLIC_HIDDEN_TABS !== "undefined" && PUBLIC_HIDDEN_TABS.has(el.dataset.tab)));
  } else {
    const refs = MTH_REFS[s.nav] || [];
    const coreIdx = deck.slice(0, _mthIdx + 1).filter(x => !x.about && !MTH_EXTRA.has(x.nav)).length;   // 1-based position among numbered slides
    view.innerHTML = `<div class="mth-card">
      <div class="mth-kicker">${s.kicker} · ${MTH_EXTRA.has(s.nav) ? "Extra" : coreIdx + " / " + coreN}</div>
      <h2 class="mth-title">${s.title}</h2>
      ${s.sections ? s.sections.map(sec => `<div class="mth-sec">${sec.cap ? `<div class="mth-seccap">${sec.cap}</div>` : ""}<div class="mth-stage">${sec.svg()}</div>${sec.text ? `<div class="mth-sectext">${sec.text}</div>` : ""}</div>`).join("") : `<div class="mth-stage">${s.svg()}</div>`}
      <div class="mth-body">${s.body}</div>
      ${MTH_MORE[s.nav] ? `<details class="mth-more"><summary>Learn more</summary><p>${MTH_MORE[s.nav]}</p></details>` : ""}
      <div class="mth-why"><b>Why it matters —</b> ${s.why}</div>
      ${refs.length ? `<div class="mth-refs">📄 ${refs.map(r => `<a href="${r[1]}" target="_blank" rel="noopener">${r[0]}</a>`).join(" &nbsp;·&nbsp; ")}</div>` : ""}
      ${(s.defs || []).length ? `<details class="mth-defs"><summary>Key terms</summary><dl>${s.defs.map(d => `<div><dt>${d[0]}</dt><dd>${d[1]}</dd></div>`).join("")}</dl></details>` : ""}
      ${nav}
    </div>`;
  }
  document.querySelectorAll(".mth-railitem").forEach((b, i) => b.classList.toggle("on", i === _mthIdx));
  const st = document.getElementById("stage"); if (st) st.scrollTop = 0;   // new slide → back to top (don't inherit the previous slide's scroll)
}
function methodsGo(i) { _mthIdx = Math.max(0, Math.min(_mthDeck().length - 1, i)); try { localStorage.setItem("opsin.mth", _mthIdx); } catch (e) { } renderMethods(); }
function methodsStep(d) { methodsGo(_mthIdx + d); }
document.addEventListener("keydown", (e) => {
  const active = document.querySelector(".tab.active");
  if (active && active.dataset.tab === "methods") { if (e.key === "ArrowRight") methodsStep(1); if (e.key === "ArrowLeft") methodsStep(-1); }
});
