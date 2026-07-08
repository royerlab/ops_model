// DiffEx traversal viewer — static, reads viewer_assets/manifest.json (or window.MANIFEST_URL).
// α scrubs precomputed WebP frames. One grid: perturbation rows (current + pinned) × cells-per-page.
const MANIFEST_URL = window.MANIFEST_URL || "manifest.json";
const BASE = MANIFEST_URL.replace(/manifest\.json$/, "");
const PAD = (i) => String(i).padStart(2, "0");
const $ = (id) => document.getElementById(id);

const state = {
  manifest: null, marker: null, markerIdx: null, targets: [], target: null, anchor: "NTC",
  cellCount: 4, page: 0, pinned: [], panels: [], alphas: [],
  idx: 0, playing: false, playSeq: [], playPos: 0, frameMs: 180,   // default 1× (180ms/frame)
  showScore: true, showReal: false, scores: {},   // scores[asset_dir] = {alphas, scores[cell][ai]} | null
  pausePoints: new Set(), pauseN: -1,   // α indices where autoplay dwells (click ticks to toggle)
  rangeLo: 0, rangeHi: 0, alphaLimit: 5,   // autoplay sweeps only within ±alphaLimit (scrub stays full)
  targetSort: "map",                       // perturbation list order: "map" (distinctiveness, default) | "alpha"
  view: "traversal",                       // active view: traversal | montage | attn (all driven by browse selection)
  attnIndex: null, attnHeadsCache: {}, attnImgCache: {},   // attention-head assets
  attnHead: "all", attnNorm: "map",     // default: show ALL heads per cell; per-cell (per-tile max) normalization
  attnClimLo: 0, attnClimHi: 1, attnAlpha: 0.6, attnImgOpacity: 1,   // clim [vmin,vmax] + constant overlay alpha (Ritvik uses 0.6) + cell-image dimming
  attnPinned: [],   // extra perturbations (geneKO) pinned for side-by-side comparison, like traversal
};

// inferno colormap (256 RGB triples, flat) — applied client-side so the attention overlay's
// normalization + opacity are live display options (no baked-in variants).
const INFERNO = Uint8ClampedArray.from([0,0,4,1,0,5,1,1,6,1,1,8,2,1,10,2,2,12,2,2,14,3,2,16,4,3,18,4,3,20,5,4,23,6,4,25,7,5,27,8,5,29,9,6,31,10,7,34,11,7,36,12,8,38,13,8,41,14,9,43,16,9,45,17,10,48,18,10,50,20,11,52,21,11,55,22,11,57,24,12,60,25,12,62,27,12,65,28,12,67,30,12,69,31,12,72,33,12,74,35,12,76,36,12,79,38,12,81,40,11,83,41,11,85,43,11,87,45,11,89,47,10,91,49,10,92,50,10,94,52,10,95,54,9,97,56,9,98,57,9,99,59,9,100,61,9,101,62,9,102,64,10,103,66,10,104,68,10,104,69,10,105,71,11,106,73,11,106,74,12,107,76,12,107,77,13,108,79,13,108,81,14,108,82,14,109,84,15,109,85,15,109,87,16,110,89,16,110,90,17,110,92,18,110,93,18,110,95,19,110,97,19,110,98,20,110,100,21,110,101,21,110,103,22,110,105,22,110,106,23,110,108,24,110,109,24,110,111,25,110,113,25,110,114,26,110,116,26,110,117,27,110,119,28,109,120,28,109,122,29,109,124,29,109,125,30,109,127,30,108,128,31,108,130,32,108,132,32,107,133,33,107,135,33,107,136,34,106,138,34,106,140,35,105,141,35,105,143,36,105,144,37,104,146,37,104,147,38,103,149,38,103,151,39,102,152,39,102,154,40,101,155,41,100,157,41,100,159,42,99,160,42,99,162,43,98,163,44,97,165,44,96,166,45,96,168,46,95,169,46,94,171,47,94,173,48,93,174,48,92,176,49,91,177,50,90,179,50,90,180,51,89,182,52,88,183,53,87,185,53,86,186,54,85,188,55,84,189,56,83,191,57,82,192,58,81,193,58,80,195,59,79,196,60,78,198,61,77,199,62,76,200,63,75,202,64,74,203,65,73,204,66,72,206,67,71,207,68,70,208,69,69,210,70,68,211,71,67,212,72,66,213,74,65,215,75,63,216,76,62,217,77,61,218,78,60,219,80,59,221,81,58,222,82,56,223,83,55,224,85,54,225,86,53,226,87,52,227,89,51,228,90,49,229,92,48,230,93,47,231,94,46,232,96,45,233,97,43,234,99,42,235,100,41,235,102,40,236,103,38,237,105,37,238,106,36,239,108,35,239,110,33,240,111,32,241,113,31,241,115,29,242,116,28,243,118,27,243,120,25,244,121,24,245,123,23,245,125,21,246,126,20,246,128,19,247,130,18,247,132,16,248,133,15,248,135,14,248,137,12,249,139,11,249,140,10,249,142,9,250,144,8,250,146,7,250,148,7,251,150,6,251,151,6,251,153,6,251,155,6,251,157,7,252,159,7,252,161,8,252,163,9,252,165,10,252,166,12,252,168,13,252,170,15,252,172,17,252,174,18,252,176,20,252,178,22,252,180,24,251,182,26,251,184,29,251,186,31,251,188,33,251,190,35,250,192,38,250,194,40,250,196,42,250,198,45,249,199,47,249,201,50,249,203,53,248,205,55,248,207,58,247,209,61,247,211,64,246,213,67,246,215,70,245,217,73,245,219,76,244,221,79,244,223,83,244,225,86,243,227,90,243,229,93,242,230,97,242,232,101,242,234,105,241,236,109,241,237,113,241,239,117,241,241,121,242,242,125,242,244,130,243,245,134,243,246,138,244,248,142,245,249,146,246,250,150,248,251,154,249,252,157,250,253,161,252,255,164]);

const PALETTE = ["#26c6ff", "#ff5252", "#f0a020", "#7ee787", "#c586ff", "#ff9edb", "#5ad1c7", "#ffd166"];
const frameURL = (dir, cell, i) => `${BASE}${dir}/cell${cell}/frame_${PAD(i)}.webp`;
const heat = (v) => {   // classifier confidence 0→1 as white → deep red (#99000d)
  const r = Math.round(255 + (153 - 255) * v), gg = Math.round(255 - 255 * v), b = Math.round(255 + (13 - 255) * v);
  return `rgb(${r},${gg},${b})`;
};
const pertOf = (markerName, t, anchor) => ({ markerName, target: t.target, anchor, slug: t.slug,
  asset_dir: t.asset_dir, alphas: t.alphas, n_cells: t.n_cells, has_real: t.has_real,
  real_dir: t.real_dir || t.asset_dir, key: markerName + "|" + t.slug });

async function boot() {
  state.manifest = await (await fetch(MANIFEST_URL)).json();
  state.geneDesc = await fetch(`${BASE}gene_desc.json`).then(r => r.ok ? r.json() : {}).catch(() => ({}));  // desc for ALL genes (incl un-cached)
  state.attnIndex = await fetch(`${BASE}attention_heads/phase/index.json`).then(r => r.ok ? r.json() : null).catch(() => null);  // attention-head availability + global_max
  wireCombo("markerfilter", "marker-list", renderMarkerList, () => markerLabel(state.markerIdx));
  wireCombo("filter", "target-list", renderTargetList, () => state.target ? targetLabel(state.target) : "");
  $("target-sort").onchange = () => { state.targetSort = $("target-sort").value; renderTargetList(); $("target-list").classList.remove("hidden"); };
  $("grain").onchange = refreshTargets;
  $("cellcount").onchange = () => { state.cellCount = Math.max(1, +$("cellcount").value | 0); state.page = 0; rebuild(); if (state.view === "attn") renderAttn(); };
  $("cprev").onclick = () => { state.page = Math.max(0, state.page - 1); rebuild(); if (state.view === "attn") renderAttn(); };
  $("cnext").onclick = () => { state.page++; rebuild(); if (state.view === "attn") renderAttn(); };
  $("anchor").onchange = () => { state.anchor = $("anchor").value; rebuild(); };
  $("addpanel").onclick = () => {
    const set = activeSet(); if (!set.length) return;
    const p = set[0];   // the current resolved (marker, anchor→target)
    if (!state.pinned.some(q => q.key === p.key)) state.pinned.push(p);
    renderPinned(); rebuild();
  };
  $("clearpanels").onclick = () => { state.pinned = []; renderPinned(); rebuild(); };
  $("alpha").oninput = () => showIdx(+$("alpha").value);
  $("play").onclick = togglePlay;
  $("showscore").onchange = () => {
    state.showScore = $("showscore").checked;
    $("score-legend").style.display = state.showScore ? "flex" : "none";
    showIdx(state.idx);
  };
  $("speed").onchange = () => { state.frameMs = +$("speed").value; };
  $("showreal").onchange = () => { state.showReal = $("showreal").checked; rebuild(); };
  $("alphalimit").onchange = () => { state.alphaLimit = +$("alphalimit").value; computeRange(); buildPlaySeq(state.alphas); };
  $("infotoggle").onclick = () => $("sidebar").classList.toggle("hidden");
  $("a-head").onchange = () => { const v = $("a-head").value; state.attnHead = v === "all" ? "all" : +v; renderAttn(); };
  $("a-norm").onchange = () => { state.attnNorm = $("a-norm").value; renderAttn(); };
  $("a-climlo").oninput = () => {   // dual-handle clim; keep lo ≤ hi
    let lo = +$("a-climlo").value; if (lo > state.attnClimHi) { lo = state.attnClimHi; $("a-climlo").value = lo; }
    state.attnClimLo = lo; renderAttn();
  };
  $("a-climhi").oninput = () => {
    let hi = +$("a-climhi").value; if (hi < state.attnClimLo) { hi = state.attnClimLo; $("a-climhi").value = hi; }
    state.attnClimHi = hi; renderAttn();
  };
  $("a-alpha").oninput = () => { state.attnAlpha = +$("a-alpha").value; renderAttn(); };
  $("a-img").oninput = () => { state.attnImgOpacity = +$("a-img").value; renderAttn(); };
  $("a-reset").onclick = () => {   // reset all attention-head display controls to defaults
    Object.assign(state, { attnHead: "all", attnNorm: "map", attnClimLo: 0, attnClimHi: 1, attnAlpha: 0.6, attnImgOpacity: 1 });
    $("a-head").value = "all"; $("a-norm").value = "map"; $("a-climlo").value = 0; $("a-climhi").value = 1;
    $("a-alpha").value = 0.6; $("a-img").value = 1;
    renderAttn();
  };
  $("a-pin").onclick = () => {   // pin the current perturbation for comparison (mirrors traversal pin)
    const g = attnCurrentGene();
    if (g && !state.attnPinned.includes(g)) { state.attnPinned.push(g); renderAttnPinned(); renderAttn(); }
  };
  $("a-pinclear").onclick = () => { state.attnPinned = []; renderAttnPinned(); renderAttn(); };
  document.querySelectorAll(".tab").forEach(b => b.onclick = () => {   // view switcher (all views share the browse selection)
    const view = b.dataset.tab; state.view = view;
    document.querySelectorAll(".tab").forEach(x => x.classList.toggle("active", x === b));
    document.querySelectorAll(".tabpane").forEach(p => p.classList.toggle("hidden", p.id !== "tab-" + view));
    $("stage").classList.toggle("montage-active", view === "montage");
    $("stage").classList.toggle("attn-active", view === "attn");
    if (view === "montage") { ensureMontage(); focusMontageOnSelection(); }
    if (view === "attn") renderAttn();
  });
  for (let c = 0; c < 20; c++) { const o = document.createElement("option"); o.value = c; o.textContent = `cell ${c}`; $("m-cell").appendChild(o); }
  $("m-cell").value = 1;   // default NTC cell = 1
  $("m-emb").onchange = loadMontage;
  $("m-alpha").onchange = loadMontage;
  $("m-cell").onchange = loadMontage;
  $("m-mode").onchange = () => setMode($("m-mode").value);
  $("m-imgalpha").oninput = () => { mont.imgAlpha = +$("m-imgalpha").value; applyLayers(); };
  $("m-ptalpha").oninput = () => { mont.ptAlpha = +$("m-ptalpha").value; drawOverlay(); };
  $("m-detail").oninput = () => {
    mont.detail = +$("m-detail").value;
    if (mont.osd) { mont.osd.minPixelRatio = mont.detail; if (mont.osd.world.getItemCount()) mont.osd.world.getItemAt(0).minPixelRatio = mont.detail; mont.osd.forceRedraw(); }
  };
  $("m-color").onchange = setField;
  $("m-labels").onchange = () => { mont.showLabels = $("m-labels").checked; drawOverlay(); };
  selectMarker(0); $("markerfilter").value = markerLabel(0);   // default = phase marker
}

// ---- Montage tab: OpenSeadragon image pyramid + synced points overlay (color-by categories) ----
const MPAL = (i, n) => `hsl(${Math.round((360 * i / Math.max(n, 1)) * 2.4) % 360},68%,60%)`;  // spread hues
const wrapLabel = (s, n = 20) => {   // word-wrap long category labels (ontology terms) at ~n chars
  const words = String(s).split(/[\s_]+/), lines = []; let cur = "";
  for (const w of words) { if (cur && (cur + " " + w).length > n) { lines.push(cur); cur = w; } else cur = cur ? cur + " " + w : w; }
  if (cur) lines.push(cur); return lines;
};
// display presets set the two opacity sliders; both layers are always drawn (faded, not hidden)
const MODES = { both: { img: 1, pt: 0.8 }, images: { img: 1, pt: 0.15 }, points: { img: 0.15, pt: 1 } };
const mont = { osd: null, labels: [], W: 0, mode: "both", imgAlpha: 1, ptAlpha: 0.8, detail: 0.7, field: "none", cmap: {}, centroids: {}, showLabels: false };

function ensureMontage() { if (!mont.osd) loadMontage(); else drawOverlay(); }
function montageBase() { return `${BASE}_montage/phase_geneKO_${$("m-emb").value}_cell${$("m-cell").value}_a${$("m-alpha").value}_tiles`; }

async function loadMontage() {
  const base = montageBase();
  const tj = await fetch(`${base}/tiles.json`).then(r => r.ok ? r.json() : null).catch(() => null);
  if (!tj) { $("m-status").textContent = "this α/cell montage isn't built yet"; if (mont.osd) mont.osd.close(); mont.labels = []; drawOverlay(); return; }
  $("m-status").textContent = `${base.split("/").pop()} · ${tj.width}×${tj.height}, ${tj.levels.length} levels`;
  $("m-embed").textContent = `Embedding: ${tj.embedding || "gene UMAP"}`;
  mont.W = tj.width;
  const maxLevel = Math.ceil(Math.log2(Math.max(tj.width, tj.height)));
  const src = { width: tj.width, height: tj.height, tileSize: tj.tileSize, tileOverlap: 0,
    minLevel: maxLevel - (tj.levels.length - 1), maxLevel,
    getTileUrl: (l, x, y) => `${base}/L${maxLevel - l}/${x}_${y}.png` };
  const dimKey = `${tj.width}x${tj.height}`;
  if (mont.osd && mont.dimKey && mont.dimKey !== dimKey) { mont.osd.destroy(); mont.osd = null; }  // aspect change (umap↔phate) → full reset, no residue
  mont.dimKey = dimKey;
  const keep = (mont.osd && mont.osd.world.getItemCount()) ? mont.osd.viewport.getBounds() : null;  // preserve viewpoint
  if (!mont.osd) {
    mont.osd = OpenSeadragon({ id: "osd", tileSources: src, showNavigationControl: false,
      crossOriginPolicy: false, gestureSettingsMouse: { clickToZoom: false, scrollToZoom: true },
      zoomPerScroll: 1.7, animationTime: 0.25, springStiffness: 9,
      minPixelRatio: mont.detail,                           // level-of-detail (live via Zoom detail slider)
      minZoomImageRatio: 0.4, maxZoomPixelRatio: 8, background: "#000" });
    ["update-viewport", "animation", "animation-finish", "resize"].forEach(ev => mont.osd.addHandler(ev, drawOverlay));
    wireHover();
  } else { mont.osd.world.removeAll(); mont.osd.open(src); }   // clear old embedding's tiles before loading new
  mont.osd.addOnceHandler("open", () => {
    if (keep) mont.osd.viewport.fitBounds(keep, true);         // α/cell switch keeps current pan/zoom
    mont.osd.world.getItemAt(0).setOpacity(mont.imgAlpha);
  });
  populateColorFields(tj.color_fields || []);
  fetch(`${base}/labels.json`).then(r => r.ok ? r.json() : []).then(l => { mont.labels = l; setField(); if (state.view === "montage") focusMontageOnSelection(); }).catch(() => { mont.labels = []; });
}

// pan the embedding to the browse-selected gene (no-op until OSD + labels are ready)
function focusMontageOnSelection() {
  if (!mont.osd || !mont.labels.length || !mont.W || !state.target) return;
  const sel = mont.labels.find(L => L.g === state.target.target);
  if (!sel) { drawOverlay(); return; }
  mont.osd.viewport.panTo(mont.osd.viewport.imageToViewportCoordinates(sel.nx * mont.W, sel.ny * mont.W), false);
  drawOverlay();
}

// color-by dropdown, populated from all anndata categorical fields; leiden/ontology resolutions grouped
function populateColorFields(cf) {
  const sel = $("m-color"), prev = sel.value;
  sel.innerHTML = '<option value="none">none</option>';
  const grp = { other: [], leiden: [], onto: [] };
  cf.forEach(f => (f.startsWith("leiden_r") ? grp.leiden : f.startsWith("top_ontology_r") ? grp.onto : grp.other).push(f));
  const add = (label, arr, fmt) => {
    if (!arr.length) return;
    const og = document.createElement("optgroup"); og.label = label;
    arr.forEach(f => { const o = document.createElement("option"); o.value = f; o.textContent = fmt ? fmt(f) : f; og.appendChild(o); });
    sel.appendChild(og);
  };
  add("fields", grp.other);
  add("Leiden resolution", grp.leiden, f => "leiden " + f.split("_r")[1]);
  add("Top ontology (per-resolution)", grp.onto, f => "ontology r" + f.split("_r")[1]);
  if (!mont.colorInit) {   // first load: default color-by = leiden_r2 (fallback to prev/none)
    mont.colorInit = true;
    sel.value = cf.includes("leiden_r2") ? "leiden_r2" : (cf.includes(prev) || prev === "none" ? prev : "none");
  } else {
    sel.value = cf.includes(prev) || prev === "none" ? prev : "none";   // keep selection across α/cell/embedding switches
  }
}

function buildCmap() {
  mont.cmap = {}; mont.centroids = {};
  if (mont.field === "none") return;
  const vals = [...new Set(mont.labels.map(L => L[mont.field]).filter(v => v))].sort();
  vals.forEach((v, i) => { mont.cmap[v] = MPAL(i, vals.length); });
  const acc = {};                                   // group centroids (image coords) for text labels
  for (const L of mont.labels) { const v = L[mont.field]; if (!v) continue; const a = acc[v] || (acc[v] = [0, 0, 0]); a[0] += L.nx * mont.W; a[1] += L.ny * mont.W; a[2]++; }
  for (const v in acc) mont.centroids[v] = { x: acc[v][0] / acc[v][2], y: acc[v][1] / acc[v][2] };
}
const colorOf = (L) => mont.field === "none" ? "#26c6ff" : (mont.cmap[L[mont.field]] || "#555");

function renderLegend() {
  const el = $("m-legend"); el.innerHTML = "";
  if (mont.field === "none") return;
  const vals = Object.keys(mont.cmap);
  el.innerHTML = `<div class="leg-hd">${mont.field} · ${vals.length}</div>`;
  vals.slice(0, 60).forEach(v => {
    const d = document.createElement("div"); d.className = "leg-i";
    d.innerHTML = `<span class="sw" style="background:${mont.cmap[v]}"></span>${v}`;
    el.appendChild(d);
  });
  if (vals.length > 60) el.insertAdjacentHTML("beforeend", `<div class="leg-i more">…+${vals.length - 60} more</div>`);
}

// points overlay synced to the OSD viewport. images mode → dots only where a gene lacks a crop; points mode → all.
function drawOverlay() {
  const cv = $("m-overlay"); if (!cv || !mont.osd) return;
  const ctx = cv.getContext("2d"), r = mont.osd.container.getBoundingClientRect();
  if (cv.width !== r.width || cv.height !== r.height) { cv.width = r.width; cv.height = r.height; }
  ctx.clearRect(0, 0, cv.width, cv.height);
  if (!mont.labels.length || !mont.W || mont.ptAlpha <= 0) return;
  ctx.globalAlpha = mont.ptAlpha;
  const rad = mont.imgAlpha < 0.5 ? 4 : 3;                        // bigger dots when images are faded (points focus)
  for (const L of mont.labels) {
    const p = mont.osd.viewport.pixelFromPoint(mont.osd.viewport.imageToViewportCoordinates(L.nx * mont.W, L.ny * mont.W), true);
    if (p.x < -5 || p.y < -5 || p.x > cv.width + 5 || p.y > cv.height + 5) continue;
    ctx.beginPath(); ctx.arc(p.x, p.y, rad, 0, 6.2832);
    ctx.fillStyle = colorOf(L); ctx.fill();
    if (L.crop) { ctx.strokeStyle = "rgba(255,255,255,.45)"; ctx.lineWidth = 0.6; ctx.stroke(); }
  }
  if (state.target) {                               // ring the browse-selected gene (embedding follows selection)
    const sel = mont.labels.find(L => L.g === state.target.target);
    if (sel) {
      const p = mont.osd.viewport.pixelFromPoint(mont.osd.viewport.imageToViewportCoordinates(sel.nx * mont.W, sel.ny * mont.W), true);
      ctx.save(); ctx.globalAlpha = 1;
      ctx.beginPath(); ctx.arc(p.x, p.y, 18, 0, 6.2832); ctx.fillStyle = "rgba(255,45,45,.18)"; ctx.fill();   // soft halo
      ctx.shadowColor = "#ff2d2d"; ctx.shadowBlur = 10;                                                       // glowing red ring
      ctx.beginPath(); ctx.arc(p.x, p.y, 14, 0, 6.2832); ctx.strokeStyle = "#ff2d2d"; ctx.lineWidth = 4; ctx.stroke();
      ctx.shadowBlur = 0;
      ctx.beginPath(); ctx.arc(p.x, p.y, 14, 0, 6.2832); ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.5; ctx.stroke();   // white outline for contrast
      ctx.restore();
    }
  }
  if (mont.showLabels && mont.field !== "none") {   // category name at each group centroid (outlined for legibility)
    ctx.globalAlpha = 1; ctx.font = "bold 13px ui-monospace,monospace"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    for (const cat in mont.centroids) {
      const c = mont.centroids[cat];
      const p = mont.osd.viewport.pixelFromPoint(mont.osd.viewport.imageToViewportCoordinates(c.x, c.y), true);
      if (p.x < 0 || p.y < 0 || p.x > cv.width || p.y > cv.height) continue;
      const lines = wrapLabel(cat, 20);
      lines.forEach((ln, li) => {
        const yy = p.y + (li - (lines.length - 1) / 2) * 15;
        ctx.lineWidth = 3; ctx.strokeStyle = "rgba(0,0,0,.85)"; ctx.strokeText(ln, p.x, yy);
        ctx.fillStyle = mont.cmap[cat] || "#fff"; ctx.fillText(ln, p.x, yy);
      });
    }
  }
  ctx.globalAlpha = 1;
}

function applyLayers() {
  if (mont.osd && mont.osd.world.getItemCount()) mont.osd.world.getItemAt(0).setOpacity(mont.imgAlpha);
  drawOverlay();
}
function setMode(m) {                              // preset → set both sliders, then apply
  mont.mode = m; const c = MODES[m] || MODES.both;
  mont.imgAlpha = c.img; mont.ptAlpha = c.pt;
  $("m-imgalpha").value = c.img; $("m-ptalpha").value = c.pt;
  applyLayers();
}
function setField() { mont.field = $("m-color").value; buildCmap(); renderLegend(); drawOverlay(); }

function nearestLabel(p) {
  let best = null, bd = Infinity;
  for (const L of mont.labels) { const dx = L.nx * mont.W - p.x, dy = L.ny * mont.W - p.y, d = dx * dx + dy * dy; if (d < bd) { bd = d; best = L; } }
  return best;
}
function findTargetEntry(gene) {   // manifest target entry for a gene (prefer phase geneKO) → info sidebar
  for (const m of state.manifest.markers) { const e = m.targets.find(t => t.target === gene && !t.control && t.grain === "geneKO"); if (e) return e; }
  for (const m of state.manifest.markers) { const e = m.targets.find(t => t.target === gene && !t.control); if (e) return e; }
  return null;
}
// clicking a montage gene selects it in the browse search box (falls back to info-only if the
// current marker/grain has no geneKO entry for it — e.g. a fluor marker lacking that gene).
function selectGeneFromMontage(gene) {
  if ($("grain").value === "complex") { $("grain").value = "all"; refreshTargets(); }   // montage genes are geneKO
  const t = state.targets.find(x => x.target === gene && x.grain === "geneKO");
  if (t) { $("filter").value = targetLabel(t); selectTarget(t.slug); }
  else renderInfo(findTargetEntry(gene) || { target: gene, grain: "geneKO", desc: (state.geneDesc || {})[gene] });
}
function wireHover() {
  const tip = $("m-tip");
  mont.osd.addHandler("canvas-exit", () => { tip.style.display = "none"; });
  mont.osd.container.addEventListener("mousemove", (e) => {
    if (!mont.labels.length || !mont.W) { tip.style.display = "none"; return; }
    const r = mont.osd.container.getBoundingClientRect();
    const best = nearestLabel(mont.osd.viewport.viewportToImageCoordinates(mont.osd.viewport.pointFromPixel(new OpenSeadragon.Point(e.clientX - r.left, e.clientY - r.top))));
    if (best) {
      const extra = mont.field !== "none" && best[mont.field] ? ` · ${best[mont.field]}` : "";
      tip.textContent = best.g + extra + (best.crop ? "" : " (no crop)");
      tip.style.display = "block"; tip.style.left = `${e.clientX + 12}px`; tip.style.top = `${e.clientY - 8}px`;
    }
  });
  mont.osd.addHandler("canvas-click", (e) => {   // click a cell/point → select that gene in the search box (+ info sidebar)
    if (!e.quick || !mont.labels.length) return;
    const best = nearestLabel(mont.osd.viewport.viewportToImageCoordinates(mont.osd.viewport.pointFromPixel(e.position)));
    if (!best) return;
    selectGeneFromMontage(best.g);
    $("sidebar").classList.remove("hidden");
  });
}

// combobox: the input shows the current selection; focus clears it to reveal the full scrollable
// (~10-row) filtered list, typing narrows it, blur restores the selection label.
function wireCombo(inputId, listId, renderList, currentLabel) {
  const inp = $(inputId), list = $(listId);
  inp.onfocus = () => { inp.value = ""; renderList(); list.classList.remove("hidden"); };
  inp.oninput = () => { renderList(); list.classList.remove("hidden"); };
  inp.onblur = () => setTimeout(() => { list.classList.add("hidden"); inp.value = currentLabel(); }, 150);
}

const markerLabel = (i) => i == null ? "" : (state.manifest.markers[i].marker_channel || "Phase");
function selectMarker(i) { state.markerIdx = i; state.marker = state.manifest.markers[i]; refreshTargets(); }
function renderMarkerList() {
  const q = $("markerfilter").value.trim().toLowerCase(), list = $("marker-list"); list.innerHTML = ""; let n = 0;
  state.manifest.markers.forEach((m, i) => {
    const name = m.marker_channel || "Phase";
    if (q && !name.toLowerCase().includes(q)) return;
    const d = document.createElement("div"); d.className = "combo-item" + (i === state.markerIdx ? " sel" : "");
    d.textContent = name; d.onmousedown = (e) => { e.preventDefault(); pickMarker(i); }; list.appendChild(d); n++;
  });
  if (!n) list.innerHTML = '<div class="combo-empty">no markers</div>';
}
function pickMarker(i) { selectMarker(i); $("markerfilter").value = markerLabel(i); $("markerfilter").blur(); $("marker-list").classList.add("hidden"); }

const targetLabel = (t) => `${t.target}${t.dist_map != null ? ` (${t.dist_map.toFixed(2)})` : ""}${t.grain === "complex" ? " ·cx" : ""}`;
function refreshTargets() {   // marker or grain changed → recompute candidates (NTC-anchored), keep/reset selection
  const g = $("grain").value;
  state.targets = state.marker.targets.filter(t => !t.control && (g === "all" || t.grain === g));
  let t = state.target && state.targets.find(x => x.slug === state.target.slug);
  if (!t) t = state.targets[0];
  if (t) { $("filter").value = targetLabel(t); selectTarget(t.slug); }
  else { state.target = null; $("filter").value = ""; rebuild(); }
}
function renderTargetList() {
  const q = $("filter").value.trim().toLowerCase(), list = $("target-list"); list.innerHTML = ""; let n = 0;
  const arr = state.targetSort === "alpha"          // manifest order is mAP-desc; alpha re-sorts by name
    ? [...state.targets].sort((a, b) => a.target.localeCompare(b.target)) : state.targets;
  arr.forEach(t => {
    if (q && !t.target.toLowerCase().includes(q)) return;
    const d = document.createElement("div"); d.className = "combo-item" + (state.target && t.slug === state.target.slug ? " sel" : "");
    d.textContent = targetLabel(t); d.onmousedown = (e) => { e.preventDefault(); pickTarget(t.slug); }; list.appendChild(d); n++;
  });
  if (!n) list.innerHTML = '<div class="combo-empty">no matches</div>';
}
function pickTarget(slug) { selectTarget(slug); const t = state.targets.find(x => x.slug === slug); $("filter").value = t ? targetLabel(t) : ""; $("filter").blur(); $("target-list").classList.add("hidden"); }

function selectTarget(slug) {
  state.target = state.targets.find(t => t.slug === slug);
  if (!state.target) return;
  populateAnchors(state.target.target);
  state.page = 0; rebuild();
  if (state.view === "attn") renderAttn();          // selection drives the attention-head view
  if (state.view === "montage") focusMontageOnSelection();   // ...and pans/highlights the embedding
}

// anchors = NTC + any class that has a precomputed A→this-target traversal in this marker.
function populateAnchors(name) {
  const anchors = ["NTC", ...new Set(state.marker.targets.filter(e => e.target === name && e.control).map(e => e.control))];
  const sel = $("anchor"); sel.innerHTML = "";
  anchors.forEach(a => {
    const o = document.createElement("option"); o.value = a;
    o.textContent = a === "NTC" ? "NTC (default)" : a; sel.appendChild(o);
  });
  if (!anchors.includes(state.anchor)) state.anchor = "NTC";
  sel.value = state.anchor;
}

function resolveEntry(name, anchor) {
  return state.marker.targets.find(e => e.target === name && (anchor === "NTC" ? !e.control : e.control === anchor));
}

// current selection is always the first row; pinned perturbations follow.
function activeSet() {
  const e = state.target ? (resolveEntry(state.target.target, state.anchor) || resolveEntry(state.target.target, "NTC")) : null;
  const cur = e ? pertOf(state.marker.marker_channel || "Phase", e, e.control || "NTC") : null;
  const set = cur ? [cur] : [];
  state.pinned.forEach(p => { if (!cur || p.key !== cur.key) set.push(p); });
  return set;
}

function renderPinned() {
  const ul = $("panellist"); ul.innerHTML = "";
  state.pinned.forEach((p, i) => {
    const li = document.createElement("li");
    li.style.color = PALETTE[(i + 1) % PALETTE.length];   // matches its grid-row color
    const a = p.anchor && p.anchor !== "NTC" ? `${p.anchor}→` : "";
    li.innerHTML = `<span>${a}${p.target} · ${p.markerName}</span>`;
    const b = document.createElement("button"); b.textContent = "✕";
    b.onclick = () => { state.pinned.splice(i, 1); renderPinned(); rebuild(); };
    li.appendChild(b); ul.appendChild(li);
  });
}

// grid = activeSet rows × cells-per-page cols, all synced to the shared α slider.
function rebuild() {
  const set = activeSet();
  const N = state.cellCount, start = state.page * N;
  const g = $("grid"); g.innerHTML = "";
  state.panels = []; let k = 0;
  set.forEach((p, gi) => {                          // one group (row) per perturbation
    const color = PALETTE[gi % PALETTE.length];
    const apfx = p.anchor && p.anchor !== "NTC" ? `${p.anchor}→` : "";
    const group = document.createElement("div"); group.className = "group";
    const hd = document.createElement("div"); hd.className = "group-hd"; hd.style.color = color;
    hd.textContent = `${apfx}${p.target} · ${p.markerName}`;   // single header, no cell count
    const cells = document.createElement("div"); cells.className = "group-cells";
    cells.style.setProperty("--cols", Math.min(N, 5));   // max 5 per row; wrap instead of stretching
    for (let c = start; c < Math.min(start + N, p.n_cells); c++) {
      const frames = p.alphas.map((_, i) => frameURL(p.asset_dir, c, i));
      frames.forEach(src => { const im = new Image(); im.src = src; });
      const panel = document.createElement("div"); panel.className = "panel" + (c === start ? " lead" : "");
      const img = document.createElement("img"); img.id = `pimg${k}`;
      if (c === start) img.style.borderLeftColor = color;      // colour bar only on left-most cell
      const badge = document.createElement("div"); badge.className = "badge"; badge.id = `pbadge${k}`; badge.style.display = "none";
      panel.appendChild(img); panel.appendChild(badge); cells.appendChild(panel);
      state.panels.push({ asset_dir: p.asset_dir, cell: c, alphas: p.alphas, frames });
      k++;
    }
    group.appendChild(hd);
    if (state.showReal && p.has_real) {          // option (a): real-cell row above the traversal
      const rl = document.createElement("div"); rl.className = "rowlbl"; rl.textContent = "real cell ↓";
      const rr = document.createElement("div"); rr.className = "group-cells"; rr.style.setProperty("--cols", Math.min(N, 5));
      for (let c = start; c < Math.min(start + N, p.n_cells); c++) {
        const rp = document.createElement("div"); rp.className = "panel";
        const rimg = document.createElement("img"); rimg.src = `${BASE}${p.real_dir}/cell${c}/real.webp`;
        rp.appendChild(rimg); rr.appendChild(rp);
      }
      const gl = document.createElement("div"); gl.className = "rowlbl"; gl.textContent = "generated α-traversal ↓";
      group.appendChild(rl); group.appendChild(rr); group.appendChild(gl);
    }
    group.appendChild(cells); g.appendChild(group);
  });
  state.alphas = state.panels.length ? state.panels[0].alphas : state.manifest.alphas;
  [...new Set(state.panels.map(p => p.asset_dir))].forEach(fetchScores);  // per-image classifier confidence

  const n = state.alphas.length;
  $("alpha").max = n - 1;
  if (state.pauseN !== n) {                        // default pauses = ends + middle (per new α grid)
    state.pausePoints = new Set([0, Math.floor(n / 2), n - 1]); state.pauseN = n;
  }
  renderTicks();
  const iP = state.alphas.indexOf(1.0), hr = $("heat-real");   // mark α=1 (true centroid) on the colorbar
  if (iP >= 0) { hr.style.left = `${(iP / (n - 1)) * 100}%`; hr.style.display = "block"; }
  else hr.style.display = "none";
  computeRange();
  const mid = Math.floor(n / 2);
  $("alpha").value = mid; buildPlaySeq(state.alphas); showIdx(mid);

  const t = state.target;
  $("meta").textContent = t ? `${set.length} row(s) × cells ${start}–${start + N - 1} · w=${state.manifest.w}` +
    (t.dist_map != null ? ` · ${t.target} dist mAP ${t.dist_map.toFixed(3)}` : "") : "no target";
  renderInfo(t);
}

function fetchScores(dir) {
  if (state.scores[dir] !== undefined) return;   // cached or pending
  state.scores[dir] = null;
  fetch(`${BASE}${dir}/scores.json`).then(r => r.ok ? r.json() : null)
    .then(j => { state.scores[dir] = j; showIdx(state.idx); }).catch(() => {});
}

function showIdx(i) {
  state.idx = i;
  state.panels.forEach((p, k) => {
    const img = $(`pimg${k}`); if (img) img.src = p.frames[Math.min(i, p.frames.length - 1)];
    const bd = $(`pbadge${k}`); if (!bd) return;
    const sc = state.scores[p.asset_dir];
    const v = state.showScore && sc && sc.scores[p.cell] ? sc.scores[p.cell][Math.min(i, sc.scores[p.cell].length - 1)] : null;
    if (v != null) {
      bd.textContent = `${Math.round(v * 100)}%`;
      bd.style.background = heat(v);                 // white → deep red by confidence
      bd.style.color = v > 0.55 ? "#fff" : "#111";
      bd.style.display = "block";
    } else bd.style.display = "none";
  });
  const n = state.alphas.length, a = state.alphas[i];
  $("alpha-read").textContent = `α = ${a.toFixed(1)}`;
  $("heat-tick").style.left = `calc(${(i / (n - 1)) * 100}% - 1.5px)`;
}

// info sidebar (wiki-style): perturbation name + grain/mAP + function/pathways or complex members.
function renderInfo(t) {
  $("info-title").textContent = t ? t.target : "";
  $("info-sub").textContent = t ? `${t.grain}` +
    (t.dist_map != null ? ` · distinctiveness mAP ${t.dist_map.toFixed(3)}` : "") : "";
  const body = $("info-body"); body.innerHTML = "";
  if (!t || !t.desc) { body.innerHTML = '<div class="empty">no annotation for this perturbation</div>'; return; }
  const sec = (lbl, txt) => {
    if (!txt) return;
    const d = document.createElement("div"); d.className = "sec";
    d.innerHTML = `<div class="sec-lbl">${lbl}</div>${txt}`;
    body.appendChild(d);
  };
  const gc = (g) => `<a href="https://www.genecards.org/cgi-bin/carddisp.pl?gene=${encodeURIComponent(g)}" target="_blank" rel="noopener">${g}</a>`;
  const s = t.desc;
  if (s.startsWith("Members")) {   // complex: members link out to GeneCards
    const mem = s.replace(/^Members \(\d+\):\s*/, "").split(", ").map(x => gc(x.trim())).join(", ");
    sec("Complex members", mem);
    return;
  }
  const g = encodeURIComponent(t.target);
  sec("Links", `<a href="https://opencell.sf.czbiohub.org/search/${g}" target="_blank" rel="noopener">OpenCell ↗</a> · <a href="https://www.genecards.org/cgi-bin/carddisp.pl?gene=${g}" target="_blank" rel="noopener">GeneCards ↗</a>`);
  const parts = s.split(" || ");   // "<function> || GO biological process: … || Reactome: … || CORUM complex: …"
  sec("Function", parts[0]);
  parts.slice(1).forEach(p => { const i = p.indexOf(": "); if (i > 0) sec(p.slice(0, i), p.slice(i + 2)); });
}

// clickable α tick marks under the slider; click toggles an autoplay pause there.
const _tip = () => $("ticktip");
function showTip(e, txt) { const tp = _tip(); tp.textContent = txt; tp.style.display = "block"; moveTip(e); }
function moveTip(e) { const tp = _tip(); tp.style.left = `${e.clientX + 10}px`; tp.style.top = `${e.clientY - 28}px`; }
function hideTip() { _tip().style.display = "none"; }

function renderTicks() {
  const n = state.alphas.length, tk = $("ticks"); tk.innerHTML = "";
  state.alphas.forEach((a, i) => {
    const t = document.createElement("div");
    t.className = "tick" + (state.pausePoints.has(i) ? " on" : "");
    t.style.left = `${(i / (n - 1)) * 100}%`;
    const lbl = () => `α=${a.toFixed(1)}${state.pausePoints.has(i) ? " · pause ✓ (click to remove)" : " · click to pause"}`;
    t.onmouseenter = (e) => showTip(e, lbl());     // instant custom tooltip (no native title delay)
    t.onmousemove = moveTip;
    t.onmouseleave = hideTip;
    t.onclick = (e) => {
      state.pausePoints.has(i) ? state.pausePoints.delete(i) : state.pausePoints.add(i);
      renderTicks(); buildPlaySeq(state.alphas); showTip(e, lbl());
    };
    tk.appendChild(t);
  });
}

// autoplay range = indices within ±alphaLimit (scrub slider is unaffected).
function computeRange() {
  const a = state.alphas, lim = state.alphaLimit;
  let lo = a.findIndex((x) => x >= -lim - 1e-9); if (lo < 0) lo = 0;
  let hi = 0; for (let i = a.length - 1; i >= 0; i--) { if (a[i] <= lim + 1e-9) { hi = i; break; } }
  state.rangeLo = lo; state.rangeHi = Math.max(hi, lo);
}

// autoplay: ping-pong within [rangeLo, rangeHi] (±alphaLimit), dwelling at pause points.
function buildPlaySeq(alphas) {
  const lo = state.rangeLo, hi = state.rangeHi, s = [];
  const hold = (v) => state.pausePoints.has(v) ? (v === lo || v === hi ? 5 : 3) : 1;
  const push = (v) => { const h = hold(v); for (let j = 0; j < h; j++) s.push(v); };
  for (let v = lo; v <= hi; v++) push(v);         // lo → hi
  for (let v = hi - 1; v >= lo; v--) push(v);     // hi → lo
  state.playSeq = s.length ? s : [Math.floor((lo + hi) / 2)]; state.playPos = 0;
}
function togglePlay() {
  state.playing = !state.playing;
  $("play").textContent = state.playing ? "❚❚" : "▶";
  if (state.playing) {
    const cur = +$("alpha").value;                 // resume the sweep from the current tick, not the left end
    const pos = state.playSeq.indexOf(cur);
    state.playPos = pos >= 0 ? pos : 0;
    tick();
  }
}
function tick() {
  if (!state.playing) return;
  const i = state.playSeq[state.playPos % state.playSeq.length];
  state.playPos++;
  $("alpha").value = i; showIdx(i);
  setTimeout(tick, state.frameMs);
}

// ---- Attention-head view: real phenotype crops + per-head inferno pixel-attribution overlay ----
// Assets = viewer/build_attention_heads.py output (phase·geneKO only). Grayscale crop + head maps
// are composited live so normalization (per-map/per-gene/fixed) + opacity are display options.
function loadAttnHeads(gene) {
  if (state.attnHeadsCache[gene] !== undefined) return Promise.resolve(state.attnHeadsCache[gene]);
  return fetch(`${BASE}attention_heads/phase/${gene}/heads.json`).then(r => r.ok ? r.json() : null)
    .catch(() => null).then(j => { state.attnHeadsCache[gene] = j; return j; });
}
function loadImg(url) {
  if (state.attnImgCache[url]) return state.attnImgCache[url];
  const p = new Promise((res, rej) => { const im = new Image(); im.onload = () => res(im); im.onerror = rej; im.src = url; });
  state.attnImgCache[url] = p; return p;
}
let _ascratch;
function imgData(im) {   // grayscale image → ImageData (a copy; safe to reuse the scratch canvas)
  const W = im.naturalWidth, H = im.naturalHeight;
  if (!_ascratch) _ascratch = document.createElement("canvas");
  _ascratch.width = W; _ascratch.height = H;
  const c = _ascratch.getContext("2d"); c.drawImage(im, 0, 0);
  return c.getImageData(0, 0, W, H);
}
function populateAttnHeadSelect(heads) {
  const sel = $("a-head"), sig = heads.gene + ":" + heads.heads.length;
  if (sel._sig === sig) return;                       // same gene's head list already populated
  sel._sig = sig; sel.innerHTML = "";
  const all = document.createElement("option"); all.value = "all"; all.textContent = "all heads"; sel.appendChild(all);
  heads.heads.forEach((hd, i) => {
    const o = document.createElement("option"); o.value = i;
    const au = hd.auroc_vs_ntc != null ? ` · AUROC ${hd.auroc_vs_ntc.toFixed(2)}` : "";
    o.textContent = `#${i + 1} · L${hd.layer}·H${hd.head}${au}`; sel.appendChild(o);
  });
  if (state.attnHead !== "all" && state.attnHead >= heads.heads.length) state.attnHead = "all";
  sel.value = String(state.attnHead);
}
// composite grayscale crop + cell-masked, clim-mapped inferno overlay (constant alpha inside the cell,
// matching Ritvik: gaussian-smoothed maps baked at build; here mask + clim + alpha are live).
async function drawAttnCell(cv, cropUrl, maskUrl, headUrl, geneMax) {
  let crop, mask, mp;
  try { [crop, mask, mp] = await Promise.all([loadImg(cropUrl), maskUrl ? loadImg(maskUrl) : null, headUrl ? loadImg(headUrl) : null]); }
  catch { return; }
  const W = crop.naturalWidth, H = crop.naturalHeight;
  cv.width = W; cv.height = H;
  const cg = imgData(crop), imgOp = state.attnImgOpacity, out = new ImageData(W, H);
  const mk = mask ? imgData(mask) : null;
  const inside = (i) => !mk || mk.data[i] >= 128;
  if (!mp) {                                          // crop-only tile
    for (let i = 0; i < out.data.length; i += 4) { const g = cg.data[i] * imgOp; out.data[i] = out.data[i + 1] = out.data[i + 2] = g; out.data[i + 3] = 255; }
    cv.getContext("2d").putImageData(out, 0, 0); return;
  }
  const mg = imgData(mp);
  let scale;                                          // head pixel (0..255) → attribution v (0..1)
  if (state.attnNorm === "gene") scale = 1 / 255;                                   // encoded per-gene already
  else if (state.attnNorm === "fixed") scale = (geneMax / state.attnIndex.global_max) / 255;   // comparable across genes
  else {                                              // per-cell = 99th percentile WITHIN the cell mask (Ritvik's vmax)
    const vals = []; for (let i = 0; i < mg.data.length; i += 4) if (inside(i)) vals.push(mg.data[i]);
    vals.sort((a, b) => a - b);
    const p99 = vals.length ? vals[Math.min(vals.length - 1, Math.floor(vals.length * 0.99))] : 0;
    scale = p99 > 0 ? 1 / p99 : 0;
  }
  const lo = state.attnClimLo, hi = Math.max(state.attnClimHi, lo + 1e-3), A = state.attnAlpha;
  for (let i = 0; i < out.data.length; i += 4) {
    const g = cg.data[i] * imgOp;                     // crop luminance
    if (!inside(i)) { out.data[i] = out.data[i + 1] = out.data[i + 2] = g; out.data[i + 3] = 255; continue; }  // no overlay outside the cell
    const v = mg.data[i] * scale;
    let t = (v - lo) / (hi - lo); if (t > 1) t = 1; else if (t < 0) t = 0;   // clim: [vmin,vmax] → full LUT
    const li = Math.round(t * 255) * 3;               // constant alpha A inside the mask (Ritvik alpha=0.6)
    out.data[i] = g * (1 - A) + INFERNO[li] * A;
    out.data[i + 1] = g * (1 - A) + INFERNO[li + 1] * A;
    out.data[i + 2] = g * (1 - A) + INFERNO[li + 2] * A;
    out.data[i + 3] = 255;
  }
  cv.getContext("2d").putImageData(out, 0, 0);
}
const attnHeadLabel = (heads, i) => { const hd = heads.heads[i]; return `#${i + 1} L${hd.layer}·H${hd.head}`; };
// the browse-selected gene, iff it has attention-head data (phase · geneKO · present in the index)
function attnCurrentGene() {
  const t = state.target, isPhase = !state.marker.marker_channel;
  const gene = t && t.grain === "geneKO" ? t.target : null;
  return (state.attnIndex && gene && isPhase && state.attnIndex.genes.includes(gene)) ? gene : null;
}
function renderAttnPinned() {   // pinned-perturbation list (mirrors the traversal pin list), color-coded
  const ul = $("a-panellist"); ul.innerHTML = "";
  state.attnPinned.forEach((g, i) => {
    const li = document.createElement("li");
    li.style.color = PALETTE[(i + 1) % PALETTE.length];   // +1: current gene owns PALETTE[0]
    li.innerHTML = `<span>${g}</span>`;
    const b = document.createElement("button"); b.textContent = "✕";
    b.onclick = () => { state.attnPinned.splice(i, 1); renderAttnPinned(); renderAttn(); };
    li.appendChild(b); ul.appendChild(li);
  });
}
async function renderAttn() {
  const grid = $("attn-grid"), lbl = $("attn-head-lbl"), status = $("a-status");
  renderAttnPinned();
  const cur = attnCurrentGene();
  if (!cur) {
    const isPhase = !state.marker.marker_channel, t = state.target;
    lbl.textContent = ""; status.textContent = "";
    grid.innerHTML = `<div class="empty">${!isPhase ? "Attention heads are phase-only (v1)."
      : !(t && t.grain === "geneKO") ? "Select a geneKO perturbation — complexes have no head rankings."
      : "No attention-head data for this gene."}</div>`;
    return;
  }
  const genes = [cur, ...state.attnPinned.filter(g => g !== cur)];   // current first, then pins
  const heads0 = await loadAttnHeads(cur);
  if (!heads0) { grid.innerHTML = '<div class="empty">failed to load attention data</div>'; return; }
  populateAttnHeadSelect(heads0);
  const nh = heads0.heads.length;
  const headMode = state.attnHead === "all" ? `all ${nh} heads` : `head ${attnHeadLabel(heads0, Math.min(state.attnHead, nh - 1))}`;
  lbl.innerHTML = `${genes.length} perturbation${genes.length > 1 ? "s" : ""} · ${headMode}` +
    `<span class="sub">rows = heads · columns = cells · left bar colors the perturbation</span>`;
  status.textContent = `norm=${state.attnNorm} · clim [${state.attnClimLo}, ${state.attnClimHi}] · α ${state.attnAlpha}`;
  grid.innerHTML = "";
  for (let gi = 0; gi < genes.length; gi++) {
    const heads = gi === 0 ? heads0 : await loadAttnHeads(genes[gi]);
    if (heads) buildAttnGroup(grid, genes[gi], PALETTE[gi % PALETTE.length], heads);
  }
}
function buildAttnGroup(grid, gene, color, heads) {   // one perturbation block: colored header + head rows
  const base = `${BASE}attention_heads/phase/${gene}`;
  const N = state.cellCount, start = state.page * N, end = Math.min(start + N, heads.n_cells);
  const ncols = Math.max(end - start, 1);
  const headIdxs = state.attnHead === "all" ? heads.heads.map((_, i) => i) : [Math.min(state.attnHead, heads.heads.length - 1)];
  const group = document.createElement("div"); group.className = "agroup";
  const hd = document.createElement("div"); hd.className = "agroup-hd"; hd.style.color = color;
  hd.textContent = `${gene} · cells ${start}–${end - 1}`;
  group.appendChild(hd);
  for (const h of headIdxs) {
    const row = document.createElement("div"); row.className = "arow";
    const rl = document.createElement("div"); rl.className = "arow-lbl"; rl.style.borderLeftColor = color;
    rl.textContent = attnHeadLabel(heads, h);
    const cells = document.createElement("div"); cells.className = "arow-cells"; cells.style.setProperty("--acols", ncols);
    for (let c = start; c < end; c++) {
      const tile = mkTile();
      cells.appendChild(tile.el);
      drawAttnCell(tile.cv, `${base}/cell${c}/crop.webp`, `${base}/cell${c}/mask.webp`, `${base}/cell${c}/head${h}.webp`, heads.gene_max);
    }
    row.appendChild(rl); row.appendChild(cells); group.appendChild(row);
  }
  grid.appendChild(group);
}
function mkTile(caption) {   // .acell wrapper with a canvas (+ optional caption)
  const el = document.createElement("div"); el.className = "acell";
  const cv = document.createElement("canvas"); el.appendChild(cv);
  if (caption) { const cap = document.createElement("div"); cap.className = "cap"; cap.textContent = caption; el.appendChild(cap); }
  return { el, cv };
}

window.addEventListener("error", (e) => { const m = $("meta"); if (m) m.textContent = "JS error: " + (e.message || e); });
boot().catch((e) => { const m = $("meta"); if (m) m.textContent = "boot error: " + (e.message || e); console.error(e); });
