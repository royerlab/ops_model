// DiffEx traversal viewer — static, reads viewer_assets/manifest.json (or window.MANIFEST_URL).
// α scrubs precomputed WebP frames. One grid: perturbation rows (current + pinned) × cells-per-page.
const MANIFEST_URL = window.MANIFEST_URL || "manifest.json";
const BASE = MANIFEST_URL.replace(/manifest\.json$/, "");
const PAD = (i) => String(i).padStart(2, "0");
const $ = (id) => document.getElementById(id);

const state = {
  manifest: null, marker: null, targets: [], target: null, anchor: "NTC",
  cellCount: 4, page: 0, pinned: [], panels: [], alphas: [],
  idx: 0, playing: false, playSeq: [], playPos: 0, frameMs: 180,   // default 1× (180ms/frame)
  showScore: true, showReal: false, scores: {},   // scores[asset_dir] = {alphas, scores[cell][ai]} | null
  pausePoints: new Set(), pauseN: -1,   // α indices where autoplay dwells (click ticks to toggle)
  rangeLo: 0, rangeHi: 0, alphaLimit: 5,   // autoplay sweeps only within ±alphaLimit (scrub stays full)
};

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
  const mkSel = $("marker");
  state.manifest.markers.forEach((m, i) => {
    const o = document.createElement("option");
    o.value = i; o.textContent = m.marker_channel || "Phase"; mkSel.appendChild(o);
  });
  mkSel.onchange = () => selectMarker(+mkSel.value);
  $("grain").onchange = refreshTargets;
  $("filter").oninput = refreshTargets;
  $("target").onchange = () => selectTarget($("target").value);
  $("cellcount").onchange = () => { state.cellCount = Math.max(1, +$("cellcount").value | 0); state.page = 0; rebuild(); };
  $("cprev").onclick = () => { state.page = Math.max(0, state.page - 1); rebuild(); };
  $("cnext").onclick = () => { state.page++; rebuild(); };
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
  document.querySelectorAll(".tab").forEach(b => b.onclick = () => {   // left-panel tabs
    document.querySelectorAll(".tab").forEach(x => x.classList.toggle("active", x === b));
    document.querySelectorAll(".tabpane").forEach(p => p.classList.toggle("hidden", p.id !== "tab-" + b.dataset.tab));
    const mont = b.dataset.tab === "montage";
    $("stage").classList.toggle("montage-active", mont);
    if (mont) ensureMontage();
  });
  for (let c = 0; c < 20; c++) { const o = document.createElement("option"); o.value = c; o.textContent = `cell ${c}`; $("m-cell").appendChild(o); }
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
  selectMarker(0);
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
const mont = { osd: null, labels: [], W: 0, mode: "both", imgAlpha: 1, ptAlpha: 0.8, detail: 0.3, field: "none", cmap: {}, centroids: {}, showLabels: false };

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
  fetch(`${base}/labels.json`).then(r => r.ok ? r.json() : []).then(l => { mont.labels = l; setField(); }).catch(() => { mont.labels = []; });
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
  sel.value = cf.includes(prev) || prev === "none" ? prev : "none";   // keep selection across α/cell/embedding switches
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
  mont.osd.addHandler("canvas-click", (e) => {   // click a cell/point → show that gene in the info sidebar
    if (!e.quick || !mont.labels.length) return;
    const best = nearestLabel(mont.osd.viewport.viewportToImageCoordinates(mont.osd.viewport.pointFromPixel(e.position)));
    if (!best) return;
    renderInfo(findTargetEntry(best.g) || { target: best.g, grain: "geneKO", desc: (state.geneDesc || {})[best.g] });
    $("sidebar").classList.remove("hidden");
  });
}

function selectMarker(i) { state.marker = state.manifest.markers[i]; refreshTargets(); }

function refreshTargets() {   // target list = the NTC-anchored phenotypes (control==null)
  const g = $("grain").value, q = $("filter").value.trim().toLowerCase();
  state.targets = state.marker.targets.filter(t =>
    !t.control && (g === "all" || t.grain === g) && t.target.toLowerCase().includes(q));
  const sel = $("target"); sel.innerHTML = "";
  state.targets.forEach(t => {
    const o = document.createElement("option");
    const m = t.dist_map != null ? ` (${t.dist_map.toFixed(2)})` : "";
    o.value = t.slug; o.textContent = `${t.target}${m}${t.grain === "complex" ? " ·cx" : ""}`;
    sel.appendChild(o);
  });
  sel.size = Math.min(Math.max(state.targets.length, 3), 10);   // ≤10 rows, scroll beyond; no black space
  if (state.targets.length) { sel.value = state.targets[0].slug; selectTarget(state.targets[0].slug); }
}

function selectTarget(slug) {
  state.target = state.targets.find(t => t.slug === slug);
  if (!state.target) return;
  populateAnchors(state.target.target);
  state.page = 0; rebuild();
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

window.addEventListener("error", (e) => { const m = $("meta"); if (m) m.textContent = "JS error: " + (e.message || e); });
boot().catch((e) => { const m = $("meta"); if (m) m.textContent = "boot error: " + (e.message || e); console.error(e); });
