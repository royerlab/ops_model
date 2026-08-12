// DiffEx traversal viewer — static, reads viewer_assets/manifest.json (or window.MANIFEST_URL).
// α scrubs precomputed WebP frames. One grid: perturbation rows (current + pinned) × cells-per-page.
const ASSET_VER = "v5";                 // v4/v5 toggle removed — the viewer always uses the v5 assets
const ASSET_PREFIX = "";                // app is served from inside the v5 assets dir — data is alongside index.html
const MANIFEST_URL = window.MANIFEST_URL || (ASSET_PREFIX + "manifest.json");
const BASE = MANIFEST_URL.replace(/manifest\.json$/, "");

// ── internal (full) vs public (demo subset) feature gating ─────────────────
// The real public deployment shows only Top cells + Traversal (+ How-it-works steps 1–6).
// Env resolution: window.VIEWER_ENV (Argus/config.js) → hostname rule → default internal.
// Prod (PR #8) serves at diffex.apps.czbiohub.org → auto public. (config.js may also set VIEWER_ENV.)
const PUBLIC_HOSTS = ["diffex.apps.czbiohub.org"];
const IS_PUBLIC_DEPLOY = (() => {
  const v = (window.VIEWER_ENV || "").toLowerCase();
  if (["public", "prod", "production"].includes(v)) return true;
  if (["internal", "staging", "rdev", "dev"].includes(v)) return false;
  return PUBLIC_HOSTS.includes(location.hostname);
})();
const PUBLIC_HIDDEN_TABS = new Set(["montage", "pc", "attn"]);   // hidden in public; Top cells/Traversal/How-it-works stay
const PUBLIC_HIDDEN_GRAINS = new Set(["minibinder", "pc"]);       // hidden from the Type selector in public
const isPublic = () => IS_PUBLIC_DEPLOY || state.publicPreview;   // publicPreview = staging-only manual toggle
function setSidePanel(m, init) {   // right panel: 'info' (selected perturbation) | 'about' (viewer overview). Re-click active → hide.
  const bar = $("sidebar");
  if (!init && state.sidePanel === m) { bar.classList.toggle("hidden"); }   // re-click toggles visibility
  else {
    state.sidePanel = m;
    $("side-info-view").style.display = m === "info" ? "" : "none";
    $("side-about-view").style.display = m === "about" ? "" : "none";
    if (!init) bar.classList.remove("hidden");
  }
  const vis = !bar.classList.contains("hidden");   // a button is blue ONLY when its panel is actually showing
  $("side-info").classList.toggle("active", vis && state.sidePanel === "info");
  $("side-about").classList.toggle("active", vis && state.sidePanel === "about");
  if (!init && typeof saveState === "function") saveState();
}
const NOCACHE = "?t=" + Date.now();   // per-load cache-bust for the small JSON metadata (manifest/index/labels/…)
                                      // so reloads always get the freshly-rebuilt data; images stay cached
const PAD = (i) => String(i).padStart(2, "0");
const $ = (id) => document.getElementById(id);

// universal image levels: rewindow all page images/canvases to [lo,hi] via the #img-levels SVG filter.
// output = clamp((in - lo)/(hi - lo)); lo=0,hi=1 → identity (no-op). Display-time RGB stretch, not raw-data.
function updateImgLevels() {
  const lo = state.imgClimLo, hi = state.imgClimHi, d = Math.max(1e-3, hi - lo);
  const slope = (1 / d).toFixed(4), icpt = (-lo / d).toFixed(4);
  for (const id of ["lvlR", "lvlG", "lvlB"]) { const f = $(id); if (f) { f.setAttribute("slope", slope); f.setAttribute("intercept", icpt); } }
}

// the single Image-brightness control lives in the active tab's left panel — under its Image-scale
// slider when present (Traversal/Top cells), else at the top of that tab's controls (hidden in How-it-works).
function placeImgClim() {
  const el = $("imgclim-field"); if (!el) return;
  const view = state.view || "traversal";
  if (view === "methods") { el.style.display = "none"; return; }
  el.style.display = "";
  const pane = $("tab-" + view); if (!pane) return;
  const scale = pane.querySelector('input[type="range"][id$="-scale"]');   // #tile-scale / #tc-scale
  if (scale) { (scale.closest("label") || scale).after(el); return; }
  const lead = [...pane.children].find(c => c.classList && c.classList.contains("hint"));   // else: below the description text, as the first control
  if (lead) lead.after(el); else pane.insertBefore(el, pane.firstChild);
}

// show/hide features for the current mode (internal vs public). Cosmetic gate — the real
// protection is that the hidden tabs' data isn't in the public S3 bucket. renderMethods()
// independently trims the deck to steps 1–6 when isPublic().
function applyFeatureGate() {
  const pub = isPublic();
  document.querySelectorAll("#tabbar .tab, .about-tabdesc").forEach(b =>
    b.classList.toggle("feat-hidden", pub && PUBLIC_HIDDEN_TABS.has(b.dataset.tab)));
  const grain = $("grain");   // hide minibinder + PC from the Type selector in public
  if (grain && grain._seg) grain._seg.querySelectorAll("button").forEach(b =>
    b.classList.toggle("feat-hidden", pub && PUBLIC_HIDDEN_GRAINS.has(b.dataset.value)));
  const ts = $("target-sort");   // public: perturbation list always sorted by SET ACC, no selector
  if (ts && ts._seg) ts._seg.classList.toggle("feat-hidden", pub);
  if (grain && pub && PUBLIC_HIDDEN_GRAINS.has(grain.value)) {   // currently on a hidden type → fall back to geneKO
    grain.value = "geneKO"; grain.dispatchEvent(new Event("change", { bubbles: true }));
  }
  if (state.manifest) updateBagUI();   // public hides the anchor-bag selector (forces multi_bag); reflect on toggle too
  document.querySelectorAll(".feat-internal").forEach(el => {   // internal-only controls (e.g. alt-anchor picker) — hide their generated toggle too
    el.classList.toggle("feat-hidden", pub);
    const cb = el.querySelector("input[type=checkbox]"); if (cb && cb._tog) cb._tog.classList.toggle("feat-hidden", pub);
  });
  document.querySelectorAll(".feat-public").forEach(el => {   // public-only controls (e.g. the classification-rank overlay toggle)
    el.classList.toggle("feat-hidden", !pub);
    const cb = el.querySelector("input[type=checkbox]"); if (cb && cb._tog) cb._tog.classList.toggle("feat-hidden", !pub);
  });
  if (pub) {   // public: traversals anchor on NTC only — drop the alt-anchor filter + reset any non-NTC anchor
    let changed = false;
    if (state.altAnchorsOnly) { state.altAnchorsOnly = false; $("altanchor").checked = false; $("altanchor")._togSync?.(); changed = true; }
    if (state.anchor !== "NTC") { state.anchor = "NTC"; changed = true; }
    const sm = $("pub-scoreoverlay") && $("pub-scoreoverlay").checked ? "rank" : "none";   // public overlay: off, or classification rank
    if (state.scoreMode !== sm) { state.scoreMode = sm; updateScoreLegend(); changed = true; }
    if (state.targetSort !== "setacc" && ts) { state.targetSort = "setacc"; ts.value = "setacc"; ts._segSync?.(); ensureSetacc(() => renderTargetList()); }   // public: always sort by SET ACC
    if (changed && state.marker && state.target) refreshTargets();
  } else if ($("scoremode") && $("scoremode").value !== state.scoreMode) {   // internal: keep the full selector in sync (e.g. after a preview round-trip)
    $("scoremode").value = state.scoreMode; $("scoremode")._segSync?.();
  }
  if (pub && PUBLIC_HIDDEN_TABS.has(state.view)) {   // current view just got hidden → fall back to Traversal
    const t = document.querySelector('#tabbar .tab[data-tab="traversal"]'); if (t) t.click();
  } else if (state.view === "methods") renderMethods();   // re-trim the deck in place
  const tg = $("envtoggle");
  if (tg) { tg.classList.toggle("pub-on", state.publicPreview);
    tg.textContent = state.publicPreview ? "👁 Previewing public" : "🔒 Preview public"; }
}

const state = {
  manifest: null, marker: null, markerIdx: null, targets: [], target: null, anchor: "NTC", sidePanel: "info",
  cellCount: 8, page: 0, pinned: [], panels: [], alphas: [],
  idx: 0, playing: false, playSeq: [], playPos: 0, frameMs: 180,   // default 1× (180ms/frame)
  scoreMode: "ptarget", showReal: false, scores: {}, scoresV5: {}, groups: [], realAcc20: null,   // scoreMode: none|linear|ptarget|rank. scores[dir]=linear per-cell; scoresV5[dir]=v5 set (bag,per-α); realAcc20[dir]=real top1@bag20

  pausePoints: new Set(), pauseN: -1,   // α indices where autoplay dwells (click ticks to toggle)
  rangeLo: 0, rangeHi: 0, alphaLimit: 5,   // autoplay sweeps only within ±alphaLimit (scrub stays full)
  targetSort: "setacc",                    // perturbation list order default: "setacc" (SetTransformer set-accuracy) | "map" | "alpha"
  publicPreview: false,                    // staging-only: preview the public feature subset without a real prod deploy
  altAnchorsOnly: false,                    // filter perturbation list to those with a non-NTC (A→B) anchor
  view: "traversal",                       // active view: traversal | montage | attn (all driven by browse selection)
  attnIndex: null, attnHeadsCache: {}, attnImgCache: {},   // attention-head assets
  attnHead: "all", attnNorm: "map",     // default: show ALL heads per cell; per-cell (per-tile max) normalization
  attnClimLo: 0, attnClimHi: 1, attnAlpha: 0.6, attnImgOpacity: 1,   // clim [vmin,vmax] + constant overlay alpha (Ritvik uses 0.6) + cell-image dimming
  imgClimLo: 0, imgClimHi: 1,   // universal display-time levels stretch on all page images/canvases (0–1 = no-op)
  attnPinned: [],   // extra perturbations (geneKO) pinned for side-by-side comparison, like traversal
};

// inferno colormap (256 RGB triples, flat) — applied client-side so the attention overlay's
// normalization + opacity are live display options (no baked-in variants).
const INFERNO = Uint8ClampedArray.from([0,0,4,1,0,5,1,1,6,1,1,8,2,1,10,2,2,12,2,2,14,3,2,16,4,3,18,4,3,20,5,4,23,6,4,25,7,5,27,8,5,29,9,6,31,10,7,34,11,7,36,12,8,38,13,8,41,14,9,43,16,9,45,17,10,48,18,10,50,20,11,52,21,11,55,22,11,57,24,12,60,25,12,62,27,12,65,28,12,67,30,12,69,31,12,72,33,12,74,35,12,76,36,12,79,38,12,81,40,11,83,41,11,85,43,11,87,45,11,89,47,10,91,49,10,92,50,10,94,52,10,95,54,9,97,56,9,98,57,9,99,59,9,100,61,9,101,62,9,102,64,10,103,66,10,104,68,10,104,69,10,105,71,11,106,73,11,106,74,12,107,76,12,107,77,13,108,79,13,108,81,14,108,82,14,109,84,15,109,85,15,109,87,16,110,89,16,110,90,17,110,92,18,110,93,18,110,95,19,110,97,19,110,98,20,110,100,21,110,101,21,110,103,22,110,105,22,110,106,23,110,108,24,110,109,24,110,111,25,110,113,25,110,114,26,110,116,26,110,117,27,110,119,28,109,120,28,109,122,29,109,124,29,109,125,30,109,127,30,108,128,31,108,130,32,108,132,32,107,133,33,107,135,33,107,136,34,106,138,34,106,140,35,105,141,35,105,143,36,105,144,37,104,146,37,104,147,38,103,149,38,103,151,39,102,152,39,102,154,40,101,155,41,100,157,41,100,159,42,99,160,42,99,162,43,98,163,44,97,165,44,96,166,45,96,168,46,95,169,46,94,171,47,94,173,48,93,174,48,92,176,49,91,177,50,90,179,50,90,180,51,89,182,52,88,183,53,87,185,53,86,186,54,85,188,55,84,189,56,83,191,57,82,192,58,81,193,58,80,195,59,79,196,60,78,198,61,77,199,62,76,200,63,75,202,64,74,203,65,73,204,66,72,206,67,71,207,68,70,208,69,69,210,70,68,211,71,67,212,72,66,213,74,65,215,75,63,216,76,62,217,77,61,218,78,60,219,80,59,221,81,58,222,82,56,223,83,55,224,85,54,225,86,53,226,87,52,227,89,51,228,90,49,229,92,48,230,93,47,231,94,46,232,96,45,233,97,43,234,99,42,235,100,41,235,102,40,236,103,38,237,105,37,238,106,36,239,108,35,239,110,33,240,111,32,241,113,31,241,115,29,242,116,28,243,118,27,243,120,25,244,121,24,245,123,23,245,125,21,246,126,20,246,128,19,247,130,18,247,132,16,248,133,15,248,135,14,248,137,12,249,139,11,249,140,10,249,142,9,250,144,8,250,146,7,250,148,7,251,150,6,251,151,6,251,153,6,251,155,6,251,157,7,252,159,7,252,161,8,252,163,9,252,165,10,252,166,12,252,168,13,252,170,15,252,172,17,252,174,18,252,176,20,252,178,22,252,180,24,251,182,26,251,184,29,251,186,31,251,188,33,251,190,35,250,192,38,250,194,40,250,196,42,250,198,45,249,199,47,249,201,50,249,203,53,248,205,55,248,207,58,247,209,61,247,211,64,246,213,67,246,215,70,245,217,73,245,219,76,244,221,79,244,223,83,244,225,86,243,227,90,243,229,93,242,230,97,242,232,101,242,234,105,241,236,109,241,237,113,241,239,117,241,241,121,242,242,125,242,244,130,243,245,134,243,246,138,244,248,142,245,249,146,246,250,150,248,251,154,249,252,157,250,253,161,252,255,164]);

const PALETTE = ["#26c6ff", "#ff5252", "#f0a020", "#7ee787", "#c586ff", "#ff9edb", "#5ad1c7", "#ffd166"];
// traversal frames/scores/anchor can switch NTC anchor pool (v5 only): accuracy = 25 hand-picked (_v5acc/), attention = v5 build (BASE). __to__ alt-anchors have no accpool → stay on BASE.
function travBase(dir) { return BASE; }   // v5 attention+accuracy cells are consolidated into one dir under BASE
const SET_MODES = ["ptarget", "rank"];   // v5 SetTransformer per-traversal (bag) score modes → row-header chip
// adaptive score-overlay legend caption per mode (see #scoremode); shown only on Traversal when scoreMode != none
const SCORE_LEGEND = {
  linear: "per-cell classifier score (NTC → knockout): 0 → 1",
  ptarget: "set accuracy for the whole bag: 0 → 100%",
  rank: "classification rank within the set: ≥100 → rank 1",
};
function updateScoreLegend() {
  const show = state.scoreMode !== "none" && (!state.view || state.view === "traversal");
  $("score-legend").style.display = show ? "flex" : "none";
  if (show) $("score-legend-txt").innerHTML = "score overlay (white → red) — " + (SCORE_LEGEND[state.scoreMode] || "");
}
function setChip(sv, i) {   // {txt, bg, fg, showReal} for the selected set-mode at α-index i, or null. Both modes use the one white→red heat.
  if (!sv || !SET_MODES.includes(state.scoreMode)) return null;
  if (state.scoreMode === "rank") {
    const arr = sv.rank_target; if (!arr) return null;
    const r = arr[Math.min(i, arr.length - 1)]; if (r == null) return null;
    const v = Math.max(0, Math.min(1, 1 - Math.log10(Math.max(1, r)) / 2));   // rank1→1 (deep red), rank10→0.5, rank≥100→0 (white)
    return { txt: `rank ${r}`, bg: heat(v), fg: v > 0.55 ? "#fff" : "#111", showReal: false };   // rank vs real-fraction differ in units → no real overlay
  }
  const arr = sv.p_target; if (!arr) return null;
  const x = arr[Math.min(i, arr.length - 1)]; if (x == null) return null;
  return { txt: `set-acc ${Math.round(x * 100)}%`, bg: heat(x), fg: x > 0.55 ? "#fff" : "#111", showReal: true };
}
const frameURL = (dir, cell, i) => `${travBase(dir)}${dir}/cell${cell}/frame_${PAD(i)}.webp`;
const heat = (v) => {   // classifier confidence 0→1 as white → deep red (#99000d)
  const r = Math.round(255 + (153 - 255) * v), gg = Math.round(255 - 255 * v), b = Math.round(255 + (13 - 255) * v);
  return `rgb(${r},${gg},${b})`;
};
const pertOf = (markerName, t, anchor) => ({ markerName, target: t.target, anchor, slug: t.slug,
  asset_dir: t.asset_dir, alphas: t.alphas, n_cells: t.n_cells, has_real: t.has_real,
  real_dir: t.real_dir || t.asset_dir, key: markerName + "|" + t.slug });

// ---- persist the browse selection + display prefs across page reloads (localStorage; works on static S3) ----
const LS_KEY = "opsin.state.v1";
let restoredAlpha = null;   // α VALUE restored from localStorage, applied on the first rebuild after a reload/version-switch
function saveState() {
  try {
    localStorage.setItem(LS_KEY, JSON.stringify({
      marker: markerLabel(state.markerIdx), grain: $("grain").value,
      target: state.target ? state.target.target : null, anchor: state.anchor,
      cellCount: state.cellCount, page: state.page, tilepx: $("tile-scale").value,
      scoreMode: state.scoreMode, showReal: state.showReal, altAnchor: state.altAnchorsOnly, sidePanel: state.sidePanel,
      cols: $("colslayout").checked, tcCols: $("tc-cols").checked, speed: $("speed").value, alphaLimit: $("alphalimit").value, view: state.view,
      bag: $("m-bag") ? $("m-bag").value : null,
      alpha: (state.alphas && state.idx != null && state.idx < state.alphas.length) ? state.alphas[state.idx] : null,   // hold α (by value) across reloads
      pinned: state.pinned.map(p => ({ target: p.target, anchor: p.anchor })),
    }));
  } catch (e) { /* private mode / quota — non-fatal */ }
}
function restoreState() {   // returns true if a saved snapshot was applied (skips the default selection)
  let s; try { s = JSON.parse(localStorage.getItem(LS_KEY) || "null"); } catch (e) { s = null; }
  if (!s) return false;
  if (s.alpha != null) restoredAlpha = s.alpha;   // consumed by the first rebuild → holds α across reload/version-switch
  // filters/prefs that affect the target list — set BEFORE marker/target resolution
  if (s.grain && s.grain !== "all") $("grain").value = s.grain;   // 'all' grain removed → fall back to default geneKO
  if (s.bag && $("m-bag")) $("m-bag").value = s.bag;   // restore anchor bag before marker resolution (montageCells depends on it)
  if (s.cols != null) { $("colslayout").checked = s.cols; $("grid").classList.toggle("cols-layout", s.cols); }
  if (s.tcCols != null) { $("tc-cols").checked = s.tcCols; $("tc-view").classList.toggle("cols-layout", s.tcCols); }
  if (s.altAnchor != null) { $("altanchor").checked = s.altAnchor; state.altAnchorsOnly = s.altAnchor; }
  if (s.scoreMode) { state.scoreMode = s.scoreMode; $("scoremode").value = s.scoreMode; updateScoreLegend(); }
  if (s.showReal != null) { $("showreal").checked = s.showReal; state.showReal = s.showReal; }
  if (s.sidePanel) setSidePanel(s.sidePanel, true);
  if (s.cellCount) { state.cellCount = s.cellCount; $("cellcount").value = s.cellCount; }
  if (s.tilepx) { $("tile-scale").value = s.tilepx; document.documentElement.style.setProperty("--tilepx", s.tilepx + "px"); }
  if (s.speed) { $("speed").value = s.speed; state.frameMs = +s.speed; }
  if (s.alphaLimit) { $("alphalimit").value = s.alphaLimit; state.alphaLimit = +s.alphaLimit; }
  if (s.anchor) state.anchor = s.anchor;   // populateAnchors keeps it if valid for the target, else resets to NTC
  let mi = state.manifest.markers.findIndex(m => (m.label || m.marker_channel || "Phase") === s.marker);
  if (mi < 0) mi = 0;
  selectMarker(mi); $("markerfilter").value = markerLabel(mi);   // refreshTargets → selects first target by default
  if (s.target) { const t = state.targets.find(x => x.target === s.target); if (t) { $("filter").value = targetLabel(t); selectTarget(t.slug); } }
  if (Array.isArray(s.pinned) && s.pinned.length) {   // re-pin same-marker comparisons
    const mc = state.marker.marker_channel || "Phase";
    state.pinned = s.pinned.map(pp => { const e = resolveEntry(pp.target, pp.anchor || "NTC"); return e ? pertOf(mc, e, e.control || "NTC") : null; }).filter(Boolean);
    state.pinned.forEach(p => { if (!tc.pinned.some(q => q.gene === p.target)) tc.pinned.push({ gene: p.target, mode: tc.mode }); });   // keep shared pins in sync on reload
    renderPinned(); rebuild();
  }
  if (s.page) { state.page = s.page; rebuild(); }   // restore the cell page (selectTarget reset it to 0)
  if (s.view && s.view !== "traversal") { const b = document.querySelector(`.tab[data-tab="${s.view}"]`); if (b) b.click(); }
  return true;
}

async function boot() {
  state.manifest = await (await fetch(MANIFEST_URL + NOCACHE)).json();
  state.geneDesc = await fetch(`${BASE}gene_desc.json${NOCACHE}`).then(r => r.ok ? r.json() : {}).catch(() => ({}));  // desc for ALL genes (incl un-cached)
  state.geneNarr = await fetch(`${BASE}gene_narrative.json${NOCACHE}`).then(r => r.ok ? r.json() : {}).catch(() => ({}));  // affinage mechanistic narratives (gene → prose), pre-fetched
  state.realAcc20 = await fetch(`${BASE}real_acc20.json${NOCACHE}`).then(r => r.ok ? r.json() : {}).catch(() => ({}));  // real-cell top1_acc@bag20 by asset_dir (feasibility ceiling)
  state.attnIndex = await fetch(`${BASE}attention_heads/index.json${NOCACHE}`).then(r => r.ok ? r.json() : null).catch(() => null);  // {global_max, assets:{modality:{grain:[keys]}}}
  mont.rmMap = await fetch(`${BASE}_montage/render_mode.json${NOCACHE}`).then(r => r.ok ? r.json() : {}).catch(() => ({}));  // per-marker renderer: tiles (per-marker montage) vs live
  ensureSetacc(() => {});   // preload set-accuracy so the default "by SET ACC" ordering is ready
  wireCombo("markerfilter", "marker-list", renderMarkerList, () => markerLabel(state.markerIdx));
  wireCombo("filter", "target-list", renderTargetList, () => state.target ? targetLabel(state.target) : "");
  $("tprev").onclick = () => stepTarget(-1);   // step through perturbations quickly
  $("tnext").onclick = () => stepTarget(1);
  $("target-sort").onchange = () => { state.targetSort = $("target-sort").value;
    const show = () => { renderTargetList(); $("target-list").classList.remove("hidden"); };
    state.targetSort === "setacc" ? ensureSetacc(show) : show(); };
  $("altanchor").onchange = () => { state.altAnchorsOnly = $("altanchor").checked; refreshTargets(); };
  $("grain").onchange = refreshTargets;
  $("cellcount").onchange = () => { state.cellCount = Math.max(1, +$("cellcount").value | 0); state.page = 0; rebuild(); if (state.view === "attn") renderAttn(); if (state.view === "top") renderTop(); };
  $("cprev").onclick = () => { state.page = Math.max(0, state.page - 1); rebuild(); if (state.view === "attn") renderAttn(); if (state.view === "top") renderTop(); };
  $("cnext").onclick = () => { state.page++; rebuild(); if (state.view === "attn") renderAttn(); if (state.view === "top") renderTop(); };
  $("anchor").onchange = () => { state.anchor = $("anchor").value; rebuild(); };
  $("addpanel").onclick = () => { const set = activeSet(); if (set.length) pinShared(set[0].target, set[0].anchor); };
  $("clearpanels").onclick = () => { state.pinned = []; tc.pinned = [{ gene: "NTC", mode: "accuracy" }]; redrawPins(); };
  $("alpha").oninput = () => showIdx(+$("alpha").value);
  $("alpha").onchange = saveState;   // persist α on release so reloads / v4↔v5 hold the current traversal position
  $("tile-scale").oninput = () => document.documentElement.style.setProperty("--tilepx", $("tile-scale").value + "px");   // traversal image scale
  $("tc-scale").oninput = () => document.documentElement.style.setProperty("--tcpx", $("tc-scale").value + "px");   // top-cells image scale
  $("colslayout").onchange = () => $("grid").classList.toggle("cols-layout", $("colslayout").checked);   // perturbations rows ↔ columns
  $("exportgif").onclick = exportGif;
  $("play").onclick = togglePlay;
  $("scoremode").onchange = () => {
    state.scoreMode = $("scoremode").value;
    updateScoreLegend();
    saveState(); showIdx(state.idx);
  };
  $("pub-scoreoverlay").onchange = () => {   // public: off ↔ classification-rank overlay
    state.scoreMode = $("pub-scoreoverlay").checked ? "rank" : "none";
    updateScoreLegend();
    saveState(); showIdx(state.idx);
  };
  $("speed").onchange = () => { state.frameMs = +$("speed").value; };
  $("showreal").onchange = () => { state.showReal = $("showreal").checked; rebuild(); };
  $("alphalimit").onchange = () => { state.alphaLimit = +$("alphalimit").value; computeRange(); buildPlaySeq(state.alphas); };
  setSidePanel(state.sidePanel, true);   // init the right-panel Info/About toggle
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
    $("a-norm")._segSync?.();
    renderAttn();
  };
  $("a-pin").onclick = () => {   // pin the current perturbation for comparison (mirrors traversal pin)
    const r = attnCurrentRef();
    if (r && !state.attnPinned.some(p => sameRef(p, r))) { state.attnPinned.push(r); renderAttnPinned(); renderAttn(); }
  };
  $("a-pinclear").onclick = () => { state.attnPinned = []; renderAttnPinned(); renderAttn(); };
  $("i-climlo").oninput = () => {   // universal image clim (dual-handle; keep lo ≤ hi)
    let lo = +$("i-climlo").value; if (lo > state.imgClimHi) { lo = state.imgClimHi; $("i-climlo").value = lo; }
    state.imgClimLo = lo; updateImgLevels();
  };
  $("i-climhi").oninput = () => {
    let hi = +$("i-climhi").value; if (hi < state.imgClimLo) { hi = state.imgClimLo; $("i-climhi").value = hi; }
    state.imgClimHi = hi; updateImgLevels();
  };
  $("i-climreset").onclick = () => {
    state.imgClimLo = 0; state.imgClimHi = 1; $("i-climlo").value = 0; $("i-climhi").value = 1; updateImgLevels();
  };
  document.querySelectorAll(".tab").forEach(b => b.onclick = () => {   // view switcher (all views share the browse selection)
    const view = b.dataset.tab; state.view = view;
    document.querySelectorAll(".tab").forEach(x => x.classList.toggle("active", x === b));
    document.querySelectorAll(".tabpane").forEach(p => p.classList.toggle("hidden", p.id !== "tab-" + view));
    placeImgClim();   // move the brightness control into this tab's controls
    $("stage").classList.toggle("montage-active", view === "montage");
    $("stage").classList.toggle("attn-active", view === "attn");
    $("stage").classList.toggle("pc-active", view === "pc");
    $("stage").classList.toggle("top-active", view === "top");
    $("stage").classList.toggle("methods-active", view === "methods");
    updateScoreLegend();   // score legend is traversal-only + adaptive to the selected overlay mode
    if (view === "montage") { updateVsUI(); if (mont.renderMode === "live") liveLoad(); else { ensureMontage(); focusMontageOnSelection(); } }
    if (view === "attn") renderAttn();
    if (view === "pc") loadPC();
    if (view === "top") loadTop();
    if (view === "methods") renderMethods();
  });
  fillCellDropdown();      // per-modality cell count (phase 45, markers 20); refilled on marker change
  fillOverlayMarkers();    // load the 42 VS marker names into the Overlay dropdown
  $("m-cell").value = 1;   // default NTC cell = 1
  $("m-alpha").value = "5";   // force default α=5 (exaggerated); overrides any browser-restored form value
  const LIVE = () => mont.renderMode === "live";
  $("m-render").onchange = setRenderMode;
  $("m-emb").onchange = () => LIVE() ? liveLoad() : loadMontage();
  $("m-alpha").onchange = () => LIVE() ? liveRefresh() : loadMontage();
  const ALPHA_MEANING = { "1": "α = 1 (centroid)", "2": "α = 2", "3": "α = 3", "4": "α = 4", "5": "α = 5 (exaggerated)" };
  const updateAlphaRead = () => { $("m-alpha-read").textContent = ALPHA_MEANING[$("m-alpha").value] || `α = ${$("m-alpha").value}`; };
  $("m-alpha").oninput = updateAlphaRead;   // live label while dragging (montage only reloads on release via onchange)
  updateAlphaRead();
  $("m-cell").onchange = () => LIVE() ? liveRefresh() : loadMontage();
  $("m-vs").onclick = () => {                                // Virtual staining toggle: tiles renderer, cell scrubbable over cells that have stained tiles (α scrubbable)
    $("m-vs").classList.toggle("active");
    const vs = vsMode(); $("m-cell").disabled = false;
    if (vs) { $("m-render").value = "tiles"; setRenderMode(); ovlSel = "__allmarkers__"; $("m-ovsel").value = overlayLabel(); }   // default overlay = all markers on entering VS
    else { ovlSel = "off"; $("m-ovsel").value = "off"; $("m-phaseoff").classList.remove("active"); }   // leaving VS clears overlay
    fillCellDropdown();                                      // VS → only cells with stained tiles; non-VS → full montage range
    updateVsUI();
    loadMontage();
  };
  wireCombo("m-ovsel", "m-ovsel-list", renderOverlayList, overlayLabel);   // searchable Overlay picker
  $("m-phaseoff").onclick = () => { $("m-phaseoff").classList.toggle("active"); loadMontage(); };   // hide/show phase base
  $("m-mode").onchange = () => { setMode($("m-mode").value); if (LIVE()) liveDraw(); };
  $("m-imgalpha").oninput = () => { mont.imgAlpha = +$("m-imgalpha").value; LIVE() ? liveDraw() : applyLayers(); };
  $("m-ovlalpha").oninput = () => { mont.ovlAlpha = +$("m-ovlalpha").value; applyLayers(); };   // stained-overlay opacity over phase
  $("m-ptalpha").oninput = () => { mont.ptAlpha = +$("m-ptalpha").value; LIVE() ? liveDraw() : drawOverlay(); };
  $("m-tilesize").oninput = () => { mont.tileSize = +$("m-tilesize").value; liveDraw(); };
  $("m-detail").oninput = () => {
    mont.detail = +$("m-detail").value;
    if (mont.osd) {                                          // apply to EVERY layer (phase base + VS overlay) so they stay in sync
      mont.osd.minPixelRatio = mont.detail;
      for (let i = 0; i < mont.osd.world.getItemCount(); i++) mont.osd.world.getItemAt(i).minPixelRatio = mont.detail;
      mont.osd.forceRedraw();
    }
  };
  wireCombo("m-color-search", "m-color-list", renderColorList, colorLabel);
  $("m-cmap").onchange = () => { mont.cmapName = $("m-cmap").value; if (isFeatField()) { renderLegend(); drawOverlay(); if (LIVE()) liveDraw(); } };
  $("pc-tfidf").onchange = () => { pc.tfidf = $("pc-tfidf").checked; if (pc.cur) showPC(pc.cur); };
  const pcSetMode = (m) => { pc.mode = m; $("pc-mode-onto").classList.toggle("active", m === "onto"); $("pc-mode-feat").classList.toggle("active", m === "feat"); $("pc-feat-opts").style.display = m === "feat" ? "" : "none"; if (pc.cur) showPC(pc.cur); };
  $("pc-mode-onto").onclick = () => pcSetMode("onto");
  $("pc-mode-feat").onclick = () => pcSetMode("feat");
  $("pc-dedup").onchange = () => { pc.dedup = $("pc-dedup").checked; buildPCList(); if (pc.cur) showPC(pc.cur); };
  $("pc-norm").onchange = () => { pc.norm = $("pc-norm").checked; buildPCList(); if (pc.cur) showPC(pc.cur); };
  $("pc-sort").onchange = () => { pc.sort = $("pc-sort").value; buildPCList(); };
  $("tc-pin").onclick = () => { if (state.target) pinShared(state.target.target); };   // shared with Traversal
  $("tc-pinclear").onclick = () => { state.pinned = []; tc.pinned = [{ gene: "NTC", mode: "accuracy" }]; redrawPins(); };
  $("tc-cols").onchange = () => $("tc-view").classList.toggle("cols-layout", $("tc-cols").checked);   // top cells rows ↔ columns
  $("tc-mask").onchange = () => { tc.mask = $("tc-mask").checked; $("tc-view").classList.toggle("masked", tc.mask); saveState(); };   // blue seg overlay on/off
  $("tc-inorm").onchange = () => { tc.inorm = $("tc-inorm").checked; renderTop(); saveState(); };   // marker-global vs per-cell intensity (fluor)
  $("tc-acc").onchange = () => { tc.showAcc = $("tc-acc").checked; ensureSetacc(renderTop); saveState(); };   // per-group classification-accuracy chip on/off
  $("tc-accbin").onchange = () => { tc.accBin = +$("tc-accbin").value; renderTop(); saveState(); };   // classifier bag size the accuracy is measured at
  $("m-labels").onchange = () => { mont.showLabels = $("m-labels").checked; drawOverlay(); };
  const onSetacc = () => {   // off / geneKO / complex × P(target)|rank — per-tile v5 set-score at the montage α (v5 cache only)
    mont.setaccMode = $("m-setacc").value;
    mont.setaccMetric = $("m-setacc-metric").value;
    const done = () => {
      if (mont.setaccMode === "complex") {                // color by EBI-complex, chip on the nearest-member dot
        if (mont.prevField == null) mont.prevField = mont.field;
        computeCxAnchors(); mont.field = "ebi_complex"; $("m-color-search").value = colorLabel(); setField();
      } else {                                            // geneKO / off — restore the prior coloring if we'd switched it
        if (mont.prevField != null) { mont.field = mont.prevField; mont.prevField = null; $("m-color-search").value = colorLabel(); setField(); }
        else drawOverlay();
      }
    };
    const need = mont.setaccMode === "geneKO" && mont.setaccMetric === "ptarget" && !mont.setacc ? ["setacc.json", j => mont.setacc = j]
      : mont.setaccMode === "geneKO" && mont.setaccMetric === "rank" && !mont.setaccRank ? ["setacc_rank.json", j => mont.setaccRank = j]
      : mont.setaccMode === "complex" && mont.setaccMetric === "ptarget" && !mont.setaccCx ? ["setacc_complex.json", j => mont.setaccCx = j]
      : mont.setaccMode === "complex" && mont.setaccMetric === "rank" && !mont.setaccCxRank ? ["setacc_complex_rank.json", j => mont.setaccCxRank = j] : null;
    if (need) fetch(`${BASE}_montage/${need[0]}${NOCACHE}`).then(r => r.ok ? r.json() : null).then(j => { need[1](j); done(); }).catch(done);
    else done();
  };
  $("m-setacc").onchange = onSetacc; $("m-setacc-metric").onchange = onSetacc;
  let _saveTimer;   // persist selection/prefs after any change settles (debounced; snapshot reads live state)
  const scheduleSave = () => { clearTimeout(_saveTimer); _saveTimer = setTimeout(saveState, 250); };
  document.addEventListener("change", scheduleSave); document.addEventListener("click", scheduleSave);
  setupBagUI();                                               // anchor-bag selector (multi_bag default) before first marker selection
  if (!restoreState()) {                                       // restore last session, else default = phase marker + HSPA5
    selectMarker(0); $("markerfilter").value = markerLabel(0);
    const defT = state.targets.find(t => t.target === "HSPA5");
    if (defT) { $("filter").value = targetLabel(defT); selectTarget(defT.slug); }
  }
  document.querySelectorAll("select[data-seg]").forEach(segmentize);   // small dropdowns → segmented pills
  document.querySelectorAll("label.chk").forEach(toggleize);           // checkboxes → off/on segmented switches
  if (IS_PUBLIC_DEPLOY) $("envtoggle").style.display = "none";   // the manual toggle is a staging-only preview aid
  else $("envtoggle").onclick = () => { state.publicPreview = !state.publicPreview; applyFeatureGate(); };
  applyFeatureGate();
  placeImgClim();        // position the brightness control in the default tab's controls
  updateScoreLegend();   // init the adaptive score-overlay legend (fresh loads default to ptarget)
  requestAnimationFrame(() => $("loading").classList.add("gone"));   // reveal the app only after first full render (no traversal flash)
}
// Turn a checkbox into an [off | <feature>] segmented switch (like Ontology/Features). The native checkbox
// stays in the DOM (hidden) so existing .checked/.onchange logic is untouched; buttons mirror + drive it.
function toggleize(label) {
  const inp = label.querySelector('input[type=checkbox]'); if (!inp || inp._tog) return;
  const onLbl = inp.dataset.on || label.textContent.trim();
  const g = document.createElement("div"); g.className = "seg-group tog"; g.title = label.textContent.trim();
  const off = document.createElement("button"), on = document.createElement("button");
  off.type = on.type = "button"; off.className = on.className = "seg"; off.textContent = "off"; on.textContent = onLbl;
  const sync = () => { off.classList.toggle("active", !inp.checked); on.classList.toggle("active", inp.checked); };
  off.onclick = () => { if (!inp.checked) return; inp.checked = false; inp.dispatchEvent(new Event("change", { bubbles: true })); sync(); };
  on.onclick = () => { if (inp.checked) return; inp.checked = true; inp.dispatchEvent(new Event("change", { bubbles: true })); sync(); };
  g.append(off, on); inp._tog = g; inp._togSync = sync;
  label.style.display = "none"; label.after(g); sync();
}
// Skin a <select> as a segmented button group without touching its logic: the native select stays in the DOM
// (hidden) so all .value/.onchange wiring keeps working; buttons just mirror it and dispatch change on click.
function segmentize(sel) {
  if (sel._seg) return;
  const g = document.createElement("div");
  g.className = "seg-group" + (sel.classList.contains("mini") ? " mini" : "");
  const opts = [...sel.options];
  const sync = () => [...g.children].forEach((b, i) => b.classList.toggle("active", opts[i].value === sel.value));
  opts.forEach(o => {
    const b = document.createElement("button");
    b.type = "button"; b.className = "seg"; b.textContent = o.textContent; b.title = o.title || o.textContent; b.dataset.value = o.value;
    b.onclick = () => { if (sel.value === o.value) return; sel.value = o.value; sel.dispatchEvent(new Event("change", { bubbles: true })); sync(); };
    g.appendChild(b);
  });
  sel.addEventListener("change", sync);   // resync if some code sets the value programmatically + dispatches
  sel._segSync = sync;                     // call after a bare `.value =` (no dispatch) to refresh the pills
  sel.style.display = "none"; sel.after(g); sel._seg = g; sync();
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
const mont = { osd: null, labels: [], W: 0, mode: "images", imgAlpha: 1, ovlAlpha: 0.55, ptAlpha: 0.15, detail: 0.3, field: "none", cmap: {}, centroids: {}, showLabels: false, setaccMode: "off", setaccMetric: "ptarget", setacc: null, setaccCx: null, setaccRank: null, setaccCxRank: null, cxAnchors: null, prevField: null, renderMode: "tiles", tileSize: 0.02, cmapName: "viridis", feat: null, colorFields: [] };
// montage set-score value → {v:0..1 for heat, txt}: P(target) as %, or target rank on the same white→red heat (rank1→red, ≥100→white)
function saVal(raw) { return mont.setaccMetric === "rank" ? { v: Math.max(0, Math.min(1, 1 - Math.log10(Math.max(1, raw)) / 2)), txt: `rank ${raw}` } : { v: raw, txt: `${Math.round(raw * 100)}%` }; }
const saData = () => mont.setaccMode === "complex" ? (mont.setaccMetric === "rank" ? mont.setaccCxRank : mont.setaccCx) : (mont.setaccMetric === "rank" ? mont.setaccRank : mont.setacc);
// continuous colormaps (10 anchors each, matplotlib) for OP/CP feature coloring
const CMAPS = {
  viridis: ["#440154", "#482878", "#3e4a89", "#31688e", "#26828e", "#1f9e89", "#35b779", "#6ece58", "#b5de2b", "#fde725"],
  inferno: ["#000004", "#1b0c41", "#4a0c6b", "#781c6d", "#a52c60", "#cf4446", "#ed6925", "#fb9a06", "#f7d13d", "#fcffa4"],
  magma: ["#000004", "#180f3d", "#451077", "#721f81", "#9f2f7f", "#cd4071", "#f1605d", "#fd9567", "#feca8d", "#fcfdbf"],
  plasma: ["#0d0887", "#47039f", "#7301a8", "#9c179e", "#bd3786", "#d8576b", "#ed7953", "#fa9e3b", "#fdc926", "#f0f921"],
};
function contColor(t, name) {   // t in 0..1 → interpolated hex from the named colormap
  const a = CMAPS[name] || CMAPS.viridis, x = Math.max(0, Math.min(1, t)) * (a.length - 1), i = Math.floor(x), f = x - i;
  if (i >= a.length - 1) return a[a.length - 1];
  const c0 = a[i], c1 = a[i + 1], h = (c, k) => parseInt(c.slice(1 + k * 2, 3 + k * 2), 16);
  const m = k => Math.round(h(c0, k) + (h(c1, k) - h(c0, k)) * f);
  return `rgb(${m(0)},${m(1)},${m(2)})`;
}
const isFeatField = () => mont.field.startsWith("feat:");
function ensureFeat(cb) {   // lazy-load the per-gene OP/CP feature values (3.4 MB) on first feature-color pick
  if (mont.feat) return cb();
  fetch(`${BASE}montage_features.json${NOCACHE}`).then(r => r.ok ? r.json() : null).then(d => {
    if (d) { d.index = {}; d.features.forEach((f, i) => d.index[f] = i); mont.feat = d; }
    cb();
  }).catch(() => cb());
}
function featColor(gene) {   // gene → its feature value → colormap; missing → dim gray
  if (!mont.feat) return "#333";
  const idx = mont.feat.index[mont.field.slice(5)], v = (mont.feat.values[gene] || [])[idx];
  return v == null ? "#333" : contColor(v, mont.cmapName);
}

function ensureMontage() { if (!mont.osd) loadMontage(); else drawOverlay(); }
// ---- anchor bag (phase-only): single_bag = disk cells 0-199 (attn+acc), multi_bag = disk cells 200-399 (multirank).
// The bag selector adds a fixed disk-cell OFFSET; display rank stays 1-based so the UI reads rank 1..N either bag.
function bagAware() { return ASSET_VER === "v5" && attnModality() === "phase" && $("m-bag") && (state.manifest.bags != null); }
function bagName() { return bagAware() ? ($("m-bag").value || (state.manifest.bags.default || "single_bag")) : "single_bag"; }
function bagCfg() { return (state.manifest.bags || {})[bagName()] || {}; }
function bagOffset() { return bagAware() ? (bagCfg().offset || 0) : 0; }
function diskCell(c) { return bagOffset() + c; }   // display rank (0-based) → disk cell index
// alt-anchor (A→B) dirs only have single-bag cells 0-44 — the multibag offset applies ONLY to NTC-anchored traversals
function diskCellFor(anchor, c) { return (anchor && anchor !== "NTC") ? c : diskCell(c); }

// montage is per-marker: phase (default) or the selected fluor marker's own gene embedding
// montage cell count = how many cells were built per modality: v5 phase single_bag=45, multi_bag=50, else 20
function montageCells() { return (ASSET_VER === "v5" && attnModality() === "phase") ? (bagCfg().montage_cells || 45) : 20; }
function vsCells() {   // disk cells that actually have VS montage_vs tiles, per bag (single_bag=cell1; multi_bag=200-209)
  return bagName() === "multi_bag" ? Array.from({ length: 10 }, (_, i) => 200 + i) : [1];
}
function fillCellDropdown() {
  const sel = $("m-cell"), prev = +sel.value;
  sel.innerHTML = "";
  if (vsMode()) {                                    // VS: only cells that have stained tiles — option value = disk cell, label = rank
    const cells = vsCells();
    cells.forEach((dc, i) => { const o = document.createElement("option"); o.value = dc; o.textContent = `cell ${i + 1}`; sel.appendChild(o); });
    sel.value = cells.includes(prev) ? prev : cells[0];
  } else {
    const n = montageCells();
    for (let c = 0; c < n; c++) { const o = document.createElement("option"); o.value = c; o.textContent = `cell ${c}`; sel.appendChild(o); }
    sel.value = Math.min(prev || 0, n - 1);
  }
}
let ovlItems = [{ value: "off", label: "off" }, { value: "__allmarkers__", label: "All markers" }];   // Overlay combo entries
let ovlSel = "off";                                                                                    // current overlay selection
function fillOverlayMarkers() {   // load the 42 VS marker names into the Overlay combo (after off / All markers)
  fetch(`${BASE}_montage_vs/vs_markers.json${NOCACHE}`).then(r => r.ok ? r.json() : []).then(ms => {
    ovlItems = ovlItems.slice(0, 2).concat(ms.map(m => ({ value: m.slug, label: m.label })));
  }).catch(() => {});
}
const overlayLabel = () => (ovlItems.find(o => o.value === ovlSel) || ovlItems[0]).label;
function renderOverlayList() {
  const q = $("m-ovsel").value.trim().toLowerCase(), list = $("m-ovsel-list"); list.innerHTML = "";
  ovlItems.filter(o => !q || o.label.toLowerCase().includes(q)).slice(0, 60).forEach(o => {
    const d = document.createElement("div"); d.className = "combo-item" + (o.value === ovlSel ? " sel" : "");
    d.textContent = o.label; d.title = o.label; d.onmousedown = (e) => { e.preventDefault(); pickOverlay(o.value); }; list.appendChild(d);
  });
}
function pickOverlay(value) {
  ovlSel = value;
  if (value !== "off" && !vsMode()) enterVsMode();                 // choosing an overlay implies VS mode
  $("m-ovsel").value = overlayLabel(); $("m-ovsel").blur(); $("m-ovsel-list").classList.add("hidden");
  loadMontage();
}
function enterVsMode() { $("m-vs").classList.add("active"); $("m-cell").disabled = false; $("m-render").value = "tiles"; setRenderMode(); fillCellDropdown(); }
function vsMode() { return !!($("m-vs") && $("m-vs").classList.contains("active")); }
function updateVsUI() {   // Virtual staining is phase-only; its overlay options appear only once VS is toggled on
  const ph = attnModality() === "phase", vo = $("vs-opts");
  if ($("m-vs")) $("m-vs").style.display = ph ? "" : "none";
  if (vo) vo.style.display = (ph && vsMode()) ? "" : "none";
}
function setupBagUI() {   // populate the anchor-bag selector from manifest.bags (default = manifest.bags.default), once at boot
  const sel = $("m-bag"), bags = state.manifest.bags; if (!sel || !bags) return;
  sel.innerHTML = "";
  for (const id of Object.keys(bags)) {
    if (id === "default") continue;
    const o = document.createElement("option"); o.value = id; o.textContent = bags[id].label || id; sel.appendChild(o);
  }
  sel.value = bags.default || sel.options[0]?.value || "single_bag";
  sel.onchange = () => { state.page = 0; fillCellDropdown(); rebuild(); if (state.view === "montage") (mont.renderMode === "live" ? liveRefresh() : loadMontage()); saveState(); };
}
function bagAvailable(bagId) {   // multi_bag was built ONLY for geneKO + complex; single_bag exists for every grain (incl. minibinder/PC)
  if (bagId !== "multi_bag") return true;
  const g = $("grain").value; return g === "geneKO" || g === "complex";
}
function updateBagUI() {   // bag is phase-only; grey out + auto-switch away from a bag the current grain lacks so the user never sees blank tiles
  const f = $("bag-field"), sel = $("m-bag"); if (!f) return false;
  if (isPublic()) {   // public: no bag selector — always multi_bag (built for geneKO + complex, the only public grains)
    f.style.display = "none";
    if (sel && state.manifest.bags && sel.value !== "multi_bag" && bagAvailable("multi_bag")) { sel.value = "multi_bag"; return true; }
    return false;
  }
  const show = state.manifest.bags && attnModality() === "phase";
  f.style.display = show ? "" : "none";
  if (!show || !sel) return false;
  for (const o of sel.options) o.disabled = !bagAvailable(o.value);      // grey out bags this grain doesn't have
  if (!bagAvailable(sel.value)) {                                        // current bag missing for this selection → switch to an available one
    const alt = [...sel.options].find(o => bagAvailable(o.value));
    if (alt && alt.value !== sel.value) { sel.value = alt.value; return true; }
  }
  return false;
}
function overlaySel() { return ovlSel; }       // "off" | "__allmarkers__" | marker slug
function phaseHidden() { return !!($("m-phaseoff") && $("m-phaseoff").classList.contains("active")); }
function overlayActive() { return vsMode() && overlaySel() !== "off"; }
function vsCell() { const v = +$("m-cell").value; return vsCells().includes(v) ? v : vsCells()[0]; }   // selected VS disk cell (scrubbable over cells that have stained tiles)
function phaseMontageBase() { return `${BASE}_montage/phase_geneKO_${$("m-emb").value}_cell${vsCell()}_a${$("m-alpha").value}_tiles`; }
function overlayBase() {   // the stained layer selected in the Overlay dropdown (all-markers merge or a single marker)
  return `${BASE}_montage_vs/${overlaySel()}_${$("m-emb").value}_cell${vsCell()}_a${$("m-alpha").value}_tiles`;
}
function montageBase() {
  if (overlayActive()) return overlayBase();   // top stained layer (phase base added separately in loadMontage)
  if (vsMode()) {   // representative VS cell per bag, α scrubbable. Phase = the input traversal montage; markers → stained tiles
    if (attnModality() === "phase")
      return phaseMontageBase();
    return `${BASE}_montage_vs/${attnModality()}_${$("m-emb").value}_cell${vsCell()}_a${$("m-alpha").value}_tiles`;
  }
  return `${BASE}_montage/${attnModality()}_geneKO_${$("m-emb").value}_cell${diskCell(+$("m-cell").value)}_a${$("m-alpha").value}_tiles`;
}

const osdSrc = (base, tj) => {   // OSD tileSource from a montage base URL + its tiles.json
  const maxLevel = Math.ceil(Math.log2(Math.max(tj.width, tj.height)));
  return { width: tj.width, height: tj.height, tileSize: tj.tileSize, tileOverlap: 0,
    minLevel: maxLevel - (tj.levels.length - 1), maxLevel,
    getTileUrl: (l, x, y) => `${base}/L${maxLevel - l}/${x}_${y}.png` };
};
async function loadMontage() {
  const base = montageBase();                                 // top layer (stained overlay, or the single montage)
  const tj = await fetch(`${base}/tiles.json${NOCACHE}`).then(r => r.ok ? r.json() : null).catch(() => null);
  if (!tj) { $("m-status").textContent = `no embedding montage built for ${state.marker.marker_channel || "Phase"} · ${$("m-emb").value} · cell ${$("m-cell").value} · α${$("m-alpha").value}`; if (mont.osd) mont.osd.close(); mont.labels = []; drawOverlay(); return; }
  // Layers (bottom→top). Overlay mode = phase base (unless hidden) + stained top at the image-opacity slider.
  // OSD honors per-specifier `opacity` in a tileSources array, so no manual setOpacity/add-item timing needed.
  const layers = [];
  if (overlayActive() && !phaseHidden()) {
    const pj = await fetch(`${phaseMontageBase()}/tiles.json${NOCACHE}`).then(r => r.ok ? r.json() : null).catch(() => null);
    if (pj) layers.push({ tileSource: osdSrc(phaseMontageBase(), pj), opacity: 1 });
  }
  layers.push({ tileSource: osdSrc(base, tj), opacity: overlayActive() ? mont.ovlAlpha : mont.imgAlpha });   // top = stained overlay (low α so phase shows) / single montage
  $("m-status").textContent = `${overlayActive() ? (phaseHidden() ? "" : "phase + ") + overlaySel().replace("__allmarkers__", "all markers") + " · " : ""}${base.split("/").pop()} · ${tj.width}×${tj.height}, ${tj.levels.length} levels`;
  $("m-embed").textContent = `Embedding: ${tj.embedding || "perturbation UMAP"}`;
  mont.W = tj.width;
  const dimKey = `${tj.width}x${tj.height}`;
  if (mont.osd && mont.dimKey && mont.dimKey !== dimKey) { mont.osd.destroy(); mont.osd = null; }  // aspect change (umap↔phate) → full reset, no residue
  mont.dimKey = dimKey;
  const keep = (mont.osd && mont.osd.world.getItemCount()) ? mont.osd.viewport.getBounds() : null;  // preserve viewpoint
  if (!mont.osd) {
    mont.osd = OpenSeadragon({ id: "osd", tileSources: layers, showNavigationControl: false,
      crossOriginPolicy: false, gestureSettingsMouse: { clickToZoom: false, scrollToZoom: true },
      zoomPerScroll: 1.7, animationTime: 0.25, springStiffness: 9,
      minPixelRatio: mont.detail,                           // level-of-detail (live via Zoom detail slider)
      minZoomImageRatio: 0.4, maxZoomPixelRatio: 8, background: "#000" });
    ["update-viewport", "animation", "animation-finish", "resize"].forEach(ev => mont.osd.addHandler(ev, drawOverlay));
    wireHover();
  } else { mont.osd.close(); mont.osd.open(layers); }        // replace all layers
  mont.osd.addOnceHandler("open", () => {
    if (keep) mont.osd.viewport.fitBounds(keep, true);        // α/cell switch keeps current pan/zoom
    layers.forEach((l, i) => { const it = mont.osd.world.getItemAt(i); if (it) { it.setOpacity(l.opacity); it.minPixelRatio = mont.detail; } });   // every layer opens at the current Zoom-detail so base + overlay stay synced
    drawOverlay();
  });
  populateColorFields(tj.color_fields || []);
  fetch(`${base}/labels.json${NOCACHE}`).then(r => r.ok ? r.json() : []).then(l => { mont.labels = l; setField(); if (state.view === "montage") focusMontageOnSelection(); }).catch(() => { mont.labels = []; });
}

// ---- Live embedding mode: place the EXISTING cache frames at gene coords on a pan/zoom canvas ----
// No montage precompute, no image duplication — reads the shared layout JSON + the traversal frames.
const live = { cv: null, ctx: null, byEmb: {}, layout: [], byGene: {}, scale: 1, tx: 0, ty: 0,
               imgCache: new Map(), cmap: null, raf: 0, drag: null };
const lsx = (nx) => nx * live.scale + live.tx, lsy = (ny) => ny * live.scale + live.ty;
function liveEnsure() {
  if (live.cv) return;
  live.cv = $("m-live"); live.ctx = live.cv.getContext("2d");
  live.cv.addEventListener("mousedown", (e) => { live.drag = { x: e.clientX, y: e.clientY, tx: live.tx, ty: live.ty }; live.cv.classList.add("drag"); });
  window.addEventListener("mouseup", () => { live.drag = null; live.cv.classList.remove("drag"); });
  live.cv.addEventListener("mousemove", (e) => {
    if (live.drag) { live.tx = live.drag.tx + (e.clientX - live.drag.x); live.ty = live.drag.ty + (e.clientY - live.drag.y); liveDraw(); return; }
    liveHover(e);
  });
  live.cv.addEventListener("wheel", (e) => {
    e.preventDefault(); const r = live.cv.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
    const f = Math.exp(-e.deltaY * 0.0015), wx = (mx - live.tx) / live.scale, wy = (my - live.ty) / live.scale;
    live.scale *= f; live.tx = mx - wx * live.scale; live.ty = my - wy * live.scale; liveDraw();
  }, { passive: false });
  live.cv.addEventListener("click", (e) => { const b = liveNearest(e); if (b) { selectGeneFromMontage(b); $("sidebar").classList.remove("hidden"); } });
  window.addEventListener("resize", () => { if (state.view === "montage" && mont.renderMode === "live") { liveResize(); liveDraw(); } });
}
async function liveLoad() {   // (re)fetch this marker's layout for the current embedding, fit, draw
  liveEnsure();
  const emb = $("m-emb").value;
  const m = state.marker && state.marker.marker_channel;
  const slug = (!m || /^phase/i.test(m)) ? "" : jsSlug(m);
  const key = slug ? `${slug}_${emb}` : emb;
  if (!live.byEmb[key]) {
    live.byEmb[key] = await fetch(`${BASE}_montage/layout_${key}.json${NOCACHE}`).then(r => r.ok ? r.json() : null).catch(() => null)
      || await fetch(`${BASE}_montage/layout_${emb}.json${NOCACHE}`).then(r => r.ok ? r.json() : { genes: [], color_fields: [] }).catch(() => ({ genes: [], color_fields: [] }));
  }
  const L = live.byEmb[key];
  live.layout = L.genes || []; live.byGene = {}; live.layout.forEach(g => live.byGene[g.g] = g);
  populateColorFields(L.color_fields || []);
  live.cmap = null; live.imgCache.clear();
  liveResize(); liveFit(); liveDraw();
}
function liveRefresh() { live.imgCache.clear(); liveDraw(); }   // marker/cell/α changed (keep pan/zoom)
function liveResize() { const r = $("montage-view").getBoundingClientRect(); live.cv.width = r.width; live.cv.height = r.height; }
function liveFit() { const s = Math.min(live.cv.width, live.cv.height) * 0.85; live.scale = s; live.tx = (live.cv.width - s) / 2; live.ty = (live.cv.height - s) / 2; }
function liveSchedule() { if (live.raf) return; live.raf = requestAnimationFrame(() => { live.raf = 0; liveDraw(); }); }
function liveImg(u) { let im = live.imgCache.get(u); if (im) return im; im = new Image(); im.onload = liveSchedule; im.src = u; live.imgCache.set(u, im); return im; }
const liveTargets = () => state.marker ? state.marker.targets.filter(t => !t.control && t.grain === "geneKO") : [];
function liveAi() { const a = +$("m-alpha").value, arr = state.manifest.alphas, i = arr.indexOf(a); return i < 0 ? arr.length - 1 : i; }
function liveColor(g) {
  if (mont.field === "none") return "#26c6ff";
  if (isFeatField()) return featColor(g.g);
  if (!live.cmap) { live.cmap = {}; const vals = [...new Set(live.layout.map(x => x[mont.field]).filter(Boolean))].sort(); vals.forEach((v, i) => live.cmap[v] = MPAL(i, vals.length)); }
  return live.cmap[g[mont.field]] || "#555";
}
function liveDraw() {
  if (!live.ctx) return;
  const ctx = live.ctx, W = live.cv.width, H = live.cv.height;
  ctx.fillStyle = "#000"; ctx.fillRect(0, 0, W, H);
  if (!live.layout.length) return;
  const cell = +$("m-cell").value, iop = mont.imgAlpha, pop = mont.ptAlpha, ai = liveAi();
  const cw = mont.tileSize * live.scale;           // world-pinned: tiles grow/shrink with zoom (like the montage)
  if (iop > 0) {                                   // place each present gene's cache frame at its coord
    ctx.globalAlpha = iop;
    for (const t of liveTargets()) {
      const g = live.byGene[t.target]; if (!g || (t.n_cells || 0) <= cell) continue;
      const x = lsx(g.nx) - cw / 2, y = lsy(g.ny) - cw / 2;
      if (x > W || y > H || x + cw < 0 || y + cw < 0) continue;
      const im = liveImg(`${BASE}${t.asset_dir}/cell${diskCell(cell)}/frame_${PAD(ai)}.webp`);
      if (im.complete && im.naturalWidth) ctx.drawImage(im, x, y, cw, cw);
    }
    const ntc = liveImg(`${BASE}${attnModality()}/_anchors/NTC/cell${diskCell(cell)}/real.webp`);   // the NTC anchor cell
    if (ntc.complete && ntc.naturalWidth) for (const g of live.layout) {
      if (!g.g.startsWith("NTC")) continue;
      const x = lsx(g.nx) - cw / 2, y = lsy(g.ny) - cw / 2;
      if (x > W || y > H || x + cw < 0 || y + cw < 0) continue;
      ctx.drawImage(ntc, x, y, cw, cw);            // same anchor cell shown at each NTC group node
    }
    ctx.globalAlpha = 1;
    if (cw >= 34) {                                // perturbation title at each crop's top-left (once readable)
      ctx.font = "600 10px ui-monospace,monospace"; ctx.textBaseline = "top";
      const lab = (name, x, y) => { ctx.lineWidth = 3; ctx.strokeStyle = "rgba(0,0,0,.85)"; ctx.strokeText(name, x, y); ctx.fillStyle = "#fff"; ctx.fillText(name, x, y); };
      for (const t of liveTargets()) {
        const g = live.byGene[t.target]; if (!g || (t.n_cells || 0) <= cell) continue;
        const x = lsx(g.nx) - cw / 2, y = lsy(g.ny) - cw / 2;
        if (x > W || y > H || x + cw < 0 || y + cw < 0) continue;
        lab(t.target, x + 3, y + 2);
      }
      for (const g of live.layout) { if (!g.g.startsWith("NTC")) continue; const x = lsx(g.nx) - cw / 2, y = lsy(g.ny) - cw / 2; if (x > W || y > H || x + cw < 0 || y + cw < 0) continue; lab("NTC", x + 3, y + 2); }
    }
  }
  if (pop > 0) {                                   // colored dots (all layout nodes)
    ctx.globalAlpha = pop;
    for (const g of live.layout) { const x = lsx(g.nx), y = lsy(g.ny); if (x < 0 || y < 0 || x > W || y > H) continue; ctx.beginPath(); ctx.arc(x, y, 2.5, 0, 6.2832); ctx.fillStyle = liveColor(g); ctx.fill(); }
    ctx.globalAlpha = 1;
  }
  if (state.target) {                              // selection ring
    const g = live.byGene[state.target.target];
    if (g) { const x = lsx(g.nx), y = lsy(g.ny); ctx.beginPath(); ctx.arc(x, y, 10, 0, 6.2832); ctx.strokeStyle = "#ff2d2d"; ctx.lineWidth = 3; ctx.stroke(); ctx.beginPath(); ctx.arc(x, y, 10, 0, 6.2832); ctx.strokeStyle = "#fff"; ctx.lineWidth = 1; ctx.stroke(); }
  }
}
function liveNearestXY(mx, my) { let b = null, bd = 1e9; for (const g of live.layout) { const dx = lsx(g.nx) - mx, dy = lsy(g.ny) - my, d = dx * dx + dy * dy; if (d < bd) { bd = d; b = g; } } return { g: b, d: bd }; }
function liveNearest(e) { const r = live.cv.getBoundingClientRect(), n = liveNearestXY(e.clientX - r.left, e.clientY - r.top); return n.g && n.d < 900 ? n.g.g : null; }
function liveHover(e) {
  const tip = $("m-tip"), r = live.cv.getBoundingClientRect(), n = liveNearestXY(e.clientX - r.left, e.clientY - r.top);
  if (n.g && n.d < 400) { const ex = mont.field !== "none" && n.g[mont.field] ? ` · ${n.g[mont.field]}` : ""; tip.textContent = n.g.g + ex; tip.style.display = "block"; tip.style.left = `${e.clientX + 12}px`; tip.style.top = `${e.clientY - 8}px`; }
  else tip.style.display = "none";
}
function setRenderMode() {   // toggle between precomputed tile montage and live cache-frame placement
  mont.renderMode = $("m-render").value;
  $("montage-view").classList.toggle("live", mont.renderMode === "live");
  $("tab-montage").classList.toggle("liverender", mont.renderMode === "live");   // swap renderer-specific controls
  if (state.view !== "montage") return;
  if (mont.renderMode === "live") liveLoad(); else { ensureMontage(); focusMontageOnSelection(); }
}

// pan the embedding to the browse-selected gene (no-op until OSD + labels are ready)
function focusMontageOnSelection() {
  if (!mont.osd || !mont.labels.length || !mont.W || !state.target) return;
  const sel = mont.labels.find(L => L.g === state.target.target);
  if (!sel) { drawOverlay(); return; }
  mont.osd.viewport.panTo(mont.osd.viewport.imageToViewportCoordinates(sel.nx * mont.W, sel.ny * mont.W), false);
  drawOverlay();
}

// color-by is a searchable combo over categorical anndata fields + OP/CP morphometric features
function populateColorFields(cf) {
  mont.colorFields = cf || [];
  if (!mont.colorInit) {   // first load: default color-by = leiden_r2 (fallback to none)
    mont.colorInit = true;
    mont.field = mont.colorFields.includes("leiden_r2") ? "leiden_r2" : "none";
  } else if (mont.field !== "none" && !isFeatField() && !mont.colorFields.includes(mont.field)) {
    mont.field = "none";   // field vanished for this montage
  }
  $("m-color-search").value = colorLabel();
  $("m-cmap-row").style.display = isFeatField() ? "" : "none";
}
function colorLabel() {   // human label for the current field (feature base name / leiden n / ontology / raw)
  const f = mont.field;
  if (f === "none") return "none";
  if (f.startsWith("feat:")) return f.slice(5);
  if (f.startsWith("leiden_r")) return "leiden " + f.split("_r")[1];
  if (f.startsWith("top_ontology_r")) return "ontology r" + f.split("_r")[1];
  return f;
}
function renderColorList() {
  const q = $("m-color-search").value.trim().toLowerCase(), list = $("m-color-list"); list.innerHTML = "";
  const item = (value, label, sub) => {
    const d = document.createElement("div"); d.className = "combo-item" + (value === mont.field ? " sel" : "");
    d.innerHTML = label + (sub ? ` <span class="ci-sub">${sub}</span>` : "");
    d.title = sub ? `${label} · ${sub}` : label;
    d.onmousedown = (e) => { e.preventDefault(); pickColor(value); }; list.appendChild(d);
  };
  const hd = t => { const d = document.createElement("div"); d.className = "combo-hd"; d.textContent = t; list.appendChild(d); };
  if (!q || "none".includes(q)) item("none", "none");
  const cats = mont.colorFields.filter(f => !q || f.toLowerCase().includes(q) || colorLabelOf(f).toLowerCase().includes(q));
  if (cats.length) { hd("fields"); cats.slice(0, 40).forEach(f => item(f, colorLabelOf(f))); }
  const feats = mont.feat ? mont.feat.features : null;   // features load lazily; prompt until then
  if (feats) {
    const m = q ? feats.filter(f => f.toLowerCase().includes(q)) : feats;
    if (m.length) { hd(`OP/CP features · ${m.length}`); m.slice(0, 60).forEach(f => item("feat:" + f, f)); }
    if (m.length > 60) list.insertAdjacentHTML("beforeend", `<div class="combo-empty">…refine to see ${m.length - 60} more</div>`);
  } else {
    hd("OP/CP features"); const d = document.createElement("div"); d.className = "combo-empty";
    d.textContent = "loading… (type to search once ready)"; list.appendChild(d);
    ensureFeat(() => { if (!$("m-color-list").classList.contains("hidden")) renderColorList(); });
  }
}
const colorLabelOf = (f) => f.startsWith("leiden_r") ? "leiden " + f.split("_r")[1] : f.startsWith("top_ontology_r") ? "ontology r" + f.split("_r")[1] : f;
function pickColor(value) {
  const apply = () => { mont.field = value; $("m-color-search").value = colorLabel(); $("m-color-search").blur();
    $("m-color-list").classList.add("hidden"); $("m-cmap-row").style.display = isFeatField() ? "" : "none";
    live.cmap = null; setField(); if (LIVE()) liveDraw(); };
  if (value.startsWith("feat:")) ensureFeat(apply); else apply();
}

function buildCmap() {
  mont.cmap = {}; mont.centroids = {};
  if (mont.field === "none" || isFeatField()) return;   // features use a continuous colormap, no categorical map
  const vals = [...new Set(mont.labels.map(L => L[mont.field]).filter(v => v))].sort();
  vals.forEach((v, i) => { mont.cmap[v] = MPAL(i, vals.length); });
  const acc = {};                                   // group centroids (image coords) for text labels
  for (const L of mont.labels) { const v = L[mont.field]; if (!v) continue; const a = acc[v] || (acc[v] = [0, 0, 0]); a[0] += L.nx * mont.W; a[1] += L.ny * mont.W; a[2]++; }
  for (const v in acc) mont.centroids[v] = { x: acc[v][0] / acc[v][2], y: acc[v][1] / acc[v][2] };
}
const colorOf = (L) => mont.field === "none" ? "#26c6ff" : isFeatField() ? featColor(L.g) : (mont.cmap[L[mont.field]] || "#555");

function computeCxAnchors() {   // per EBI-complex: the member geneKO dot nearest the members' centroid → where the complex chip sits
  const grp = {};
  for (const L of mont.labels) { const c = L.ebi_complex; if (c) (grp[c] = grp[c] || []).push(L); }
  mont.cxAnchors = {};
  for (const c in grp) {
    const ms = grp[c], cx = ms.reduce((s, L) => s + L.nx, 0) / ms.length, cy = ms.reduce((s, L) => s + L.ny, 0) / ms.length;
    let best = ms[0], bd = Infinity;
    for (const L of ms) { const d = (L.nx - cx) ** 2 + (L.ny - cy) ** 2; if (d < bd) { bd = d; best = L; } }
    mont.cxAnchors[c] = { nx: best.nx, ny: best.ny, gene: best.g };
  }
}

function renderLegend() {
  const el = $("m-legend"); el.innerHTML = "";
  if (mont.field === "none") return;
  if (isFeatField()) {   // continuous gradient bar with the feature's real 2–98 pct range
    const base = mont.field.slice(5), rng = (mont.feat && mont.feat.range[base]) || [0, 1];
    const stops = CMAPS[mont.cmapName].join(",");
    el.innerHTML = `<div class="leg-hd">${base}</div>` +
      `<div class="leg-grad" style="background:linear-gradient(90deg,${stops})"></div>` +
      `<div class="leg-cont"><span>${(+rng[0]).toPrecision(3)}</span><span>${(+rng[1]).toPrecision(3)}</span></div>`;
    return;
  }
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
  const aKey = String(+$("m-alpha").value);
  if (mont.setaccMode === "geneKO" && saData()) {   // per-tile geneKO set-score (P(target) gen%/real% | rank) at the montage α
    const data = saData();
    ctx.globalAlpha = 1; ctx.font = "700 11px ui-monospace,monospace"; ctx.textAlign = "left"; ctx.textBaseline = "top";
    for (const L of mont.labels) {
      const s = data[L.g], raw = s ? s[aKey] : null; if (raw == null) continue;
      const p = mont.osd.viewport.pixelFromPoint(mont.osd.viewport.imageToViewportCoordinates(L.nx * mont.W, L.ny * mont.W), true);
      if (p.x < -30 || p.y < -30 || p.x > cv.width + 30 || p.y > cv.height + 30) continue;
      const { v, txt: base } = saVal(raw);
      const rl = mont.setaccMetric !== "rank" && state.realAcc20 ? state.realAcc20["phase/geneKO/" + L.g] : null;
      const txt = mont.setaccMetric === "rank" ? base : `${Math.round(raw * 100)}${rl != null ? "/" + Math.round(rl * 100) : ""}`;
      const pad = 3, bw = ctx.measureText(txt).width + pad * 2;
      ctx.fillStyle = heat(v);
      if (ctx.roundRect) { ctx.beginPath(); ctx.roundRect(p.x + 5, p.y - 7, bw, 14, 3); ctx.fill(); } else ctx.fillRect(p.x + 5, p.y - 7, bw, 14);
      ctx.fillStyle = v > 0.55 ? "#fff" : "#111"; ctx.fillText(txt, p.x + 5 + pad, p.y - 5);
    }
  }
  if (mont.setaccMode === "complex" && saData() && mont.cxAnchors) {   // one chip per complex on its nearest-member dot
    const data = saData();
    ctx.globalAlpha = 1; ctx.font = "800 13px ui-monospace,monospace"; ctx.textAlign = "left"; ctx.textBaseline = "top";
    for (const c in mont.cxAnchors) {
      const s = data[c], raw = s ? s[aKey] : null; if (raw == null) continue;
      const A = mont.cxAnchors[c];
      const p = mont.osd.viewport.pixelFromPoint(mont.osd.viewport.imageToViewportCoordinates(A.nx * mont.W, A.ny * mont.W), true);
      if (p.x < -40 || p.y < -40 || p.x > cv.width + 40 || p.y > cv.height + 40) continue;
      const { v, txt } = saVal(raw), pad = 4, bw = ctx.measureText(txt).width + pad * 2;
      ctx.fillStyle = heat(v);
      if (ctx.roundRect) { ctx.beginPath(); ctx.roundRect(p.x + 5, p.y - 9, bw, 18, 4); ctx.fill(); } else ctx.fillRect(p.x + 5, p.y - 9, bw, 18);
      ctx.fillStyle = v > 0.55 ? "#fff" : "#111"; ctx.fillText(txt, p.x + 5 + pad, p.y - 6);
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
  if (mont.osd && mont.osd.world.getItemCount()) {           // top layer = overlay-opacity (overlay mode) or image-opacity (single); phase base stays opaque
    mont.osd.world.getItemAt(mont.osd.world.getItemCount() - 1).setOpacity(overlayActive() ? mont.ovlAlpha : mont.imgAlpha);
  }
  drawOverlay();
}
function setMode(m) {                              // preset → set both sliders, then apply
  mont.mode = m; const c = MODES[m] || MODES.both;
  mont.imgAlpha = c.img; mont.ptAlpha = c.pt;
  $("m-imgalpha").value = c.img; $("m-ptalpha").value = c.pt;
  applyLayers();
}
function setField() { buildCmap(); renderLegend(); drawOverlay(); }   // mont.field is set by pickColor/populateColorFields

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
  if ($("grain").value === "complex") { $("grain").value = "geneKO"; $("grain")._segSync?.(); refreshTargets(); }   // montage genes are geneKO
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
      const aK = String(+$("m-alpha").value); let scoreStr = ""; const data = saData();
      if (mont.setaccMode === "geneKO" && data && data[best.g]) {
        const raw = data[best.g][aK];
        if (raw != null) { const rl = mont.setaccMetric !== "rank" && state.realAcc20 ? state.realAcc20["phase/geneKO/" + best.g] : null;
          scoreStr = mont.setaccMetric === "rank" ? ` | ${saVal(raw).txt}` : ` | set-acc ${Math.round(raw * 100)}%${rl != null ? ` / real ${Math.round(rl * 100)}%` : ""}`; }
      } else if (mont.setaccMode === "complex" && data && best.ebi_complex && data[best.ebi_complex]) {
        const raw = data[best.ebi_complex][aK];
        if (raw != null) scoreStr = mont.setaccMetric === "rank" ? ` | ${best.ebi_complex} ${saVal(raw).txt}` : ` | ${best.ebi_complex} set-acc ${Math.round(raw * 100)}%`;
      }
      tip.textContent = best.g + scoreStr + extra + (best.crop ? "" : " (no crop)");
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
  inp.onfocus = () => { inp.value = ""; renderList(); list.classList.remove("hidden");
    const sel = list.querySelector(".sel");   // open scrolled to the current selection (context above/below), not the top
    if (sel) list.scrollTop = Math.max(0, sel.offsetTop - list.clientHeight / 2 + sel.offsetHeight / 2); };
  inp.oninput = () => { renderList(); list.classList.remove("hidden"); };
  inp.onblur = () => setTimeout(() => { list.classList.add("hidden"); inp.value = currentLabel(); }, 150);
}

const markerLabel = (i) => i == null ? "" : (state.manifest.markers[i].label || state.manifest.markers[i].marker_channel || "Phase");
function selectMarker(i) {
  state.markerIdx = i; state.marker = state.manifest.markers[i];
  $("markerfilter").title = markerLabel(i);   // hover shows the full marker name when truncated in the input
  refreshTargets();
  updateBagUI();        // anchor-bag selector is phase-only (before fillCellDropdown: montageCells depends on bag)
  fillCellDropdown();   // phase → 45/50 cells (bag), markers → 20 (reflect what was built)
  updateVsUI();         // Virtual staining is phase-only
  if (state.view === "montage") (mont.renderMode === "live" ? liveRefresh() : loadMontage());   // switch to this marker (keeps chosen renderer)
  if (state.view === "pc") loadPC();
}
function renderMarkerList() {
  const q = $("markerfilter").value.trim().toLowerCase(), list = $("marker-list"); list.innerHTML = ""; let n = 0;
  state.manifest.markers.forEach((m, i) => {
    const name = m.label || m.marker_channel || "Phase";
    if (q && !name.toLowerCase().includes(q)) return;
    const d = document.createElement("div"); d.className = "combo-item" + (i === state.markerIdx ? " sel" : "");
    d.textContent = name; d.title = name; d.onmousedown = (e) => { e.preventDefault(); pickMarker(i); }; list.appendChild(d); n++;
  });
  if (!n) list.innerHTML = '<div class="combo-empty">no markers</div>';
}
function pickMarker(i) { selectMarker(i); $("markerfilter").value = markerLabel(i); $("markerfilter").blur(); $("marker-list").classList.add("hidden"); }

const targetLabel = (t) => {
  const sa = state.targetSort === "setacc" ? targetAcc(t) : -1;   // sorting by SET ACC → show set-accuracy in parens instead of mAP
  const paren = t.grain === "pc" && t.explained_variance != null ? ` (${t.explained_variance.toFixed(1)}%)`
    : sa >= 0 ? ` (${Math.round(sa * 100)}% acc)`
    : t.dist_map != null ? ` (${t.dist_map.toFixed(2)})` : "";
  return `${t.target}${paren}${t.grain === "complex" ? " ·cx" : t.grain === "minibinder" ? " ·mb" : ""}`;
};
function refreshTargets() {   // marker or grain changed → recompute candidates (NTC-anchored), keep/reset selection
  updateBagUI();              // correct the anchor bag for this grain BEFORE building the grid (minibinder/PC → single_bag; never blank)
  const g = $("grain").value;
  const altSet = new Set(state.marker.targets.filter(e => e.control).map(e => e.target));   // names with a non-NTC anchor
  state.targets = state.marker.targets.filter(t => !t.control && (g === "all" || t.grain === g)
    && (!state.altAnchorsOnly || altSet.has(t.target)));
  if (g !== "complex" && !state.altAnchorsOnly && !state.targets.some(t => t.target === "NTC"))   // NTC selectable (Top Cells / embedding / PC); no traversal assets
    state.targets.push({ grain: "geneKO", target: "NTC", slug: "NTC", dist_map: null, n_cells: 0, alphas: [], desc: "non-targeting control" });
  let t = state.target && state.targets.find(x => x.slug === state.target.slug);
  if (!t) t = state.targets[0];
  if (t) { $("filter").value = targetLabel(t); selectTarget(t.slug); }
  else { state.target = null; $("filter").value = ""; rebuild(); }
}
let accBins = null;   // {markerSlug|"phase": {bins:[...], acc:{perturbation:[acc@bin...]}}} — SetTransformer real-cell set-accuracy across bag sizes
function ensureSetacc(cb) {
  if (accBins) return cb();
  fetch(`${BASE}_montage/setacc_bins_bymarker.json${NOCACHE}`).then(r => r.ok ? r.json() : null).then(j => {
    if (j) { accBins = j; return cb(); }
    // fallback: legacy single-bin (@100) file → wrap in the bins form so the rest of the code is uniform
    return fetch(`${BASE}_montage/setacc_bymarker.json${NOCACHE}`).then(r => r.ok ? r.json() : {}).then(old => {
      accBins = {}; for (const mk in old) accBins[mk] = { bins: [100], acc: Object.fromEntries(Object.entries(old[mk]).map(([p, v]) => [p, [v]])) }; cb();
    });
  }).catch(() => { accBins = {}; cb(); });
}
const saBins = markerKey => { const r = accBins && accBins[markerKey]; return r ? r.bins : []; };   // available bag sizes for a marker (adaptive)
function saAcc(markerKey, pert, bin) {   // set-accuracy for a perturbation at a bag size (nearest available); null if unknown
  const r = accBins && accBins[markerKey]; if (!r || !r.acc[pert]) return null;
  let bi = r.bins.indexOf(bin);
  if (bi < 0) bi = r.bins.reduce((best, b, i) => Math.abs(b - bin) < Math.abs(r.bins[best] - bin) ? i : best, 0);
  const v = r.acc[pert][bi]; return v == null ? null : v;
}
function targetAcc(t) { const v = saAcc(attnModality(), t.target, 100); return v == null ? -1 : v; }   // ordering: set-accuracy @100 (nearest); -1 if unknown
function sortedTargets() {   // current list order: mAP-desc (manifest) | A–Z | set-accuracy — shared by the list + the ◀/▶ stepper
  return state.targetSort === "alpha" ? [...state.targets].sort((a, b) => a.target.localeCompare(b.target))
    : state.targetSort === "setacc" ? [...state.targets].sort((a, b) => targetAcc(b) - targetAcc(a))
    : state.targets;
}
function stepTarget(d) {   // move to the previous/next perturbation in the current sorted order
  const arr = sortedTargets(); if (!arr.length) return;
  let i = state.target ? arr.findIndex(t => t.slug === state.target.slug) : -1;
  i = Math.max(0, Math.min(arr.length - 1, (i < 0 ? 0 : i + d)));
  const t = arr[i]; if (!t) return;
  $("filter").value = targetLabel(t); selectTarget(t.slug);
}
function renderTargetList() {
  const q = $("filter").value.trim().toLowerCase(), list = $("target-list"); list.innerHTML = ""; let n = 0;
  const arr = sortedTargets();
  arr.forEach(t => {
    if (q && !t.target.toLowerCase().includes(q)) return;
    const d = document.createElement("div"); d.className = "combo-item" + (state.target && t.slug === state.target.slug ? " sel" : "");
    d.title = t.target;   // hover shows the full name (protein-complex names get ellipsis-truncated)
    if (state.targetSort === "setacc") {   // show the set-score % (P(target) @ α=1), not the mAP
      const a = targetAcc(t), suffix = t.grain === "complex" ? " ·cx" : t.grain === "minibinder" ? " ·mb" : "";
      d.textContent = `${t.target}${suffix}`;
      const s = document.createElement("span"); s.className = "ci-sub"; s.textContent = a < 0 ? " —" : ` ${Math.round(a * 100)}%`; d.appendChild(s);
    } else {
      d.textContent = targetLabel(t);
    }
    d.onmousedown = (e) => { e.preventDefault(); pickTarget(t.slug); }; list.appendChild(d); n++;
  });
  if (!n) list.innerHTML = '<div class="combo-empty">no matches</div>';
}
function pickTarget(slug) { selectTarget(slug); const t = state.targets.find(x => x.slug === slug); $("filter").value = t ? targetLabel(t) : ""; $("filter").blur(); $("target-list").classList.add("hidden"); }

function selectTarget(slug) {
  state.target = state.targets.find(t => t.slug === slug);
  if (!state.target) return;
  $("filter").title = targetLabel(state.target);   // hover shows the full name (long complex names get truncated in the input)
  populateAnchors(state.target.target);
  state.page = 0; rebuild();
  if (state.view === "attn") renderAttn();          // selection drives the attention-head view
  if (state.view === "montage") (mont.renderMode === "live" ? liveDraw() : focusMontageOnSelection());   // update embedding selection
  if (state.view === "pc" && pc.data && pc.data.geneData[state.target.target]) pcGeneOverlay(state.target.target);   // top search → gene's PCs
  if (state.view === "top") loadTop();              // selection drives the top-cells view (loadTop refetches if the marker changed)
}

// anchors = NTC + any class that has a precomputed A→this-target traversal in this marker.
function populateAnchors(name) {
  const anchors = isPublic() ? ["NTC"]   // public: traversals anchor on NTC only (no alternative-anchor A→B pairs)
    : ["NTC", ...new Set(state.marker.targets.filter(e => e.target === name && e.control).map(e => e.control))];
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
  const set = state.pinned.slice();
  if (cur && !set.some(p => p.key === cur.key)) set.unshift(cur);   // prepend current ONLY if not already pinned → grid order == pin-list order
  return set;
}

// pins are SHARED across Top Cells + Traversal (same perturbation set). NTC stays top-cells-only (an NTC traversal is degenerate).
function pinShared(target, anchor = "NTC") {
  if (!target) return;
  if (target !== "NTC" && state.marker) {
    const e = resolveEntry(target, anchor) || resolveEntry(target, "NTC");
    if (e) { const p = pertOf(state.marker.marker_channel || "Phase", e, e.control || "NTC");
      if (p && !state.pinned.some(q => q.key === p.key)) state.pinned.push(p); }
  }
  if (!tc.pinned.some(p => p.gene === target)) tc.pinned.push({ gene: target, mode: tc.mode });
  redrawPins();
}
function unpinShared(target) {
  state.pinned = state.pinned.filter(p => p.target !== target);
  tc.pinned = tc.pinned.filter(p => p.gene !== target);
  redrawPins();
}
function redrawPins() {   // refresh both pin lists + the active grid, and persist
  renderPinned(); renderTopPins();
  if (state.view === "top") renderTop(); else rebuild();
  if (typeof saveState === "function") saveState();
}

async function exportGif() {   // client-side GIF of the current traversal grid across all α (works on static S3)
  const grid = $("grid"); if (!state.panels.length || typeof GIF === "undefined") return;
  const btn = $("exportgif"), old = btn.textContent; btn.disabled = true; btn.textContent = "rendering…";
  const gr = grid.getBoundingClientRect();
  const MARGIN = 18;                 // general padding around the whole gif
  const TOP = 72;                    // heatbar band on its own line above the grid (extra gap below its end labels)
  // walk each group (row/column) keeping its header label + colored tiles, in raw grid coords
  const groups = [];
  grid.querySelectorAll(".group").forEach(gEl => {
    const hd = gEl.querySelector(".group-hd"); if (!hd) return;
    const hr = hd.getBoundingClientRect(), tiles = [];
    gEl.querySelectorAll(".panel img").forEach(im => {
      const k = +im.id.replace("pimg", ""); const p = state.panels[k]; if (!p) return;
      const r = im.getBoundingClientRect();
      tiles.push({ x: r.left - gr.left + grid.scrollLeft, y: r.top - gr.top + grid.scrollTop, w: r.width, h: r.height, frames: p.frames, dir: p.asset_dir, cell: p.cell });
    });
    if (tiles.length) groups.push({ text: hd.textContent, color: getComputedStyle(hd).color, dir: tiles[0].dir,   // skip the tile-less real-cells group
      hx: hr.left - gr.left + grid.scrollLeft, hy: hr.top - gr.top + grid.scrollTop, tiles });
  });
  const allT = groups.flatMap(g => g.tiles);
  if (!allT.length) { btn.disabled = false; btn.textContent = old; return; }
  // size the canvas to the real tile bounding box (+margin) so empty auto-fill tracks aren't rendered
  const minX = Math.min(...allT.map(t => t.x)), maxX = Math.max(...allT.map(t => t.x + t.w));
  const minY = Math.min(...allT.map(t => t.y), ...groups.map(g => g.hy)), maxY = Math.max(...allT.map(t => t.y + t.h));
  const W = Math.ceil(maxX - minX) + MARGIN * 2, H = Math.ceil(maxY - minY) + TOP + MARGIN * 2;
  const offX = MARGIN - minX, offY = TOP + MARGIN - minY;   // shift content to the margin origin
  groups.forEach(g => { g.hx += offX; g.hy += offY; g.tiles.forEach(t => { t.x += offX; t.y += offY; }); });
  const cache = {};
  const srcs = [...new Set(groups.flatMap(g => g.tiles.flatMap(t => t.frames)))];
  await Promise.all(srcs.map(src => new Promise(res => {
    const im = new Image(); im.onload = im.onerror = () => { cache[src] = im; res(); }; im.src = src; })));
  // frame plan = viewer autoplay: ping-pong over the ±alphaLimit range, dwelling at the user's pause ticks (dwell → longer per-frame delay, not duplicate frames)
  computeRange();
  const lo = state.rangeLo, hi = state.rangeHi;
  const base = Math.max(40, ((+$("speed").value) || state.frameMs || 120) * 0.38);   // faster baseline than the on-screen player
  const hold = v => (v === lo || v === hi) ? 6 : (state.pausePoints.has(v) ? 3 : 1);   // always dwell at the sweep ends, slightly longer than interior pauses
  const path = [];
  for (let v = lo; v <= hi; v++) path.push(v);
  for (let v = hi - 1; v > lo; v--) path.push(v);        // ping-pong back (drop final lo; GIF loops to it)
  if (!path.length) path.push(Math.floor((lo + hi) / 2));
  const cv = document.createElement("canvas"); cv.width = W; cv.height = H; const cx = cv.getContext("2d");
  const gif = new GIF({ workers: 2, quality: 10, workerScript: "gif.worker.js", width: W, height: H, globalPalette: true });
  const LH = 14;   // wrap each group label to its own tile-column width, centered above its images (matches on-screen)
  const wrap = (txt, maxW) => { const words = String(txt).split(/\s+/), lines = []; let cur = "";
    for (const w of words) { const t = cur ? cur + " " + w : w; if (cur && cx.measureText(t).width > maxW) { lines.push(cur); cur = w; } else cur = t; } if (cur) lines.push(cur); return lines; };
  cx.font = "12px sans-serif";
  groups.forEach(g => { const gx0 = Math.min(...g.tiles.map(t => t.x)), gx1 = Math.max(...g.tiles.map(t => t.x + t.w));
    g.gcx = (gx0 + gx1) / 2; g.gtop = Math.min(...g.tiles.map(t => t.y)); g.lines = wrap(g.text, gx1 - gx0 - 4); });
  const tgt = (state.target && state.target.target) || "target";
  const anch = state.anchor || "NTC";
  // heatbar geometry: its own centered line, matching the viewer's --neg/--mid/--pos scale
  const barW = Math.min(W - MARGIN * 2, 520), bx0 = (W - barW) / 2, bx1 = bx0 + barW, cy = MARGIN + 15;
  const chip = (txt, ax, ay, v, align, col) => {   // colored score chip; col={bg,fg} overrides the default heat(v) white→red
    cx.font = "700 12px ui-monospace,monospace"; cx.textBaseline = "top"; cx.textAlign = "left";
    const pad = 4, bw = cx.measureText(txt).width + pad * 2, bh = 17, bx = align === "right" ? ax - bw - 3 : ax + 3, by = ay + 3;
    cx.fillStyle = col ? col.bg : heat(v);
    if (cx.roundRect) { cx.beginPath(); cx.roundRect(bx, by, bw, bh, 4); cx.fill(); } else cx.fillRect(bx, by, bw, bh);
    cx.fillStyle = col ? col.fg : (v > 0.55 ? "#fff" : "#111"); cx.fillText(txt, bx + pad, by + 3);
  };
  for (const a of path) {
    cx.fillStyle = "#0d0f14"; cx.fillRect(0, 0, W, H);
    const grd = cx.createLinearGradient(bx0, 0, bx1, 0);                          // orange (anti-phenotype) → blue (anchor) → red (phenotype)
    grd.addColorStop(0, "#f0a020"); grd.addColorStop(0.5, "#26c6ff"); grd.addColorStop(1, "#ff5252");
    cx.fillStyle = grd; cx.fillRect(bx0, cy - 4, barW, 8);
    const frac = hi > lo ? (a - lo) / (hi - lo) : 0.5, mx = bx0 + frac * barW;    // moving marker (only thing that changes → no flicker)
    cx.strokeStyle = "#fff"; cx.lineWidth = 2; cx.beginPath(); cx.moveTo(mx, cy - 10); cx.lineTo(mx, cy + 10); cx.stroke();
    cx.fillStyle = "#fff"; cx.beginPath(); cx.moveTo(mx, cy - 10); cx.lineTo(mx - 4, cy - 15); cx.lineTo(mx + 4, cy - 15); cx.closePath(); cx.fill();
    cx.font = "600 12px sans-serif"; cx.textBaseline = "top";                     // three static labels colored like the bar
    cx.textAlign = "left";   cx.fillStyle = "#f0a020"; cx.fillText("anti-phenotype", bx0, cy + 10);
    cx.textAlign = "center"; cx.fillStyle = "#26c6ff"; cx.fillText(anch, (bx0 + bx1) / 2, cy + 10);
    cx.textAlign = "right";  cx.fillStyle = "#ff5252"; cx.fillText("phenotype", bx1, cy + 10);
    for (const g of groups) {                                                    // wrapped label centered above the group's images, then the tiles
      cx.font = "12px sans-serif"; cx.textAlign = "center"; cx.textBaseline = "bottom"; cx.fillStyle = g.color;
      g.lines.forEach((ln, li) => cx.fillText(ln, g.gcx, g.gtop - 5 - (g.lines.length - 1 - li) * LH));
      const lead = g.tiles.reduce((m, t) => t.x < m.x ? t : m, g.tiles[0]);
      for (const t of g.tiles) {
        const im = cache[t.frames[a]]; if (im && im.width) cx.drawImage(im, t.x, t.y, t.w, t.h);
        if (state.scoreMode === "linear") {                                      // per-cell linear classifier score (top-right of tile)
          const sc = state.scores[t.dir], v = sc && sc.scores[t.cell] ? sc.scores[t.cell][Math.min(a, sc.scores[t.cell].length - 1)] : null;
          if (v != null) chip(`${Math.round(v * 100)}%`, t.x + t.w, t.y, v, "right");
        }
      }
      { const sch = setChip(state.scoresV5[g.dir], a);                            // v5 set score (selected mode) on the row's leftmost tile
        if (sch) { const rl = sch.showReal && state.realAcc20 ? state.realAcc20[g.dir] : null;
          chip(`${sch.txt}${rl != null ? `/real ${Math.round(rl * 100)}%` : ""}`, lead.x, lead.y, 0, "left", sch); } }
    }
    gif.addFrame(cx, { copy: true, delay: base * hold(a) });
  }
  gif.on("finished", blob => { const a = document.createElement("a"); a.href = URL.createObjectURL(blob);
    a.download = `traversal_${tgt}.gif`; a.click();
    URL.revokeObjectURL(a.href); btn.disabled = false; btn.textContent = old; });
  gif.render();
}
function renderPinned() {
  const ul = $("panellist"); ul.innerHTML = "";
  const aset = activeSet();
  state.pinned.forEach((p, i) => {
    const li = document.createElement("li");
    const gi = aset.findIndex(q => q.key === p.key);
    li.style.color = PALETTE[((gi < 0 ? i : gi)) % PALETTE.length];   // exact color of its grid row
    li.draggable = true; li.style.cursor = "move";        // drag to reorder
    const a = p.anchor && p.anchor !== "NTC" ? `${p.anchor}→` : "";
    li.innerHTML = `<span>⠿ ${a}${p.target} · ${p.markerName}</span>`;
    const b = document.createElement("button"); b.textContent = "✕";
    b.onclick = () => unpinShared(p.target);
    li.ondragstart = e => { e.dataTransfer.setData("text/plain", i); e.dataTransfer.effectAllowed = "move"; };
    li.ondragover = e => { e.preventDefault(); li.style.opacity = ".5"; };
    li.ondragleave = () => { li.style.opacity = ""; };
    li.ondrop = e => { e.preventDefault(); li.style.opacity = ""; const from = +e.dataTransfer.getData("text/plain");
      if (from !== i) { const [m] = state.pinned.splice(from, 1); state.pinned.splice(i, 0, m); renderPinned(); rebuild(); } };
    li.appendChild(b); ul.appendChild(li);
  });
}

// grid = activeSet rows × cells-per-page cols, all synced to the shared α slider.
function rebuild() {
  const set = activeSet();
  const N = state.cellCount, start = state.page * N;
  const cr = state.target ? state.target.n_cells : (set[0] ? set[0].n_cells : 0);   // rank range of the current perturbation's cells
  $("cellrange").textContent = cr ? `${Math.min(start + 1, cr)}-${Math.min(start + N, cr)} (${cr})` : "";
  $("pagenum").textContent = `Page ${state.page + 1}${cr ? " / " + Math.max(1, Math.ceil(cr / N)) : ""}`;
  const g = $("grid"); g.innerHTML = "";
  state.panels = []; state.groups = []; let k = 0;
  if (state.showReal) {                             // real cells shown ONCE as their own group (shared starting cells → identical across NTC-anchored perturbations)
    const rp0 = set.find(p => p.has_real);
    if (rp0) {
      const rgroup = document.createElement("div"); rgroup.className = "group";
      const rhd = document.createElement("div"); rhd.className = "group-hd"; rhd.style.color = "#9aa0a6";
      rhd.textContent = "real cells"; rhd.title = rhd.textContent;
      const rr = document.createElement("div"); rr.className = "group-cells"; rr.style.setProperty("--cols", Math.min(N, 5));
      for (let c = start; c < Math.min(start + N, rp0.n_cells); c++) {
        const rp = document.createElement("div"); rp.className = "panel";
        const rimg = document.createElement("img"); rimg.src = `${travBase(rp0.real_dir)}${rp0.real_dir}/cell${diskCellFor(rp0.anchor, c)}/real.webp`;
        rp.appendChild(rimg); rr.appendChild(rp);
      }
      rgroup.appendChild(rhd); rgroup.appendChild(rr); g.appendChild(rgroup);
    }
  }
  set.forEach((p, gi) => {                          // one group (row) per perturbation
    const color = PALETTE[gi % PALETTE.length];
    const apfx = p.anchor && p.anchor !== "NTC" ? `${p.anchor}→` : "";
    const group = document.createElement("div"); group.className = "group"; group.style.setProperty("--cols", Math.min(N, 5));   // cap header/box to the actual cell columns
    const hd = document.createElement("div"); hd.className = "group-hd"; hd.style.color = color;
    const tt = document.createElement("span"); tt.className = "gh-title"; tt.textContent = `${apfx}${p.target} · ${p.markerName}`;
    hd.appendChild(tt); hd.title = tt.title = tt.textContent;   // title on the visible name span (+ header) so hover shows the full name
    const sa = document.createElement("span"); sa.className = "setacc"; sa.style.display = "none"; hd.appendChild(sa);
    state.groups.push({ dir: p.asset_dir, hd, sa, nCells: p.n_cells, label: hd.title });   // v5 set-accuracy chip (bag, per-α)
    const cells = document.createElement("div"); cells.className = "group-cells";
    cells.style.borderLeft = `4px solid ${color}`;   // colored bar; box styling in CSS (matches Top Cells strip)
    cells.style.setProperty("--cols", Math.min(N, 5));   // max 5 per row; wrap instead of stretching
    for (let c = start; c < Math.min(start + N, p.n_cells); c++) {
      const frames = p.alphas.map((_, i) => frameURL(p.asset_dir, diskCellFor(p.anchor, c), i));
      frames.forEach(src => { const im = new Image(); im.src = src; });
      const panel = document.createElement("div"); panel.className = "panel" + (c === start ? " lead" : "");
      const img = document.createElement("img"); img.id = `pimg${k}`; img.style.cursor = "zoom-in";
      { const kk = k, pp = p, cc = c;                          // click → pop-out the current-α frame (same glassy popout as PC/Top-cells)
        img.onclick = () => { const pn = state.panels[kk]; if (!pn) return;
          const pre = pp.anchor && pp.anchor !== "NTC" ? `${pp.anchor}→` : "";
          const av = (state.alphas && state.idx != null) ? state.alphas[Math.min(state.idx, state.alphas.length - 1)] : "";
          const sch = setChip(state.scoresV5[pp.asset_dir], state.idx);
          popOut(pn.frames[Math.min(state.idx, pn.frames.length - 1)], `${pre}${pp.target} · ${pp.markerName}`,
            `<div class="po-sub">α ${av} · cell ${cc + 1}${sch ? " · " + sch.txt : ""}</div>`); }; }
      const badge = document.createElement("div"); badge.className = "badge"; badge.id = `pbadge${k}`; badge.style.display = "none";
      panel.appendChild(img); panel.appendChild(badge); cells.appendChild(panel);
      state.panels.push({ asset_dir: p.asset_dir, cell: c, alphas: p.alphas, frames });
      k++;
    }
    group.appendChild(hd);
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
  computeRange();
  const mid = Math.floor(n / 2);
  let keep = (state.idx != null && state.idx >= 0 && state.idx < n) ? state.idx : mid;   // hold α across perturbation/cell-page changes; reset only on load / α-count change
  if (restoredAlpha != null) { const j = state.alphas.indexOf(restoredAlpha); if (j >= 0) keep = j; restoredAlpha = null; }   // reload/version-switch: restore the saved α
  $("alpha").value = keep; buildPlaySeq(state.alphas); showIdx(keep);

  const t = state.target;
  const scoreLbl = t && t.grain === "minibinder" ? `${t.phenotype || "39S"} cell-score` : t && t.grain === "complex" ? "EBI mAP" : "dist mAP";
  $("meta").textContent = t ? `${set.length} row(s) × cells ${start}–${start + N - 1} · w=${state.manifest.w}` +
    (t.dist_map != null ? ` · ${t.target} ${scoreLbl} ${t.dist_map.toFixed(3)}` : "") +
    (t.grain === "minibinder" ? ` · binder prob ${(t.binder_prob || 0).toFixed(3)} · target ${t.gene_target || ""}` : "") : "no target";
  renderInfo(t);
}

function fetchScores(dir) {
  if (state.scores[dir] !== undefined) return;   // cached or pending
  state.scores[dir] = null; state.scoresV5[dir] = null;
  fetch(`${travBase(dir)}${dir}/scores.json${NOCACHE}`).then(r => r.ok ? r.json() : null)
    .then(j => { state.scores[dir] = j; showIdx(state.idx); }).catch(() => {});
  fetch(`${travBase(dir)}${dir}/scores_v5.json${NOCACHE}`).then(r => r.ok ? r.json() : null)   // v5 SetTransformer set-accuracy (bag)
    .then(j => { state.scoresV5[dir] = j; showIdx(state.idx); }).catch(() => {});
}

function showIdx(i) {
  state.idx = i;
  state.panels.forEach((p, k) => {
    const img = $(`pimg${k}`); if (img) img.src = p.frames[Math.min(i, p.frames.length - 1)];
    const bd = $(`pbadge${k}`); if (!bd) return;
    const sc = state.scores[p.asset_dir];
    const v = state.scoreMode === "linear" && sc && sc.scores[p.cell] ? sc.scores[p.cell][Math.min(i, sc.scores[p.cell].length - 1)] : null;
    if (v != null) {
      bd.textContent = `${Math.round(v * 100)}%`;
      bd.style.background = heat(v);                 // white → deep red by confidence
      bd.style.color = v > 0.55 ? "#fff" : "#111";
      bd.style.display = "block";
    } else bd.style.display = "none";
  });
  state.groups.forEach(gp => {   // per-traversal v5 set score chip (selected mode: P(target) heat | rank RdYlGn)
    const sch = setChip(state.scoresV5[gp.dir], i);
    if (sch) {
      const rl = sch.showReal && state.realAcc20 ? state.realAcc20[gp.dir] : null;   // real-cell ceiling (P(target) only)
      gp.sa.textContent = `${sch.txt}${rl != null ? ` / real ${Math.round(rl * 100)}%` : ""} · bag ${gp.nCells}`;
      gp.sa.style.background = sch.bg; gp.sa.style.color = sch.fg;
      gp.sa.style.display = "inline-block";
    } else gp.sa.style.display = "none";
  });
  const n = state.alphas.length, a = state.alphas[i];
  $("alpha-read").textContent = `α = ${a.toFixed(1)}`;
  $("heat-tick").style.left = `calc(${(i / (n - 1)) * 100}% - 1.5px)`;
}

// complex "Members (N): …" desc is the same for every marker but only stored on markers that have it
// (phase). Build a name→desc map once so non-phase complex targets still show the member links.
function complexDescOf(name) {
  if (!state._cxDesc) {
    state._cxDesc = {};
    for (const m of (state.manifest?.markers || []))
      for (const e of (m.targets || []))
        if (e.grain === "complex" && e.desc && e.desc.startsWith("Members") && !state._cxDesc[e.target]) state._cxDesc[e.target] = e.desc;
  }
  return state._cxDesc[name] || null;
}

// info sidebar (wiki-style): perturbation name + grain/mAP + function/pathways or complex members.
function renderInfo(t) {
  $("info-title").textContent = t ? t.target : "";
  $("info-sub").textContent = !t ? "" :
    t.grain === "minibinder"
      ? `minibinder · ${t.phenotype || "39S"} cell-score ${(t.cell_score ?? t.dist_map ?? 0).toFixed(3)} · binder prob ${(t.binder_prob || 0).toFixed(3)} · target ${t.gene_target || ""}`
      : `${t.grain === "complex" ? "protein complex" : t.grain}` + (t.dist_map != null ? ` · ${t.grain === "complex" ? "EBI" : "geneKO"} mAP ${t.dist_map.toFixed(3)}` : "");
  const body = $("info-body"); body.innerHTML = "";
  const s = t && (t.desc || (t.grain === "complex" ? complexDescOf(t.target) : null));   // complex members are marker-independent → fall back across markers (non-phase targets lack desc)
  if (!t || !s) { body.innerHTML = '<div class="empty">no annotation for this perturbation</div>'; return; }
  const sec = (lbl, txt) => {
    if (!txt) return;
    const d = document.createElement("div"); d.className = "sec";
    d.innerHTML = `<div class="sec-lbl">${lbl}</div>${txt}`;
    body.appendChild(d);
  };
  const gc = (g) => `<a href="https://www.genecards.org/cgi-bin/carddisp.pl?gene=${encodeURIComponent(g)}" target="_blank" rel="noopener">${g}</a>`;
  if (s.startsWith("Members")) {   // complex info page: portal link + each member gene with its function
    const members = s.replace(/^Members \(\d+\):\s*/, "").split(", ").map(x => x.trim()).filter(Boolean);
    sec("Links", `<a href="https://www.ebi.ac.uk/complexportal/complex/search?query=${encodeURIComponent(t.target)}" target="_blank" rel="noopener">EBI Complex Portal ↗</a>`);
    const rows = members.map(mg => {
      const gd = (state.geneDesc || {})[mg];
      const fn = gd ? gd.split(" || ")[0].replace(/^GO biological process:\s*/i, "") : "";
      const nar = (state.geneNarr || {})[mg];   // collapsed-by-default affinage narrative per member
      const det = nar ? `<details class="cx-narr"><summary><a href="https://affinage.wi.mit.edu/gene/${encodeURIComponent(mg)}" target="_blank" rel="noopener" onclick="event.stopPropagation()">Affinage mechanistic narrative ↗</a></summary><div class="narr">${pmidLink(nar)}</div></details>` : "";
      return `<div style="margin:5px 0;line-height:1.4">${gc(mg)}${fn ? ` — <span style="color:var(--mut)">${fn}</span>` : ""}${det}</div>`;
    }).join("");
    sec(`Complex members (${members.length})`, rows);
    return;
  }
  const g = encodeURIComponent(t.target);
  sec("Links", `<a href="https://opencell.sf.czbiohub.org/search/${g}" target="_blank" rel="noopener">OpenCell ↗</a> · <a href="https://www.genecards.org/cgi-bin/carddisp.pl?gene=${g}" target="_blank" rel="noopener">GeneCards ↗</a> · <a href="https://affinage.wi.mit.edu/gene/${g}" target="_blank" rel="noopener">Affinage ↗</a>`);
  const parts = s.split(" || ");   // "<function> || GO biological process: … || Reactome: … || CORUM complex: …"
  sec("Function", parts[0]);
  const narr = (state.geneNarr || {})[t.target];   // affinage mechanistic narrative (long prose, PMID-cited) — collapsible
  if (narr) {
    const LIM = 320;
    const narrLbl = `<a href="https://affinage.wi.mit.edu/gene/${g}" target="_blank" rel="noopener">Affinage mechanistic narrative ↗</a>`;
    if (narr.length <= LIM) sec(narrLbl, `<div class="narr">${pmidLink(narr)}</div>`);
    else {
      let cut = narr.lastIndexOf(" ", LIM); if (cut < 0) cut = LIM;
      const head = narr.slice(0, cut).replace(/[\s\[(]*PMID:\d*$/i, "").trim();   // drop any dangling partial citation
      sec(narrLbl,
        `<div class="narr"><span class="narr-short">${pmidLink(head)}… </span>` +
        `<span class="narr-full" hidden>${pmidLink(narr)}</span>` +
        `<button class="narr-more" onclick="toggleNarr(this)">continue reading ▾</button></div>`);
    }
  }
  parts.slice(1).forEach(p => { const i = p.indexOf(": "); if (i > 0) sec(p.slice(0, i), p.slice(i + 2)); });
}
const pmidLink = (s) => s.replace(/PMID:(\d+)/g, '<a href="https://pubmed.ncbi.nlm.nih.gov/$1" target="_blank" rel="noopener">PMID:$1</a>');
function toggleNarr(btn) {   // expand/collapse the affinage narrative
  const w = btn.closest(".narr"), sh = w.querySelector(".narr-short"), fu = w.querySelector(".narr-full");
  const open = fu.hidden;
  fu.hidden = !open; sh.hidden = open;
  btn.textContent = open ? "show less ▴" : "continue reading ▾";
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
// attention assets are keyed by (modality, grain, key): modality = "phase" | slugify(marker_channel),
// grain = geneKO|complex, key = gene (geneKO) or complex-slug. A "ref" bundles that + a display label.
const jsSlug = (s) => String(s).replace(/[^A-Za-z0-9]/g, "_").replace(/^_+|_+$/g, "");
const attnModality = () => (state.marker && state.marker.marker_channel) ? jsSlug(state.marker.marker_channel) : "phase";
const attnBase = (ref) => `${BASE}attention_heads/${ref.modality}/${ref.grain}/${ref.key}`;
const sameRef = (a, b) => a.modality === b.modality && a.grain === b.grain && a.key === b.key;
function haveAttn(modality, grain, key) {
  const a = state.attnIndex && state.attnIndex.assets && state.attnIndex.assets[modality];
  return !!(a && a[grain] && a[grain].includes(key));
}
function attnRefOf(t, modality) {   // manifest target → ref if its attention assets exist
  if (!t) return null;
  const key = t.grain === "geneKO" ? t.target : t.slug;
  return haveAttn(modality, t.grain, key) ? { modality, grain: t.grain, key, label: t.target } : null;
}
const attnCurrentRef = () => attnRefOf(state.target, attnModality());
function loadAttnHeads(ref) {
  const base = attnBase(ref);
  if (state.attnHeadsCache[base] !== undefined) return Promise.resolve(state.attnHeadsCache[base]);
  return fetch(`${base}/heads.json${NOCACHE}`).then(r => r.ok ? r.json() : null)
    .catch(() => null).then(j => { state.attnHeadsCache[base] = j; return j; });
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
function populateAttnHeadSelect(heads, sig) {
  const sel = $("a-head");
  if (sel._sig === sig) return;                       // same target's head list already populated
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
function renderAttnPinned() {   // pinned-perturbation list (mirrors the traversal pin list), color-coded
  const ul = $("a-panellist"); ul.innerHTML = "";
  state.attnPinned.forEach((r, i) => {
    const li = document.createElement("li");
    li.style.color = PALETTE[(i + 1) % PALETTE.length];   // +1: current selection owns PALETTE[0]
    const ctx = r.modality === "phase" ? "" : ` · ${r.modality}`;
    li.innerHTML = `<span>${r.label}${r.grain === "complex" ? " ·cx" : ""}${ctx}</span>`;
    const b = document.createElement("button"); b.textContent = "✕";
    b.onclick = () => { state.attnPinned.splice(i, 1); renderAttnPinned(); renderAttn(); };
    li.appendChild(b); ul.appendChild(li);
  });
}
async function renderAttn() {
  const grid = $("attn-grid"), lbl = $("attn-head-lbl"), status = $("a-status");
  renderAttnPinned();
  const cur = attnCurrentRef();
  if (!cur) {
    lbl.textContent = ""; status.textContent = "";
    grid.innerHTML = `<div class="empty">${!state.target ? "Select a perturbation."
      : "No attention-head data for this marker × perturbation."}</div>`;
    return;
  }
  const refs = [cur, ...state.attnPinned.filter(r => !sameRef(r, cur))];   // current first, then pins
  const heads0 = await loadAttnHeads(cur);
  if (!heads0) { grid.innerHTML = '<div class="empty">failed to load attention data</div>'; return; }
  populateAttnHeadSelect(heads0, attnBase(cur));
  const nh = heads0.heads.length;
  const headMode = state.attnHead === "all" ? `all ${nh} heads` : `head ${attnHeadLabel(heads0, Math.min(state.attnHead, nh - 1))}`;
  lbl.innerHTML = `${refs.length} perturbation${refs.length > 1 ? "s" : ""} · ${headMode}` +
    `<span class="sub">rows = heads · columns = cells · left bar colors the perturbation</span>`;
  status.textContent = `norm=${state.attnNorm} · clim [${state.attnClimLo}, ${state.attnClimHi}] · α ${state.attnAlpha}`;
  grid.innerHTML = "";
  for (let gi = 0; gi < refs.length; gi++) {
    const heads = gi === 0 ? heads0 : await loadAttnHeads(refs[gi]);
    if (heads) buildAttnGroup(grid, refs[gi], PALETTE[gi % PALETTE.length], heads);
  }
}
function buildAttnGroup(grid, ref, color, heads) {   // one perturbation block: colored header + head rows
  const base = attnBase(ref);
  const N = state.cellCount, start = state.page * N, end = Math.min(start + N, heads.n_cells);
  const ncols = Math.max(end - start, 1);
  const headIdxs = state.attnHead === "all" ? heads.heads.map((_, i) => i) : [Math.min(state.attnHead, heads.heads.length - 1)];
  const group = document.createElement("div"); group.className = "agroup";
  const hd = document.createElement("div"); hd.className = "agroup-hd"; hd.style.color = color;
  hd.textContent = `${ref.label}${ref.grain === "complex" ? " ·cx" : ""} · cells ${start}–${end - 1}`;
  group.appendChild(hd);
  for (const h of headIdxs) {
    const row = document.createElement("div"); row.className = "arow";
    const rl = document.createElement("div"); rl.className = "arow-lbl"; rl.style.borderLeftColor = color;
    rl.textContent = attnHeadLabel(heads, h);
    const cells = document.createElement("div"); cells.className = "arow-cells"; cells.style.setProperty("--acols", ncols);
    for (let c = start; c < end; c++) {
      const tile = mkTile(); tile.el.style.cursor = "zoom-in";
      { const cc = c, hh = h;                                  // click → pop-out a snapshot of the rendered crop+attention composite
        tile.el.onclick = () => { let src; try { src = tile.cv.toDataURL(); } catch (e) { src = `${base}/cell${cc}/crop.webp`; }
          popOut(src, `${ref.label}`, `<div class="po-sub">cell ${cc + 1} · ${attnHeadLabel(heads, hh)}</div>`); }; }
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

// ---- PCs tab: PC-strip explorer (principal components of the CellDINO gene embedding) ----
const PC_CROP_V = "?m=2";   // bump when PC crops are re-rendered (masked v1) so browsers refetch same-named PNGs
const pc = { data: null, enrich: null, feat: null, cur: null, cells: [], tfidf: false, mode: "onto", dedup: true, norm: false, sort: "pc", df: null, gcol: {}, marker: undefined, base: null };
function pcSlug() {   // per-marker PC cache slug (null = phase, which uses the shared pcs/ cache)
  const m = state.marker && state.marker.marker_channel;
  return (!m || /^phase/i.test(m)) ? null : m.toLowerCase().replace(/[^a-z0-9]/g, "");
}
async function loadPC() {
  const slug = pcSlug();
  if (pc.marker !== slug || pc.data === null) {   // marker changed → load that marker's isolated PC cache
    pc.marker = slug; pc.cur = 1;
    pc.base = slug ? `${BASE}pcs/markers/${slug}/` : `${BASE}pcs/`;
    pc.data = await fetch(`${pc.base}index.json${NOCACHE}`).then(r => r.ok ? r.json() : null).catch(() => null);
    pc.enrich = await fetch(`${pc.base}enrichment.json${NOCACHE}`).then(r => r.ok ? r.json() : null).catch(() => null);
    pc.feat = await fetch(`${pc.base}features.json${NOCACHE}`).then(r => r.ok ? r.json() : null).catch(() => null);
    if (pc.feat) { pcBuildColors(); pcBuildSort(); }
    if (pc.data) { buildPCList(); pcComputeDf(); }
  }
  if (!pc.data) { $("pc-view").innerHTML = `<div class="empty">No PC cache for ${slug ? state.marker.marker_channel : "phase"} yet.</div>`; return; }
  showPC(pc.cur || 1);
}
function pcComputeDf() {   // document frequency: # of PCs whose top high/low lists a gene appears in (for tf-idf)
  pc.df = {};
  for (const p in pc.data.pcData) { const d = pc.data.pcData[p];
    for (const g of [...d.high_genes, ...d.low_genes]) pc.df[g.gene] = (pc.df[g.gene] || 0) + 1; }
}
function pcRank(genes) {   // tf-idf: |loading| * log(nPCs/(1+df)) so genes shared across PCs sink, unique ones rise
  if (!pc.tfidf) return genes;
  const n = pc.data.overview.n_pcs;
  return [...genes].map(g => ({ ...g, _w: Math.abs(g.score) * Math.log(n / (1 + (pc.df[g.gene] || 0))) }))
    .sort((a, b) => b._w - a._w);
}
function pcBuildSort() {   // populate the PC-list sort dropdown from the feature groups (default = explained variance)
  const sel = $("pc-sort"); if (!sel || !pc.feat) return; const m = pc.feat.meta || {};
  const og = (label, kind, groups) => !groups || !groups.length ? "" :
    `<optgroup label="${label}">` + groups.map(g => `<option value="${kind}:${g}">${g}</option>`).join("") + "</optgroup>";
  sel.innerHTML = `<option value="pc">explained variance</option>` +
    og("feature class %", "cls", m.classes) + og("organelle group %", "org", m.orgGroups) + og("profiling tool %", "src", m.srcGroups);
  sel.value = pc.sort;
}
function buildPCList() {
  const ov = pc.data.overview, ul = $("pc-list"); ul.innerHTML = "";
  const evmx = Math.max(...ov.explained_variance);
  let rows = ov.explained_variance.map((v, i) => ({ pc: i + 1, ev: v }));
  const byGroup = pc.sort && pc.sort !== "pc" && pc.feat;
  if (byGroup) {   // sort PCs by a chosen class/organelle/tool composition %, honoring dedup/tf-idf/norm state
    const [kind, grp] = pc.sort.split(":"), vv = pc.dedup ? "dedup" : "full", N = pc.norm ? "N" : "";
    rows.forEach(o => { const fp = pc.feat[o.pc]; const comp = fp && (pc.tfidf ? fp.tfidf[vv] : fp.raw[vv])[kind + N] || {}; o.m = comp[grp] || 0; });
    rows.sort((a, b) => b.m - a.m);
  }
  const mmx = byGroup ? (Math.max(...rows.map(o => o.m)) || 1) : evmx;
  rows.forEach(o => {
    const li = document.createElement("li"); li.className = "pc-item"; li.dataset.pc = o.pc;
    const w = (byGroup ? o.m / mmx : o.ev / evmx) * 100, txt = byGroup ? `${Math.round(o.m * 100)}%` : `${o.ev.toFixed(1)}%`;
    li.innerHTML = `<span class="pc-lbl">PC${o.pc}</span><span class="pc-bar"><span style="width:${w}%"></span></span><span class="pc-pct">${txt}</span>`;
    li.onclick = () => showPC(o.pc);
    ul.appendChild(li);
  });
  if (pc.cur) { const a = ul.querySelector(`.pc-item[data-pc="${pc.cur}"]`); if (a) a.classList.add("active"); }
}
function showPC(n) {
  const ov = pc.data && pc.data.overview; if (!ov || n < 1 || n > ov.n_pcs) return;
  pc.cur = n;
  document.querySelectorAll("#pc-list .pc-item").forEach(li => li.classList.toggle("active", +li.dataset.pc === n));
  const a = document.querySelector(`#pc-list .pc-item[data-pc="${n}"]`); if (a) a.scrollIntoView({ block: "nearest" });
  renderPC(pc.data.pcData[n]);
}
function renderPC(d) {
  const ov = pc.data.overview, view = $("pc-view"); pc.cells = [];
  let h = `<div class="pc-head"><h2>PC${d.pc}</h2><span class="hint">${d.explained_variance}% explained variance · ${ov.n_bins} bins low→high</span>` +
    `<span class="row"><button onclick="showPC(${d.pc - 1})"${d.pc <= 1 ? " disabled" : ""}>◀ prev</button><button onclick="showPC(${d.pc + 1})"${d.pc >= ov.n_pcs ? " disabled" : ""}>next ▶</button></span></div><div class="pc-strip">`;
  for (let row = 0; row < ov.n_rows; row++) {
    h += '<div class="pc-strip-row">';
    for (const bin of d.strip) {
      const c = bin.cells[row];
      if (c && c.has_crop) { const i = pc.cells.push(c) - 1; h += `<img class="pc-cell" src="${pc.base}crops/${c.img}${PC_CROP_V}" title="${c.gene} · ${c.score}" onclick="pcCell(${i})">`; }
      else h += '<div class="pc-cell ph"></div>';
    }
    h += "</div>";
  }
  h += '<div class="pc-axis"><span>low</span><span>high</span></div></div>';
  const chip = (g, cls, sign) => `<span class="chip ${cls}" onclick="pcGeneOverlay('${g.gene}')">${g.gene} <small>${sign}${g.score.toFixed(1)}</small></span>`;
  const feat = pc.mode === "feat";
  const en = pc.enrich && pc.enrich[d.pc];                        // tf-idf toggle swaps to the PC-unique gene-set enrichment
  const enH = en && (pc.tfidf ? en.high_tfidf : en.high), enL = en && (pc.tfidf ? en.low_tfidf : en.low);
  const belowH = feat ? pcFeatHtml(d.pc, "high") : pcEnrichHtml(enH);
  const belowL = feat ? pcFeatHtml(d.pc, "low") : pcEnrichHtml(enL);
  h += '<div class="pc-genes">' +
    `<div><div class="sec-lbl">High-loading perturbations (positive)</div><div class="chips">${pcRank(d.high_genes).map(g => chip(g, "pos", "+")).join("")}</div>${belowH}</div>` +
    `<div><div class="sec-lbl">Low-loading perturbations (negative)</div><div class="chips">${pcRank(d.low_genes).map(g => chip(g, "neg", "")).join("")}</div>${belowL}</div>` +
    "</div>";
  if (feat) h += pcCompHtml(d.pc);
  view.innerHTML = h;
}
// ---- Features mode: morphometric-feature enrichment per PC (OP/CP phase features) ----
// three palettes in disjoint hue regions — class = warm, organelle = cool, source = purple/mint
const CLS_PAL = ["#ef476f", "#f3722c", "#f8961e", "#f9c74f", "#e0a458", "#bc6c25", "#e07a5f", "#c98986"]; // warm
const ORG_PAL = ["#2ec4b6", "#48cae4", "#4361ee", "#7209b7", "#4895ef", "#06d6a0", "#5e60ce", "#8ecae6"]; // cool
const SRC_PAL = ["#c77dff", "#80ed99", "#adb5bd"]; // CellProfiler / OrganelleProfiler / other
function pcBuildColors() {   // fixed color per feature-class / organelle-group / source for the composition bars
  const m = pc.feat.meta || {}; pc.gcol = {};
  (m.classes || []).forEach((g, i) => pc.gcol[g] = CLS_PAL[i % CLS_PAL.length]);
  (m.orgGroups || []).forEach((g, i) => pc.gcol[g] = ORG_PAL[i % ORG_PAL.length]);
  (m.srcGroups || []).forEach((g, i) => pc.gcol[g] = SRC_PAL[i % SRC_PAL.length]);
}
// per-side (high/low) feature bars: length = |r|, annotated with signed r; tf-idf → distinctive features split by r sign
function pcFeatHtml(pcnum, side) {
  const fp = pc.feat && pc.feat[pcnum]; if (!fp) return "";
  const v = pc.dedup ? "dedup" : "full";
  let list, lbl;
  if (pc.tfidf) {
    const pos = side === "high"; list = (fp.tfidf[v].dist || []).filter(t => pos ? t.r >= 0 : t.r < 0);
    lbl = `distinctive features ${pos ? "↑ (+r)" : "↓ (−r)"}`;
  } else { list = side === "high" ? fp.raw[v].pos : fp.raw[v].neg; lbl = side === "high" ? "features ↑ with PC (+r)" : "features ↓ with PC (−r)"; }
  if (!list || !list.length) return `<div class="enr-lib"><div class="enr-hd">${lbl}</div><div class="hint">none</div></div>`;
  const warn = fp.dirConf ? "" : ' <span class="enr-warn" title="low-variance PC: this axis is only weakly aligned to the feature-correlation embedding, so ↑/↓ direction is approximate">± dir approx</span>';
  let h = `<div class="enr-lib"><div class="enr-hd">${lbl}${warn}</div>`;
  for (const t of list) {
    const w = Math.max(2, Math.abs(t.r) * 100), pos = t.r >= 0;
    h += `<div class="enr-row"><span class="enr-track"><span class="enr-bar${pos ? "" : " neg"}" style="width:${w}%"></span><span class="enr-term" title="${t.f}">${t.f}</span></span><span class="enr-n" title="Pearson r${t.tfidf != null ? ` · tf-idf ${t.tfidf}` : ""}">r=${t.r >= 0 ? "+" : ""}${t.r}</span></div>`;
  }
  return h + "</div>";
}
// stacked composition bars: fraction of each PC's top features falling in each feature-class / organelle-group
function pcCompHtml(pcnum) {
  const fp = pc.feat && pc.feat[pcnum]; if (!fp) return "";
  const v = pc.dedup ? "dedup" : "full";
  const src = pc.tfidf ? fp.tfidf[v] : fp.raw[v], n = (pc.feat.meta || {}).compTopN || 50, N = pc.norm ? "N" : "";
  const bar = (comp, title) => {
    const segs = Object.entries(comp).map(([g, v2]) => {
      const pct = Math.round(v2 * 100);   // "99" alpha → translucent so the bar reads as see-through frosted glass
      return `<span class="comp-seg" style="width:${v2 * 100}%;background-color:${(pc.gcol[g] || "#888")}99" title="${g}: ${pct}%"><span class="comp-lbl">${g} ${pct}%</span></span>`;
    }).join("");
    return `<div class="comp-block"><div class="enr-hd">${title}</div><div class="comp-bar">${segs}</div></div>`;
  };
  const how = pc.norm ? "size-normalized share" : `share of top ${n}${pc.dedup ? " deduped" : ""} features`;
  return `<div class="pc-comp"><div class="sec-lbl">Feature composition <span class="hint">(${how}${pc.tfidf ? ", tf-idf-weighted" : ", by |r|"})</span></div>` +
    bar(src["cls" + N], "by feature class") + bar(src["org" + N], "by organelle group") + bar(src["src" + N], "by profiling tool") + "</div>";
}
// speedrichr-style bars per library: ordered by adjusted p-value, length = -log10(adj p), annotated with #overlap
function pcEnrichHtml(dir) {
  if (!dir) return "";
  let h = "";
  for (const lib of ["GO BP", "GO compartment", "Reactome", "KEGG"]) {
    const terms = dir[lib]; if (!terms || !terms.length) continue;
    const mx = Math.max(...terms.map(t => -Math.log10(t.adjp || 1)), 1);
    h += `<div class="enr-lib"><div class="enr-hd">${lib}</div>`;
    for (const t of terms.slice(0, 6)) {
      const w = Math.max(2, -Math.log10(t.adjp || 1) / mx * 100), sig = (t.adjp || 1) <= 0.05;
      const p = t.adjp < 0.001 ? t.adjp.toExponential(1) : t.adjp.toFixed(3);
      const kk = t.K ? `${t.n}/${t.K}` : `${t.n}`, pct = t.K ? ` (${Math.round(t.n / t.K * 100)}%)` : "";
      h += `<div class="enr-row"${sig ? "" : ' style="opacity:.45"'}><span class="enr-track"><span class="enr-bar" style="width:${w}%"></span><span class="enr-term">${t.term}</span></span><span class="enr-n" title="${t.n} of ${t.K || "?"} term genes${pct} · adj p ${p}">${kk} · ${p}</span></div>`;
    }
    h += "</div>";
  }
  return h;
}
function pcCell(i) {   // click a strip cell → glassy pop-out for close inspection (stackable, draggable)
  const c = pc.cells[i]; if (!c) return;
  popOut(`${pc.base}crops/${c.img}${PC_CROP_V}`, c.gene,
    `<div class="po-sub">PC${pc.cur} · score ${(+c.score).toFixed(2)}</div>` +
    `<div>${c.experiment} · well ${c.well} · (${c.x}, ${c.y})</div>` +
    `<a href="#" onclick="pcGeneOverlay('${c.gene}');return false">${c.gene} → top PCs ↗</a>`);
}
// glassy, draggable, stackable inset — you can still see the viewer through it
const PO_W = 450, PO_GAIN = 1.7;   // pop-out base width (matches .popout CSS); drag gain (>1 = tile moves faster than the cursor)
function popOut(imgUrl, title, metaHtml, ovUrl) {
  const el = document.createElement("div"); el.className = "popout";
  const pops = document.querySelectorAll(".popout").length;   // stagger horizontally by a full tile width, wrap rows
  const perRow = Math.max(1, Math.floor((window.innerWidth - 180) / (PO_W + 12)));
  el.style.left = `${150 + (pops % perRow) * (PO_W + 12)}px`;
  const rowStep = 30, rows = Math.max(1, Math.floor((window.innerHeight - 220) / rowStep));   // ~title-bar-height down-step per wrapped row (behind window's × bar stays fully visible); wraps so it never runs off the bottom
  el.style.top = `${100 + (Math.floor(pops / perRow) % rows) * rowStep}px`;
  const ov = (ovUrl && tc.mask) ? `<img class="po-ov" src="${ovUrl}">` : "";   // follow the live mask toggle at click time
  el.innerHTML = `<div class="po-bar"><span>${title}</span><button title="close">×</button></div>` +
    `<div class="po-img"><img src="${imgUrl}">${ov}</div><div class="po-body">${metaHtml}</div>`;
  el.querySelector("button").onclick = () => el.remove();
  const bar = el.querySelector(".po-bar"); let s = null;
  bar.addEventListener("mousedown", (e) => { s = { x: e.clientX, y: e.clientY, l: el.offsetLeft, t: el.offsetTop }; e.preventDefault(); });
  window.addEventListener("mousemove", (e) => { if (!s) return; el.style.left = `${s.l + (e.clientX - s.x) * PO_GAIN}px`; el.style.top = `${s.t + (e.clientY - s.y) * PO_GAIN}px`; });
  window.addEventListener("mouseup", () => { s = null; });
  document.body.appendChild(el);
}
function pcGeneOverlay(name) {
  const g = pc.data && pc.data.geneData[name]; if (!g) return;
  const mx = Math.max(...g.top_pcs.map(p => Math.abs(p.score))) || 1;
  const rows = g.top_pcs.map(p => {
    const col = p.score >= 0 ? "#7ee787" : "#ff5252", w = Math.abs(p.score) / mx * 46;
    const bar = p.score >= 0 ? `right:50%;width:${w}%;background:${col}` : `left:50%;width:${w}%;background:${col}`;
    return `<div class="ov-row" onclick="pcCloseOverlay();showPC(${p.pc})"><span class="ov-pc">PC${p.pc}</span><span class="ov-track"><span style="${bar}"></span></span><span class="ov-val" style="color:${col}">${p.score > 0 ? "+" : ""}${p.score.toFixed(2)}</span></div>`;
  }).join("");
  pcCloseOverlay();
  const ov = document.createElement("div"); ov.id = "pc-overlay"; ov.className = "pc-overlay";
  ov.onclick = (e) => { if (e.target === ov) pcCloseOverlay(); };
  ov.innerHTML = `<div class="pc-ov-card"><div class="pc-ov-hd"><span>${name}</span><button onclick="pcCloseOverlay()">×</button></div><div class="hint">Top PCs by |score| — click to open that PC</div>${rows}</div>`;
  document.body.appendChild(ov);
}
function pcCloseOverlay() { const o = $("pc-overlay"); if (o) o.remove(); }

// ---- Top Cells tab: per-gene top phenotype cells by attention or accuracy (phase), masked like the PC tab ----
const TC_CROP_V = "?m=3";
const tc = { data: null, marker: undefined, mode: "accuracy", mask: true, inorm: true, showAcc: true, accBin: 100, pinned: [{ gene: "NTC", mode: "accuracy" }] };   // inorm: marker-global intensity (fluor only); showAcc: per-group set-acc chip (default on) @accBin
function tcBase() {   // per-marker top-cells for fluor (marker channel crops), shared phase otherwise
  const m = state.marker && state.marker.marker_channel;
  return (!m || /^phase/i.test(m)) ? "top_cells/" : `top_cells/markers/${jsSlug(m)}/`;
}
async function loadTop() {
  const slug = tcBase();
  if (tc.data === null || tc.marker !== slug) {   // marker changed → load that marker's top-cells
    tc.marker = slug;
    tc.data = await fetch(`${BASE}${slug}index.json${NOCACHE}`).then(r => r.ok ? r.json() : null).catch(() => null);
  }
  if (!tc.data) { $("tc-view").innerHTML = '<div class="empty">No top-cell assets for this marker yet.</div>'; return; }
  ensureSetacc(() => { renderTopPins(); renderTop(); });   // load set-accuracy (bins) for the accuracy chip + adaptive bin dropdown
}
function tcData(name) {   // resolve a perturbation name → its ranking entry (genes, or complexes by base name)
  if (!tc.data) return null;
  if (tc.data.genes && tc.data.genes[name]) return tc.data.genes[name];
  const cx = tc.data.complexes;
  if (cx) { if (cx[name]) return cx[name]; const base = name.split(",")[0].trim(); for (const k in cx) if (k === base || name.startsWith(k)) return cx[k]; }
  return null;
}
function tcEntries() {   // pinned rows in their order; current prepended ONLY if not already pinned → pin-list order == grid order
  const cur = state.target ? state.target.target : null, out = [];
  const curPinned = cur && tc.pinned.some(p => p.gene === cur && p.mode === tc.mode);
  if (cur && !curPinned) out.push({ gene: cur, mode: tc.mode, current: true });
  for (const p of tc.pinned) out.push({ gene: p.gene, mode: p.mode, current: cur === p.gene && tc.mode === p.mode });
  return out;
}
function renderTop() {
  const v = $("tc-view"); if (!tc.data) return;
  const cap = tc.data.top_n || 20;
  const n = Math.max(1, Math.min(cap, state.cellCount || 10));   // count = header "Cells per page"
  const pg = Math.min(state.page, Math.max(0, Math.ceil(cap / n) - 1)), lo = pg * n;   // ◀ ▶ paginate the ranking
  const mk = attnModality(), bins = saBins(mk), sel = $("tc-accbin");   // adaptive bag-size dropdown (bins depend on marker: phase vs fluor)
  const inw = $("tc-inorm") && $("tc-inorm")._tog; if (inw) inw.style.display = mk === "phase" ? "none" : "";   // marker-intensity normalization is fluor-only — hide its toggle for phase
  if (sel) {
    if (isPublic()) tc.accBin = (bins.length && bins.includes(100)) ? 100 : (bins[bins.length - 1] || 100);   // public: fixed 100-cell bag, no selector
    else if (bins.length && !bins.includes(tc.accBin)) tc.accBin = bins.includes(100) ? 100 : bins[bins.length - 1];
    if (sel.dataset.bins !== bins.join(",")) { sel.innerHTML = bins.map(b => `<option value="${b}">${b} cells</option>`).join(""); sel.dataset.bins = bins.join(","); }
    sel.value = tc.accBin;
    const wrap = $("tc-accbin-wrap"); if (wrap) wrap.style.display = (tc.showAcc && bins.length > 1) ? "" : "none";   // public hidden via feat-internal (applyFeatureGate)
  }
  let h = "", ci = 0;
  for (const e of tcEntries()) {
    const gd = tcData(e.gene), cells = gd ? (gd[e.mode] || []).slice(lo, lo + n) : [];
    const sk = e.mode === "attention" ? "attn" : "conf";
    const color = PALETTE[ci++ % PALETTE.length];   // per-group color, matching the traversal groups
    const av = tc.showAcc ? saAcc(mk, e.gene, tc.accBin) : null;   // per-group SetTransformer set-accuracy at the selected bag size
    const accChip = tc.showAcc ? `<span class="tc-acc" title="SetTransformer real-cell set-accuracy · bag ${tc.accBin}" style="background:${av != null ? heat(av) : "rgba(255,255,255,.06)"};color:${av != null ? (av > 0.55 ? "#fff" : "#111") : "var(--fg)"}">${av != null ? Math.round(av * 100) + "%" : "—"}</span>` : "";
    h += `<div class="tc-row"><div class="tc-hd" style="color:${color}"><span class="gh-title" title="${e.gene}">${e.gene}</span>${accChip}</div><div class="pc-strip-row tc-strip" style="border-left:4px solid ${color}">`;
    if (!cells.length) h += `<div class="hint">no ${e.mode} cells${e.gene === "NTC" && e.mode === "accuracy" ? " (NTC has no accuracy ranking)" : ""}</div>`;
    for (const c of cells) {
      const cropDir = (tc.inorm && tcBase().includes("markers/")) ? "crops_norm" : "crops";   // marker-global intensity (fluor only)
      const url = `${BASE}${tcBase()}${cropDir}/${c.img}${TC_CROP_V}`;
      const ovUrl = c.ov ? `${BASE}${tcBase()}overlays/${c.ov}${TC_CROP_V}` : null;
      const ov = ovUrl ? `<img class="tc-ov" src="${ovUrl}">` : "";
      const meta = `<div>${c.exp} · well ${c.well} · (${c.x}, ${c.y})</div><div>${e.mode} ${sk}=${c[sk]} · rank ${c.rank}</div>`;
      const poArgs = `${JSON.stringify(url)},${JSON.stringify(e.gene + " · rank " + c.rank)},${JSON.stringify(meta)}${ovUrl ? "," + JSON.stringify(ovUrl) : ""}`;
      h += `<div class="tc-cell"><img class="pc-cell" src="${url}" title="${e.gene} · rank ${c.rank} · ${sk} ${c[sk]}" onclick='popOut(${poArgs})'>${ov}<span class="tc-rank">${c.rank}</span></div>`;
    }
    h += "</div></div>";
  }
  v.innerHTML = h || '<div class="empty">select a perturbation</div>';
  v.classList.toggle("masked", tc.mask !== false);   // show/hide the blue seg overlay layer
  $("tc-status").textContent = `ranks ${lo + 1}–${Math.min(cap, lo + n)} of ${cap} · ${tcEntries().length} row(s)`;
  $("cellrange").textContent = `${lo + 1}-${Math.min(cap, lo + n)} (${cap})`;   // top-bar rank range (matches traversal)
  $("pagenum").textContent = `Page ${pg + 1} / ${Math.max(1, Math.ceil(cap / n))}`;
}
function renderTopPins() {
  const ul = $("tc-panellist"); if (!ul) return; ul.innerHTML = "";
  const ents = tcEntries();   // color each pin to match its group color in the grid (same as the traversal pins)
  tc.pinned.forEach((p, i) => {
    const li = document.createElement("li"); li.innerHTML = `<span>⠿ ${p.gene} <span class="hint">· ${p.mode}</span></span>`;
    const gi = ents.findIndex(e => e.gene === p.gene && e.mode === p.mode);
    li.style.color = PALETTE[(gi < 0 ? i : gi) % PALETTE.length];
    li.draggable = true; li.style.cursor = "move";        // drag to reorder (like the traversal pins)
    const b = document.createElement("button"); b.textContent = "×";
    b.onclick = () => unpinShared(p.gene);
    li.ondragstart = e => { e.dataTransfer.setData("text/plain", i); e.dataTransfer.effectAllowed = "move"; };
    li.ondragover = e => { e.preventDefault(); li.style.opacity = ".5"; };
    li.ondragleave = () => { li.style.opacity = ""; };
    li.ondrop = e => { e.preventDefault(); li.style.opacity = ""; const from = +e.dataTransfer.getData("text/plain");
      if (from !== i) { const [m] = tc.pinned.splice(from, 1); tc.pinned.splice(i, 0, m); renderTopPins(); renderTop(); } };
    li.appendChild(b); ul.appendChild(li);
  });
}

window.addEventListener("error", (e) => { const m = $("meta"); if (m) m.textContent = "JS error: " + (e.message || e); });
boot().catch((e) => { const m = $("meta"); if (m) m.textContent = "boot error: " + (e.message || e); console.error(e); });
