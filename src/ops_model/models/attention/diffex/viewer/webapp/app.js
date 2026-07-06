// DiffEx traversal viewer — static, reads viewer_assets/manifest.json (sibling of this app,
// or set window.MANIFEST_URL). No build step, no backend; α scrubs precomputed WebP frames.
const MANIFEST_URL = window.MANIFEST_URL || "manifest.json";
// asset paths in the manifest are relative to the manifest's directory (viewer_assets/)
const BASE = MANIFEST_URL.replace(/manifest\.json$/, "");

const $ = (id) => document.getElementById(id);
const state = { manifest: null, marker: null, targets: [], target: null, cell: 0, frames: [], playing: false };

async function boot() {
  state.manifest = await (await fetch(MANIFEST_URL)).json();
  const mkSel = $("marker");
  state.manifest.markers.forEach((m, i) => {
    const o = document.createElement("option");
    o.value = i; o.textContent = m.marker_channel || "Phase";
    mkSel.appendChild(o);
  });
  mkSel.onchange = () => selectMarker(+mkSel.value);
  $("grain").onchange = refreshTargets;
  $("filter").oninput = refreshTargets;
  $("target").onchange = () => selectTarget($("target").value);
  $("cell").onchange = () => { state.cell = +$("cell").value; loadFrames(); };
  $("alpha").oninput = () => showFrame(+$("alpha").value);
  $("play").onclick = togglePlay;
  selectMarker(0);
}

function selectMarker(i) {
  state.marker = state.manifest.markers[i];
  refreshTargets();
}

function refreshTargets() {
  const g = $("grain").value, q = $("filter").value.trim().toLowerCase();
  state.targets = state.marker.targets.filter(t =>
    (g === "all" || t.grain === g) && t.target.toLowerCase().includes(q));
  const sel = $("target");
  sel.innerHTML = "";
  state.targets.forEach(t => {
    const o = document.createElement("option");
    const m = t.dist_map != null ? ` (${t.dist_map.toFixed(2)})` : "";
    o.value = t.slug; o.textContent = `${t.target}${m}${t.grain === "complex" ? "  ·cx" : ""}`;
    sel.appendChild(o);
  });
  if (state.targets.length) { sel.value = state.targets[0].slug; selectTarget(state.targets[0].slug); }
}

function selectTarget(slug) {
  state.target = state.targets.find(t => t.slug === slug);
  if (!state.target) return;
  const cSel = $("cell");
  cSel.innerHTML = "";
  for (let c = 0; c < state.target.n_cells; c++) {
    const o = document.createElement("option"); o.value = c; o.textContent = `cell ${c}`; cSel.appendChild(o);
  }
  state.cell = 0; cSel.value = 0;
  loadFrames();
}

function loadFrames() {
  const alphas = state.target.alphas || state.manifest.alphas;  // per-target (cache fills incrementally)
  state.alphas = alphas;
  state.frames = alphas.map((_, i) =>
    `${BASE}${state.target.asset_dir}/cell${state.cell}/frame_${String(i).padStart(2, "0")}.webp`);
  state.frames.forEach(src => { const im = new Image(); im.src = src; });  // preload
  $("alpha").max = alphas.length - 1;
  const mid = Math.floor(alphas.length / 2);
  $("alpha").value = mid; showFrame(mid);
  $("meta").textContent =
    `${state.marker.marker_channel || "Phase"} · ${state.target.grain} · ${state.target.target}` +
    (state.target.dist_map != null ? ` · distinctiveness mAP ${state.target.dist_map.toFixed(3)}` : "") +
    ` · cell ${state.cell} · w=${state.manifest.w}`;
}

function showFrame(i) {
  const a = state.alphas[i];
  $("frame").src = state.frames[i];
  $("alpha-read").textContent = `α = ${a.toFixed(1)}`;
  const lbl = $("dir-label");
  const name = state.target.target;
  if (a > 0.05) { lbl.textContent = `→ ${name} KO`; lbl.style.color = "var(--pos)"; }
  else if (a < -0.05) { lbl.textContent = `← anti-${name}`; lbl.style.color = "var(--neg)"; }
  else { lbl.textContent = "NTC (unperturbed)"; lbl.style.color = "var(--mid)"; }
}

function togglePlay() {
  state.playing = !state.playing;
  $("play").textContent = state.playing ? "❚❚" : "▶";
  if (state.playing) animate();
}
let dir = 1;
function animate() {
  if (!state.playing) return;
  const s = $("alpha"); let v = +s.value + dir;
  if (v >= +s.max) { v = +s.max; dir = -1; } else if (v <= 0) { v = 0; dir = 1; }
  s.value = v; showFrame(v);
  setTimeout(() => requestAnimationFrame(animate), 160);
}

boot();
