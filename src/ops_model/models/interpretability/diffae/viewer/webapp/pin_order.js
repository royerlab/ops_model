// Pure pin-ordering logic for the DiffAE viewer's shared pin system.
// Loaded before app.js (browser global `PinOrder`) AND require()-d by pin_order.test.js (node) — the test
// exercises the SAME code the app runs, so it can't drift. See pin_order.test.js for the full feature spec.
//
// A pin = { key, target, markerName, colorIdx, ... }. state.pinned is newest-first (new pins unshift to index 0).
// NTC (Top Cells only) is positioned by a single anchor: tc.ntcBefore = key of the pin NTC sits directly ABOVE
// (null = bottom, the default). Pin order itself is always state.pinned, so cross-tab reorder syncs for free.
(function (root) {
  // Row tokens for Top Cells: pins in shared order + NTC inserted before its anchor pin (or bottom).
  function tokens(pinned, ntcBefore) {
    const toks = pinned.map(p => ({ pin: p }));
    let idx = toks.length;                                              // bottom: default, or anchor pin unpinned
    if (ntcBefore != null) { const bi = pinned.findIndex(p => p.key === ntcBefore); if (bi >= 0) idx = bi; }
    toks.splice(idx, 0, { ntc: true });
    return toks;
  }
  // Stable color: lowest palette slot not used by existing pins (freed slots get reused).
  function firstFreeColor(pinned, paletteLen) {
    const used = new Set(pinned.map(p => p.colorIdx).filter(x => x != null));
    for (let i = 0; i < paletteLen; i++) if (!used.has(i)) return i;
    return pinned.length % paletteLen;
  }
  // Removing `target`: if NTC's anchor is among the removed pins, re-anchor to the next pin below (else bottom).
  function reanchorOnUnpin(pinned, ntcBefore, target) {
    if (!pinned.some(p => p.target === target && p.key === ntcBefore)) return ntcBefore;
    const bi = pinned.findIndex(p => p.key === ntcBefore);
    for (let j = bi + 1; j < pinned.length; j++) if (pinned[j].target !== target) return pinned[j].key;
    return null;
  }
  // After a drag that produced this token array, the pin NTC now sits above (null = bottom).
  function ntcBeforeFromTokens(tokenArr) {
    const ni = tokenArr.findIndex(t => t.ntc);
    return (ni + 1 < tokenArr.length && tokenArr[ni + 1].pin) ? tokenArr[ni + 1].pin.key : null;
  }
  // Canonical grid order EVERY tab renders from: the current perturbation is prepended ONLY if it isn't already
  // pinned (a pinned current stays in its state.pinned slot) → identical panel order across Traversal / Top Cells /
  // Attention. Per-tab views then just filter (Attention drops no-attention pins) or augment (Top Cells adds NTC),
  // but the shared-pin subsequence is always this order.
  function gridOrder(pinnedKeys, currentKey) {
    return (currentKey != null && !pinnedKeys.includes(currentKey)) ? [currentKey].concat(pinnedKeys) : pinnedKeys.slice();
  }
  const api = { tokens, firstFreeColor, reanchorOnUnpin, ntcBeforeFromTokens, gridOrder };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.PinOrder = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
