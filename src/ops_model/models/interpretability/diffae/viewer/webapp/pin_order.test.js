// Feature spec + tests for the viewer's shared pin-ordering logic. Run:  node pin_order.test.js
// These call the SAME pin_order.js functions app.js runs, so a regression here IS a regression in the app.
//
// FEATURES of the pin-selection system:
//  A. Shared pin set across Traversal / Top Cells / Attention; pin order == state.pinned (newest-first).
//  B. New pins go to the TOP (unshift).
//  C. Stable colors: colorIdx = first free palette slot at pin time; existing pins never change; freed slots reused.
//  D. Reorder syncs across tabs (order is state.pinned, which every tab renders from).
//  E. NTC (Top Cells only), anchored by tc.ntcBefore = key of the pin NTC sits directly above (null = bottom):
//     E1 default = BOTTOM · E2 new pins to top DON'T move NTC · E3 NTC holds position relative to its neighbor on add
//     E4 removing a pin ABOVE NTC doesn't move it · E5 removing NTC's anchor re-anchors down (no jump) · E6 NTC follows its anchor on reorder.
const PO = require("./pin_order.js");
const PN = 8;
const mk = (k, colorIdx) => ({ key: k, target: k, markerName: "phase", colorIdx });
const pinAdd = (pins, k) => pins.some(p => p.key === k) ? pins : [mk(k, PO.firstFreeColor(pins, PN)), ...pins];   // = pinShared
const unpin = (pins, nb, target) => ({ pins: pins.filter(p => p.target !== target), ntcBefore: PO.reanchorOnUnpin(pins, nb, target) });
const drag = (pins, nb, from, to) => { const a = PO.tokens(pins, nb); const [m] = a.splice(from, 1); a.splice(to, 0, m); return { pins: a.filter(t => t.pin).map(t => t.pin), ntcBefore: PO.ntcBeforeFromTokens(a) }; };
const layout = (pins, nb) => PO.tokens(pins, nb).map(t => t.ntc ? "NTC" : t.pin.key).join(",");
const colors = pins => pins.map(p => `${p.key}:${p.colorIdx}`).join(",");

let pass = 0, fail = 0;
const eq = (got, want, name) => { const ok = got === want; console.log(`${ok ? "✓" : "✗"} ${name}${ok ? "" : `  got[${got}] want[${want}]`}`); ok ? pass++ : fail++; };

let pins = [], nb = null;
pins = pinAdd(pins, "A"); pins = pinAdd(pins, "B");
eq(layout(pins, nb), "B,A,NTC", "E1 default NTC bottom; B on top of A");
pins = pinAdd(pins, "C");
eq(layout(pins, nb), "C,B,A,NTC", "E2 new pin to top, NTC stays bottom");
eq(colors(pins), "C:2,B:1,A:0", "C stable colors (A=0,B=1,C=2)");
let r = drag(pins, nb, 3, 2); pins = r.pins; nb = r.ntcBefore;          // C,B,A,NTC -> C,B,NTC,A
eq(layout(pins, nb), "C,B,NTC,A", "E3a NTC dragged between B and A");
eq(nb, "A", "E3b ntcBefore == A");
pins = pinAdd(pins, "D");
eq(layout(pins, nb), "D,C,B,NTC,A", "E3c new pin D to top, NTC still before A");
eq(colors(pins), "D:3,C:2,B:1,A:0", "E3d D takes free slot 3");
r = unpin(pins, nb, "B"); pins = r.pins; nb = r.ntcBefore;
eq(layout(pins, nb), "D,C,NTC,A", "E4 remove B (above NTC), NTC still before A");
eq(colors(pins), "D:3,C:2,A:0", "E4b colors stable, slot 1 freed");
pins = pinAdd(pins, "E");
eq(colors(pins), "E:1,D:3,C:2,A:0", "C reuse freed slot 1 for E");
eq(layout(pins, nb), "E,D,C,NTC,A", "E4c new pin E to top, NTC before A");
r = unpin(pins, nb, "A"); pins = r.pins; nb = r.ntcBefore;
eq(layout(pins, nb), "E,D,C,NTC", "E5 remove anchor A (last), NTC drops to bottom (no jump)");
eq(nb, null, "E5b ntcBefore null (bottom)");
pins = ["X", "Y", "Z"].map((k, i) => mk(k, i)); nb = "Y";                // X,NTC,Y,Z
eq(layout(pins, nb), "X,NTC,Y,Z", "E5c setup NTC before Y (middle)");
r = unpin(pins, nb, "Y"); pins = r.pins; nb = r.ntcBefore;
eq(layout(pins, nb), "X,NTC,Z", "E5d remove middle anchor Y, NTC re-anchors before Z (stays put)");
eq(nb, "Z", "E5e ntcBefore == Z");
r = drag(pins, nb, 2, 0); pins = r.pins; nb = r.ntcBefore;              // X,NTC,Z -> Z,X,NTC
eq(layout(pins, nb), "Z,X,NTC", "E6 reorder Z to top; NTC follows its anchor");

// ---- cross-tab PANEL ORDER consistency (feature D): every tab renders the same shared-pin order ----
// Each tab's grid keys: Traversal = gridOrder; Attention = gridOrder filtered to has-attention; Top Cells = gridOrder + NTC.
const pinnedKeys = ["C", "B", "A"];                    // newest-first
const subseq = keys => keys.filter(k => pinnedKeys.includes(k)).join(",");   // pin subsequence
const hasAttn = k => k !== "B";                        // pretend B has no attention data
// current NOT pinned:
let trav = PO.gridOrder(pinnedKeys, "CUR");
let attn = PO.gridOrder(pinnedKeys, "CUR").filter(k => hasAttn(k) || k === "CUR").filter(hasAttn);
let topPins = PO.tokens(["C", "B", "A"].map(k => mk(k, 0)), null).filter(t => t.pin).map(t => t.pin.key);
eq(trav.join(","), "CUR,C,B,A", "D1 traversal: current prepended (not pinned) + pins");
eq(subseq(trav), "C,B,A", "D2 traversal pin order == state.pinned");
eq(subseq(attn), "C,A", "D3 attention pin order == state.pinned (B filtered, order preserved)");
eq(subseq(topPins), "C,B,A", "D4 top cells pin order == state.pinned");
// current IS pinned (B): must NOT be duplicated/moved to front in ANY tab
trav = PO.gridOrder(pinnedKeys, "B");
attn = PO.gridOrder(pinnedKeys, "B").filter(hasAttn);
eq(trav.join(","), "C,B,A", "D5 pinned-current NOT prepended (stays in slot) — traversal");
eq(subseq(trav), "C,B,A", "D6 traversal pin order intact with pinned current");
eq(attn.join(","), "C,A", "D7 attention same slot order (B was the current & filtered)");

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
