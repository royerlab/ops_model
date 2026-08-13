"""Create the sister Confluence page mirroring the v4/v5 DiffAE validation page, with the new multibag bag-sweep
figures. REST flow (create -> upload attachments -> PUT storage body with <ac:image>). Idempotent-ish: pass an
existing PAGE_ID env to update instead of create."""
import os, json, base64, urllib.request

USER = "gav.sturm@czbiohub.org"; TOKEN = os.environ["CONFLUENCE_API_TOKEN"]
BASE = "https://czbiohub.atlassian.net/wiki"
SPACE_ID = "3319857206"; PARENT = "5538218009"
TITLE = "DiffAE multibag-traversal validation — bag-size sweep (Figure 4)"
PLOTS = "/hpc/projects/icd.fast.ops/analysis/figure4_traversals/bag_sweep_plots_v5new"
IMGS = ["settransformer_bagsweep_all.png", "settransformer_bagsweep_realdist.png",
        "centroid_bagsweep.png", "centroid_bagsweep_global.png", "centroid_pooled_bagsweep.png", "centroid_pooled_bagsweep_global.png",
        "control_halves_zscore.png", "anchor_halves.png", "st_anchor_halves.png", "retrieval_map_proper1k.png",
        "distinct_sweep_median.png", "distinct_sweep_mean.png", "distinct_violin.png"]


def _auth():
    return "Basic " + base64.b64encode(f"{USER}:{TOKEN}".encode()).decode()


def _req(url, data=None, method="GET", ctype="application/json", raw=False):
    h = {"Authorization": _auth()}
    if ctype:
        h["Content-Type"] = ctype
    r = urllib.request.Request(url, data=data, method=method, headers=h)
    with urllib.request.urlopen(r) as resp:
        return resp.read() if raw else json.load(resp)


def create():
    body = {"type": "page", "title": TITLE, "space": {"key": "dashboard"},
            "ancestors": [{"id": PARENT}], "body": {"storage": {"value": "<p>scaffold</p>", "representation": "storage"}}}
    d = _req(f"{BASE}/rest/api/content", json.dumps(body).encode(), "POST")
    return d["id"]


def upload(pid, path):
    """Upload new, or update-by-id if the filename already exists (idempotent re-runs). Returns the attachment id
    (or None if missing) so the body can build a working REST download link."""
    import subprocess
    if not os.path.exists(path):
        print("skip (missing):", os.path.basename(path)); return None
    fn = os.path.basename(path)
    ex = _req(f"{BASE}/rest/api/content/{pid}/child/attachment?filename={fn}")
    url = f"{BASE}/rest/api/content/{pid}/child/attachment"
    if ex.get("results"):
        url += f"/{ex['results'][0]['id']}/data"
    subprocess.run(["curl", "-s", "-u", f"{USER}:{TOKEN}", "-X", "POST", "-H", "X-Atlassian-Token: nocheck",
                    "-F", f"file=@{path}", url], check=True, capture_output=True)
    q = _req(f"{BASE}/rest/api/content/{pid}/child/attachment?filename={fn}")
    return q["results"][0]["id"] if q.get("results") else None


PID = os.environ.get("PAGE_ID", "")
SVG_ATT = {}                                                # svg filename -> attachment id (populated during upload)


def img(fn, w=1000):
    svg = fn.rsplit(".", 1)[0] + ".svg"
    att = SVG_ATT.get(svg)
    link = (f'<p style="text-align:center"><a href="{BASE}/rest/api/content/{PID}/child/attachment/{att}/download">'
            f'⬇ download SVG (vector)</a></p>' if att else "")
    return (f'<ac:image ac:layout="wide" ac:align="center" ac:width="{w}">'
            f'<ri:attachment ri:filename="{fn}"/></ac:image>{link}')


def img_cell(fn, w=620):
    """Inline (non-wide) image for a table cell — two of these sit side by side in a 2-col table."""
    svg = fn.rsplit(".", 1)[0] + ".svg"
    att = SVG_ATT.get(svg)
    link = (f'<p style="text-align:center"><a href="{BASE}/rest/api/content/{PID}/child/attachment/{att}/download">'
            f'⬇ SVG</a></p>' if att else "")
    return f'<ac:image ac:width="{w}"><ri:attachment ri:filename="{fn}"/></ac:image>{link}'


def build_body():
    return f"""
<ac:structured-macro ac:name="info"><ac:rich-text-body>
<p><strong>Headline.</strong> The <strong>new multibag-ranked</strong> DiffAE phase traversals (viewer_assets_v5, 400 cells × 17 α, w=1.5, 100 DDIM steps) <strong>recover the strong v4-era validation</strong> that the earlier single-anchor "valid200" run had lost. Across all three independent checks the recovery signature is back: SetTransformer real-distinguishable recovery returns to ~46% top-1 (geneKO) / ~86% (complex) and P(target) near the real-cell ceiling; pooled centroid recovery (now <strong>per-bag α=0 standardized</strong> — the domain-honest metric established by the control below) reaches bag-200 top-1 ~48% (geneKO) / ~69% (complex) against a per-bag real-cell ceiling of 66% / 89%; and the classifier peak returns to α≈0.5–1.5.</p>
<p>This page is the bag-size-sweep sister of the <a href="{BASE}/spaces/dashboard/pages/5559091239">v4/v5 DiffAE generative-cell validation page</a>: every quantitative measure is re-scored at bag sizes <strong>{{20, 50, 100, 200, 400}}</strong> on the full <strong>1,000 geneKO + 95 complex</strong> panel.</p>
</ac:rich-text-body></ac:structured-macro>

<ac:structured-macro ac:name="warning"><ac:rich-text-body>
<p><strong>MAJOR FINDING — the anchor-half "gap" is a centroid-metric standardization artifact, not phenotype loss.</strong> The 400-cell traversals concatenate two 200-cell anchor sets with <em>identical directions and traversals</em>: cells <strong>0–199 = old hand-picked (curated favourite) NTC anchors</strong>; cells <strong>200–399 = strict multibag top-200 NTC anchors</strong>. The apparent recovery gap appears only in the position-sensitive centroid metric, and only under its global standardization — it vanishes once standardization is matched to the classifier's:</p>
<table><tbody>
<tr><td></td><td><strong>hand-picked (first 200)</strong></td><td><strong>strict multibag NTC (second 200)</strong></td><td><strong>gap</strong></td></tr>
<tr><td>centroid top-1, <strong>global</strong> α=0 std — geneKO</td><td><strong>76%</strong></td><td><strong>13%</strong></td><td>63 pts</td></tr>
<tr><td>centroid top-1, <strong>global</strong> α=0 std — complex</td><td><strong>86%</strong></td><td><strong>40%</strong></td><td>46 pts</td></tr>
<tr><td>centroid top-1, <strong>per-bag</strong> α=0 std — geneKO</td><td>48%</td><td>42%</td><td><strong>7 pts</strong></td></tr>
<tr><td>centroid top-1, <strong>per-bag</strong> α=0 std — complex</td><td>69%</td><td>64%</td><td><strong>5 pts</strong></td></tr>
<tr><td>SetTransformer top-1 — geneKO</td><td>10%</td><td>9%</td><td>~0</td></tr>
<tr><td>SetTransformer top-1 — complex</td><td>52%</td><td>54%</td><td>~0</td></tr>
</tbody></table>
<p><strong>Mechanism.</strong> The centroid metric standardizes generated cells against a <em>global / panel</em> α=0 mean (shared <code>{{grain}}_mu.npz</code>), so a half-specific CellDINO offset survives and inflates the gap — it even pushes the second half's optimal α out to α=3 (extreme α needed to drag the offset half onto its target). <code>score_embs_v5</code> (SetTransformer) instead <em>self-standardizes each bag on its own α=0 generated frames</em>, cancelling that offset — which is exactly why it sees no gap. Applying the same <strong>per-bag α=0 standardization to the centroid metric collapses the gap from 63→7 pts (geneKO) and 46→5 pts (complex)</strong>, and restores the normal α≈0.5–1 peak.</p>
<p><strong>Conclusion.</strong> The two anchor sets produce the <strong>same perturbation phenotype</strong> (the supervised classifier — the gold standard — cannot tell them apart). The 76/13 is a standardization choice in the centroid metric, so the <strong>bag=400 "dilution" is not real phenotype loss.</strong> The other bag-sweep plots still use bags <strong>20–200</strong> for a consistent global-standardized reference.</p>
</ac:rich-text-body></ac:structured-macro>
{img("control_halves_zscore.png", 1000)}
<p><em>The control. Peak-α centroid top-1 under global (panel α=0) vs per-bag (each half's own α=0) standardization. The large first/second gap under global collapses under per-bag — definitively a metric standardization artifact.</em></p>
{img("anchor_halves.png", 1150)}
<p><em>Centroid recovery under the current global standardization — first-200 (hand-picked, blue) vs second-200 (strict multibag top-NTC, red). Same directions/traversals; only the anchor cells differ. The gap here is the artifact the control above dissolves.</em></p>
{img("st_anchor_halves.png", 1150)}
<p><em>SetTransformer on the same two halves (P(target) · median rank · mean rank · top-5/top-1). Near-identical — the self-standardizing classifier sees the same phenotype from both anchor sets.</em></p>

<h2>Source on Bruno</h2>
<table><tbody>
<tr><td><strong>Cache build</strong></td><td><code>ops_model/.../diffex/figures/gen_validation/valid200_cache_build.py</code> (V200_ASSETS=viewer_assets_v5, NCELL=400)</td></tr>
<tr><td><strong>Scoring</strong></td><td><code>bag_sweep_score.py</code> (SetTransformer) · <code>centroid_bagsweep.py</code> (centroid recovery) · <code>gen_real_distinct.py</code> (distinctiveness)</td></tr>
<tr><td><strong>Plots</strong></td><td><code>bag_sweep_plots.py</code></td></tr>
<tr><td><strong>Cache</strong></td><td><code>/hpc/projects/icd.fast.ops/analysis/figure4_traversals/gen_real_map_cache_v5new/</code></td></tr>
<tr><td><strong>Outputs</strong></td><td><code>/hpc/projects/icd.fast.ops/analysis/figure4_traversals/{{bag_sweep_v5new, centroid_bagsweep_v5new, gen_real_distinct_v5new_K*, bag_sweep_plots_v5new}}/</code></td></tr>
</tbody></table>

<h2>How to read these plots</h2>
<p>Each measure is scored on <strong>bags of generated cells</strong> and plotted vs the traversal step α (x-axis, integer ticks −5…5; α=0 = real control, α≈1 = the class mean, |α|&gt;1 = exaggerated). One <strong>line per bag size</strong> (viridis: dark=20 → yellow=400). geneKO / complex are the two rows. "Higher is better" on every y-axis except <em>target rank</em> (1 = the classifier's top pick, lower is better).</p>
<p>Bags are the first-B cells (nested), restricted to <strong>20–200 = the hand-picked anchor half</strong> for a consistent reference (bag=400 mixes in the second-200 strict-multibag anchors; per the finding above their apparent "dilution" is a centroid-standardization artifact, not weaker phenotypes). Within that half, the pooled SetTransformer set-classifier and pooled centroid recovery both <em>improve</em> with bag size (more cells = more evidence), saturating by ~100–200.</p>

<h2>1 · SetTransformer set-classifier (rank / P(target) / top-k)</h2>
<p>An independently-trained v5 SetTransformer (Alex Lin; real cells only, never exposed to DiffAE traversals) scores each generated bag to its target class. Four columns: P(target), median target rank, mean target rank (outlier-sensitive), and % top-5 / top-1 recovered. Dashed real-cell references where applicable.</p>
{img("settransformer_bagsweep_all.png", 1200)}
<p><em>All classes (1,000 geneKO / 95 complex).</em></p>
{img("settransformer_bagsweep_realdist.png", 1200)}
<p><em>Real-distinguishable subset only (genes whose REAL cells score top-1 accuracy &gt; 0.5 @ bag-20). Here generated traversals track close to the real-cell ceiling — geneKO ~46% top-1, complex ~86% top-1 at peak α≈1–1.5; complex P(target) rises to ~0.85 at bag 400.</em></p>

<h2>2 · Centroid recovery (generated → nearest real centroid)</h2>
<p>Classifier-independent: does a generated cell land nearest its true class's faithful real centroid (Cell-DINO), rather than a neighbour? mAP (1/rank of the true centroid), % top-1, % top-5. <strong>Now scored with per-bag α=0 standardization</strong> (each gene's own α=0 frames — matching <code>score_embs_v5</code>, the domain-honest metric established by the control above; the old global-mu numbers are preserved on disk).</p>
<table><tbody>
<tr><th>Per-bag α=0 standardized (domain-honest — correct)</th><th>Global-mu standardized (previous — inflated)</th></tr>
<tr><td>{img_cell("centroid_bagsweep.png")}</td><td>{img_cell("centroid_bagsweep_global.png")}</td></tr>
</tbody></table>
<p><em>Per-cell centroid recovery, per-bag (left) vs global-mu (right). <strong>Left (correct):</strong> geneKO mAP ~0.24 (top-1 ~15%) at α≈+4, complex ~0.53 (top-1 ~39%) at α≈+3; negative α recovers nothing. <strong>Right (old):</strong> geneKO mAP ~0.335 (the "v4 baseline"), complex ~0.65 — higher, but standardization-inflated per the control above, not truer recovery.</em></p>

<h3>2b · Pooled bag-level centroid recovery (with per-bag real ceiling)</h3>
<p>The bag-level view you actually want for a bag-size sweep: pool B cells → the generated class centroid → does <em>that</em> land nearest the true real centroid? More cells = a better centroid estimate, so recovery rises with bag; the dotted line is the matched per-bag <strong>real-cell</strong> ceiling (bootstrap B real cells → mean). Per-bag α=0 standardized.</p>
<table><tbody>
<tr><th>Per-bag α=0 standardized (domain-honest — correct)</th><th>Global-mu standardized (previous — inflated)</th></tr>
<tr><td>{img_cell("centroid_pooled_bagsweep.png")}</td><td>{img_cell("centroid_pooled_bagsweep_global.png")}</td></tr>
</tbody></table>
<p><em>Pooled bag-level recovery, per-bag (left) vs global-mu (right). Solid = generated, dotted = per-bag real-cell ceiling; all 1,000 geneKO / 95 complex. <strong>Left (correct):</strong> bag-200 top-1 ~48% (geneKO) / ~69% (complex), peak α≈+0.5, below the ~66% / ~89% real ceilings. <strong>Right (old):</strong> geneKO ~76% (slightly <em>exceeds</em> its ceiling), complex ~86% — the more impressive-looking result, but standardization-inflated per the control above.</em></p>

<h2>3 · Distinctiveness (how separable generated cells are from other classes)</h2>
<p>Within-domain copairs mAP: do a class's generated cells cluster together and apart from other classes? Dotted = real-cell reference. Both median (robust) and mean (outlier-sensitive) shown. <strong>K (cells/class) coverage differs by grain:</strong> complex (95 classes) extends to K = {{20, 50, 100, 200}}; geneKO (1,000 classes) reaches <strong>K = 100</strong> on a 700 GB himem node (peak ~412 GB). geneKO <strong>K = 200 is not run</strong> — it needs &gt; 700 GB (peak hit 705 GB at the limit), exceeding the largest available node (773 GB), so it would require a GPU/chunked-AP rewrite rather than just more memory. The cached real reference is 30 cells/class, so the dotted real line is the same 30-cell reference at every K.</p>
{img("distinct_sweep_median.png", 1000)}
<p><em>Median distinctiveness mAP. (geneKO: K=20/50/100; complex: K=20/50/100/200.)</em></p>
{img("distinct_sweep_mean.png", 1000)}
<p><em>Mean distinctiveness mAP.</em></p>
{img("distinct_violin.png", 950)}
<p><em>Per-class distinctiveness distribution (top-50 cells/class) at each grain's peak α — real (blue) vs generated (green), median bar. Generated exceeds real for gene-KOs and matches/exceeds for complexes, reflecting the low-diversity effect (centroid-directed generated cells cluster tighter than real cells).</em></p>

<h2>Not yet ported from the v4/v5 page</h2>
<p>For completeness — two measures on the <a href="{BASE}/spaces/dashboard/pages/5559091239">sister v4/v5 page</a> are <strong>not</strong> reproduced here yet:</p>
<ul>
<li><strong>Cross-domain retrieval mAP</strong> (mAP_proper1k) — for each generated class-X cell, the average precision of retrieving real class-X cells ahead of all others (gen→real 1000-cell gallery). A genuine third mAP-family measure. <strong>Complex is done; geneKO timed out</strong> (the full 1,000-cell/class GPU gallery pass hit the 2 h wall). It also uses the same global-mu path as §2 and so needs the per-bag redo; rebuild pending (capped gallery + per-bag standardization).</li>
<li><strong>Reach-fraction vs bag</strong> (v5_reachfrac) — % of real-distinguishable classes whose generated accuracy reaches the real-cell level, per bag. Needs a fresh <code>bag_scaling</code> run on the new traversals (the old bagtest data has been cleared); real-cell per-bag accuracy is no longer on disk to reuse.</li>
</ul>
<p>The embedding-projection sections (UMAP/PHATE placement, complex montages, §4–6 of the v4/v5 page) are separate analyses, not part of this bag-size sweep.</p>
"""


def main():
    pid = os.environ.get("PAGE_ID") or create()
    print("page:", pid)
    for fn in IMGS:
        upload(pid, f"{PLOTS}/{fn}"); print("uploaded", fn)
        svg = f"{fn.rsplit('.', 1)[0]}.svg"
        aid = upload(pid, f"{PLOTS}/{svg}")                          # vector for download link
        if aid:
            SVG_ATT[svg] = aid
    cur = _req(f"{BASE}/rest/api/content/{pid}?expand=version")["version"]["number"]
    payload = {"version": {"number": cur + 1, "message": "multibag bag-sweep figures"},
               "title": TITLE, "type": "page", "body": {"storage": {"value": build_body(), "representation": "storage"}}}
    v = _req(f"{BASE}/rest/api/content/{pid}", json.dumps(payload).encode(), "PUT")["version"]["number"]
    print(f"PUT ok, version {v}")
    print(f"URL: {BASE}/spaces/dashboard/pages/{pid}")


if __name__ == "__main__":
    main()
