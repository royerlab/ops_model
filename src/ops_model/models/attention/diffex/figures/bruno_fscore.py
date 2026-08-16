"""F-rescale variant of bruno_final's violins/cellgrid, per MORPH_F_RESCALE_HANDOFF.md: replace the raw
traversal alpha axis with phenotype-fraction phi = alpha/f (f = each perturbation's own centroid-recovery
peak-alpha), so phi=0/1/2/3 sit at the SAME biological milestone across mTOR/POLR1B/TIM23.

f values (alpha units, from f_centroid_recovery/f_all.json + centroid_recovery_fluor/*.json):
  POLR1B=2.45, MTOR=1.38, TIM23=2.25
alpha_k = k*f, snapped to the nearest already-generated grid alpha within tolerance (0.25 in [-3,3], 0.5
outside -- per the handoff's own rule); phi=3 exceeds the model's max alpha (5) for POLR1B/TIM23, so it's
CAPPED at alpha=5 and rendered as a duplicate of phi=2 -- explicitly labeled "capped", not silently exact.

Merges each group's existing stats.npz/panel.npz (already has alpha 0/1.5/2.5/3 etc.) with a small
new_alpha.npz/new_alpha_panel.npz (the ONE new grid point each perturbation needed: alpha=4 for mTOR,
alpha=5 for POLR1B/TIM23) -- no remeasurement of anything already on disk.

Run: python bruno_fscore.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import bruno_final as bf
import morpho_native as mn

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

NAT = mn.OUT
OUT = "/hpc/projects/icd.fast.ops/analysis/figure4_traversals_violin/bruno_fscore"
PHIS = [0, 0.5, 1, 1.5, 2, 3]
PHI_LABELS = [f"φ={p:g}" for p in PHIS]

# (display group, base dirname (existing full measurement), fscore gname (holds the ONE new alpha), feats)
FGROUPS = [
    ("mTOR", "mtor_mo_hm_100", "mtor_mo_hm_fscore", bf.GROUPS[0][2]),
    ("POLR1B", "polr1b_vsnpm3_stringentcpu", "polr1b_vsnpm3_fscore", bf.GROUPS[1][2]),   # stringent (mo_local_adjust 1.4) gen seg, validated against real NTC/KO
    ("TIM23", "tim23_100", "tim23_fscore", bf.GROUPS[2][2]),
    ("TAF1B", "taf1b_vsnpm3_stringent", "taf1b_vsnpm3_fscore",
     [("circularity", "Nucleolar circularity", ""), ("area", "Nucleolar area", "(µm²)")]),
]


_GKEY = {"mTOR": "mtor_mo_hm", "POLR1B": "polr1b_vsnpm3", "TIM23": "tim23", "TAF1B": "taf1b_vsnpm3",
         "SAMM50": "samm50_chromalive", "MICOS13": "micos13_chromalive"}


def _gkey(group):
    return _GKEY[group]


def _key_variants(feat, a):
    """The base group's DEFAULT_ALPHAS mixes int/float literals (gen_area_a3) while the new-alpha
    measurement used a plain float (gen_area_a4.0) -- try both canonical forms."""
    return (f"gen_{feat}_a{a}", f"gen_{feat}_a{a:g}")


def _get(z, feat, a):
    for k in _key_variants(feat, a):
        if k in z:
            return z[k]
    raise KeyError(f"no key for gen_{feat}_a{a} in {list(z.files)[:5]}...")


def _merged_stats(dirname, fscore_gname):
    base = dict(bf._load(dirname))
    new = np.load(f"{NAT}/{fscore_gname}/new_alpha.npz")
    for k in new.files:
        base[k] = new[k]
    return base


def _phi_series(z, group, feat, normalize, real_top100=False):
    key = f"rn100_{feat}" if real_top100 else f"rn_{feat}"
    rkey = f"rk100_{feat}" if real_top100 else f"rk_{feat}"
    rn, rk = z[key], z[rkey]
    phi_alpha = mn.FSCORE_PHI_ALPHA[f"{_gkey(group)}_fscore"]
    phis = [(_get(z, feat, phi_alpha[p]) if phi_alpha[p] is not None else np.array([])) for p in PHIS]
    vals = [rn, rk, *phis]
    if feat in bf.AREA_FEATS:
        vals = [v * bf.PX_UM ** 2 if len(v) else v for v in vals]
    if not normalize:
        return vals
    base = float(np.nanmean(vals[0]))
    return [bf._pct(v, base) if len(v) else v for v in vals]


def render_violin(group, dirname, fscore_gname, feat, disp, unit, normalize, real_top100=False, suffix="", data_override=None, unit_newline=False):
    if data_override is not None:
        data = [np.asarray(d, float) for d in data_override]
    else:
        z = _merged_stats(dirname, fscore_gname)
        data = [np.asarray(d, float) for d in _phi_series(z, group, feat, normalize, real_top100=real_top100)]
    data = [d[np.isfinite(d)] for d in data]                            # KDE and median below use this FULL data, unfiltered --
    labs = ["real\nNTC", "real\nKO"] + PHI_LABELS                       # only the axis view is clipped (see ax.set_ylim below), matching
                                                                          # the protein-complex panels' "crop the display only" convention
    keep = [i for i, d in enumerate(data) if len(d)]                    # missing phi columns are DROPPED, not left as a gap -- compact the remaining ones together
    data = [data[i] for i in keep]; labs = [labs[i] for i in keep]
    pos = list(range(len(keep)))
    fig, ax = plt.subplots(figsize=(9.6, 5.4), facecolor="white")
    C = ["#999999", "#2e8b57", "#c6dbef", "#aed4ec", "#9ecae1", "#6baed6", "#3182bd", "#08519c"]
    C = [C[i] for i in keep]
    parts = ax.violinplot(data, positions=pos, showmeans=False, showextrema=False, showmedians=False, widths=0.82)
    for pc, c in zip(parts["bodies"], C):
        pc.set_facecolor(c); pc.set_alpha(0.6); pc.set_edgecolor(c); pc.set_linewidth(1.5)
    for i in pos:
        ax.hlines(np.median(data[i]), i - 0.34, i + 0.34, color="#222", lw=3, zorder=5)
    if normalize:
        ax.axhline(0, color="#999", lw=1.5, zorder=1)
        from matplotlib.ticker import FuncFormatter
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.0f}%" if abs(v) >= 0.5 else "0%"))
        ylab = f"{disp}\n(% change vs real NTC)"
    else:
        ylab = f"{disp}\n{unit}".strip() if unit_newline and unit else f"{disp} {unit}".strip()
    ax.set_xticks(pos); ax.set_xticklabels(labs, fontsize=24)
    ax.set_ylabel(ylab, fontsize=(36 if len(ylab.split(chr(10))[0]) <= 30 else 30))
    ax.tick_params(axis="y", labelsize=28)
    pooled = np.concatenate(data)                                        # view window only (1st-98th pct of the POOLED data, same convention as the
    p_lo, p_hi = np.percentile(pooled, [1, 98])                          # protein-complex panels) -- KDE/median above already used the full data
    pad = (p_hi - p_lo) * 0.12
    lo = p_lo - pad
    if not normalize:
        lo = max(lo, 0)                                                  # these features (area/count/circularity) are never negative -- don't pad into meaningless empty space below 0
    ax.set_ylim(lo, p_hi + pad)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    tag = "normalized" if normalize else "raw"
    stem = f"{OUT}/{group}_{feat}_{tag}{suffix}"
    for ext in ("png", "svg"):
        fig.savefig(f"{stem}.{ext}", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {stem}.png/svg")


def build():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for group, dirname, fscore_gname, feats in FGROUPS:
        for feat, disp, unit in feats:
            render_violin(group, dirname, fscore_gname, feat, disp, unit, normalize=False)
            render_violin(group, dirname, fscore_gname, feat, disp, unit, normalize=True)
            rows.append((group, disp, feat))
    _write_index(rows)


def _write_index(rows):
    lines = ["# Figure 4 -- F-rescaled violins (phi = alpha/f, MORPH_F_RESCALE_HANDOFF.md), bruno review, not yet on Confluence\n",
             "f (centroid-recovery peak-alpha): POLR1B=2.45, MTOR=1.38, TIM23=2.25.\n",
             "phi columns with no measured/snappable alpha (POLR1B phi=1.5,3; TIM23 phi=3 -- true k*f exceeds the generated range or isn't measured yet) are left blank, labeled '(no data)', not substituted with another phi's data.\n",
             "Paper-style image panels (actual traversal images, not this violin's cellgrid): see bruno_fscore_panels.py output, {group}_cellpanel_fscore.svg.\n",
             "\n| Group | Feature | Raw | Normalized |", "|---|---|---|---|"]
    for group, disp, feat in rows:
        lines.append(f"| {group} | {disp} | [{feat}_raw.svg]({group}_{feat}_raw.svg) | [{feat}_normalized.svg]({group}_{feat}_normalized.svg) |")
    open(f"{OUT}/index.md", "w").write("\n".join(lines))
    print(f"saved {OUT}/index.md")


if __name__ == "__main__":
    build()
