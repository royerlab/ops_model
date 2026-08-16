"""F-rescaled variant of raw_alpha_panels.py -- the SAME mechanism (scratch asset tree -> unmodified
figure4_morpho_traversal.image_panels/render_images -> qualitative per-object palette), but showing
phi=0,0.5,1,1.5,2,3 (= alpha/f at each perturbation's own centroid-recovery f) instead of raw alpha=0,1,2.

phi columns with no measured/snappable alpha (POLR1B phi=1.5,3; TIM23 phi=3 -- true k*f exceeds the
generated range or isn't measured yet) are rendered as a BLANK placeholder, labeled "(no data)" -- never
substituted with another phi's image.

Run: python fscore_panels.py
"""
import json
import os

import numpy as np
from PIL import Image
from skimage.transform import resize

import raw_alpha_panels as brp
import figure4_morpho_traversal as T
import morpho_native as mn
from ops_model.models.attention.diffex.viewer.morpho_pipeline import _clip_border

NAT = brp.NAT
PROD_VA = brp.PROD_VA
SCRATCH = "/hpc/projects/icd.fast.ops/analysis/figure4_traversals_violin/bruno_fscore/_scratch_assets"
OUT = "/hpc/projects/icd.fast.ops/analysis/figure4_traversals_violin/bruno_fscore"

GKEY = {"mTOR": "mtor_mo_hm", "POLR1B": "polr1b_vsnpm3", "TIM23": "tim23", "TAF1B": "taf1b_vsnpm3",
        "SAMM50": "samm50_chromalive", "MICOS13": "micos13_chromalive"}
PHIS = [0, 0.5, 1, 1.5, 2, 3]


def _phi_alpha(group):
    return mn.FSCORE_PHI_ALPHA[f"{GKEY[group]}_fscore"]


def _load_merged_gpanel(cfg):
    d = np.load(f"{NAT}/{cfg['native_dir']}/panel.npz", allow_pickle=True)
    gpanel = dict(d["gpanel"].item())
    new_pf = f"{NAT}/{GKEY[cfg['group']]}_fscore/new_alpha_panel.npz"
    if os.path.exists(new_pf):
        newp = np.load(new_pf, allow_pickle=True)["gpanel"].item()
        for ci, entries in newp.items():
            gpanel.setdefault(ci, {}).update(entries)
    return gpanel, d


def _build_scratch(group, cfg, scratch_cell, local_cell):
    dir_ = f"{cfg['marker_dir']}/{cfg['grain']}/{cfg['target']}"
    md = f"{SCRATCH}/_morphometrics/{dir_}"
    os.makedirs(f"{md}/cell{scratch_cell}", exist_ok=True)
    os.makedirs(f"{SCRATCH}/{dir_}/cell{scratch_cell}", exist_ok=True)
    json.dump({"alphas": brp.ALPHAS}, open(f"{md}/full_features.json", "w"))

    gpanel, _ = _load_merged_gpanel(cfg)
    raw = np.load(f"{PROD_VA}/{dir_}/cell{scratch_cell}/frames_f32.npz")["gen"]
    phi_alpha = _phi_alpha(group)
    for phi in PHIS:
        a = phi_alpha[phi]
        if a is None:
            continue
        zi = mn._aidx(a)
        keys_try = (f"gen_a{a}", f"gen_a{a:g}")
        k_found = next((k for k in keys_try if k in gpanel.get(local_cell, {})), None)
        if k_found is None:
            print(f"  [warn] no gpanel entry for {group} phi={phi} (alpha={a}) -- skipping"); continue
        _, lc, mask = gpanel[local_cell][k_found]
        lc = _clip_border(np.where(np.asarray(mask), np.asarray(lc), 0)).astype(np.int32)
        img01 = np.clip((raw[zi] + 1) / 2, 0, 1).astype(np.float32)
        img256 = resize(img01, (256, 256), preserve_range=True)
        Image.fromarray((np.clip(img256, 0, 1) * 255).astype(np.uint8)).save(
            f"{SCRATCH}/{dir_}/cell{scratch_cell}/frame_{zi:02d}.webp", quality=90)
        feats = brp._regionprops_dict(lc)
        json.dump(feats, open(f"{md}/cell{scratch_cell}/a{zi:02d}_feats.json", "w"))
        lc256 = resize(lc, (256, 256), order=0, preserve_range=True, anti_aliasing=False).astype(np.uint16)
        Image.fromarray(lc256, mode="I;16").save(f"{md}/cell{scratch_cell}/a{zi:02d}_labels.png")
    os.makedirs(f"{SCRATCH}/{cfg['marker_dir']}/_anchors/NTC/cell{scratch_cell}", exist_ok=True)
    print(f"[scratch] built {group} F-rescaled (panel.npz cell {local_cell} -> scratch cell{scratch_cell}) -> {md}/cell{scratch_cell}")


def render(group, cfg, scratch_cell, local_cell, op=0.75, suffix=""):
    dir_ = f"{cfg['marker_dir']}/{cfg['grain']}/{cfg['target']}"
    md = f"{SCRATCH}/_morphometrics/{dir_}"
    real_anchor = f"{PROD_VA}/{cfg['marker_dir']}/_anchors/NTC/cell{scratch_cell}/real.webp"
    scratch_anchor = f"{SCRATCH}/{cfg['marker_dir']}/_anchors/NTC/cell{scratch_cell}/real.webp"
    if not os.path.exists(scratch_anchor) and os.path.exists(real_anchor):
        import shutil; shutil.copy(real_anchor, scratch_anchor)
    T.VA = SCRATCH
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    phi_alpha = _phi_alpha(group)
    valid_phis = [p for p in PHIS if phi_alpha[p] is not None]
    alphas_show = [phi_alpha[p] for p in valid_phis]
    panels, okey, lo, hi = T.image_panels(md, dir_, dir_, "", scratch_cell, alphas_show)

    has_anchor = panels and panels[0][0] == "original NTC"
    if has_anchor:
        d = np.load(f"{NAT}/{cfg['native_dir']}/panel.npz", allow_pickle=True)
        rn = list(d["rn"])
        if local_cell < len(rn):
            rimg, rlab, rmask = rn[local_cell]
            rimg = np.asarray(rimg, np.float64)
            p1, p99 = np.percentile(rimg, [1, 99])
            rimg01 = np.clip((rimg - p1) / max(p99 - p1, 1e-6), 0, 1)
            gray256 = (resize(rimg01, (256, 256), preserve_range=True) * 255).astype(np.uint8)
            rlab_clip = _clip_border(np.where(np.asarray(rmask), np.asarray(rlab, np.int32), 0))
            lab256 = resize(rlab_clip, (256, 256), order=0, preserve_range=True, anti_aliasing=False).astype(np.int32)
            panels[0] = ("original NTC", gray256, lab256, {})

    # label each panel with its phi (image_panels labels them "α=+N" using the resolved α, which is confusing
    # here); phi values with no data are already absent from `panels` (image_panels was never asked to show
    # them) -- skipped entirely, not left as a gap
    valid_panels = panels[1:] if has_anchor else panels
    labeled = [(f"φ={p:g}", gray, lab, fj) for p, (t, gray, lab, fj) in zip(valid_phis, valid_panels)]
    panels = ([panels[0]] if has_anchor else []) + labeled

    nc = len(panels)
    fig = plt.figure(figsize=(nc * 2.6, 5.4), facecolor="white")
    g = fig.add_gridspec(2, nc, hspace=0.02, wspace=0.02)
    oaxes = []
    for j, (t, gray, lab, fj) in enumerate(panels):
        axi = fig.add_subplot(g[0, j]); axi.set_xticks([]); axi.set_yticks([])
        axi.set_title(t, fontsize=18)
        for s in axi.spines.values():
            s.set_visible(False)
        axo = fig.add_subplot(g[1, j]); axo.set_xticks([]); axo.set_yticks([])
        for s in axo.spines.values():
            s.set_visible(False)
        axi.imshow(gray, cmap="gray"); axo.imshow(gray, cmap="gray")
        if lab is not None:
            axo.imshow(brp._qualitative_rgba(lab, op=op), aspect="auto")
        axi.set_aspect("auto"); axo.set_aspect("auto")
        oaxes.append(axo)
    fig.suptitle(f"{group}{suffix} (F-rescaled: φ=α/f)", fontsize=16)
    stem = f"{OUT}/{group}_cellpanel_fscore{suffix}"
    for ext in ("png", "svg"):
        fig.savefig(f"{stem}.{ext}", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {stem}.png/svg")


NATIVE_DIR_OVERRIDE = {"POLR1B": "polr1b_vsnpm3_stringentcpu"}   # stringent (mo_local_adjust 1.35) gen seg -- raw_alpha_panels.GROUPS still points at the lenient one for the main paper panel


def _cfg_for(group):
    cfg0 = brp.GROUPS[group]
    return dict(cfg0, group=group, native_dir=NATIVE_DIR_OVERRIDE.get(group, cfg0["native_dir"]))


def build():
    for group in brp.GROUPS:
        cfg = _cfg_for(group)
        scratch_cell = cfg["cell"]; local_cell = cfg["local_cell"]
        _build_scratch(group, cfg, scratch_cell, local_cell)
        render(group, cfg, scratch_cell, local_cell)


def render_candidates(group, local_cells, offset=0):
    """Render the SAME F-rescaled panel for several candidate cells -- for picking. `local_cells` are
    panel.npz gpanel keys; scratch_cell = local_cell + offset (POLR1B's global numbering is offset +200
    from its local panel.npz rank; mTOR/TIM23 have no offset)."""
    cfg = _cfg_for(group)
    for lc in local_cells:
        sc = lc + offset
        _build_scratch(group, cfg, scratch_cell=sc, local_cell=lc)
        render(group, cfg, scratch_cell=sc, local_cell=lc, suffix=f"_candidate{lc}")


if __name__ == "__main__":
    build()
