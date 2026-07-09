"""Mimic Alex's CellDINO extraction so our cells land in the SetTransformer classifier's
input space (see katamari extract_embeddings_phase_celldino_fast.yaml + CellDinoWrapper).

Pipeline per cell:  128x128 Phase2D crop @ (x_pheno,y_pheno)  →  seg-mask (cell_seg==id)
  →  percentile-norm (x-p1)/(p99-p1)  →  CellDINO (resize224 + per-image z-score, our embed_crops)
  →  z-standardize per (channel,experiment) on NTC-control stats.

The CellDINO forward is identical to our existing `embed_crops`; the added mask + percentile are
cheap pixel ops, and crops are chunk-local windowed reads — so cost/cell ≈ the CellDINO forward
we already pay. `validate()` reproduces a few genes in one experiment, scores bags with the real
classifier, and prints accuracy + timing/cell.
"""
from __future__ import annotations

import time
import types

import numpy as np
import pandas as pd
import torch
import zarr

from .set_classifier import load_set_classifier, score_bags

PT_PHASE = "/hpc/projects/icd.fast.ops/models/alex_lin_attention/v4/val_ops_zstdcontrol_cdino_v2"
ZARR = "/hpc/projects/icd.fast.ops/{exp}/3-assembly/phenotyping_v3.zarr"
SIZE = 128
PHASE_CH = 0                       # Phase2D channel index
PCT_LEVEL = "4"                    # pyramid level for the per-well p1/p99 estimate


def flatten_pt(gene, pt_root=PT_PHASE):
    """Alex's per-gene .pt cell_metadata (bags of lists) → (flat per-cell df, his per-cell embeddings
    aligned to df rows). His embeddings are the ground-truth target space (already control-z-std)."""
    d = torch.load(f"{pt_root}/{gene}.pt", map_location="cpu")
    cm = d["cell_metadata"]
    cols = ["experiment", "well", "x_pheno", "y_pheno", "segmentation_id"]
    flat = {c: [v for bag in cm[c] for v in bag] for c in cols}
    df = pd.DataFrame(flat)
    df["gene"] = gene
    alex = np.asarray(d["embeddings"])
    if len(alex) != len(df):
        print(f"  [warn] {gene}: {len(alex)} embs vs {len(df)} flattened cells — alignment off")
    return df, alex


def _well_pos(root, well):
    """well 'A3' -> position group 'A/3/0'."""
    return root[f"{well[0]}/{well[1:]}/0"]


def _pct(pos):
    """per-well (p1, p99) of Phase2D from a low-res pyramid level (one cheap full read)."""
    lo = np.asarray(pos[PCT_LEVEL][0, PHASE_CH, 0])
    return np.percentile(lo, [1, 99])


def load_raw(exp, cells, size=SIZE):
    """cells for ONE experiment → (raw crops (N,1,s,s), masks (N,s,s) bool, per-well (p1,p99), keep).
    Returns the components uncomposed so `compose()` can build mask/percentile variants."""
    root = zarr.open(ZARR.format(exp=exp), mode="r")
    h = size // 2
    crops, masks, pcts, keep = [], [], [], []
    for well, g in cells.groupby("well"):
        pos = _well_pos(root, well)
        img = pos["0"]; seg = pos["labels/cell_seg/0"]
        Y, X = img.shape[-2:]
        p1, p99 = _pct(pos)
        for idx, r in g.iterrows():
            y, x = int(round(r.y_pheno)), int(round(r.x_pheno))
            if y - h < 0 or x - h < 0 or y + h > Y or x + h > X:
                continue
            crop = np.asarray(img[0, PHASE_CH, 0, y - h:y + h, x - h:x + h]).astype(np.float32)
            m = np.asarray(seg[0, 0, 0, y - h:y + h, x - h:x + h]) == int(r.segmentation_id)
            crops.append(crop); masks.append(m); pcts.append((p1, p99)); keep.append(idx)
    return (np.stack(crops), np.stack(masks), np.array(pcts), keep) if crops else (None, None, None, [])


def compose(crops, masks, pcts, mask=True, pct="well"):
    """Build (N,1,s,s) from raw crops. mask: apply seg mask. pct: 'well'|'crop'|'none' intensity norm."""
    x = crops.copy()
    if mask:
        x = x * masks
    if pct == "well":
        p1, p99 = pcts[:, 0][:, None, None], pcts[:, 1][:, None, None]
        x = (x - p1) / (p99 - p1 + 1e-6)
    elif pct == "crop":
        p1 = np.percentile(crops.reshape(len(x), -1), 1, axis=1)[:, None, None]
        p99 = np.percentile(crops.reshape(len(x), -1), 99, axis=1)[:, None, None]
        x = (x - p1) / (p99 - p1 + 1e-6)
    return x[:, None].astype(np.float32)


def load_masked_crops(exp, cells, size=SIZE):
    crops, masks, pcts, keep = load_raw(exp, cells, size)
    if crops is None:
        return np.zeros((0, 1, size, size), np.float32), []
    return compose(crops, masks, pcts), keep


def _celldino(crops, batch=256):
    """(N,1,H,W) → (N,1024) via the SAME CellDinoModel embed_crops uses. Returns (embs, sec/cell)."""
    from ops_model.models.cell_dino import CellDinoModel
    model = CellDinoModel(z_score=True)
    embs = []
    t0 = time.time()
    with torch.inference_mode():
        for i in range(0, len(crops), batch):
            out = model.extract_features({"data": torch.as_tensor(crops[i:i + batch])})
            embs.append(out.float().cpu().numpy())
    e = np.concatenate(embs).astype(np.float32)
    return e, (time.time() - t0) / max(len(crops), 1)


def zstd_control(embs, control_mask):
    """z-standardize per feature using CONTROL-only mean/std (matches z_standardize_control_only)."""
    ctrl = embs[control_mask]
    mu, sd = ctrl.mean(0), ctrl.std(0) + 1e-6
    return (embs - mu) / sd


def validate(exp="ops0031_20250424", genes=("HSPA5", "KIF11", "POLR1B", "TIMM23"),
             per_gene=150, run="miwkg1cy"):
    """Reproduce cells for a few genes (+NTC for control stats) in ONE experiment, embed via the
    mimic, z-std on NTC, score bags → per-gene argmax accuracy + timing/cell."""
    m, g2i, c2i = load_set_classifier(run)
    i2g = {v: k for k, v in g2i.items()}
    genes = list(genes) + (["NTC"] if "NTC" in g2i else [])

    frames, alex_rows = [], []
    io_t0 = time.time()
    for gene in genes:
        df, alex = flatten_pt(gene)
        mask_exp = (df.experiment == exp).to_numpy()
        df = df[mask_exp].head(per_gene); alex = alex[mask_exp][:per_gene]
        if not len(df):
            print(f"  {gene}: no cells in {exp} — skip"); continue
        crops, keep = load_masked_crops(exp, df)
        if not len(crops):
            print(f"  {gene}: all crops OOB — skip"); continue
        pos = [df.index.get_loc(i) for i in keep]           # positional idx into df/alex
        sub = df.loc[keep].copy(); sub["_crops"] = list(crops)
        frames.append(sub); alex_rows.append(alex[pos])
        print(f"  {gene}: {len(sub)} cells loaded")
    all_df = pd.concat(frames, ignore_index=True)
    alex_emb = np.concatenate(alex_rows)                    # his (control-z-std) embeddings, aligned
    io_per = (time.time() - io_t0) / len(all_df)

    crops = np.stack(list(all_df["_crops"]))
    raw, cd_per = _celldino(crops)
    embs = zstd_control(raw, (all_df.gene == "NTC").to_numpy())

    # fidelity: my reproduction vs Alex's SAME-cell embedding (cosine)
    def _cos(a, b):
        a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
        b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
        return (a * b).sum(1)
    cos_all = _cos(embs, alex_emb)
    print(f"\n[fidelity] mean cosine(my zstd, Alex .pt) = {cos_all.mean():.3f} "
          f"(p50 {np.median(cos_all):.3f}, >0.9: {(cos_all>0.9).mean()*100:.0f}%)")

    print(f"\n[timing] I/O+mask+pct: {io_per*1000:.1f} ms/cell | CellDINO: {cd_per*1000:.1f} ms/cell "
          f"| total ~{(io_per+cd_per)*1000:.1f} ms/cell  (N={len(all_df)})")
    print(f"[scale] ~{(io_per+cd_per):.3f} s/cell → 119k cells ≈ {(io_per+cd_per)*119000/3600:.1f} GPU-hr\n")

    rng = np.random.default_rng(0)
    ci = c2i.get("Phase", 0)
    print(f"{'gene':8s} | {'mine argmax':12s} P(t)  | {'Alex argmax':12s} P(t)  | cos")
    for gene in genes:
        gi = np.where((all_df.gene == gene).to_numpy())[0]
        if len(gi) < 20:
            continue
        sel = [rng.choice(gi, min(100, len(gi))) for _ in range(6)]
        pm = score_bags(m, np.stack([embs[s] for s in sel]), channel_idx=ci)
        pa = score_bags(m, np.stack([alex_emb[s] for s in sel]), channel_idx=ci)
        tm, ta = i2g[int(pm.mean(0).argmax())], i2g[int(pa.mean(0).argmax())]
        print(f"  {gene:6s} | {tm:12s} {pm[:, g2i[gene]].mean():.3f} {'HIT' if tm==gene else '   '} "
              f"| {ta:12s} {pa[:, g2i[gene]].mean():.3f} {'HIT' if ta==gene else '   '} "
              f"| {cos_all[gi].mean():.3f}")


def cellpose_masks(crops01, diameter=None, flow_threshold=0.4, batch=64):
    """Segment GENERATED phase crops (N,H,W) in [0,1] with Cellpose-SAM → central-cell boolean masks.
    The morphed cell is centered, so keep the label covering the crop centre (fallback: nearest label).
    Lets us mask fake images the same way training masks real cells (the load-bearing step)."""
    import torch
    from cellpose import models
    from scipy import ndimage as ndi
    m = models.CellposeModel(gpu=True, device=torch.device("cuda"))
    H, W = crops01.shape[-2:]; cy, cx = H // 2, W // 2
    out = []
    for i in range(0, len(crops01), batch):
        labs = m.eval(list(crops01[i:i + batch]), diameter=diameter, flow_threshold=flow_threshold)[0]
        for lab in labs:
            c = lab[cy, cx]
            if c == 0 and lab.max() > 0:                       # centre is background → nearest cell
                idx = ndi.distance_transform_edt(lab == 0, return_distances=False, return_indices=True)
                c = lab[tuple(v[cy, cx] for v in idx)]
            out.append(lab == c if c > 0 else np.zeros_like(lab, bool))
    return np.stack(out)


def test_cellpose(genes=("HSPA5", "POLR1B", "TIMM23"), n=20, alpha_frame=8):
    """Load real GENERATED α-frames from the viewer cache, Cellpose-SAM segment them, keep the
    central cell → report mask coverage + timing/cell (feasibility of masking fake images)."""
    from PIL import Image
    cache = "/hpc/projects/icd.fast.ops/models/diffex/viewer_assets/phase/geneKO"
    crops = []
    for g in genes:
        for c in range(n):
            f = f"{cache}/{g}/cell{c}/frame_{alpha_frame:02d}.webp"
            try:
                crops.append(np.asarray(Image.open(f).convert("L"), np.float32) / 255.0)
            except Exception:
                pass
    crops = np.stack(crops)
    print(f"loaded {len(crops)} generated crops {crops.shape[1:]}")
    t0 = time.time()
    masks = cellpose_masks(crops)
    per = (time.time() - t0) / len(crops)
    cov = masks.reshape(len(masks), -1).mean(1)
    nonempty = (cov > 0).mean()
    print(f"[cellpose] {per*1000:.1f} ms/cell | central-cell found: {nonempty*100:.0f}% | "
          f"mean coverage {cov.mean()*100:.1f}% of the {crops.shape[1]}px crop")
    print(f"[scale] {per:.3f} s/cell → 119k ≈ {per*119000/3600:.1f} GPU-hr for masking")


def _zstd_per_exp(embs, exps, is_ntc):
    """z-standardize per experiment on that experiment's NTC control (fallback: all cells)."""
    out = embs.copy()
    for e in np.unique(exps):
        me = exps == e
        ctrl = embs[me & is_ntc]
        if len(ctrl) < 10:
            ctrl = embs[me]
        out[me] = (embs[me] - ctrl.mean(0)) / (ctrl.std(0) + 1e-6)
    return out


def validate_bags(genes=("HSPA5", "POLR1B", "KIF11", "TIMM23"), per_exp=30, max_exp=25,
                  bag=100, draws=20, run="miwkg1cy"):
    """FIDELITY-AT-SCALE test: reproduce cells across MANY experiments (per-exp z-std), then compare
    mimic vs Alex's own embeddings at realistic bag sizes — does 0.91 fidelity give correct argmax?"""
    m, g2i, c2i = load_set_classifier(run)
    i2g = {v: k for k, v in g2i.items()}
    glist = list(genes) + (["NTC"] if "NTC" in g2i else [])
    crops_l, mk_l, pc_l, alex_l, gcol, ecol = [], [], [], [], [], []
    for gene in glist:
        df, alex = flatten_pt(gene)
        # cap per experiment, cap #experiments — keeps I/O bounded
        df = df.reset_index(drop=True)
        parts = []
        for e, g in df.groupby("experiment"):
            parts.append(g.head(per_exp))
            if len(parts) >= max_exp:
                break
        df = pd.concat(parts); alex = alex[df.index.to_numpy()]
        for e, g in df.groupby("experiment"):
            c, mk, pc, keep = load_raw(e, g)
            if c is None:
                continue
            pos = [g.index.get_loc(i) for i in keep]
            crops_l.append(c); mk_l.append(mk); pc_l.append(pc)
            alex_l.append(alex[[df.index.get_loc(i) for i in keep]])
            gcol += [gene] * len(keep); ecol += [e] * len(keep)
        print(f"  {gene}: {sum(x==gene for x in gcol)} cells across experiments")
    crops = np.concatenate(crops_l); masks = np.concatenate(mk_l); pcts = np.concatenate(pc_l)
    alex_emb = np.concatenate(alex_l); gcol = np.array(gcol); ecol = np.array(ecol)

    raw, per = _celldino(compose(crops, masks, pcts, mask=True, pct="none"))
    print(f"[timing] {per*1000:.1f} ms/cell CellDINO, N={len(raw)}")
    is_ntc = gcol == "NTC"
    mine = _zstd_per_exp(raw, ecol, is_ntc)
    alex_z = alex_emb                                        # already control-z-std by Alex
    cos = ((mine/ (np.linalg.norm(mine,axis=1,keepdims=True)+1e-9)) *
           (alex_z/(np.linalg.norm(alex_z,axis=1,keepdims=True)+1e-9))).sum(1)
    print(f"[fidelity] mean cos(mine, Alex) = {cos.mean():.3f}\n")

    ci = c2i.get("Phase", 0); rng = np.random.default_rng(0)
    print(f"{'gene':8s} | {'MINE hit-rate  meanP':22s} | {'ALEX hit-rate  meanP':22s}")
    for gene in genes:
        gi = np.where(gcol == gene)[0]
        if len(gi) < bag:
            print(f"  {gene}: only {len(gi)} cells (<{bag}) — skip"); continue
        def _eval(E):
            bags = np.stack([E[rng.choice(gi, bag, replace=False)] for _ in range(draws)])
            p = score_bags(m, bags, channel_idx=ci)
            hits = np.mean([i2g[int(r.argmax())] == gene for r in p])
            return hits, p[:, g2i[gene]].mean()
        hm, pm = _eval(mine); ha, pa = _eval(alex_z)
        print(f"  {gene:6s} | {hm*100:5.0f}%  P={pm:.3f}          | {ha*100:5.0f}%  P={pa:.3f}")


CTRL_REF = "/hpc/projects/icd.fast.ops/models/diffex/control_ref_phase.npz"
CACHE = "/hpc/projects/icd.fast.ops/models/diffex/viewer_assets"


def build_control_ref(per_exp=40, max_exp=25, out=CTRL_REF):
    """Global NTC control reference (per-feature mean/std of mimic-embedded REAL NTC cells) → used to
    z-standardize GENERATED cells into the classifier space (they have no experiment of their own)."""
    import os
    if os.path.exists(out):
        d = np.load(out); return d["mu"], d["sd"]
    df, _ = flatten_pt("NTC")
    parts = [g.head(per_exp) for _, g in df.groupby("experiment")][:max_exp]
    df = pd.concat(parts)
    crops = []
    for e, g in df.groupby("experiment"):
        c, mk, pc, keep = load_raw(e, g)
        if c is not None:
            crops.append(compose(c, mk, pc, mask=True, pct="none"))
    x = np.concatenate(crops)
    raw, _ = _celldino(x)
    mu, sd = raw.mean(0), raw.std(0) + 1e-6
    np.savez(out, mu=mu, sd=sd)
    print(f"[control_ref] {len(raw)} NTC cells -> {out}")
    return mu, sd


def _load_gen(gene, grain="geneKO", modality="phase", n_cells=20):
    """Load a traversal's cache α-frames → (n_alpha, n_cells, 256, 256) FULL frames + the α list.
    Cellpose segments on the FULL FOV (robust); the crop to the 128 training FOV happens AFTER masking."""
    import json
    from PIL import Image
    base = f"{CACHE}/{modality}/{grain}/{gene}"
    meta = json.load(open(f"{base}/cell0/meta.json")) if __import__("os").path.exists(f"{base}/cell0/meta.json") \
        else json.load(open(f"{base}/meta.json"))
    alphas = meta["alphas"]
    frames = [[] for _ in alphas]
    for c in range(n_cells):
        for ai in range(len(alphas)):
            frames[ai].append(np.asarray(Image.open(f"{base}/cell{c}/frame_{ai:02d}.webp").convert("L"), np.float32) / 255.0)
    return np.array(frames), alphas


def score_generated(genes=("HSPA5", "POLR1B", "KIF11", "TIMM23"), grain="geneKO",
                    modality="phase", n_cells=20, run="miwkg1cy"):
    """End-to-end per-α classifier score for GENERATED traversals: cellpose-mask each fake frame →
    mimic-CellDINO → z-std(control ref) → bag P(target) per α. Prints the α-curve (should rise toward
    the target as |α| increases if the counterfactual convinces the classifier)."""
    m, g2i, c2i = load_set_classifier(run)
    i2g = {v: k for k, v in g2i.items()}
    mu, sd = build_control_ref()
    ci = c2i.get("Phase", 0)
    for gene in genes:
        if gene not in g2i:
            print(f"{gene}: not in classifier"); continue
        try:
            frames, alphas = _load_gen(gene, grain, modality, n_cells)
        except Exception as e:
            print(f"{gene}: load failed ({e})"); continue
        na, nc = frames.shape[:2]
        flat = frames.reshape(na * nc, *frames.shape[2:])   # full 256 frames
        masks = cellpose_masks(flat)                         # segment on the FULL FOV (robust)
        masked = flat * masks
        h = (masked.shape[1] - 128) // 2                     # crop to the 128 training FOV AFTER masking
        masked = masked[:, h:h + 128, h:h + 128] if h > 0 else masked
        raw, _ = _celldino(masked[:, None].astype(np.float32))
        emb = ((raw - mu) / sd).reshape(na, nc, -1)
        ptarget = [float(score_bags(m, e[None], channel_idx=ci)[0, g2i[gene]]) for e in emb]
        top = [i2g[int(score_bags(m, e[None], channel_idx=ci)[0].argmax())] for e in emb]
        print(f"\n{gene} (grain={grain}) per-α P(target):")
        for a, p, t in zip(alphas, ptarget, top):
            bar = "#" * int(p * 40)
            print(f"  α={a:+.1f}  P={p:.3f} {bar:40s} argmax={t}{'  <TARGET' if t == gene else ''}")


def sweep(exp="ops0031_20250424", genes=("HSPA5", "POLR1B"), per_gene=150, run="miwkg1cy"):
    """Load raw cells ONCE, embed several (mask × percentile) variants, report cosine-to-Alex +
    P(target) for each → find the preprocessing recipe that best reproduces Alex's space."""
    m, g2i, c2i = load_set_classifier(run)
    glist = list(genes) + (["NTC"] if "NTC" in g2i else [])
    raws, msks, pcs, alex_rows, gene_col = [], [], [], [], []
    for gene in glist:
        df, alex = flatten_pt(gene)
        me = (df.experiment == exp).to_numpy()
        df = df[me].head(per_gene); alex = alex[me][:per_gene]
        c, mk, pc, keep = load_raw(exp, df)
        if c is None:
            continue
        pos = [df.index.get_loc(i) for i in keep]
        raws.append(c); msks.append(mk); pcs.append(pc)
        alex_rows.append(alex[pos]); gene_col += [gene] * len(keep)
        print(f"  {gene}: {len(keep)} cells")
    crops = np.concatenate(raws); masks = np.concatenate(msks); pcts = np.concatenate(pcs)
    alex_emb = np.concatenate(alex_rows); gene_col = np.array(gene_col)
    ntc = gene_col == "NTC"; ci = c2i.get("Phase", 0); rng = np.random.default_rng(0)

    variants = [("mask+pct_well", True, "well"), ("mask+pct_crop", True, "crop"),
                ("mask+no_pct", True, "none"), ("nomask+pct_well", False, "well"),
                ("nomask+no_pct", False, "none")]
    print(f"\n{'variant':16s} | cos(Alex) | " + " ".join(f"P({g})" for g in genes))
    for name, mask, pct in variants:
        x = compose(crops, masks, pcts, mask=mask, pct=pct)
        raw, _ = _celldino(x)
        e = zstd_control(raw, ntc)
        a = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)
        b = alex_emb / (np.linalg.norm(alex_emb, axis=1, keepdims=True) + 1e-9)
        cos = (a * b).sum(1).mean()
        ps = []
        for g in genes:
            gi = np.where(gene_col == g)[0]
            bags = np.stack([e[rng.choice(gi, min(100, len(gi)))] for _ in range(6)])
            ps.append(score_bags(m, bags, channel_idx=ci)[:, g2i[g]].mean())
        print(f"{name:16s} |   {cos:.3f}   | " + " ".join(f"{p:.3f}" for p in ps))


if __name__ == "__main__":
    validate()
