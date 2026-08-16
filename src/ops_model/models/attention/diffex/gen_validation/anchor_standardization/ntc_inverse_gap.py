"""Real NTC vs inverse-α=0 NTC in CellDINO space — paired, same-cell.

Uses the NEW DDIM-inverted v5 traversals (viewer_assets_v5_inv), NOT the old random-xT cache.
For the phase channel, geneKO anchors are NTC cells: anchor cellN (real.webp) and any gene's traversal cellN
frame_08 (α=0) are the SAME cell — so we get matched pairs (real NTC vs its α=0 reconstruction). We measure:
  (1) per-cell gap: cosine(real_i, gen_i);
  (2) offset consistency: are the N offset vectors (gen_i - real_i) parallel + equal-magnitude (rigid) or scattered;
  (3) where each lands in the phase gene embedding (real-pop standardization) — does inverse-α=0 now sit on NTC?
α=0 is gene-independent (no direction), so one finished phase gene suffices.
"""
import os, glob, json
import numpy as np
from PIL import Image

INV = "/hpc/projects/icd.fast.ops/models/diffex/viewer_assets_v5_inv"
A0 = 8                              # α=0 frame index (17 frames, -5..+5)
OUT = "/hpc/projects/icd.fast.ops/analysis/figure4_embedding/ntc_inverse_gap"


def _celln(d):
    return sorted(glob.glob(f"{d}/cell*"), key=lambda p: int(p.rsplit("cell", 1)[1]))


def _emb_webp(paths, cfg, embed_crops):
    imgs = [np.asarray(Image.open(p).convert("L"), np.float32) / 255.0 * 2 - 1 for p in paths]
    return embed_crops(np.stack(imgs)[:, None].astype(np.float32), cfg, cache_path=None)


def _first_done(channel=None):
    """Find (channel, gene) of a finished geneKO traversal with α=0 frames + NTC anchors. Prefer phase."""
    chans = [channel] if channel else (["phase"] + sorted(os.listdir(INV)))
    for ch in chans:
        gk = f"{INV}/{ch}/geneKO"
        if not os.path.isdir(gk) or not os.path.isdir(f"{INV}/{ch}/_anchors/NTC"):
            continue
        for g in sorted(os.listdir(gk)):
            if glob.glob(f"{gk}/{g}/cell0/frame_{A0:02d}.webp"):
                return ch, g
    return None, None


def run(channel=None, gene=None):
    import torch  # noqa
    from ops_model.models.attention.diffex.classifier.celldino_features import embed_crops
    from ops_model.models.attention.diffex.directions.config import DirConfig
    os.makedirs(OUT, exist_ok=True)
    ch, g = _first_done(channel)
    gene = gene or g
    if ch is None:
        print("[ntc-gap] no finished traversal yet — rerun when one lands"); return
    print(f"[ntc-gap] channel={ch} gene={gene} for α=0 (NTC-anchor reconstruction, gene-independent)", flush=True)

    anc = _celln(f"{INV}/{ch}/_anchors/NTC")
    trav = f"{INV}/{ch}/geneKO/{gene}"
    n = min(len(anc), len(_celln(trav)))
    real_paths = [f"{INV}/{ch}/_anchors/NTC/cell{i}/real.webp" for i in range(n)]
    gen_paths = [f"{trav}/cell{i}/frame_{A0:02d}.webp" for i in range(n)]
    keep = [i for i in range(n) if os.path.exists(real_paths[i]) and os.path.exists(gen_paths[i])]
    real_paths = [real_paths[i] for i in keep]; gen_paths = [gen_paths[i] for i in keep]
    print(f"[ntc-gap] {len(keep)} matched anchor/α=0 pairs", flush=True)

    cfg = DirConfig(grain="geneKO", target=gene, device="cuda")
    R = np.asarray(_emb_webp(real_paths, cfg, embed_crops), np.float64)     # real NTC (N,1024)
    G = np.asarray(_emb_webp(gen_paths, cfg, embed_crops), np.float64)      # inverse α=0 NTC (N,1024)
    np.savez(f"{OUT}/emb_{ch}.npz", R=R, G=G, gene=gene, channel=ch)
    analyze(ch)


def analyze(ch="phase"):
    from numpy.linalg import norm
    d = np.load(f"{OUT}/emb_{ch}.npz", allow_pickle=True)
    R, G = d["R"], d["G"]
    cos = lambda u, v: float(u @ v / (norm(u) * norm(v)))
    # (1) per-cell paired gap
    pc = np.array([cos(R[i], G[i]) for i in range(len(R))])
    # (2) offset consistency
    off = G - R                                                            # per-cell offset vectors
    mo = off.mean(0); mo_hat = mo / norm(mo)
    cos_to_mean = np.array([cos(off[i], mo) for i in range(len(off))])
    mag = norm(off, axis=1)
    # (3) centroid gap
    print("\n=== Real NTC  vs  inverse-α=0 NTC  (CellDINO, paired same-cell) ===")
    print(f" pairs                         : {len(R)}")
    print(f" (1) per-cell cosine(real,gen) : mean {pc.mean():.3f}  min {pc.min():.3f}  max {pc.max():.3f}")
    print(f" (3) centroid cosine           : {cos(R.mean(0), G.mean(0)):.3f}")
    print(f"     centroid ||offset||       : {norm(G.mean(0) - R.mean(0)):.2f}")
    print(f" (2) offset ||·||              : mean {mag.mean():.2f}  std {mag.std():.2f}  (rigid ⇒ low std)")
    print(f"     offset dir consistency    : cos(offset_i, mean offset) mean {cos_to_mean.mean():.3f} std {cos_to_mean.std():.3f}")
    print(f"     (cos→1 & low ||·|| std ⇒ CONSISTENT rigid offset, correctable by one translation)")
    if ch == "phase":
        _project(R, G)
    else:
        print(f"\n[note] channel={ch} (phase not finished yet); UMAP placement skipped (phase-only embedding).")


def _project(R, G):
    """Place real-NTC and inverse-α=0 into the phase gene embedding (real-pop std) — distance to real NTC cluster."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import gen_phate_passthrough as gp
    from numpy.linalg import norm
    a, comp, mean = gp._load_embedding()
    Xpca = np.asarray(a.obsm["X_pca"], np.float64)
    U = np.asarray(a.obsm["X_umap"], np.float64)
    m = gp._ntc_mask(a); ntc2d = U[m].mean(0)
    mu, sd = gp._real_baseline(gp._cache_files())                          # real-population standardization
    def place(X):
        pc = (((X - mu) / sd) - mean) @ comp.T
        return gp._landmark(pc.mean(0), Xpca, U)
    pr, pg = place(R), place(G)
    print("\n=== Phase UMAP placement (real-pop std) ===")
    print(f" real NTC cluster           : {ntc2d.round(2)}")
    print(f" real-anchor centroid lands : {pr.round(2)}  dist to NTC {norm(pr - ntc2d):.2f}  (sanity: should be small)")
    print(f" inverse-α=0 centroid lands : {pg.round(2)}  dist to NTC {norm(pg - ntc2d):.2f}  (old random-xT was 8.43)")


CTRL = f"{INV}/phase/_anchors/NTC/ctrl.npz".replace("viewer_assets_v5_inv", "viewer_assets_v5")


def webp_ab():
    """8-bit-webp vs proper-float A/B: embed the SAME 45 real NTC crops (float, from ctrl.npz) two ways —
    (A) float straight into CellDINO, (B) round-tripped through the traversal's 8-bit _save_webp path —
    and compare CellDINO cosine. Answers: does saving as a proper (float/zarr) image bring generated closer to real?"""
    import torch, tempfile  # noqa
    from ops_model.models.attention.diffex.classifier.celldino_features import embed_crops
    from ops_model.models.attention.diffex.directions.config import DirConfig
    from ops_model.models.attention.diffex.viewer.precompute import _save_webp
    os.makedirs(OUT, exist_ok=True)
    d = np.load(CTRL, allow_pickle=True)
    imgs = d["anchor_imgs"].astype(np.float32)                          # (45,1,160,160) float [-1,1]
    ctrl = d["ctrl_embs"].astype(np.float64)                            # pipeline float-path embeddings
    cfg = DirConfig(grain="geneKO", target="NTC", device="cuda")
    A = np.asarray(embed_crops(imgs, cfg, cache_path=None), np.float64)               # (A) float path
    tmp = tempfile.mkdtemp(); B = []
    for i in range(len(imgs)):
        p = f"{tmp}/c{i}.webp"; _save_webp(p, imgs[i, 0], 256)                         # exact traversal 8-bit path
        B.append(np.asarray(Image.open(p).convert("L"), np.float32) / 255.0 * 2 - 1)
    B = np.asarray(embed_crops(np.stack(B)[:, None].astype(np.float32), cfg, cache_path=None), np.float64)
    np.savez(f"{OUT}/webp_ab.npz", A=A, B=B, ctrl=ctrl)
    _report_ab()


def _report_ab():
    from numpy.linalg import norm
    d = np.load(f"{OUT}/webp_ab.npz"); A, B, ctrl = d["A"], d["B"], d["ctrl"]
    cs = lambda U, V: np.array([float(U[i] @ V[i] / (norm(U[i]) * norm(V[i]))) for i in range(len(U))])
    fw = cs(A, B)
    print("\n=== 8-bit webp vs proper float (same 45 real NTC cells, CellDINO) ===")
    print(f" float-path  vs  8bit-webp-path : cosine mean {fw.mean():.4f}  min {fw.min():.4f}  max {fw.max():.4f}")
    print(f" (sanity) float-path vs pipeline ctrl_embs : mean {cs(A, ctrl).mean():.4f}")
    print(" → cosine≈1 ⇒ webp is NOT the gap (proper/zarr image won't help; residual is generative/OOD)")
    print("   cosine notably <1 ⇒ 8-bit webp shifts the embedding; saving as float/zarr would help")


def submit(func=run, name="ntc_inverse_gap"):
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    submit_parallel_jobs([{"name": name, "func": func, "kwargs": {}}],
                         experiment="ntc_inverse_gap",
                         slurm_params={"slurm_partition": "gpu", "slurm_gres": "gpu:1", "cpus_per_task": 8,
                                       "mem_gb": 48, "timeout_min": 60}, log_dir="ntc_inverse_gap",
                         wait_for_completion=False)


if __name__ == "__main__":
    import sys
    if "--analyze" in sys.argv:
        analyze()
    elif "--report-ab" in sys.argv:
        _report_ab()
    elif "--webp" in sys.argv:
        submit(func=webp_ab, name="ntc_webp_ab")
    elif "--local" in sys.argv:
        run()
    else:
        submit()
