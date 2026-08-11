"""GPU check: does Alex's per-gene .pt CellDINO embedding == embed_crops (the DiffAE conditioning
space)? Embed top cells via the current gather, match to .pt by segmentation_id, compare.
Run as a one-off SLURM job; prints cosine + gap agreement. Delete after."""
import torch, numpy as np
from .precompute import _gather_class
from ..directions.config import DirConfig
from ..directions.data import _top_cells
from ..classifier.config import GRAINS

PT = "/hpc/projects/icd.fast.ops/models/alex_lin_attention/v4/train_ops_zstdcontrol_cdino_v2"


def run():
    cfg = DirConfig(grain="geneKO", target="KIF11", control="NTC", device="cuda")
    cfg.num_workers = 12
    pq = GRAINS["geneKO"]["parquet"]
    for g in ["NTC", "KIF11"]:
        rows = _top_cells(pq, "gene", g, 40).reset_index(drop=True)
        imgs, embs = _gather_class(cfg, g, 40)               # fresh embed_crops
        rows = rows.iloc[:len(embs)]
        # COMPOSITE key (segmentation_id is only unique within an image)
        key_fresh = list(zip(rows["experiment"].astype(str), rows["well"].astype(str),
                             rows["segmentation"].astype(np.int64)))
        o = torch.load(f"{PT}/{g}.pt", map_location="cpu")
        E = np.asarray(o["embeddings"], np.float32)
        md = o["cell_metadata"]
        exp_pt = [x for bag in md["experiment"] for x in bag]
        well_pt = [x for bag in md["well"] for x in bag]
        seg_pt = [x for bag in md["segmentation_id"] for x in bag]
        pt_by_key = {(str(e), str(w), int(s)): E[i] for i, (e, w, s) in enumerate(zip(exp_pt, well_pt, seg_pt))}
        cos, nr = [], []
        for e, k in zip(embs, key_fresh):
            if k in pt_by_key:
                p = pt_by_key[k]
                cos.append(float(e @ p / (np.linalg.norm(e) * np.linalg.norm(p) + 1e-9)))
                nr.append(np.linalg.norm(e) / (np.linalg.norm(p) + 1e-9))
        cos = np.array(cos)
        print(f"[{g}] fresh embed_crops |row|={np.linalg.norm(embs,axis=1).mean():.2f}  "
              f".pt |row|={np.linalg.norm(E,axis=1).mean():.2f}  "
              f"matched {len(cos)}  cos(fresh,.pt) mean={cos.mean():.4f} min={cos.min():.4f}  "
              f"norm-ratio={np.mean(nr):.3f}")
    print("VERDICT: cos≈1.0 + norm-ratio≈1.0 → .pt IS embed_crops space (safe drop-in). "
          "cos≈1 but ratio≠1 → same direction, rescale needed. cos<0.9 → different space, do NOT use.")


if __name__ == "__main__":
    run()
