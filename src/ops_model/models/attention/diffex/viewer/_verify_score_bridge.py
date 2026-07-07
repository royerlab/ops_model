"""Does embed_crops + z-standardize-on-control land in the SetTransformer's input space?
If yes, we can score generated cells: embed_crops(gen) → (x-μ_NTC)/σ_NTC → classifier bag.
Run as a GPU job; prints P(KIF11) for raw vs zstd-control embed_crops bags. Delete after."""
import numpy as np

from ..directions.config import DirConfig
from .precompute import _gather_class
from .set_classifier import load_set_classifier, score_bags


def run():
    cfg = DirConfig(grain="geneKO", target="KIF11", control="NTC", device="cuda")
    cfg.num_workers = 12
    _, ntc = _gather_class(cfg, "NTC", 400)       # embed_crops (per-image z-score) features
    _, kif = _gather_class(cfg, "KIF11", 400)
    mu, sd = ntc.mean(0), ntc.std(0) + 1e-6
    m, g2i, c2i = load_set_classifier("miwkg1cy", device="cuda")
    ci, tgt = c2i["Phase2D"], g2i["KIF11"]
    rng = np.random.default_rng(0)
    print(f"embed_crops |row|={np.linalg.norm(kif, axis=1).mean():.1f}")
    for name, feats in [("raw embed_crops", kif), ("zstd-on-control", (kif - mu) / sd)]:
        bags = np.stack([feats[rng.choice(len(feats), 100)] for _ in range(5)])
        p = score_bags(m, bags, ci, device="cuda")
        print(f"  {name}: P(KIF11)={p[:, tgt].mean():.3f}  top1=={int((p.argmax(1) == tgt).sum())}/5  "
              f"argmax={[list(g2i)[i] for i in p.argmax(1)]}")
    print("VERDICT: zstd-on-control P(KIF11) high → bridge works (embed_crops+zstd = classifier space).")
