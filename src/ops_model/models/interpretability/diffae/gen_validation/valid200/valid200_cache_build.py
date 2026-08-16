"""Build a gen_real_map-format cache for the 200-cell validation bag (viewer_assets_valid200).

Per gene: reuse the real top-K cells from the existing gen_real_map_cache, and embed the 200 generated
webp frames at each of the 7 valid200 α through Cell-DINO (loaded once per shard). Output npz per gene
{real, gen (list[7] of (200,1024)), alphas (7), gene} — the exact format gen_real_map / gen_real_centroid /
gen_real_distinct consume, so the mAP suite can run on the 200-cell bag by pointing CACHE here.
"""
import os, glob
import numpy as np

GRAIN = os.environ.get("V200_GRAIN", "geneKO")     # geneKO | complex
ASSETS = os.environ.get("V200_ASSETS", "viewer_assets_valid200")            # new multibag: viewer_assets_v5
OUTCACHE = os.environ.get("V200_OUTCACHE", "gen_real_map_cache_valid200")   # new multibag: gen_real_map_cache_v5new
V = f"/hpc/projects/icd.fast.ops/models/diffex/{ASSETS}/phase/{GRAIN}"
OLD = f"/hpc/projects/icd.fast.ops/analysis/figure4_traversals/gen_real_map_cache/{GRAIN}"
OUT = f"/hpc/projects/icd.fast.ops/analysis/figure4_traversals/{OUTCACHE}/{GRAIN}"
NCELL = int(os.environ.get("V200_NCELL", "200"))


def _alphas():
    """Read the actual α grid from a gene's meta.json (new v5 = 17-pt [-5..5]); fallback to the 7-pt valid200 grid."""
    import json
    for d in sorted(glob.glob(f"{V}/*")):
        mp = f"{d}/meta.json"
        if os.path.exists(mp):
            return np.asarray(json.load(open(mp))["alphas"], np.float32)
    return np.array([0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0], np.float32)


ALPHAS = _alphas()
NA = len(ALPHAS)


def build_shard(genes):
    import torch
    from PIL import Image
    from ops_model.models.cell_dino import CellDinoModel
    os.makedirs(OUT, exist_ok=True)
    model = CellDinoModel(z_score=True)

    def emb(imgs):                                          # (N,1,H,W) float32 → (N,1024)
        out = []
        with torch.inference_mode():
            for i in range(0, len(imgs), 256):
                out.append(model.extract_features({"data": torch.as_tensor(imgs[i:i + 256])}).float().cpu().numpy())
        return np.concatenate(out).astype(np.float32)

    for g in genes:
        if os.path.exists(f"{OUT}/{g}.npz"):
            continue                                        # resume: already built
        oc = f"{OLD}/{g}.npz"                               # reuse old real ref if present; else empty (real not needed for
        real = np.asarray(np.load(oc, allow_pickle=True)["real"], np.float32) if os.path.exists(oc) else np.zeros((0, 1024), np.float32)  # SetTransformer/distinct; centroid-recovery uses centroids.npz)
        gen = []
        for ai in range(NA):
            imgs = []
            for c in range(NCELL):
                f = f"{V}/{g}/cell{c}/frame_{ai:02d}.webp"
                if os.path.exists(f):
                    imgs.append(np.asarray(Image.open(f).convert("L"), np.float32) / 255.0 * 2 - 1)
            gen.append(emb(np.stack(imgs)[:, None].astype(np.float32)) if imgs else None)
        np.savez(f"{OUT}/{g}.npz", real=real, gen=np.array(gen, dtype=object), alphas=ALPHAS, gene=g)
    return len(genes)


def main():
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    gf = os.environ.get("V200_GENES_FILE")
    if gf:                                                              # explicit gene list (e.g. the 42 dropped)
        genes = [g.strip() for g in open(gf) if g.strip()]
    else:
        genes = sorted(os.path.basename(d) for d in glob.glob(f"{V}/*")
                       if os.path.isdir(d) and "__to__" not in os.path.basename(d) and not os.path.basename(d).startswith("_"))
    ch = int(os.environ.get("V200_CHUNK", "40"))
    shards = [genes[i:i + ch] for i in range(0, len(genes), ch)]
    jobs = [{"name": f"v200cache_{i}", "func": build_shard, "kwargs": {"genes": s}} for i, s in enumerate(shards)]
    print(f"[valid200-cache] {len(genes)} genes → {len(jobs)} GPU shards")
    submit_parallel_jobs(jobs, experiment="valid200_cache",
                         slurm_params={"slurm_partition": "preempted", "slurm_gres": "gpu:1", "cpus_per_task": 10,
                                       "mem_gb": 64, "timeout_min": 180,
                                       "slurm_constraint": "[a40|a6000|l40s]"},   # preempted queue + weak GPUs — light embed job
                         log_dir="valid200_cache", wait_for_completion=False)


if __name__ == "__main__":
    main()
