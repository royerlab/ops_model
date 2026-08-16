"""Marker-specific centroid-recovery (the metric in centroid_pooled_bagsweep) for a few (marker, class) pairs.
Embed (CellDINO) every class's top-k real crops in the marker → class centroids; embed the target's generated
frames per α; per-domain standardize (real by real-pop, gen by gen-α0), L2, and per α compute the fraction of
generated cells whose NEAREST real centroid (over all marker classes) is the true class (top-1) + its mAP.
Peak-α of top-1 = the marker-correct analog of POLR1B's ~2.2. GPU (CellDINO embed only)."""
import glob
import json
import os
import types

import numpy as np
from PIL import Image

V5 = "/hpc/projects/icd.fast.ops/models/diffex/viewer_assets_v5"
OUT = "/hpc/projects/icd.fast.ops/analysis/figure4_traversals/centroid_recovery_fluor"
TOPK = 100

# (label, marker_slug, grain, index_block, class_name, gen_dir)
CASES = [
    ("5xUPRE-HSPA5", "5xUPRE", "geneKO", "genes", "HSPA5", "5xUPRE/geneKO/HSPA5"),
    ("LysoTracker-MTOR", "lysosome_LysoTracker_live_cell_dye", "geneKO", "genes", "MTOR", "lysosome_LysoTracker_live_cell_dye/geneKO/MTOR"),
    ("ChromaLIVE-TIM23", "mitochondria_ChromaLIVE_561_excitation", "complex", "complexes",
     "TIM23 mitochondrial inner membrane pre-sequence translocase complex, TIM17A variant",
     "mitochondria_ChromaLIVE_561_excitation/complex/TIM23_mitochondrial_inner_membrane_pre_sequence_translocase_complex__TIM17A_variant"),
    ("BODIPY-EMC", "lipid_droplet_BODIPY_live_cell_dye", "complex", "complexes",
     "Endoplasmic reticulum membrane complex, EMC8 variant",
     "lipid_droplet_BODIPY_live_cell_dye/complex/Endoplasmic_reticulum_membrane_complex__EMC8_variant"),
]


def _load(paths):
    ims = []
    for p in paths:
        try:
            ims.append(np.asarray(Image.open(p).convert("L"), np.float32))
        except Exception:
            pass
    return np.stack(ims)[:, None] if ims else np.zeros((0, 1, 256, 256), np.float32)


def _emb(paths, cfg):
    from ops_model.models.interpretability.diffae.classifier.celldino_features import embed_crops
    x = _load(paths)
    return embed_crops(x, cfg) if len(x) else np.zeros((0, 1024), np.float32)


def run_case(label, mod, grain, block, cls, gendir):
    cfg = types.SimpleNamespace(batch_size=64, celldino_z_score=True)
    idx = json.load(open(f"{V5}/top_cells/markers/{mod}/index.json"))
    cropdir = f"{V5}/top_cells/markers/{mod}/crops"
    classes = idx[block]                                                   # {class: [cell keys ranked]}
    # embed every class's top-k real crops → centroids + real-pop stats
    names, cents, allreal = [], [], []
    for c, rec in classes.items():
        keys = rec.get("accuracy") or rec.get("attention") or []          # ranked cells: [{'img': '<file>.webp', ...}]
        e = _emb([f"{cropdir}/{r['img']}" for r in keys[:TOPK]], cfg)
        if not len(e):
            continue
        names.append(c); cents.append(e.mean(0)); allreal.append(e)
    R = np.concatenate(allreal); mu_r, sd_r = R.mean(0), R.std(0) + 1e-6
    cz = (np.stack(cents) - mu_r) / sd_r
    cz = cz / (np.linalg.norm(cz, axis=1, keepdims=True) + 1e-9)
    ti = names.index(cls)
    # gen frames per α
    cells = sorted(glob.glob(f"{V5}/{gendir}/cell*/"))
    frames = sorted(os.path.basename(f) for f in glob.glob(f"{cells[0]}frame_*.webp"))
    meta = json.load(open(f"{V5}/{gendir}/meta.json")); al = [float(a) for a in meta["alphas"]]
    a0 = int(np.argmin(np.abs(np.array(al))))
    genA = {ai: [] for ai in range(len(al))}
    for cdir in cells:
        for ai, fr in enumerate(frames):
            p = f"{cdir}{fr}"
            if os.path.exists(p):
                genA[ai].append(p)
    # gen-α0 stats
    g0 = _emb(genA[a0], cfg); mu_g, sd_g = g0.mean(0), g0.std(0) + 1e-6
    top1, mp = [], []
    for ai in range(len(al)):
        e = _emb(genA[ai], cfg)
        if not len(e):
            top1.append(None); mp.append(None); continue
        gz = (e - mu_g) / sd_g; gz = gz / (np.linalg.norm(gz, axis=1, keepdims=True) + 1e-9)
        sim = gz @ cz.T; order = np.argsort(-sim, axis=1)
        rank_true = np.where(order == ti)[1] + 1
        top1.append(float(np.mean(order[:, 0] == ti))); mp.append(float(np.mean(1.0 / rank_true)))
    os.makedirs(OUT, exist_ok=True)
    json.dump({"label": label, "n_classes": len(names), "alphas": al, "top1": top1, "map": mp},
              open(f"{OUT}/{label}.json", "w"))
    return {"label": label, "classes": len(names)}


def submit():
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    jobs = [{"name": f"crf_{i}", "func": run_case, "kwargs": dict(zip(["label", "mod", "grain", "block", "cls", "gendir"], c))}
            for i, c in enumerate(CASES)]
    submit_parallel_jobs(jobs, experiment="crf",
                         slurm_params={"slurm_partition": "preempted", "slurm_gres": "gpu:1", "cpus_per_task": 8,
                                       "mem_gb": 48, "timeout_min": 120, "slurm_constraint": "[a40|a6000|l40s]"},
                         log_dir="crf", wait_for_completion=False)


if __name__ == "__main__":
    submit()
