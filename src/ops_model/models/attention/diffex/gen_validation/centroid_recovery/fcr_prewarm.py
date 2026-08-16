"""Parallel pre-warm of the fcr CellDINO embcache for the heavy markers: shard each (marker,grain) into
`nchunks` stride-chunks of galleries + gen-traversals, embed each chunk on its own GPU. Resume-safe (skips
already-cached keys). After this, re-run the fcr scoring shards → all cache hits → finish in minutes.
"""
import glob
import json
import os
import types

from ops_model.models.attention.diffex.gen_validation.centroid_recovery.f_centroid_recovery import _emb, _slug, TOPK, V5

# the 7 heavy geneKO markers still missing their fcr JSON
HEAVY = [("nucleus_NucleoLIVE_Live_Cell_dye", "geneKO"), ("nuclear_speckles_SRRM2", "geneKO"),
         ("lysosome_LysoTracker_live_cell_dye", "geneKO"), ("cis_Golgi_mStayGold_CENPRaltORF", "geneKO"),
         ("mitochondria_ChromaLIVE_561_excitation", "geneKO"), ("nucleolus_GC_NPM3", "geneKO"),
         ("lipid_droplet_BODIPY_live_cell_dye", "geneKO")]
NCHUNKS = 8


def embed_chunk(mod, grain, i, nchunks=NCHUNKS, device="cuda"):
    cfg = types.SimpleNamespace(batch_size=128, celldino_z_score=True)
    block = "genes" if grain == "geneKO" else "complexes"
    idx = json.load(open(f"{V5}/top_cells/markers/{mod}/index.json"))
    cropdir = f"{V5}/top_cells/markers/{mod}/crops"
    # galleries: stride-chunk of the marker's classes
    classes = sorted(idx.get(block, {}).keys())[i::nchunks]
    for c in classes:
        rec = idx[block][c]; keys = (rec.get("accuracy") or rec.get("attention") or [])[:TOPK]
        _emb([f"{cropdir}/{r['img']}" for r in keys], cfg, key=f"gal/{mod}/{grain}/{_slug(c)}")
    # gen frames: stride-chunk of the marker's traversals
    metas = sorted(glob.glob(f"{V5}/{mod}/{grain}/*/meta.json"))[i::nchunks]
    for mp in metas:
        gd = os.path.dirname(mp); meta = json.load(open(mp)); cls = meta.get("target") or os.path.basename(gd)
        cells = sorted(glob.glob(f"{gd}/cell*/"))
        if not cells:
            continue
        frames = sorted(os.path.basename(f) for f in glob.glob(f"{cells[0]}frame_*.webp"))
        gk = f"gen/{mod}/{grain}/{_slug(cls)}"
        for ai, fr in enumerate(frames):
            _emb([f"{cd}{fr}" for cd in cells if os.path.exists(f"{cd}{fr}")], cfg, key=f"{gk}/a{ai}")
    return {"mod": mod, "grain": grain, "chunk": i, "gal": len(classes), "gen": len(metas)}


def submit():
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    jobs = [{"name": f"pw_{m[:10]}_{i}", "func": embed_chunk, "kwargs": {"mod": m, "grain": g, "i": i}}
            for m, g in HEAVY for i in range(NCHUNKS)]
    print(f"[prewarm] {len(jobs)} GPU embed-chunks ({len(HEAVY)} markers × {NCHUNKS})")
    submit_parallel_jobs(jobs, experiment="pw",
                         slurm_params={"slurm_partition": "gpu", "slurm_gres": "gpu:1", "cpus_per_task": 8,
                                       "mem_gb": 48, "timeout_min": 90, "slurm_constraint": "[a40|a6000|l40s|a100]",
                                       "slurm_exclude": "gpu-b-4"},
                         log_dir="pw", wait_for_completion=False)


if __name__ == "__main__":
    submit()
