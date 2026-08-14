"""Per-marker black point for the fluor Traversal clip = that marker's p1 over its generated frames —
the SAME marker-global percentile recipe Top Cells' crops_norm uses (viewer/_fluor_topcells.py: lo=p1).
Emits the TRAV_CLIP JS literal to paste into webapp/app.js (keyed by marker_channel).

  python -m ops_model.models.attention.diffex.viewer.build_trav_clip [assets_root]
"""
import glob
import json
import os
import sys
from re import sub

import numpy as np
from PIL import Image

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/hpc/projects/icd.fast.ops/models/diffex/viewer_assets_v5"
EXCL = {"phase", "_montage", "_montage_vs", "_rankings", "_directions", "top_cells", "_anchors", "pcs", "attention", "_morphometrics"}


def jsslug(s):
    return sub(r"[^0-9A-Za-z]+", "_", s)


def main():
    man = json.load(open(f"{ROOT}/manifest.json"))
    slug2ch = {jsslug(mk["marker_channel"]): mk["marker_channel"] for mk in man["markers"] if mk.get("marker_channel")}
    markers = [d for d in sorted(os.listdir(ROOT)) if os.path.isdir(f"{ROOT}/{d}") and d not in EXCL and os.path.isdir(f"{ROOT}/{d}/geneKO")]
    clip = {}
    for m in markers:
        fr = []
        for g in sorted(glob.glob(f"{ROOT}/{m}/geneKO/*/"))[:8]:
            for c in sorted(glob.glob(f"{g}cell*/"))[:1]:
                fr += sorted(glob.glob(f"{c}frame_*.webp"))[::3]
        px = [np.asarray(Image.open(f).convert("L")).ravel() for f in fr[:60]]
        if not px:
            continue
        p1 = np.percentile(np.concatenate(px).astype(float) / 255.0, 1)
        clip[slug2ch.get(m, m)] = round(float(p1), 3)
    js = "const TRAV_CLIP = {" + ", ".join(f'"{ch}": {v}' for ch, v in sorted(clip.items())) + "};"
    print(js)


if __name__ == "__main__":
    main()
