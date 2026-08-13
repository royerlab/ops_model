"""Prune the public manifest: drop non-public perturbation grains (minibinder, PC) and their
design metadata (binder_prob/gene_target) so the public bucket's manifest.json can't leak
embargoed perturbations via a login-free curl.

Staging is untouched — staging serves the FULL manifest.json from its own bucket (its "Preview
public" toggle is UI-gating only). This script produces the pruned copy that feeds the PROD bucket.

Usage:
  python -m ops_model.models.attention.diffex.viewer.prune_public_manifest \
      <src manifest.json> <dst manifest.public.json>
"""
import json
import sys

PUBLIC_GRAINS = {"geneKO", "complex"}          # everything else (minibinder, pc, …) is dropped from the public copy
STRIP_KEYS = ("binder_prob", "gene_target")    # design metadata that must never reach the public copy


def prune(src, dst):
    m = json.load(open(src))
    kept = dropped = 0
    for mk in m.get("markers", []):
        out = []
        for t in mk.get("targets", []):
            if t.get("grain") not in PUBLIC_GRAINS:
                dropped += 1
                continue
            for k in STRIP_KEYS:
                t.pop(k, None)
            out.append(t)
            kept += 1
        mk["targets"] = out
    json.dump(m, open(dst, "w"))

    chk = json.load(open(dst))                  # verify nothing embargoed survived the round-trip
    grains = {t.get("grain") for mk in chk["markers"] for t in mk["targets"]}
    leaked_g = grains - PUBLIC_GRAINS
    leaked_k = any(k in t for mk in chk["markers"] for t in mk["targets"] for k in STRIP_KEYS)
    assert not leaked_g, f"leaked non-public grains: {leaked_g}"
    assert not leaked_k, f"leaked design metadata {STRIP_KEYS}"
    print(f"[prune] kept {kept} targets ({sorted(grains)}), dropped {dropped}; wrote {dst} — verified public-only")


if __name__ == "__main__":
    prune(sys.argv[1], sys.argv[2])
