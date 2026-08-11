"""Plot final training loss + conditioning ratio (best_ratio) for every DiffAE generator, read
straight from each run's diffae_train_state.pt (history[-1].loss, best_ratio, epoch)."""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

plt.rcParams["pdf.fonttype"] = 42

DD = "/hpc/projects/icd.fast.ops/models/diffex/diffae"
OUT = "/hpc/projects/icd.fast.ops/models/diffex/model_metrics"


def collect():
    rows = []
    for sp in sorted(glob.glob(f"{DD}/*/diffae_train_state.pt")):
        name = os.path.basename(os.path.dirname(sp))
        try:
            s = torch.load(sp, map_location="cpu", mmap=True)
        except Exception as e:
            print("skip", name, e); continue
        h = s.get("history", [])
        cr = [(e["epoch"], e["cond_ratio"]) for e in h if e.get("cond_ratio", -1) >= 0]
        rows.append(dict(name=name, ep=[e["epoch"] for e in h], loss=[e["loss"] for e in h],
                         cr_ep=[x[0] for x in cr], cr=[x[1] for x in cr],
                         is_phase=name.startswith("phase"), best=float(s.get("best_ratio", np.nan))))
    return rows


def main():
    rows = collect()
    fluor = [r for r in rows if not r["is_phase"]]
    fcolors = plt.cm.viridis(np.linspace(0, 0.95, max(len(fluor), 1)))
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    for i, r in enumerate(fluor):                                  # fluor: thin viridis lines (each its own)
        ax[0].plot(r["ep"], r["loss"], color=fcolors[i], lw=0.8, alpha=0.5)
        if r["cr"]:
            ax[1].plot(r["cr_ep"], r["cr"], color=fcolors[i], lw=0.8, alpha=0.5)
    for r in rows:                                                 # phase: thick orange, labeled, on top
        if not r["is_phase"]:
            continue
        ax[0].plot(r["ep"], r["loss"], lw=2.4, alpha=0.95, label=r["name"])
        if r["cr"]:
            ax[1].plot(r["cr_ep"], r["cr"], lw=2.4, alpha=0.95, label=r["name"])

    ax[0].set_yscale("log"); ax[0].set_title("training loss"); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss (log)")
    ax[1].axhline(0.468, ls="--", c="#888", lw=1, label="prod 0.468")
    ax[1].set_title("conditioning ratio"); ax[1].set_xlabel("epoch"); ax[1].set_ylabel("cond_ratio")
    for a in ax:
        a.grid(alpha=0.3); a.legend(fontsize=7, ncol=2, loc="best")
    fig.suptitle(f"DiffAE generators (n={len(rows)}) — loss + conditioning-ratio curves "
                 f"(phase highlighted; {len(fluor)} fluor in viridis)", y=0.99)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(f"{OUT}_curves.{ext}", dpi=140, bbox_inches="tight")
    print(f"[plot] {len(rows)} models ({len(fluor)} fluor) -> {OUT}_curves.png/.svg")


if __name__ == "__main__":
    main()
