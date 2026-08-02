"""A5: seasonal-vs-idiom ember-kill, tidied — 4 stacked trajectory panels (base, early/mid/late
single-step ablation of the idiom's S-mass at the 5 contested slots, seed s3) + a summary
outcome matrix over all kill seeds x ablation steps.

Data: diffusiongemma/exp/dg_planning/ember_kill.json (dg-planning seasonal study archive).
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

EXP = Path("/workspace-vast/jbauer/diffusiongemma/exp/dg_planning")
OUT = Path("/workspace-vast/jbauer/dg_blog/figs")
runs = json.load(open(EXP / "ember_kill.json"))
by = {(r["seed"], r["tag"]): r for r in runs}

IDIOM, SEAS = "k", "#9c36b5"
GREEN, ORANGE = "#2f9e44", "#e8590c"
PANELS = [("base", "base", None), ("kill@t2", "early kill", 2),
          ("kill@t5", "mid kill", 5), ("kill@t10", "late kill", 10)]
SEEDS, TS = (3, 4, 7), list(range(1, 13))

fig = plt.figure(layout="constrained",
                 figsize=(plt.rcParams["figure.figsize"][0],
                          plt.rcParams["figure.figsize"][1] * 1.55))
G = fig.add_gridspec(5, 1, height_ratios=[1, 1, 1, 1, 1.5])
axB = None
for i, (tag, lab, t_abl) in enumerate(PANELS):
    ax = fig.add_subplot(G[i, 0], sharex=axB, sharey=axB)
    axB = axB or ax
    r = by[(3, tag)]
    ax.plot(r["m_idt"], color=IDIOM, label="idiom")
    ax.plot(r["m_set"], color=SEAS, alpha=0.9, label="seasonal")
    if t_abl is not None:
        ax.plot([t_abl], [r["m_idt"][t_abl]], "x", color="#d32f2f", mew=1.8, ms=8, zorder=5)
    oc = r["outcome"]
    ax.text(0.99, 0.5, f"{lab}{f' @t{t_abl}' if t_abl else ''} → {oc}",
            transform=ax.transAxes, ha="right", va="center",
            color=GREEN if oc == "idiom" else ORANGE, fontweight="bold")
    ax.set_ylim(-0.08, 1.08)
    ax.set_yticks([0, 1])
    ax.set_ylabel(r"$\mathbf{S}$-mass")
    ax.spines[["top", "right"]].set_visible(False)
    if i == 0:
        ax.legend(loc="upper left", frameon=False, ncols=2)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    if i < 3:
        ax.tick_params(labelbottom=False)
    else:
        ax.set_xlabel("denoising step $t$")

axS = fig.add_subplot(G[4, 0])
M = np.full((len(SEEDS), len(TS)), np.nan)
OC = {"idiom": 0, "seasonal": 1, "other": 2}
for i, s in enumerate(SEEDS):
    for t in TS:
        r = by.get((s, f"kill@t{t}"))
        if r:
            M[i, t - 1] = OC[r["outcome"]]
axS.imshow(np.ma.masked_invalid(M), cmap=ListedColormap([GREEN, ORANGE, "#868e96"]),
           vmin=0, vmax=2, aspect="auto")
for t in (2, 5, 10):  # the panels above
    axS.add_patch(plt.Rectangle((t - 1.5, -0.5), 1, 1, fill=False, edgecolor="k", linewidth=1.6))
axS.set_xticks(range(len(TS)), TS)
axS.set_yticks(range(len(SEEDS)), [f"seed s{s}" for s in SEEDS])
axS.set_xlabel(r"ablation step $t_{\mathrm{abl}}$")
axS.legend(handles=[Patch(color=GREEN, label="final: idiom (escape survives)"),
                    Patch(color=ORANGE, label="final: seasonal (escape killed)")],
           loc="upper left", bbox_to_anchor=(0, -0.55), frameon=False, ncols=2)
fig.savefig(OUT / "figA5_seasonal_ember_kill.png", dpi=200)
print(OUT / "figA5_seasonal_ember_kill.png")
