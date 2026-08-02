"""A5: seasonal-vs-idiom ember-kill — 4 stacked trajectory panels (base separated by a gap
+ hline; early/mid/late single-step ablations, seed s3). The outcome matshow was dropped
2026-08-02 per user request.

Data: src_data/ember_kill.json (dg-planning seasonal study archive).
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "src_data"
OUT = ROOT / "figs"
runs = json.load(open(EXP / "ember_kill.json"))
by = {(r["seed"], r["tag"]): r for r in runs}

IDIOM, SEAS = "k", "#9c36b5"
GREEN, ORANGE = "#2f9e44", "#e8590c"
PANELS = [("base", "base", None), ("kill@t2", "early kill", 2),
          ("kill@t5", "mid kill", 5), ("kill@t10", "late kill", 10)]

fig = plt.figure(layout="constrained",
                 figsize=(plt.rcParams["figure.figsize"][0] * 1.15,
                          plt.rcParams["figure.figsize"][1] * 1.25))
G = fig.add_gridspec(5, 1, height_ratios=[1, 0.22, 1, 1, 1])

# ---- trajectories (base | gap | early, mid, late) ----
row_of = [0, 2, 3, 4]
axB, traj_axes = None, []
for (tag, lab, t_abl), gr in zip(PANELS, row_of):
    ax = fig.add_subplot(G[gr], sharex=axB, sharey=axB)
    traj_axes.append(ax)
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
    ax.set_ylabel(r"$\mathbf{s}$-mass")
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    if gr == 0:
        ax.legend(loc="upper left", frameon=False, ncols=2)
    if gr < 4:
        ax.tick_params(labelbottom=False)
    else:
        ax.set_xlabel("denoising step $t$")

# separator between the base panel and the ablation panels
fig.canvas.draw()
fig.set_layout_engine("none")
b0, b1 = traj_axes[0].get_position(), traj_axes[1].get_position()
ymid = (b0.y0 + b1.y1) / 2
fig.add_artist(plt.Line2D([b0.x0, b0.x1], [ymid, ymid], color="0.5", linewidth=0.8,
                          transform=fig.transFigure))
fig.savefig(OUT / "figA5_seasonal_ember_kill.png", dpi=200)
print(OUT / "figA5_seasonal_ember_kill.png")
