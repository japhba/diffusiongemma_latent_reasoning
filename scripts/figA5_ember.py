"""A5: seasonal-vs-idiom ember-kill — left: 4 stacked trajectory panels (base separated by a gap
+ hline; early/mid/late single-step ablations, seed s3); right: extended outcome matshow over
all arms (1-step kills, persistent kills, s0 rescues) x ablation steps, cell text = draft-flip step.

Data: diffusiongemma/exp/dg_planning/ember_kill.json (dg-planning seasonal study archive).
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "src_data"
OUT = ROOT / "figs"
runs = json.load(open(EXP / "ember_kill.json"))
by = {(r["seed"], r["tag"]): r for r in runs}

IDIOM, SEAS = "k", "#9c36b5"
GREEN, ORANGE, GREY = "#2f9e44", "#e8590c", "#868e96"
PANELS = [("base", "base", None), ("kill@t2", "early kill", 2),
          ("kill@t5", "mid kill", 5), ("kill@t10", "late kill", 10)]
SINGLE_T, PERSIST_T = list(range(1, 13)), [2, 4, 6, 8, 10]
ROWS = ([(s, "kill", "1-step", SINGLE_T) for s in (3, 4, 7)] +
        [(s, "kill", "persist", PERSIST_T) for s in (3, 4, 7)] +
        [(0, "rescue", "1-step", SINGLE_T), (0, "rescue", "persist", PERSIST_T)])
OC = {"idiom": 0, "seasonal": 1, "other": 2}

fig = plt.figure(layout="constrained",
                 figsize=(plt.rcParams["figure.figsize"][0] * 2.1,
                          plt.rcParams["figure.figsize"][1] * 1.25))
G = fig.add_gridspec(5, 2, height_ratios=[1, 0.22, 1, 1, 1], width_ratios=[1.05, 1])

# ---- left: trajectories (base | gap | early, mid, late) ----
row_of = [0, 2, 3, 4]
axB, traj_axes = None, []
for (tag, lab, t_abl), gr in zip(PANELS, row_of):
    ax = fig.add_subplot(G[gr, 0], sharex=axB, sharey=axB)
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
    ax.set_ylabel(r"$\mathbf{S}$-mass")
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    if gr == 0:
        ax.legend(loc="upper left", frameon=False, ncols=2)
    if gr < 4:
        ax.tick_params(labelbottom=False)
    else:
        ax.set_xlabel("denoising step $t$")

# ---- right: extended outcome matshow ----
axS = fig.add_subplot(G[:, 1])
M = np.full((len(ROWS), len(SINGLE_T)), np.nan)
for i, (s, arm, kind, ts) in enumerate(ROWS):
    for t in ts:
        r = by.get((s, f"{arm}@t{t}" + ("+" if kind == "persist" else "")))
        if r is None:
            continue
        M[i, t - 1] = OC[r["outcome"]]
        axS.text(t - 1, i, "—" if r["flip"] is None else str(r["flip"]),
                 ha="center", va="center", fontsize="small",
                 color="white" if OC[r["outcome"]] < 2 else "black")
cmap = ListedColormap([GREEN, ORANGE, GREY])
cmap.set_bad("#00000010")
axS.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=0, vmax=2, aspect="auto")
for t in (2, 5, 10):  # the trajectory panels shown left
    axS.add_patch(plt.Rectangle((t - 1.5, -0.5), 1, 1, fill=False, edgecolor="k", linewidth=1.6))
for yy in (2.5, 5.5):  # group separators: 1-step kills | persistent kills | rescues
    axS.axhline(yy, color="white", linewidth=2.5)
axS.set_xticks(range(len(SINGLE_T)), SINGLE_T)
axS.set_yticks(range(len(ROWS)),
               [f"s{s} {arm} {kind}" for s, arm, kind, _ in ROWS])
axS.set_xlabel(r"ablation step $t_{\mathrm{abl}}$")
axS.legend(handles=[Patch(color=GREEN, label="final: idiom"),
                    Patch(color=ORANGE, label="final: seasonal"),
                    Patch(color=GREY, label="final: other")],
           loc="upper left", bbox_to_anchor=(0, -0.12), frameon=False, ncols=3)

# separator between the base panel and the ablation panels (left column)
fig.canvas.draw()
fig.set_layout_engine("none")
b0, b1 = traj_axes[0].get_position(), traj_axes[1].get_position()
ymid = (b0.y0 + b1.y1) / 2
fig.add_artist(plt.Line2D([b0.x0, b0.x1], [ymid, ymid], color="0.5", linewidth=0.8,
                          transform=fig.transFigure))
fig.savefig(OUT / "figA5_seasonal_ember_kill.png", dpi=200)
print(OUT / "figA5_seasonal_ember_kill.png")
