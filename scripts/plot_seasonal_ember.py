"""A5: seasonal-vs-idiom preservation — 2 stacked panels (seed s5): base autonomous run
(idiom takes over) vs persistent idiom-kill from t=2 (dotted onset + shading) preserving the
native seasonal draft. Replaces the earlier single-step-kill figure (data regenerated
2026-08-03, "preservation, not flipping").

Data: src_data/ember_base_traj.json + src_data/ember_kill2.json (palindrome_words__3 capture;
builder diffusiongemma/planning/ember_preserve_fig.py).
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figs"
SEED, T_ABL = 5, 2
IDIOM, SEAS = "k", "#9c36b5"
GREEN, ORANGE, RED = "#2f9e44", "#e8590c", "#d32f2f"

base = json.load(open(ROOT / "src_data" / "ember_base_traj.json"))[str(SEED)]
kill = json.load(open(ROOT / "src_data" / "ember_kill2.json"))[f"s{SEED}|kill@t{T_ABL}+"]

fig = plt.figure(layout="constrained",
                 figsize=(plt.rcParams["figure.figsize"][0] * 1.15,
                          plt.rcParams["figure.figsize"][1] * 0.85))
G = fig.add_gridspec(3, 1, height_ratios=[1, 0.22, 1])

axB, panel_axes = None, []
for gr, r, lab, shaded in ((0, base, "base", False),
                           (2, kill, f"persistent ablation @t{T_ABL}+", True)):
    ax = fig.add_subplot(G[gr], sharex=axB, sharey=axB)
    panel_axes.append(ax)
    axB = axB or ax
    ax.plot(r["m_idt"], color=IDIOM, label="idiom")
    ax.plot(r["m_set"], color=SEAS, alpha=0.9, label="seasonal")
    if shaded:
        ax.axvspan(T_ABL, len(r["m_idt"]) - 1, color=RED, alpha=0.08, lw=0)
        ax.axvline(T_ABL, color=RED, lw=1.0, ls=":")
    oc = r["outcome"]
    ax.text(0.99, 0.5, f"{lab} → {oc}", transform=ax.transAxes, ha="right", va="center",
            color=GREEN if oc == "idiom" else ORANGE, fontweight="bold")
    ax.set_ylim(-0.08, 1.08)
    ax.set_yticks([0, 1])
    ax.set_ylabel(r"$\mathbf{s}$-mass")
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
panel_axes[0].legend(loc="upper left", frameon=False, ncols=2)
panel_axes[0].tick_params(labelbottom=False)
panel_axes[1].set_xlabel("denoising step $t$")

# separator between the base panel and the intervention panel
fig.canvas.draw()
fig.set_layout_engine("none")
b0, b1 = panel_axes[0].get_position(), panel_axes[1].get_position()
ymid = (b0.y0 + b1.y1) / 2
fig.add_artist(plt.Line2D([b0.x0, b0.x1], [ymid, ymid], color="0.5", linewidth=0.8,
                          transform=fig.transFigure))
fig.savefig(OUT / "seasonal_ember_kill.png", dpi=200)
print(OUT / "seasonal_ember_kill.png")
print("base:", base["outcome"], "kill:", kill["outcome"], "flip:", kill.get("flip"))
