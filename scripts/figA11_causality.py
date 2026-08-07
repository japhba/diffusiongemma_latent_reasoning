"""A11: how causal (left-to-right) is DG's commitment order? Per task type (hot sampler,
T=64): center of mass of the canvas indices of the currently-committed content positions,
normalized to the content span, per denoising step. Dashed = matched left-to-right filler
(same committed count, leftmost positions first): a purely causal (AR-like) fill would sit on
it; a holistic fill sits flat at 0.5. Faint = single rollouts, bold = task mean.

Data: src_data/planning/commit_com.json (extract_commit_com.py over the raw cruns archive).
The default (colder) regime is omitted: it commits the full canvas by step ~3, leaving no
order signal to read.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figs"
rows = [r for r in json.load(open(ROOT / "src_data" / "planning" / "commit_com.json"))
        if r["regime"] == "hot"]
TTS = sorted({r["ttype"] for r in rows})
XMAX = 36
OBS, REF = "#1971c2", "0.35"

fig, axes = plt.subplots(2, 4, sharex=True, sharey=True, layout="constrained",
                         figsize=(plt.rcParams["figure.figsize"][0] * 2.0,
                                  plt.rcParams["figure.figsize"][1] * 1.1))
for ax, tt in zip(axes.flat, TTS):
    rs = [r for r in rows if r["ttype"] == tt]
    com = np.full((len(rs), XMAX), np.nan)
    ref = np.full((len(rs), XMAX), np.nan)
    for i, r in enumerate(rs):
        for t in range(min(XMAX, r["T"])):
            if r["com"][t] is not None:
                com[i, t] = r["com"][t]
                ref[i, t] = r["ref"][t]
        ax.plot(com[i], color=OBS, alpha=0.15, linewidth=0.7)
    ax.plot(np.nanmean(com, axis=0), color=OBS, linewidth=2.0)
    ax.plot(np.nanmean(ref, axis=0), color=REF, linestyle="--", linewidth=1.4)
    ax.axhline(0.5, color="0.85", linewidth=0.6, zorder=0)
    ax.text(0.97, 0.05, f"{tt}\n(n={len(rs)})", transform=ax.transAxes, ha="right", va="bottom",
            fontsize="small")
    ax.set_ylim(-0.02, 1.02)
    ax.spines[["top", "right"]].set_visible(False)
for ax in axes.flat[len(TTS):]:
    ax.axis("off")
for ax in axes[-1]:
    ax.set_xlabel("denoising step $t$")
for ax in axes[:, 0]:
    ax.set_ylabel("committed CoM\n(0 = left, 1 = right)")
handles = [plt.Line2D([], [], color=OBS, linewidth=2, label="observed"),
           plt.Line2D([], [], color=REF, linestyle="--", linewidth=1.4, label="left-to-right filler (matched count)")]
axes[0, 0].legend(handles=handles, frameon=False, fontsize="small", loc="upper left")
fig.savefig(OUT / "figA11_commit_causality.png", dpi=200)
print(OUT / "figA11_commit_causality.png")
print("ttypes:", TTS)
