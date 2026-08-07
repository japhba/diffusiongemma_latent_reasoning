"""A11: how causal (left-to-right) is DG's commitment order? Commit-order diagram per task
(hot sampler): x = commit rank (order in which content positions finally commit, normalized),
y = normalized canvas position of that commitment. A strictly left-to-right (AR-like) order is
the diagonal; descending segments are anticausal stages. Panel annotation: median Spearman
rank correlation between commit step and position (chain rho; +1 = causal, -1 = anticausal).

This replaces the earlier center-of-mass version: CoM averages symmetric spreads to 0.5 and
hides end-anchoring. Data: src_data/planning/commit_order.json (extract_commit_order.py over
the raw cruns archive; final-commit step = start of the accepted-mask's terminal True suffix).
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figs"
rows = [r for r in json.load(open(ROOT / "src_data" / "planning" / "commit_order.json"))
        if r["regime"] == "hot"]
TTS = sorted({r["ttype"] for r in rows})
OBS = "#1971c2"

def rankdata(v):
    v = np.asarray(v, float); order = np.argsort(v); ranks = np.empty(len(v)); sv = v[order]; i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks

GRID = np.linspace(0, 1, 101)
fig, axes = plt.subplots(2, 4, sharex=True, sharey=True, layout="constrained",
                         figsize=(plt.rcParams["figure.figsize"][0] * 2.0,
                                  plt.rcParams["figure.figsize"][1] * 1.1))
for ax, tt in zip(axes.flat, TTS):
    rs = [r for r in rows if r["ttype"] == tt]
    curves, rhos = [], []
    for r in rs:
        pos = np.array(r["content"], float)
        posn = (pos - pos.min()) / max(pos.max() - pos.min(), 1)
        fc = np.array(r["first_commit"], float)
        rhos.append(float(np.corrcoef(rankdata(pos), rankdata(fc))[0, 1]))
        order = np.argsort(fc, kind="stable")
        xr = np.arange(len(order)) / max(len(order) - 1, 1)
        ax.plot(xr, posn[order], color=OBS, alpha=0.15, linewidth=0.7)
        curves.append(np.interp(GRID, xr, posn[order]))
    ax.plot(GRID, np.mean(curves, axis=0), color=OBS, linewidth=2.0)
    ax.plot([0, 1], [0, 1], color="0.6", linestyle="--", linewidth=1.2)
    ax.text(0.03, 0.95, rf"$\rho_{{\mathrm{{chain}}}} = {np.median(rhos):+.2f}$",
            transform=ax.transAxes, va="top", fontsize="small")
    ax.text(0.97, 0.05, f"{tt}\n(n={len(rs)})", transform=ax.transAxes, ha="right", va="bottom",
            fontsize="small")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 1.02)
    ax.spines[["top", "right"]].set_visible(False)
for ax in axes.flat[len(TTS):]:
    ax.axis("off")
for ax in axes[-1]:
    ax.set_xlabel("commit rank (fraction)")
for ax in axes[:, 0]:
    ax.set_ylabel("canvas position\n(0 = left, 1 = right)")
fig.savefig(OUT / "figA11_commit_causality.png", dpi=200)
print(OUT / "figA11_commit_causality.png")
