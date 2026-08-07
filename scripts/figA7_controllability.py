"""A7: constraint-margin trajectories (hot regime) — DG transiently violates a global
constraint on the canvas and patches it steps later. One line per rollout; margin = exact-checker
violation count of the decoded canvas (0 = satisfied). Outcome classes per analyze_constrained.py:
clean (never violating from t=3), escape (>=5-step violation spell, then satisfied), trapped.

Data: src_data/planning/canalysis.json (372 runs; builder diffusiongemma/planning/
analyze_constrained.py, original plot plots_constrained.py).
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figs"
RECS = json.load(open(ROOT / "src_data" / "planning" / "canalysis.json"))
TYPES = ["palindrome_words", "acrostic_word", "self_count_words", "snowball"]
OC = {"clean": ("#7aa2ff", 0.35, 0.8), "escape": ("#2f9e44", 1.0, 2.2), "trapped": ("#e8590c", 0.45, 0.8)}

fig, axes = plt.subplots(1, len(TYPES), sharey=True, layout="constrained",
                         figsize=(plt.rcParams["figure.figsize"][0] * 2.0,
                                  plt.rcParams["figure.figsize"][1] * 0.8))
for ax, tt in zip(axes, TYPES):
    rs = [r for r in RECS if r["ttype"] == tt and r["regime"] == "hot"]
    for r in sorted(rs, key=lambda r: r["outcome"] == "escape"):  # draw escapes on top
        col, al, lw = OC[r["outcome"]]
        m = [np.nan if x is None else x for x in r["m_out"]]
        ax.plot(m, color=col, alpha=al, linewidth=lw)
    ax.set_xlabel("denoising step $t$")
    ax.set_ylim(-0.3, 8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.97, 0.95, f"{tt}\n(n={len(rs)})", transform=ax.transAxes, ha="right", va="top",
            fontsize="small")
axes[0].set_ylabel("constraint margin (violations; 0 = satisfied)")
handles = [plt.Line2D([], [], color=OC[k][0], linewidth=2, label=k) for k in ("clean", "escape", "trapped")]
axes[0].legend(handles=handles, frameon=False, fontsize="small", loc="center right")
fig.savefig(OUT / "figA7_constraint_margins.png", dpi=200)
print(OUT / "figA7_constraint_margins.png")
esc = [r["cond"] for r in RECS if r["outcome"] == "escape"]
print("escape runs:", esc)
