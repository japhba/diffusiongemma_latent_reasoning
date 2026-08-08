"""A9: answer resolution over denoising steps — post-hoc (bat_ball, monty, prod_then_digitsum)
vs load-bearing (reverse_then_add, sq1000, cubes_10_1000; the third case in each panel is from
the 2026-08-08 extension battery, captured via posthoc_ext/ext_anim.py, same GRID). Solid = mean token entropy over the ANSWER positions, dashed = over
the CoT positions; the answer-value path is annotated on the load-bearing panel.

Data: src_data/posthoc/com_posthoc_anim.json (posthoc/anim_entropy.py capture; C=256 T=128,
answer-first framing, warm regime). Curve = mean(entropy[k][ans_pos]) per step, first 9 steps
(denoising is converged after that; the tail is frozen).
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figs"
W = 9

cases = json.load(open(ROOT / "src_data" / "posthoc" / "com_posthoc_anim.json"))["cases"]
C2 = plt.rcParams["axes.prop_cycle"].by_key()["color"]

fig, axes = plt.subplots(1, 2, sharey=True, layout="constrained",
                         figsize=(plt.rcParams["figure.figsize"][0] * 1.6,
                                  plt.rcParams["figure.figsize"][1] * 0.85))
PANELS = [("post-hoc", axes[0], [C2[0], C2[9] if len(C2) > 9 else C2[2], C2[5]]),
          ("true-checking", axes[1], [C2[1], C2[3], C2[2]])]
for regime, ax, cols in PANELS:
    for c, col in zip([x for x in cases if x["regime"] == regime], cols):
        E = np.array(c["entropy"])
        ks = range(min(W, E.shape[0]))
        ax.plot(ks, [float(E[k][c["ans_pos"]].mean()) for k in ks], "-o", color=col,
                markersize=3.5, label=f'{c["pid"]} — answer')
        ax.plot(ks, [float(E[k][c["cot_pos"]].mean()) for k in ks], "--", color=col,
                alpha=0.65, label=f'{c["pid"]} — CoT')
        if regime == "true-checking":
            dy = {"reverse_then_add": 0.97, "sq1000": 0.88, "cubes_10_1000": 0.79}[c["pid"]]
            ax.annotate(" → ".join(c["answer_path"]), xy=(0.97, dy), xycoords="axes fraction",
                        ha="right", va="top", color=col, fontsize="small")
    ax.set_xlabel("denoising step $t$")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize="small")
axes[0].set_ylabel("mean token entropy (nats)")
fig.savefig(OUT / "figA9_resolution.png", dpi=200)
print(OUT / "figA9_resolution.png")
for c in cases:
    print(c["regime"], c["pid"], "flips", c["n_flips"], "path", "→".join(c["answer_path"]),
          "ans_lock", c["ans_lock"], "cot_lock", c["cot_lock"])
