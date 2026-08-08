"""A9: answer resolution over denoising steps, averaged over the full n=40 battery.

Problems are grouped by the susceptibility analysis (the load-bearing definition of this
section): post-hoc = S <= 0.1, load-bearing = S >= 0.3 (the S in (0.1, 0.3) middle band is
excluded and reported). Per problem: one rollout (seed 0, suscept GRID, answer-first framing),
per-step mean token entropy over the answer span (solid) and CoT region (dashed), positions
localized from the final canvas. Faint = single problems, bold = group mean, first 9 steps.

Data: src_data/posthoc/anim_curves.json (pod capture posthoc_ext/ext_anim_batch.py,
2026-08-09) + suscept.json for the grouping.
"""
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figs"
SP = ROOT / "src_data" / "posthoc"
W = 9
ANS, COT = "#1971c2", "0.45"

curves = json.load(open(SP / "anim_curves.json"))
susc = json.load(open(SP / "suscept.json"))
bypid = defaultdict(list)
for s in susc.values():
    bypid[s["pid"]].append(s)
S = {}
for pid, cells in bypid.items():
    bycs = defaultdict(dict)
    for s in cells:
        bycs[s["rho"]][s["corr_seed"]] = s["A_hat"]
    base = bycs[0.0]
    drift = [st.mean([1.0 if bycs[r].get(c) != base.get(c) else 0.0 for c in base if c in bycs[r]])
             for r in sorted(bycs) if r > 0]
    S[pid] = st.mean(drift)

groups = {"post-hoc": [p for p, v in S.items() if v <= 0.10 and curves.get(p)],
          "load-bearing": [p for p, v in S.items() if v >= 0.30 and curves.get(p)]}
mid = [p for p, v in S.items() if 0.10 < v < 0.30]
skipped = [p for p in S if curves.get(p) is None]
print(f"groups: { {k: len(v) for k, v in groups.items()} }, middle-band excluded {len(mid)} {mid}, "
      f"capture-skipped {skipped}")

fig, axes = plt.subplots(1, 2, sharey=True, layout="constrained",
                         figsize=(plt.rcParams["figure.figsize"][0] * 1.6,
                                  plt.rcParams["figure.figsize"][1] * 0.85))
for ax, (label, pids) in zip(axes, groups.items()):
    A = np.array([curves[p]["ans_curve"][:W] for p in pids])
    C = np.array([curves[p]["cot_curve"][:W] for p in pids])
    ks = np.arange(W)
    for row in A:
        ax.plot(ks, row, color=ANS, alpha=0.15, linewidth=0.7)
    for row in C:
        ax.plot(ks, row, color=COT, linestyle="--", alpha=0.12, linewidth=0.7)
    ax.plot(ks, A.mean(0), "-o", color=ANS, markersize=3.5, linewidth=2.0, label="answer positions")
    ax.plot(ks, C.mean(0), "--", color=COT, linewidth=2.0, label="CoT positions")
    ax.set_title(f"{label} (n={len(pids)})")
    ax.set_xlabel("denoising step $t$")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("mean token entropy (nats)")
axes[0].legend(frameon=False, fontsize="small")
fig.savefig(OUT / "figA9_resolution.png", dpi=200)
print(OUT / "figA9_resolution.png")
