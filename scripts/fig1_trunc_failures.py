"""Fig 1: GPQA top-k truncation — % failed rollouts, standard vs gentle sampler, stacked by failure mode.

Data: fig2-trunc study manifests (robust grade), n=64/arm.
  standard: commit_ds/acts_bench/gpqa/<rung>/manifest.json
  gentle:   soft/k1 from acts_stab/gpqa/<rung>_slow3/manifest.json,
            k8/k4/k2 from acts_psweep/gpqa/<rung>_slow3/manifest_w*.jsonl
Failure modes (per lockin/fig2_report.py norm_record):
  loop   = budget exhausted (not finished) and dup8 >= 0.5
  capped = budget exhausted, still writing fresh text (dup8 < 0.5)
  wrong  = self-terminated but graded incorrect (robust grade)
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CD = ROOT / "src_data" / "commit_ds"
OUT = ROOT / "figs"
RUNGS = ["soft", "k8", "k4", "k2", "k1"]
RUNG_LAB = {"soft": "soft", "k8": "top-8", "k4": "top-4", "k2": "top-2", "k1": "top-1"}
FMS = ["wrong", "capped", "loop"]  # stacking order bottom -> top
FMCOL = {"loop": "#8a5a2b", "capped": "#eda100", "wrong": "#e87ba4"}
FMLAB = {"loop": "degenerate loop", "capped": "budget-capped mid-answer", "wrong": "wrong answer (clean finish)"}


def tail_dup(text, n=16):
    t = text or ""
    if len(t) < 4 * n:
        return 0.0
    grams = [t[i:i + n] for i in range(len(t) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def classify(m):
    ok = bool(m.get("correct", m.get("ok")))
    if ok:
        return "ok"
    if not m.get("finished", True):
        d8 = m.get("dup8")
        return "loop" if (d8 if d8 is not None else tail_dup(m.get("final_text", ""))) >= 0.5 else "capped"
    return "wrong"


def load_arm(rung, sampler):
    if sampler == "std":
        rows = list(json.loads((CD / "acts_bench" / "gpqa" / rung / "manifest.json").read_text()).values())
    elif rung in ("soft", "k1"):
        rows = list(json.loads((CD / "acts_stab" / "gpqa" / f"{rung}_slow3" / "manifest.json").read_text()).values())
    else:
        rows = [json.loads(l) for f in sorted((CD / "acts_psweep" / "gpqa" / f"{rung}_slow3").glob("manifest_w*.jsonl"))
                for l in f.read_text().splitlines()]
    assert len(rows) == 64, (rung, sampler, len(rows))
    return [classify(m) for m in rows]


counts = {(r, s): {fm: 0 for fm in FMS + ["ok"]} for r in RUNGS for s in ("std", "gentle")}
for r in RUNGS:
    for s in ("std", "gentle"):
        for fm in load_arm(r, s):
            counts[(r, s)][fm] += 1

fig, ax = plt.subplots()
x = np.arange(len(RUNGS))
w = 0.38
tops = {}
for s, off, alpha in (("std", -w / 2, 1.0), ("gentle", w / 2, 1.0)):
    bottom = np.zeros(len(RUNGS))
    for fm in FMS:
        vals = np.array([100.0 * counts[(r, s)][fm] / 64 for r in RUNGS])
        ax.bar(x + off, vals, w, bottom=bottom, color=FMCOL[fm], alpha=alpha,
               edgecolor="white", linewidth=0.5,
               hatch=None if s == "std" else "//",
               label=FMLAB[fm] if s == "std" else None)
        bottom += vals
    tops[s] = (x + off, bottom.copy())
ax.plot(*tops["std"], color="black", linestyle="-")
ax.plot(*tops["gentle"], color="black", linestyle="--")

ax.set_xticks(x, [RUNG_LAB[r] for r in RUNGS])
ax.set_xlabel(r"state-vocabulary truncation (top-$k$ of $\mathbf{S}_t$)")
ax.set_ylabel("% of rollouts failed")
ax.set_ylim(0, 100)
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
handles = [Patch(facecolor=FMCOL[fm]) for fm in FMS[::-1]]
labels = [FMLAB[fm] for fm in FMS[::-1]]
handles += [(Patch(facecolor="0.65"), Line2D([], [], color="black")),
            (Patch(facecolor="0.65", hatch="//", edgecolor="white"), Line2D([], [], color="black", linestyle="--"))]
labels += ["standard sampler (left bars)", "gentle sampler (right bars)"]
ax.legend(handles, labels, loc="upper left", frameon=False, ncols=2,
          handler_map={tuple: HandlerTuple(ndivide=None)})
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "fig1_gpqa_trunc_failures.png", dpi=200)
print(OUT / "fig1_gpqa_trunc_failures.png")
for r in RUNGS:
    for s in ("std", "gentle"):
        c = counts[(r, s)]
        print(r, s, "ok", c["ok"], "wrong", c["wrong"], "capped", c["capped"], "loop", c["loop"])
