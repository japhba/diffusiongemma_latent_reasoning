"""A11 (merged): commit-order diagrams, benchmark-style tasks (top row) vs idiosyncratic
tasks (bottom row). x = commit rank (order in which content positions finalize, normalized),
y = normalized canvas position; the diagonal is a strictly left-to-right (AR-like) order,
descending segments are anticausal stages. Panel annotation: median chain rho (Spearman of
finalize step vs position).

Top row (fresh pod captures, capture_bench_order.py; default sampler, argmax lock-in proxy):
GPQA, MATH, HumanEval, WildChat user prompts — 12 rollouts each. Bottom row: acrostic_word / palindrome_words /
ends_with (constrained battery, hot sampler, accept-mask commitment) and reverse_chain
(films; correct runs green — they commit back-to-front, cf. the body text).

Data: src_data/planning/{films_order,commit_order}.json.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figs"
films = json.load(open(ROOT / "src_data" / "planning" / "films_order.json"))
bench = json.load(open(ROOT / "src_data" / "planning" / "bench_order.json"))
constr = [r for r in json.load(open(ROOT / "src_data" / "planning" / "commit_order.json"))
          if r["regime"] == "hot"]
OBS, OKC, BADC = "#1971c2", "#2f9e44", "0.62"
GRID = np.linspace(0, 1, 101)

def rankdata(v):
    v = np.asarray(v, float); order = np.argsort(v); ranks = np.empty(len(v)); sv = v[order]; i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks

def curve(ax, content, lock, col, al, lw, z=2):
    pos = np.array(content, float)
    posn = (pos - pos.min()) / max(pos.max() - pos.min(), 1)
    order = np.argsort(np.array(lock), kind="stable")
    xr = np.arange(len(order)) / max(len(order) - 1, 1)
    ax.plot(xr, posn[order], color=col, alpha=al, linewidth=lw, zorder=z)
    return (np.interp(GRID, xr, posn[order]),
            float(np.corrcoef(rankdata(pos), rankdata(np.array(lock, float)))[0, 1]))

TOP = [("gpqa", "GPQA"), ("math", "MATH"), ("humaneval", "HumanEval")]
BOT = ["palindrome_words", "ends_with"]

fig, axes = plt.subplots(2, 3, sharex=True, sharey=True, layout="constrained",
                         figsize=(plt.rcParams["figure.figsize"][0] * 1.7,
                                  plt.rcParams["figure.figsize"][1] * 1.1))
def finish(ax, name, n, rho_txt):
    ax.plot([0, 1], [0, 1], color="0.6", linestyle="--", linewidth=1.2)
    ax.text(0.03, 0.95, rho_txt, transform=ax.transAxes, va="top", fontsize="small")
    ax.text(0.97, 0.05, f"{name}\n(n={n})", transform=ax.transAxes, ha="right", va="bottom",
            fontsize="small")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 1.02)
    ax.spines[["top", "right"]].set_visible(False)

for ax, (key, label) in zip(axes[0], TOP):
    rs = [r for r in bench if r["bench"] == key and len(set(r["lock"])) > 1 and len(r["content"]) >= 6]
    cs, rhos = zip(*[curve(ax, r["content"], r["lock"], OBS, 0.15, 0.7) for r in rs])
    ax.plot(GRID, np.mean(cs, axis=0), color=OBS, linewidth=2.0, zorder=4)
    finish(ax, label, len(rs), rf"$\rho_{{\mathrm{{chain}}}} = {np.median(rhos):+.2f}$")

for ax, tt in zip(axes[1], BOT):
    rs = [r for r in constr if r["ttype"] == tt]
    cs, rhos = zip(*[curve(ax, r["content"], r["first_commit"], OBS, 0.15, 0.7) for r in rs])
    ax.plot(GRID, np.mean(cs, axis=0), color=OBS, linewidth=2.0, zorder=4)
    finish(ax, tt, len(rs), rf"$\rho_{{\mathrm{{chain}}}} = {np.median(rhos):+.2f}$")

# bottom-right: reverse_chain, correct (backward) vs wrong (forward)
ax = axes[1, 2]
rc = [r for r in films if r["task"] == "reverse_chain" and r["depth"] in (4, 5)
      and len(set(r["lock"])) > 1]
stats = {}
for ok, col, al, lw, z in ((False, BADC, 0.25, 0.7, 2), (True, OKC, 0.85, 1.5, 3)):
    rs = [r for r in rc if r["ok"] == ok]
    cs, rhos = zip(*[curve(ax, r["content"], r["lock"], col, al, lw, z) for r in rs])
    ax.plot(GRID, np.mean(cs, axis=0), color=col, linewidth=2.2, zorder=z + 2)
    stats[ok] = (np.median(rhos), len(rs))
finish(ax, "reverse_chain (d4–5)", len(rc),
       rf"$\rho$: correct ${stats[True][0]:+.2f}$, wrong ${stats[False][0]:+.2f}$")
handles = [plt.Line2D([], [], color=OKC, linewidth=2, label="correct"),
           plt.Line2D([], [], color=BADC, linewidth=1.2, label="wrong")]
ax.legend(handles=handles, frameon=False, fontsize="small", loc="center left")

for a in axes[-1]:
    a.set_xlabel("commit rank (fraction)")
for a in axes[:, 0]:
    a.set_ylabel("canvas position\n(0 = left, 1 = right)")
fig.savefig(OUT / "figA11_commit_causality.png", dpi=200)
print(OUT / "figA11_commit_causality.png")
