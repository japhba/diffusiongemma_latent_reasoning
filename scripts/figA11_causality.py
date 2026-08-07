"""A11: commitment center-of-mass over diffusion progress — classic benchmarks (top row) vs
idiosyncratic tasks (bottom row). x = diffusion progress t/T; y = center of mass of the
canvas positions committed by step t, normalized to the content span (0 = left, 1 = right).
Dashed = matched left-to-right filler (same committed count per step, leftmost positions
first): a purely causal (AR-like) fill sits on it; early end-anchoring sits above it.
Panel annotation: median chain rho (Spearman of final-commit step vs position).

Top row (fresh pod captures, capture_bench_order.py; default sampler, argmax lock-in proxy):
GPQA, MATH, HumanEval — 12 rollouts each. Bottom row: palindrome_words / ends_with
(constrained battery, hot sampler, accept-mask commitment) and reverse_chain d4-5 (films;
correct runs green — the committed CoM starts at the anchored END and walks backward).

Data: src_data/planning/{bench_order,commit_order,films_order}.json.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figs"
bench = json.load(open(ROOT / "src_data" / "planning" / "bench_order.json"))
constr = [r for r in json.load(open(ROOT / "src_data" / "planning" / "commit_order.json"))
          if r["regime"] == "hot"]
films = json.load(open(ROOT / "src_data" / "planning" / "films_order.json"))
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

def com_curves(content, lock, T):
    """Cumulative committed-CoM and matched-L2R-filler CoM, interpolated onto the progress grid."""
    pos = np.array(content, float)
    posn = (pos - pos.min()) / max(pos.max() - pos.min(), 1)
    srt = np.sort(posn)
    lk = np.array(lock)
    com = np.full(T, np.nan)
    ref = np.full(T, np.nan)
    for t in range(T):
        sel = posn[lk <= t]
        if len(sel):
            com[t] = sel.mean()
            ref[t] = srt[:len(sel)].mean()
    x = (np.arange(T) + 1) / T
    rho = float(np.corrcoef(rankdata(pos), rankdata(lk.astype(float)))[0, 1])
    return np.interp(GRID, x, com), np.interp(GRID, x, ref), rho

def panel(ax, runs, col=OBS, z=2, alpha=0.15, lw=0.7, draw_ref=True):
    cs, fs, rhos = [], [], []
    for content, lock, T in runs:
        c, f, rho = com_curves(content, lock, T)
        ax.plot(GRID, c, color=col, alpha=alpha, linewidth=lw, zorder=z)
        cs.append(c); fs.append(f); rhos.append(rho)
    ax.plot(GRID, np.nanmean(cs, axis=0), color=col, linewidth=2.0, zorder=z + 2)
    if draw_ref:
        ax.plot(GRID, np.nanmean(fs, axis=0), color="0.35", linestyle="--", linewidth=1.4, zorder=z + 1)
    return float(np.median(rhos))

fig, axes = plt.subplots(2, 3, sharex=True, sharey=True, layout="constrained",
                         figsize=(plt.rcParams["figure.figsize"][0] * 1.7,
                                  plt.rcParams["figure.figsize"][1] * 1.1))
def finish(ax, name, n, rho_txt):
    ax.axhline(0.5, color="0.88", linewidth=0.6, zorder=0)
    ax.text(0.03, 0.95, rho_txt, transform=ax.transAxes, va="top", fontsize="small")
    ax.text(0.97, 0.05, f"{name}\n(n={n})", transform=ax.transAxes, ha="right", va="bottom",
            fontsize="small")
    ax.set_xlim(0, 0.6)  # commitment is done by ~t/T=0.5 everywhere; crop the frozen tail
    ax.set_ylim(-0.02, 1.02)
    ax.spines[["top", "right"]].set_visible(False)

for ax, (key, label) in zip(axes[0], [("gpqa", "GPQA"), ("math", "MATH"), ("humaneval", "HumanEval")]):
    rs = [(r["content"], r["lock"], r["T"]) for r in bench
          if r["bench"] == key and len(set(r["lock"])) > 1 and len(r["content"]) >= 6]
    rho = panel(ax, rs)
    finish(ax, label, len(rs), rf"$\rho_{{\mathrm{{chain}}}} = {rho:+.2f}$")

for ax, tt in zip(axes[1], ["palindrome_words", "ends_with"]):
    rs = [(r["content"], r["first_commit"], r["T"]) for r in constr if r["ttype"] == tt]
    rho = panel(ax, rs)
    finish(ax, tt, len(rs), rf"$\rho_{{\mathrm{{chain}}}} = {rho:+.2f}$")

# bottom-right: reverse_chain, correct (backward) vs wrong (forward)
ax = axes[1, 2]
rc = [r for r in films if r["task"] == "reverse_chain" and r["depth"] in (4, 5)
      and len(set(r["lock"])) > 1]
rho_w = panel(ax, [(r["content"], r["lock"], r["T"]) for r in rc if not r["ok"]],
              col=BADC, z=2, alpha=0.25, draw_ref=False)
rho_c = panel(ax, [(r["content"], r["lock"], r["T"]) for r in rc if r["ok"]],
              col=OKC, z=3, alpha=0.6, lw=1.2, draw_ref=True)
finish(ax, "reverse_chain (d4–5)", len(rc), rf"$\rho$: correct ${rho_c:+.2f}$, wrong ${rho_w:+.2f}$")
handles = [plt.Line2D([], [], color=OKC, linewidth=2, label="correct"),
           plt.Line2D([], [], color=BADC, linewidth=1.2, label="wrong")]
ax.legend(handles=handles, frameon=False, fontsize="small", loc="upper right")

for a in axes[-1]:
    a.set_xlabel("diffusion progress $t/T$")
for a in axes[:, 0]:
    a.set_ylabel("committed CoM\n(0 = left, 1 = right)")
fig.savefig(OUT / "figA11_commit_causality.png", dpi=200)
print(OUT / "figA11_commit_causality.png")
