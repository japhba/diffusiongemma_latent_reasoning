"""A11: commitment center-of-mass over diffusion progress, one panel per task class:
logically left-to-right (GPQA), direction-indifferent (poem writing), logically right-to-left
(reverse_chain, end-anchored). y = center of mass of the canvas positions committed by step t,
normalized to the content span; dashed = matched left-to-right filler (same committed count,
leftmost first) — a purely causal fill sits on it. Panel annotation: median chain rho
(Spearman of final-commit step vs position).

GPQA + poem: fresh pod captures (capture_bench_order.py / poem topics), default sampler,
argmax lock-in proxy, 12 rollouts each. reverse_chain d4-5: thinkfast films, correct runs
green (committed CoM starts at the anchored END and walks backward).

Data: src_data/planning/{bench_order,poem_order,films_order}.json.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figs"
bench = json.load(open(ROOT / "src_data" / "planning" / "bench_order.json"))
poems = json.load(open(ROOT / "src_data" / "planning" / "poem_order.json"))
films = json.load(open(ROOT / "src_data" / "planning" / "films_order.json"))
judged = json.load(open(ROOT / "src_data" / "planning" / "judged_logical_order.json"))
jmed = {b: np.median([v["rho_logic"] for v in judged.values() if v["bench"] == b])
        for b in ("gpqa", "poem")}
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

fig, axes = plt.subplots(1, 3, sharex=True, sharey=True, layout="constrained",
                         figsize=(plt.rcParams["figure.figsize"][0] * 1.8,
                                  plt.rcParams["figure.figsize"][1] * 0.85))
def finish(ax, klass, name, n, rho_txt, judge_txt=None):
    ax.axhline(0.5, color="0.88", linewidth=0.6, zorder=0)
    ax.text(0.03, 0.97, rho_txt, transform=ax.transAxes, va="top", fontsize="small")
    if judge_txt:
        ax.text(0.03, 0.885, judge_txt, transform=ax.transAxes, va="top",
                fontsize="small", color="0.35", style="italic")
    ax.text(0.97, 0.03, f"{klass}\n{name} (n={n})", transform=ax.transAxes, ha="right",
            va="bottom", fontsize="small")
    ax.set_xlim(0, 0.6)  # commitment is done by ~t/T=0.5 everywhere; crop the frozen tail
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("diffusion progress $t/T$")
    ax.spines[["top", "right"]].set_visible(False)

rs = [(r["content"], r["lock"], r["T"]) for r in bench
      if r["bench"] == "gpqa" and len(set(r["lock"])) > 1 and len(r["content"]) >= 6]
rho = panel(axes[0], rs)
finish(axes[0], "logically left-to-right", "GPQA", len(rs),
       rf"$\rho_{{\mathrm{{chain}}}} = {rho:+.2f}$",
       rf"judge: $\rho_{{\mathrm{{logic}}}} = {jmed['gpqa']:+.2f}$")

rs = [(r["content"], r["lock"], r["T"]) for r in poems
      if len(set(r["lock"])) > 1 and len(r["content"]) >= 6]
rho = panel(axes[1], rs)
finish(axes[1], "direction-indifferent", "poem writing", len(rs),
       rf"$\rho_{{\mathrm{{chain}}}} = {rho:+.2f}$",
       rf"judge: $\rho_{{\mathrm{{logic}}}} = {jmed['poem']:+.2f}$")

ax = axes[2]
rc = [r for r in films if r["task"] == "reverse_chain" and r["depth"] in (4, 5)
      and len(set(r["lock"])) > 1]
rho_w = panel(ax, [(r["content"], r["lock"], r["T"]) for r in rc if not r["ok"]],
              col=BADC, z=2, alpha=0.25, draw_ref=False)
rho_c = panel(ax, [(r["content"], r["lock"], r["T"]) for r in rc if r["ok"]],
              col=OKC, z=3, alpha=0.6, lw=1.2, draw_ref=True)
finish(ax, "logically right-to-left", "reverse_chain (d4–5)", len(rc),
       rf"$\rho$: correct ${rho_c:+.2f}$, wrong ${rho_w:+.2f}$",
       r"judge: $\rho_{\mathrm{logic}} = -1$ (by construction)")
handles = [plt.Line2D([], [], color=OKC, linewidth=2, label="correct"),
           plt.Line2D([], [], color=BADC, linewidth=1.2, label="wrong")]
ax.legend(handles=handles, frameon=False, fontsize="small", loc="upper right")

axes[0].set_ylabel("committed CoM\n(0 = left, 1 = right)")
fig.savefig(OUT / "figA11_commit_causality.png", dpi=200)
print(OUT / "figA11_commit_causality.png")
