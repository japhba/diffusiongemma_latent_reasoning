"""A12: anticausal commitment on the end-anchored chain (reverse_chain, transparency-paper
replication battery). The task fixes the LAST element (x(k+1) = f(x(k)), x_n given) but asks
for x1..xn in forward order — dependency runs right-to-left, emission left-to-right.

Left:  digit commit-order diagrams, depths 4-5, T>=16 — correct runs (green, bold) crystallize
       from the anchored end backward (hug the anti-diagonal); wrong runs (grey) fill forward.
Right: chain rho (Spearman of lock step vs digit index) per depth, split by correctness.

Data: src_data/planning/reverse_chain_order.json (extract_reverse_chain_order.py over the
thinkfast denoising films; lock step = argmax stability onset).
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figs"
rows = [r for r in json.load(open(ROOT / "src_data" / "planning" / "reverse_chain_order.json"))
        if r["T"] >= 16 and len(set(r["lock"])) > 1]

def rankdata(v):
    v = np.asarray(v, float); order = np.argsort(v); ranks = np.empty(len(v)); sv = v[order]; i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks

for r in rows:
    r["rho"] = float(np.corrcoef(rankdata(range(len(r["lock"]))), rankdata(r["lock"]))[0, 1])

OKC, BADC = "#2f9e44", "0.6"
fig, (axL, axR) = plt.subplots(1, 2, layout="constrained",
                               figsize=(plt.rcParams["figure.figsize"][0] * 1.7,
                                        plt.rcParams["figure.figsize"][1] * 0.9))

# ---- left: lock time per chain element, depths 4-5 ----
GRID = np.linspace(0, 1, 41)
for ok, col, al, lw, z in ((False, BADC, 0.3, 0.8, 2), (True, OKC, 0.85, 1.6, 3)):
    curves = []
    for r in [x for x in rows if x["depth"] in (4, 5) and x["ok"] == ok]:
        lk = np.array(r["lock"], float)
        lk = lk / lk.max()
        xr = np.arange(len(lk)) / (len(lk) - 1)
        axL.plot(xr, lk, color=col, alpha=al, linewidth=lw, zorder=z)
        curves.append(np.interp(GRID, xr, lk))
    axL.plot(GRID, np.mean(curves, axis=0), color=col, linewidth=2.6, zorder=z + 2)
axL.set_xlabel("chain element ($0 = x_1$, $1 = x_n$ = the given anchor)")
axL.set_ylabel("lock-in time (fraction of the run's last lock)")
axL.spines[["top", "right"]].set_visible(False)
handles = [plt.Line2D([], [], color=OKC, linewidth=2, label="correct — locks back-to-front"),
           plt.Line2D([], [], color=BADC, linewidth=1.2, label="wrong — fills forward")]
axL.legend(handles=handles, frameon=False, fontsize="small", loc="upper center")

# ---- right: chain rho per depth, split by correctness ----
DEPTHS = sorted({r["depth"] for r in rows})
rng = np.random.default_rng(0)
for i, d in enumerate(DEPTHS):
    for ok, col, dx in ((False, BADC, -0.16), (True, OKC, 0.16)):
        vs = [r["rho"] for r in rows if r["depth"] == d and r["ok"] == ok]
        if not vs:
            continue
        x = i + dx + rng.uniform(-0.06, 0.06, len(vs))
        axR.scatter(x, vs, s=16, color=col, alpha=0.75, zorder=3)
        axR.plot([i + dx - 0.12, i + dx + 0.12], [np.median(vs)] * 2, color=col, linewidth=2, zorder=4)
axR.axhline(0, color="0.85", linewidth=0.8, zorder=0)
axR.set_xticks(range(len(DEPTHS)), [f"d{d}" for d in DEPTHS])
axR.set_xlabel("chain depth")
axR.set_ylabel(r"$\rho_{\mathrm{chain}}$  ($+1$ forward, $-1$ backward)")
axR.set_ylim(-1.08, 1.08)
axR.spines[["top", "right"]].set_visible(False)
fig.savefig(OUT / "figA12_reverse_chain.png", dpi=200)
print(OUT / "figA12_reverse_chain.png")
ok4 = [r["rho"] for r in rows if r["depth"] in (4, 5) and r["ok"]]
print("d4-5 correct rhos:", [round(v, 2) for v in sorted(ok4)])
