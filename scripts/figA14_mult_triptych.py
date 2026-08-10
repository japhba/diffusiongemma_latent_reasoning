"""Fig A14: parallelism triptych vs n for the multiplicative family (mu, UPPER->UPPER).

Identical protocol to fig2b but with the multiplicative image map x' = letter at k*pos(x),
k in {2,3,4}. Panels: (1) <R>_T and <R>_N, (2) E = <R>_T − <R>_N, (3) NE = E/(n eps0).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from payload import load_payload

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figs"
FLOOR = 1e-5
D = load_payload()
UPP = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

dom = "mu"
t = D["doms"][dom]
eps0 = t["eps0"]

cells = []
for c in t["cells"]:
    st = t["states"][c["state"]]
    R = {w: np.log10(max(c["arm"][w] or FLOOR, FLOOR) / max(st["base"][w] or FLOOR, FLOOR))
         for w in t["field"]}
    tjs = set()
    for w in c["subset"]:
        i = (UPP.index(w) + 1) * c["k"]
        if 1 <= i <= 26:
            tjs.add(UPP[i - 1])
    ex = (st["ja"], st["jb"])
    rt = [R[w] for w in tjs if w not in ex and w in R]
    rn = [R[w] for w in t["scope"] if w not in tjs and w not in ex]
    if not (rt and rn):
        continue
    E = float(np.mean(rt) - np.mean(rn))
    assert abs(E - c["E"]) < 0.02, (c["state"], E, c["E"])
    cells.append(dict(n=c["n"], rt=float(np.mean(rt)), rn=float(np.mean(rn)), E=E, NE=E / (eps0 * c["n"])))

ns = sorted({c["n"] for c in cells})
mci = lambda vals: (np.mean(vals), 1.96 * np.std(vals, ddof=1) / np.sqrt(len(vals)))

fig, axes = plt.subplots(1, 3, layout="constrained",
                         figsize=(plt.rcParams["figure.figsize"][0] * 2.0,
                                  plt.rcParams["figure.figsize"][1] * 0.85))
panels = [
    (axes[0], [("rt", "#e8590c", r"$\langle R\rangle_{T}$ (targets)"),
               ("rn", "#7aa2ff", r"$\langle R\rangle_{N}$ (non-targets)")],
     r"components", r"$\langle R\rangle\ (\log_{10})$"),
    (axes[1], [("E", "0.2", None)], r"$E=\langle R\rangle_{T}-\langle R\rangle_{N}$", r"$E$"),
    (axes[2], [("NE", "0.2", None)], r"$\mathrm{NE} = E/(n\,\varepsilon_0)$", r"$\mathrm{NE}$"),
]
for ax, series, title, ylab in panels:
    for key, col, lab in series:
        m, ci = zip(*[mci([c[key] for c in cells if c["n"] == n]) for n in ns])
        ax.errorbar(ns, m, yerr=ci, color=col, marker="o", markersize=3.5, capsize=2.5, label=lab)
    ax.axhline(0, color="0.85", linewidth=0.6, zorder=0)
    ax.set_xticks(ns)
    ax.set_xlabel(r"$n$ simultaneous injections")
    ax.set_ylabel(ylab)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].legend(frameon=False)
fig.savefig(OUT / "figA14_mult_triptych.png", dpi=200)
print(OUT / "figA14_mult_triptych.png")
print("NE mean by n:", {n: round(float(np.mean([c['NE'] for c in cells if c['n'] == n])), 2) for n in ns})
