"""Fig A15: worst-case (strict) parallelism triptych vs n for the multiplicative family (mu).

Same cells as figA14, min-over-targets vs max-over-non-targets aggregation as in figA12.
Panels: (1) R_T^min and R_N^max, (2) E^min = R_T^min − R_N^max, (3) NE^min = E^min/(n eps0).
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
    assert abs((np.mean(rt) - np.mean(rn)) - c["E"]) < 0.02, (c["state"], c["E"])
    Es = float(min(rt) - max(rn))
    cells.append(dict(n=c["n"], rtn=float(min(rt)), rnx=float(max(rn)), Es=Es, Vs=Es / (eps0 * c["n"])))

ns = sorted({c["n"] for c in cells})
mci = lambda vals: (np.mean(vals), 1.96 * np.std(vals, ddof=1) / np.sqrt(len(vals)))

fig, axes = plt.subplots(1, 3, layout="constrained",
                         figsize=(plt.rcParams["figure.figsize"][0] * 2.0,
                                  plt.rcParams["figure.figsize"][1] * 0.85))
panels = [
    (axes[0], [("rtn", "#e8590c", r"$R_{T}^{\min}$ (weakest target)"),
               ("rnx", "#7aa2ff", r"$R_{N}^{\max}$ (strongest non-target)")],
     r"components", r"$R\ (\log_{10})$"),
    (axes[1], [("Es", "0.2", None)], r"$E^{\min}=R_{T}^{\min}-R_{N}^{\max}$", r"$E^{\min}$"),
    (axes[2], [("Vs", "0.2", None)], r"$\mathrm{NE}^{\min} = E^{\min}/(n\,\varepsilon_0)$", r"$\mathrm{NE}^{\min}$"),
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
axes[0].legend(frameon=False, fontsize="small")
fig.savefig(OUT / "figA15_mult_strict.png", dpi=200)
print(OUT / "figA15_mult_strict.png")
print("E^min mean by n:", {n: round(float(np.mean([c['Es'] for c in cells if c['n'] == n])), 2) for n in ns})
