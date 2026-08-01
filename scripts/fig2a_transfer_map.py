"""Fig 2a-1: letter-arithmetic single-injection transfer maps (n=1) — argmax histogram + mean effect R.

Headline variant from the symbol_arithmetic report: UPPER->UPPER, eps=0.45 strictly-subleading
injections, k in {3,5,7,11}, 8 paired draws, t=2. Data: report payload DATA.tmap.let (UU states).
Rows = aligned output x'-k (A..W + other), cols = perturbed source letter x.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from payload import load_payload

OUT = Path("/workspace-vast/jbauer/dg_blog/figs")
UPP = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
FLOOR = 1e-5
NC = 23  # A..W
D = load_payload()
tm = D["tmap"]["let"]
EPS = tm["eps"]

conf = np.zeros((NC + 1, NC))
esum = np.zeros((NC + 1, NC)); ecnt = np.zeros((NC + 1, NC))
for c in tm["cells"]:
    st = tm["states"][c["st"]]
    if st["v"] != "UU" or c["x"] not in UPP or UPP.index(c["x"]) >= NC:
        continue
    k, xi = st["k"], UPP.index(c["x"])
    base, arm = st["base"], c["arm"]
    L = {q: np.log10(max(arm[q] or FLOOR, FLOOR) / max(base[q] or FLOOR, FLOOR))
         for q in range(52) if q not in (st["ja"], st["jb"])}
    row = lambda q: (q - 26 - k) if (q >= 26 and 0 <= q - 26 - k < NC) else NC
    for q, v in L.items():
        esum[row(q), xi] += v; ecnt[row(q), xi] += 1
    conf[row(max(L, key=L.get)), xi] += 1

coln = conf.sum(axis=0).astype(int)
cn = np.ma.masked_invalid(np.where(coln > 0, conf / np.maximum(coln, 1), np.nan))
em = np.ma.masked_invalid(np.where(ecnt > 0, esum / np.maximum(ecnt, 1), np.nan))
xt = list(UPP[:NC]); yt = list(UPP[:NC]) + ["other"]

fig, axes = plt.subplots(1, 2, sharey=True, layout="constrained",
                         figsize=(plt.rcParams["figure.figsize"][0] * 1.7,
                                  plt.rcParams["figure.figsize"][1] * 1.15))
for ax, M, cmap, vmin, vmax, title in (
        (axes[0], cn, "viridis", 0, 1,
         r"$\langle \mathbf{1}[x'=\mathrm{argmax}\; R_c(\,\cdot \mid x)]\rangle_c$"),
        (axes[1], em, "coolwarm", None, None,
         r"$\langle R_c(x' \mid x)\rangle_c$")):
    if vmin is None:
        vm = np.percentile(np.abs(M.compressed()), 98)
        vmin, vmax = -vm, vm
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
    ax.set_xticks(range(NC), xt)
    ax.set_yticks(range(NC + 1), yt)
    fig.colorbar(im, ax=ax, shrink=0.75)
axes[0].set_ylabel(r"aligned target $x' - k$   (uppercase band)")
fig.supxlabel(rf"perturbed source letter $x$ (strictly subleading, $\varepsilon={EPS:g}$)")
fig.savefig(OUT / "fig2a_transfer_map.png", dpi=200)
print(OUT / "fig2a_transfer_map.png")
d = [em[i, i] for i in range(NC) if ecnt[i, i] > 0]
o = [em[i, j] for i in range(NC) for j in range(NC) if i != j and ecnt[i, j] > 0]
print("diag mean R:", round(float(np.mean(d)), 3), "offdiag:", round(float(np.mean(o)), 3))
