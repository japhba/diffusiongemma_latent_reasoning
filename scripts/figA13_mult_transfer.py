"""Fig A13: multiplicative-family single-injection transfer map (n=1) — mean effect R.

Same eps=0.45 strictly-subleading protocol as fig2a, but the image map is x' = letter at
k*pos(x), k in {2,3,4} (UPPER->UPPER). Data: report payload DATA.tmap.mu. Rows = pre-image
pos(x')/k pooled over k (so the diagonal is the multiplicative transfer), cols = source x.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from payload import load_payload

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figs"
UPP = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
FLOOR = 1e-5
NC = 13  # A..M: the k=2 source pool; k=3/4 use prefixes of it
D = load_payload()
tm = D["tmap"]["mu"]
EPS = tm["eps"]

esum = np.zeros((NC + 1, NC)); ecnt = np.zeros((NC + 1, NC))
for c in tm["cells"]:
    st = tm["states"][c["st"]]
    if c["x"] not in UPP or UPP.index(c["x"]) >= NC:
        continue
    k, xi = st["k"], UPP.index(c["x"])
    base, arm = st["base"], c["arm"]
    L = {q: np.log10(max(arm[q] or FLOOR, FLOOR) / max(base[q] or FLOOR, FLOOR))
         for q in range(52) if q not in (st["ja"], st["jb"])}

    def row(q):
        if q >= 26 and (q - 26 + 1) % k == 0 and 0 <= (q - 26 + 1) // k - 1 < NC:
            return (q - 26 + 1) // k - 1
        return NC

    for q, v in L.items():
        esum[row(q), xi] += v; ecnt[row(q), xi] += 1

em = np.ma.masked_invalid(np.where(ecnt > 0, esum / np.maximum(ecnt, 1), np.nan))
xt = list(UPP[:NC]); yt = list(UPP[:NC]) + ["other"]

fig, ax = plt.subplots(layout="constrained")
vm = np.percentile(np.abs(em.compressed()), 98)
im = ax.imshow(em, cmap="coolwarm", vmin=-vm, vmax=vm, origin="lower")
ax.set_xticks(range(NC), xt)
ax.set_yticks(range(NC + 1), yt)
ax.set_xlabel(r"$x^{t}$")
ax.set_ylabel(r"$\mathrm{pos}(x^{\prime\,t+1})/k$")  # rows are k-aligned (pooled over k): the pre-image
ax.set_title(r"response $\mathbf{R}[x^{\prime\,t+1} \vert\, \mathrm{pert}(x^t)]$, image $x' = k\cdot\mathrm{pos}(x)$")
fig.colorbar(im, ax=ax, shrink=0.8)
fig.savefig(OUT / "figA13_mult_transfer.png", dpi=200)
print(OUT / "figA13_mult_transfer.png")
d = [em[i, i] for i in range(NC) if ecnt[i, i] > 0]
o = [em[i, j] for i in range(NC) for j in range(NC) if i != j and ecnt[i, j] > 0]
print("diag mean R:", round(float(np.mean(d)), 3), "offdiag:", round(float(np.mean(o)), 3))
