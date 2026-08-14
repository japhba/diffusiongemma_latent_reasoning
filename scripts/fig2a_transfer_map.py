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

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figs"
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

em = np.ma.masked_invalid(np.where(ecnt > 0, esum / np.maximum(ecnt, 1), np.nan))
d = [em[i, i] for i in range(NC) if ecnt[i, i] > 0]
o = [em[i, j] for i in range(NC) for j in range(NC) if i != j and ecnt[i, j] > 0]

# G is every UU state's natural operand: never injected (empty column) and its image is the
# always-excluded natural target (empty row) — drop both
gi = UPP.index("G")
assert ecnt[:, gi].sum() == 0 and ecnt[gi, :].sum() == 0, "G column/row unexpectedly has data"
keep = [j for j in range(NC) if j != gi]
em = em[keep + [NC], :][:, keep]
xt = [UPP[j] for j in keep]; yt = [UPP[i] for i in keep] + ["other"]

fig, ax = plt.subplots(layout="constrained")
vm = np.percentile(np.abs(em.compressed()), 98)
im = ax.imshow(em, cmap="coolwarm", vmin=-vm, vmax=vm, origin="lower")
ax.set_xticks(range(len(keep)), xt)
ax.set_yticks(range(len(keep) + 1), yt)
ax.set_xlabel(r"$x^{t}$")
# rows are k-aligned (pooled over k): row = the operand whose image x+k this response is at
ax.set_ylabel(r"operand $x$ with image $x^{\prime\,t+1} = x{+}k$")
ax.set_title(r"response $\mathbf{R}[x^{\prime\,t+1} \vert\, \mathrm{pert}(x^t)]$")
fig.colorbar(im, ax=ax, shrink=0.8)
fig.savefig(OUT / "fig2a_transfer_map.png", dpi=200)
print(OUT / "fig2a_transfer_map.png")
print("diag mean R:", round(float(np.mean(d)), 3), "offdiag:", round(float(np.mean(o)), 3))
