"""Fig 2a-1: letter-arithmetic single-injection transfer map (n=1) — mean effect R, k=3 ONLY.

Headline variant from the symbol_arithmetic report: UPPER->UPPER, eps=0.45 strictly-subleading
injections, 8 paired draws, t=2, restricted to the k=3 task (UU3; single-k since 2026-08-14 per
user — rows then resolve to unique target letters). Data: report payload DATA.tmap.let.
Rows = output x' = x+3 labeled by the resolved target letter, cols = perturbed source letter x.
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
K = 3
for c in tm["cells"]:
    st = tm["states"][c["st"]]
    if st["v"] != "UU" or st["k"] != K or c["x"] not in UPP or UPP.index(c["x"]) >= NC:
        continue
    k, xi = st["k"], UPP.index(c["x"])
    base, arm = st["base"], c["arm"]
    # ja/jb (the natural image J and committed answer) are NOT excluded here (since 2026-08-14,
    # user): row J then displays the incumbent's displacement under every injection
    L = {q: np.log10(max(arm[q] or FLOOR, FLOOR) / max(base[q] or FLOOR, FLOOR))
         for q in range(52)}
    row = lambda q: (q - 26 - k) if (q >= 26 and 0 <= q - 26 - k < NC) else NC
    for q, v in L.items():
        esum[row(q), xi] += v; ecnt[row(q), xi] += 1
    conf[row(max(L, key=L.get)), xi] += 1

em = np.ma.masked_invalid(np.where(ecnt > 0, esum / np.maximum(ecnt, 1), np.nan))
gi = UPP.index("G")
# specificity stats over transfer rows only (row J = incumbent displacement, not transfer)
d = [em[i, i] for i in range(NC) if i != gi and ecnt[i, i] > 0]
o = [em[i, j] for i in range(NC) for j in range(NC) if i != gi and i != j and ecnt[i, j] > 0]

# G is every UU state's natural operand and never injected — drop the empty source column
# (its ROW, the natural image J, is kept: that's the displacement readout)
assert ecnt[:, gi].sum() == 0, "G column unexpectedly has data"
keep = [j for j in range(NC) if j != gi]
em = em[:, keep]
xt = [UPP[j] for j in keep]; yt = [UPP[i + K] for i in range(NC)] + ["other"]

fig, ax = plt.subplots(layout="constrained")
vm = np.percentile(np.abs(em.compressed()), 98)
im = ax.imshow(em, cmap="coolwarm", vmin=-vm, vmax=vm, origin="lower")
ax.set_xticks(range(len(keep)), xt)
ax.set_yticks(range(NC + 1), yt)
ax.set_xlabel(r"$x^{t}$")
ax.set_ylabel(r"$x^{\prime\,t+1}$")
ax.set_title(r"response $\mathbf{R}[x^{\prime\,t+1} \vert\, \mathrm{pert}(x^t)]$,  $k=3$")
fig.colorbar(im, ax=ax, shrink=0.8)
fig.savefig(OUT / "fig2a_transfer_map.png", dpi=200)
print(OUT / "fig2a_transfer_map.png")
print("diag mean R:", round(float(np.mean(d)), 3), "offdiag:", round(float(np.mean(o)), 3))
