"""Fig 2b: parallel-computation budget ladder (letters +3) — fraction of injection sets
with min_i E(x_i) > 0 vs number of routable operands n, K=3, dashed chance 0.5^n.

Faithful UU3-only port of diffusiongemma/planning/xtask_par3_read.py + build_par_ladderfrac.py
(the upstream pair that generated the original figure; capture: experiments/symbol_arithmetic/
xtask_par3.py). Ladder arms at matched TOTAL injection mass 3*eps0 = 0.66: j2 = 1 routable
hypothesis + 2 junks, r2j = 2 + 1 junk, r3 = 3 routable. E(x_i) = R(img(x_i)) - <R(img(y))>_P
with R(w) = log10 pbar_pert(w)/pbar_base(w); placebo pool P = images of non-injected pool
operands >= dstar=4 from every injected token, excluding ja (base image) and jb (incumbent).
No titles: E is defined in the post's caption.

Data: src_data/planning/{xtask_par3,xtask_samecase_nsweep,par3_incumbents}.json.
"""
import json
import string
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SDP = ROOT / "src_data" / "planning"
OUT = ROOT / "figs"
FLOOR = 1e-5
LOW, UPP = list(string.ascii_lowercase), list(string.ascii_uppercase)
F52 = LOW + UPP
TAG, K, POOL, DSTAR, SEEDS = "UU3", 3, UPP[:23], 4, range(10)

par = json.load(open(SDP / "xtask_par3.json"))
dns = json.load(open(SDP / "xtask_samecase_nsweep.json"))
jbs = json.load(open(SDP / "par3_incumbents.json"))

img = lambda w: (UPP[UPP.index(w) + K] if UPP.index(w) + K < 26 else None)
pi = {w: i for i, w in enumerate(UPP)}

recs, drop = [], 0
for s in SEEDS:
    bsrc = par.get(f"{TAG}|s{s}|par3|base") or dns.get(f"{TAG}|s{s}|base")
    pairs = par.get(f"{TAG}|s{s}|par3|pairs")
    if not bsrc or not pairs:
        continue
    nat = bsrc["nat_op"]
    bmap = {w: float(v) for w, v in zip(F52, np.array(bsrc["rows"]).mean(axis=0))}
    ja, jb = img(nat), jbs[f"{TAG}|s{s}"]
    for r, rp in enumerate(pairs):
        arms = {}
        for arm in ("n1", "j2", "r2j", "r3"):
            c = par.get(f"{TAG}|s{s}|par3|r{r}|{arm}")
            if not c:
                continue
            if any(v == 0 for v in c["ranks"].values()):  # subleadingness rank guard
                drop += 1
                continue
            amap = {w: float(v) for w, v in zip(F52, np.array(c["rows"]).mean(axis=0))}
            arms[arm] = {w: np.log10(max(amap[w], FLOOR) / max(bmap[w], FLOOR)) for w in UPP}
        if len(arms) < 4:
            continue
        xs, junks = rp["x"], rp["junks"]
        allsrc = [w for w in xs + junks if w in pi]
        P = [img(y) for y in POOL if y not in xs and y != nat and img(y) not in (ja, jb, None)]
        P = [w for w in set(P) - set(xs) - set(junks)
             if all(abs(pi[w] - pi[sx]) >= DSTAR for sx in allsrc)]
        tgts = [img(x) for x in xs]
        if len(P) < 2 or any(t not in arms["n1"] for t in tgts):
            continue
        spec = lambda arm, tgt: arms[arm][tgt] - np.mean([arms[arm][w] for w in P])
        recs.append(dict(state=f"{TAG}|{s}",
                         s1_j2=spec("j2", tgts[0]),
                         s1_r2j=spec("r2j", tgts[0]), s2_r2j=spec("r2j", tgts[1]),
                         s1_r3=spec("r3", tgts[0]), s2_r3=spec("r3", tgts[1]), s3_r3=spec("r3", tgts[2])))

print(f"UU3 recs: {len(recs)}, rank-guard drops: {drop}")
assert len(recs) == 34

OK = {1: [(r["s1_j2"] > 0, r["state"]) for r in recs],
      2: [(min(r["s1_r2j"], r["s2_r2j"]) > 0, r["state"]) for r in recs],
      3: [(min(r["s1_r3"], r["s2_r3"], r["s3_r3"]) > 0, r["state"]) for r in recs]}


def agg(pairs):
    by = {}
    for ok, stt in pairs:
        by.setdefault(stt, []).append(ok)
    sm = np.array([np.mean(v) for v in by.values()])
    se = sm.std(ddof=1) / np.sqrt(len(sm)) if len(sm) > 1 else 0.0
    return np.mean([ok for ok, _ in pairs]), 1.96 * se


pts = {r: agg(OK[r]) for r in (1, 2, 3)}
print({r: round(p[0], 3) for r, p in pts.items()})

fig, ax = plt.subplots(figsize=(5.2, 4.4))
rs = [1, 2, 3]
ax.plot(rs, [0.5 ** r for r in rs], ls="--", color="k", lw=1.2, label=r"chance $0.5^{\,n}$")
ax.errorbar(rs, [pts[r][0] for r in rs], yerr=[pts[r][1] for r in rs], marker="o",
            capsize=3, color="tab:blue", label="_nolegend_")
ax.set_xticks(rs)
ax.set_ylim(0, 1)
ax.legend(fontsize=7)
ax.set_xlabel(r"operands $n$")
ax.set_ylabel(r"frac of injection sets with $\min_i E(x_i) > 0$")
fig.tight_layout()
fig.savefig(OUT / "letters_parallel_frac.png", dpi=150)
print(OUT / "letters_parallel_frac.png")
