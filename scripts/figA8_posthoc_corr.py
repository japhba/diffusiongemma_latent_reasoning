"""A8: post-hoc vs load-bearing CoT — the three correlations + case-study card data.

Panels: (1) blind difficulty vs commitment time, (2) blind difficulty vs susceptibility S,
(3) commitment time vs S. Per problem (n=40; original 20 + the 2026-08-08 extension battery): commitment = median-over-seeds of median answer
lock-in step (clean.json); S = mean over rho>0 of P(answer differs from the rho=0 clean-clamp
baseline, matched by corruption seed) (suscept.json); difficulty = mean of 3 blind subagent
ratings from the problem text alone (difficulty.json). Spearman hand-rolled (rank + corrcoef);
asserts pin the n=40 values +0.37 / +0.28 / +0.60 (n=20 report values were +0.37 / +0.42 / +0.66).

Also emits data/posthoc_case.json: the squares_400_800 dissociation (random rho=1.0 corruption
denoised away -> answer stays 8; coherent off-by-one lure CoT -> answer follows to 9, 5/5).
"""
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figs"
SP = ROOT / "src_data" / "posthoc"
clean = json.load(open(SP / "clean.json"))
susc = json.load(open(SP / "suscept.json"))
cf = json.load(open(SP / "counterfactual.json"))
diff = json.load(open(SP / "difficulty.json"))["mean"]
med = st.median

by_pid = {}
for c in clean.values():
    by_pid.setdefault(c["pid"], {"cat": c["cat"], "clean": []})["clean"].append(c)
rows = []
for pid, g in by_pid.items():
    seeds = g["clean"]
    a = med([med(c["ans_lock"]) for c in seeds if c["ans_lock"]])
    acc = st.mean([1.0 if c["is_correct"] else 0.0 for c in seeds])
    sc = [s for s in susc.values() if s["pid"] == pid]
    bycs = defaultdict(dict)
    for s in sc:
        bycs[s["rho"]][s["corr_seed"]] = s["A_hat"]
    base = bycs.get(0.0, {})
    drift = {r: (0.0 if r == 0 else st.mean([1.0 if bycs[r].get(csi) != base.get(csi) else 0.0
                                             for csi in base if csi in bycs[r]]))
             for r in sorted({s["rho"] for s in sc})}
    S = st.mean([v for r, v in drift.items() if r > 0])
    rows.append(dict(pid=pid, cat=g["cat"], commit=a, S=S, acc=acc))

def rankdata(v):
    v = np.asarray(v, float)
    order = np.argsort(v)
    ranks = np.empty(len(v))
    sv = v[order]
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks

def spearman(xs, ys):
    return float(np.corrcoef(rankdata(xs), rankdata(ys))[0, 1])

CATCOL = {"crt": "#c2255c", "count": "#1971c2", "hard": "#2f9e44", "transform": "#9c36b5"}
PANELS = [(lambda r: diff[r["pid"]], lambda r: r["commit"],
           "blind difficulty rating", "commitment time (answer lock-in step)"),
          (lambda r: diff[r["pid"]], lambda r: r["S"],
           "blind difficulty rating", "susceptibility $S$"),
          (lambda r: r["commit"], lambda r: r["S"],
           "commitment time (answer lock-in step)", "susceptibility $S$")]

fig, axes = plt.subplots(1, 3, layout="constrained",
                         figsize=(plt.rcParams["figure.figsize"][0] * 2.1,
                                  plt.rcParams["figure.figsize"][1] * 0.85))
rhos = []
for ax, (fx, fy, xlab, ylab) in zip(axes, PANELS):
    xs, ys = [fx(r) for r in rows], [fy(r) for r in rows]
    for r, x, y in zip(rows, xs, ys):
        ax.scatter(x, y, color=CATCOL[r["cat"]], s=45, zorder=3,
                   marker="o" if r["acc"] >= 0.5 else "x")
        ax.annotate(r["pid"], (x, y), fontsize=5.5, xytext=(3, 3), textcoords="offset points",
                    color="0.45")
    b, a = np.polyfit(xs, ys, 1)
    gx = np.linspace(min(xs), max(xs), 20)
    ax.plot(gx, a + b * gx, color="0.55", linestyle="--", linewidth=1.1, zorder=2)
    rho = spearman(xs, ys)
    rhos.append(rho)
    ax.text(0.03, 0.97, rf"$\rho_S = {rho:+.2f}$", transform=ax.transAxes, va="top")
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.spines[["top", "right"]].set_visible(False)
handles = [plt.Line2D([], [], marker="o", linestyle="", color=c, label=k) for k, c in CATCOL.items()]
axes[0].legend(handles=handles, frameon=False, fontsize="small", loc="lower right")
assert abs(rhos[0] - 0.369) < 0.01 and abs(rhos[1] - 0.279) < 0.01 and abs(rhos[2] - 0.598) < 0.01, rhos
fig.savefig(OUT / "figA8_posthoc_correlations.png", dpi=200)
print(OUT / "figA8_posthoc_correlations.png")
print("spearman:", [round(r, 3) for r in rhos])

# ---- case-study card data: squares_400_800 dissociation ----
lure = json.load(open(SP / "lure_cots.json"))["squares_400_800"]
q = next(c["q"] for c in clean.values() if c["pid"] == "squares_400_800")
rand = susc["squares_400_800__r1.0__k0__c0"]
cfr = [cf[f"squares_400_800__s{s}"] for s in range(5)]
followed = sum(1 for v in cfr if v["followed_lure"])
case = {"pid": "squares_400_800", "q": q, "correct": lure["correct"], "lure_ans": lure["lure"],
        "free_answer": "8",
        "random": {"n_corrupted": rand["n_corrupted"], "n_cot": rand["n_cot"],
                   "snippet": rand["final_text"][:300], "answer": rand["A_hat"],
                   "drift_rhos": "0.00 at rho 0.25/0.5/1.0 (S = 0.05)"},
        "lure": {"cot": lure["cot"], "answer": cfr[0]["A_hat"], "followed": f"{followed}/5"}}
DATA = ROOT / "data"; DATA.mkdir(exist_ok=True)
json.dump(case, open(DATA / "posthoc_case.json", "w"), indent=1)
print(DATA / "posthoc_case.json")
