"""A8: post-hoc vs load-bearing CoT — the three correlations + case-study card data.

Panels: (1) blind difficulty vs commitment time, (2) blind difficulty vs susceptibility S,
(3) commitment time vs S. Per problem (n=40; original 20 + the 2026-08-08 extension battery): commitment = median-over-seeds of median answer
lock-in step (clean.json); S = mean over rho>0 of P(answer differs from the rho=0 clean-clamp
baseline, matched by corruption seed) (suscept.json); difficulty = mean of 3 blind subagent
ratings from the problem text alone (difficulty.json). Spearman hand-rolled (rank + corrcoef);
asserts pin the n=40 values +0.37 / +0.28 / +0.60 (n=20 report values were +0.37 / +0.42 / +0.66).

p-values: two-sided permutation test on the Spearman rho (20000 permutations, seeded).

Also emits data/posthoc_case.json for the easy-vs-hard case card: bat_ball (commit 0, S~0.05,
answer survives rho=1.0 full-CoT randomization) vs sq1000 (commit ~4, S~0.70, answer flips).
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
    rng = np.random.default_rng(0)
    ry = rankdata(ys)
    rx = rankdata(xs)
    perm = np.array([np.corrcoef(rx, rng.permutation(ry))[0, 1] for _ in range(20000)])
    pval = (1 + np.sum(np.abs(perm) >= abs(rho))) / (1 + len(perm))
    ax.text(0.03, 0.97, rf"$\rho_S = {rho:+.2f}$" + f"\np = {pval:.4f}".replace("0.", "."),
            transform=ax.transAxes, va="top")
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.spines[["top", "right"]].set_visible(False)
handles = [plt.Line2D([], [], marker="o", linestyle="", color=c, label=k) for k, c in CATCOL.items()]
axes[0].legend(handles=handles, frameon=False, fontsize="small", loc="lower right")
assert abs(rhos[0] - 0.369) < 0.01 and abs(rhos[1] - 0.279) < 0.01 and abs(rhos[2] - 0.598) < 0.01, rhos
fig.savefig(OUT / "posthoc_correlations.png", dpi=200)
print(OUT / "posthoc_correlations.png")
print("spearman:", [round(r, 3) for r in rhos])

# ---- case-study card data: easy (bat_ball) vs hard (sq1000), clean + rho=1.0 clamp ----
Srow = {r["pid"]: r["S"] for r in rows}
case = {}
for tag, pid in [("easy", "bat_ball"), ("hard", "sq1000")]:
    reps = [clean[f"{pid}__{s}"] for s in range(5)]
    rep = next(c for c in reps if c["is_correct"] and c["ans_lock"] and c["cot_pos"])
    cells = [susc[f"{pid}__r1.0__k0__c{c}"] for c in range(5)]
    sc = cells[0]
    case[tag] = dict(
        pid=pid, q=rep["q"], S=round(Srow[pid], 2),
        clean=dict(text=rep["final_text"][:300], answer=rep["model_ans"],
                   commit=med(rep["ans_lock"]), traj=rep["answer_traj"]),
        suscept=dict(text=sc["final_text"][:300], answer=sc["A_hat"],
                     answers=[c["A_hat"] for c in cells],
                     match=sc["match"], n_cot=sc["n_cot"]))
DATA = ROOT / "data"; DATA.mkdir(exist_ok=True)
json.dump(case, open(DATA / "posthoc_case.json", "w"), indent=1)
print(DATA / "posthoc_case.json")
