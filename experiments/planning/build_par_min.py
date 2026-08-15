"""Static mpl summary of the strict all-members parallelism readout:
min_{injected x_i} E(x_i) per joint cell, E(x_i) = R(img(x_i)) - <R(img(non-injected))>_P.
-> figs/par_min.png + printed per-task stats (state-clustered means, frac(min>0) vs 0.5^n).
"""
import os
import importlib.util
from pathlib import Path

import numpy as np
from scipy import stats as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PLAN = Path(__file__).resolve().parent
FIGS = Path(os.environ.get("DG_FIGS_DIR", str(Path(__file__).resolve().parent / "figs")))


def load_mod(name):
    spec = importlib.util.spec_from_file_location(name, PLAN / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


p2 = load_mod("xtask_par2_read")
p3 = load_mod("xtask_par3_read")

# per family: series -> {n: [(minval, state)]}, junk -> {n: [(val, state)]}
DATA = {}
def add(fam, series, n, v, state, junk=False):
    d = DATA.setdefault(fam, {}).setdefault(series, dict(real={}, junk={}))
    d["junk" if junk else "real"].setdefault(n, []).append((float(v), state))


for r in p2.recs:
    fam = {"UU3": "letters +3", "UU5": "letters +5", "UU7": "letters +7", "UU11": "letters +11"}[r["tag"]]
    stt = f"{r['tag']}|{r['s']}"
    add(fam, "0.30", 1, r["s1_n1"], stt); add(fam, "0.30", 1, r["s2_n1"], stt)
    add(fam, "0.30", 2, min(r["s1_n2"], r["s2_n2"]), stt)
    add(fam, "0.30", 2, r["s1_jk"], stt, junk=True)
GRP3 = {"mu x2": "letters ×2", "mu x3": "letters ×3", "reflect": "reflection", "num +3": "numbers +3"}
EPS3 = {"mu x2": "0.30", "mu x3": "0.30", "reflect": "0.30", "num +3": "0.25"}
for r in p3.recs2:
    if r["group"] not in GRP3:
        continue
    fam, e = GRP3[r["group"]], EPS3[r["group"]]
    add(fam, e, 1, r["s1_n1"], r["state"]); add(fam, e, 1, r["s2_n1"], r["state"])
    add(fam, e, 2, min(r["s1_n2"], r["s2_n2"]), r["state"])
    add(fam, e, 2, r["s1_jk"], r["state"], junk=True)
for r in p3.recs3:
    fam = {"UU3": "letters +3", "UU5": "letters +5"}[r["group"]]
    add(fam, "0.22 ladder", 1, r["s1_n1"], r["state"])
    add(fam, "0.22 ladder", 2, min(r["s1_r2j"], r["s2_r2j"]), r["state"])
    add(fam, "0.22 ladder", 3, min(r["s1_r3"], r["s2_r3"], r["s3_r3"]), r["state"])
    add(fam, "0.22 ladder", 3, r["s1_j2"], r["state"], junk=True)


def agg(pairs):
    by = {}
    for v, s in pairs:
        by.setdefault(s, []).append(v)
    sm = np.array([np.mean(v) for v in by.values()])
    se = sm.std(ddof=1) / np.sqrt(len(sm)) if len(sm) > 1 else 0.0
    p = st.ttest_1samp(sm, 0).pvalue if len(sm) > 1 else float("nan")
    return sm.mean(), se, p, len(sm)


ORDER = ["letters +3", "letters +5", "letters +7", "letters +11",
         "letters ×2", "letters ×3", "reflection", "numbers +3"]
COLS = {"0.30": "tab:blue", "0.25": "tab:blue", "0.22 ladder": "tab:orange"}

fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True)
print(f"{'family':>12} {'series':>12} {'n':>2} {'mean min-spec':>16} {'p':>8} "
      f"{'frac(min>0)':>11} {'chance':>7} {'cells':>5}")
for ax, fam in zip(axes.flat, ORDER):
    for series, d in DATA.get(fam, {}).items():
        col = COLS[series]
        ns = sorted(d["real"])
        pts = [agg(d["real"][n]) for n in ns]
        ax.errorbar(ns, [p[0] for p in pts], yerr=[p[1] for p in pts], marker="o",
                    capsize=3, color=col, label=f"ε₀={series}")
        for n, pt in zip(ns, pts):
            vals = [v for v, _ in d["real"][n]]
            frac = np.mean([v > 0 for v in vals])
            chance = 0.5 ** n
            ax.annotate(f"{frac:.0%}", (n, pt[0]), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=7, color=col)
            print(f"{fam:>12} {series:>12} {n:>2} {pt[0]:>+10.3f}±{pt[1]:.3f} {pt[2]:>8.1e} "
                  f"{frac:>10.0%} {chance:>7.0%} {len(vals):>5}")
        for n, pairs in d["junk"].items():
            m, se, _, _ = agg(pairs)
            ax.errorbar([n + 0.15], [m], yerr=[se], marker="D", mfc="none", ls="none",
                        capsize=3, color="gray")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks([1, 2, 3])
    ax.set_title(fam)
    ax.legend(fontsize=7)
for ax in axes[1]:
    ax.set_xlabel("n (real operands)")
for ax in axes[:, 0]:
    ax.set_ylabel("min over members of spec\n(± clustered SE; % = frac all-members>0)")
fig.suptitle("strict parallelism readout: EVERY injected member's image elevated "
             "(hollow ◇ = single-member junk-matched ceiling)", fontsize=10)
fig.tight_layout()
fig.savefig(FIGS / "par_min.png", dpi=150)
print(FIGS / "par_min.png")
