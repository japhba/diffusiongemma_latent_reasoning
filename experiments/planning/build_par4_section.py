"""Assets for the co-dominant (no-guard) n=2 section of symbol_arithmetic.html:
figs/par4_noguard.png + exp/dg_planning/par4_tables.html (spliced by build_superpos.py)."""
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
EXP = Path(os.environ.get("DG_PLANNING_DIR", str(Path(__file__).resolve().parent / "exp")))

spec = importlib.util.spec_from_file_location("p4", PLAN / "xtask_par4_read.py")
p4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p4)
recs = p4.recs


def agg(pairs):
    by = {}
    for v, s in pairs:
        by.setdefault(s, []).append(v)
    sm = np.array([np.mean(v) for v in by.values()])
    se = sm.std(ddof=1) / np.sqrt(len(sm)) if len(sm) > 1 else float("nan")
    p = st.ttest_1samp(sm, 0).pvalue if len(sm) > 1 else float("nan")
    return sm.mean(), se, p, len(sm)


GROUPS = ["UU +3", "UU +7", "mu x2", "mu x3"]
LAB = {"UU +3": "letters +3", "UU +7": "letters +7", "mu x2": "letters ×2", "mu x3": "letters ×3"}
ARMS = [("s1_n1", "solo @0.45 (guarded)", "tab:blue"),
        ("s1_n2", "joint co-dominant", "tab:red"),
        ("s1_jk", "junk co-dominant", "tab:gray")]

fig, ax = plt.subplots(figsize=(7.5, 4.2))
xs = np.arange(len(GROUPS))
for i, (key, lab, color) in enumerate(ARMS):
    y, e = [], []
    for g in GROUPS:
        m, se, _, _ = agg([(r[key], r["state"]) for r in recs if r["group"] == g])
        y.append(m); e.append(se)
    ax.errorbar(xs + (i - 1) * 0.15, y, yerr=e, fmt="o", capsize=3, label=lab, color=color)
ax.axhline(0, color="k", lw=0.6)
ax.set_xticks(xs)
ax.set_xticklabels([LAB[g] for g in GROUPS])
ax.set_ylabel("operand-1 spec (placebo- & distance-matched)")
ax.set_title("co-dominant n=2 @ ε₀=0.45, subleading guard dropped (total 0.9)")
ax.legend()
fig.tight_layout()
fig.savefig(FIGS / "par4_noguard.png", dpi=150)
print(FIGS / "par4_noguard.png")


def fmt(a):
    m, se, p, n = a
    star = "***" if p < 1e-3 else "**" if p < .01 else "*" if p < .05 else "&dagger;" if p < .1 else ""
    return f"{m:+.2f}&plusmn;{se:.2f}{star}"


def row(cells, tag="td"):
    return "<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>"


TS = ' style="border-collapse:collapse;margin:8px 0" border="1" cellpadding="5"'
frag = [f"<table{TS}>" + row(["family", "solo @0.45 (guarded)", "joint co-dom", "junk co-dom",
                              "joint−solo", "joint−junk", "op2 joint−solo",
                              "both tgts top-4", "R(ja) joint", "states"], "th")]
for g in GROUPS + ["ALL"]:
    R = [r for r in recs if g == "ALL" or r["group"] == g]
    if not R:
        continue
    sts = [r["state"] for r in R]
    frag.append(row([LAB.get(g, "pooled"),
                     fmt(agg([(r["s1_n1"], r["state"]) for r in R])),
                     fmt(agg([(r["s1_n2"], r["state"]) for r in R])),
                     fmt(agg([(r["s1_jk"], r["state"]) for r in R])),
                     fmt(agg([(r["s1_n2"] - r["s1_n1"], r["state"]) for r in R])),
                     fmt(agg([(r["s1_n2"] - r["s1_jk"], r["state"]) for r in R])),
                     fmt(agg([(r["s2_n2"] - r["s2_n1"], r["state"]) for r in R])),
                     f"{np.mean([r['top4'] for r in R]):.0%}",
                     fmt(agg([(r["ja_n2"], r["state"]) for r in R])),
                     str(len(set(sts)))]))
frag.append("</table>")
(EXP / "par4_tables.html").write_text("\n".join(frag))
print(EXP / "par4_tables.html")
