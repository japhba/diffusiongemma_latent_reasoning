"""Build the confound-free n>1 assets for symbol_arithmetic.html:
  figs/par_confoundfree.png  — two-panel proper-E curve (per-op dose fixed | total fixed)
  exp/dg_planning/par_tables.html — HTML fragment (arm tables + null checks), spliced
                                    into the report by build_superpos.py.
Numbers are re-derived from xtask_par2/par3.json via the reader scripts (imported).
"""
import os
import importlib.util
import numpy as np
from scipy import stats as st
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PLAN = Path(__file__).resolve().parent
FIGS = Path(os.environ.get("DG_FIGS_DIR", str(Path(__file__).resolve().parent / "figs")))
EXP = Path(os.environ.get("DG_PLANNING_DIR", str(Path(__file__).resolve().parent / "exp")))


def load_mod(name):
    spec = importlib.util.spec_from_file_location(name, PLAN / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


p2 = load_mod("xtask_par2_read")     # .recs  (UU n=2 @0.3)
p3 = load_mod("xtask_par3_read")     # .recs2 (other fams n=2), .recs3 (UU n=3 ladder)


def agg(pairs):
    """pairs = [(value, state)] -> state-clustered (mean, se, p, n_states)"""
    by = {}
    for v, s in pairs:
        by.setdefault(s, []).append(v)
    sm = np.array([np.mean(v) for v in by.values()])
    se = sm.std(ddof=1) / np.sqrt(len(sm)) if len(sm) > 1 else float("nan")
    p = st.ttest_1samp(sm, 0).pvalue if len(sm) > 1 else float("nan")
    return sm.mean(), se, p, len(sm)


def stt(r):
    return f"{r['tag']}|{r['s']}"


# ---- panel A series: per-operand dose fixed, all-real arms only (both operands pooled)
def uu_point(key_solo, key_joint):
    solo = agg([(r[k], stt(r)) for r in p2.recs for k in key_solo])
    joint = agg([(r[k], stt(r)) for r in p2.recs for k in key_joint])
    return solo, joint


A = {}
A["UU +k pooled (ε₀=0.30)"] = dict(
    x=[1, 2],
    pts=[agg([(r[k], stt(r)) for r in p2.recs for k in ("s1_n1", "s2_n1")]),
         agg([(r[k], stt(r)) for r in p2.recs for k in ("s1_n2", "s2_n2")])],
    junk=(2, agg([(r["s1_jk"], stt(r)) for r in p2.recs])), color="tab:blue")
A["reflect (ε₀=0.30)"] = dict(
    x=[1, 2],
    pts=[agg([(r[k], r["state"]) for r in p3.recs2 if r["group"] == "reflect" for k in ("s1_n1", "s2_n1")]),
         agg([(r[k], r["state"]) for r in p3.recs2 if r["group"] == "reflect" for k in ("s1_n2", "s2_n2")])],
    junk=(2, agg([(r["s1_jk"], r["state"]) for r in p3.recs2 if r["group"] == "reflect"])), color="tab:green")
A["numbers +3 (ε₀=0.25)"] = dict(
    x=[1, 2],
    pts=[agg([(r[k], r["state"]) for r in p3.recs2 if r["group"] == "num +3" for k in ("s1_n1", "s2_n1")]),
         agg([(r[k], r["state"]) for r in p3.recs2 if r["group"] == "num +3" for k in ("s1_n2", "s2_n2")])],
    junk=(2, agg([(r["s1_jk"], r["state"]) for r in p3.recs2 if r["group"] == "num +3"])), color="tab:purple")
A["UU3 (ε₀=0.22)"] = dict(
    x=[1, 3],
    pts=[agg([(r["s1_n1"], r["state"]) for r in p3.recs3 if r["group"] == "UU3"]),
         agg([(r[k], r["state"]) for r in p3.recs3 if r["group"] == "UU3" for k in ("s1_r3", "s2_r3", "s3_r3")])],
    junk=None, color="tab:red")
A["UU5 (ε₀=0.22)"] = dict(
    x=[1, 3],
    pts=[agg([(r["s1_n1"], r["state"]) for r in p3.recs3 if r["group"] == "UU5"]),
         agg([(r[k], r["state"]) for r in p3.recs3 if r["group"] == "UU5" for k in ("s1_r3", "s2_r3", "s3_r3")])],
    junk=None, color="tab:gray")

# ---- panel B: fixed total mass 0.66 (3 tokens @0.22), routable count 1/2/3
B = {}
for grp, color in (("UU3", "tab:red"), ("UU5", "tab:gray")):
    R = [r for r in p3.recs3 if r["group"] == grp]
    B[grp] = dict(color=color, pts=[
        agg([(r["s1_j2"], r["state"]) for r in R]),
        agg([(r["s1_r2j"], r["state"]) for r in R]),
        agg([(r["s1_r3"], r["state"]) for r in R])])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
for lab, d in A.items():
    y = [p[0] for p in d["pts"]]; e = [p[1] for p in d["pts"]]
    ax1.errorbar(d["x"], y, yerr=e, marker="o", capsize=3, label=lab, color=d["color"])
    if d["junk"]:
        xj, pj = d["junk"]
        ax1.errorbar([xj + 0.06], [pj[0]], yerr=[pj[1]], marker="D", mfc="none", capsize=3,
                     color=d["color"], ls="none")
ax1.axhline(0, color="k", lw=0.6)
ax1.set_xticks([1, 2, 3])
ax1.set_xlabel("n (simultaneously injected operands, per-operand dose fixed)")
ax1.set_ylabel("per-operand spec (placebo- & distance-matched)")
ax1.set_title("per-operand dose fixed — joint vs solo\n(open ◇ = junk-matched control at same total mass)")
ax1.legend(fontsize=8)
for lab, d in B.items():
    y = [p[0] for p in d["pts"]]; e = [p[1] for p in d["pts"]]
    ax2.errorbar([1, 2, 3], y, yerr=e, marker="o", capsize=3, label=lab, color=d["color"])
    ax2.plot([1, 2, 3], [y[0], y[0] / 2, y[0] / 3], ls="--", lw=1, color=d["color"], alpha=0.5)
ax2.axhline(0, color="k", lw=0.6)
ax2.set_xticks([1, 2, 3])
ax2.set_xlabel("routable hypotheses among 3 injected tokens (total mass 0.66)")
ax2.set_title("total mass fixed — budget test\n(dashed = strict budget-sharing prediction B/n)")
ax2.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIGS / "par_confoundfree.png", dpi=150)
print(FIGS / "par_confoundfree.png")


# ---- HTML tables fragment -------------------------------------------------------------
def fmt(a):
    m, se, p, n = a
    star = "***" if p < 1e-3 else "**" if p < .01 else "*" if p < .05 else "&dagger;" if p < .1 else ""
    return f"{m:+.2f}&plusmn;{se:.2f}{star}"


def row(cells, tag="td"):
    return "<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>"


TS = ' style="border-collapse:collapse;margin:8px 0" border="1" cellpadding="5"'
frag = []
frag.append("<h3>n=2, four arms per (state, rep) &mdash; operand-1 spec</h3>")
frag.append(f"<table{TS}>" + row(["family (ε₀)", "solo", "joint", "junk-matched",
                                  "joint−solo", "joint−junk", "states"], "th"))
fams = [("UU +k pooled (0.30)", p2.recs, "s1_n1", "s1_n2", "s1_jk", stt)]
for g, eps in (("mu x2", "0.30"), ("mu x3", "0.30"), ("reflect", "0.30"),
               ("qwerty", "0.30"), ("minus3", "0.30"), ("num +3", "0.25")):
    fams.append((f"{g} ({eps})", [r for r in p3.recs2 if r["group"] == g],
                 "s1_n1", "s1_n2", "s1_jk", lambda r: r["state"]))
for lab, R, k1, k2, k3, sf in fams:
    if not R:
        continue
    frag.append(row([lab, fmt(agg([(r[k1], sf(r)) for r in R])),
                     fmt(agg([(r[k2], sf(r)) for r in R])),
                     fmt(agg([(r[k3], sf(r)) for r in R])),
                     fmt(agg([(r[k2] - r[k1], sf(r)) for r in R])),
                     fmt(agg([(r[k2] - r[k3], sf(r)) for r in R])),
                     str(len({sf(r) for r in R}))]))
frag.append("</table>")
(EXP / "par_tables.html").write_text("\n".join(frag))
frag = []
frag.append("<h4>n=3 ladder (letters +3/+5, ε₀=0.22, non-solo arms at matched total mass 0.66)</h4>")
frag.append(f"<table{TS}>" + row(["task", "solo @0.22", "1 hyp (+2 junk)", "2 hyp (+1 junk)",
                                  "3 hyp", "r3−j2 (paired)"], "th"))
for g in ("UU3", "UU5"):
    R = [r for r in p3.recs3 if r["group"] == g]
    sf = lambda r: r["state"]
    frag.append(row([g, fmt(agg([(r["s1_n1"], sf(r)) for r in R])),
                     fmt(agg([(r["s1_j2"], sf(r)) for r in R])),
                     fmt(agg([(r["s1_r2j"], sf(r)) for r in R])),
                     fmt(agg([(r["s1_r3"], sf(r)) for r in R])),
                     fmt(agg([(r["s1_r3"] - r["s1_j2"], sf(r)) for r in R]))]))
frag.append("</table>")
(EXP / "par_ladder.html").write_text("\n".join(frag))
print(EXP / "par_tables.html", EXP / "par_ladder.html")
