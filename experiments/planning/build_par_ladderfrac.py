"""Junk-padded ladder curves, strict frac readout -> figs/par_ladder_frac.png
Per task: x = r (routable members among K injected tokens, total mass fixed = K*eps0),
y = frac of cells with min_i E(x_i) > 0. K=2 series from the four-arm battery
(jk arm = r=1, joint = r=2); K=3 series (letters +3/+5) from the ladder
(j2 = r=1, r2j = r=2, r3 = r=3). Dashed line = chance 0.5^r. Errorbars = 95% CI over
state means. No significance markers.
"""
import os
import importlib.util
from pathlib import Path

import numpy as np
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

# task -> series label -> {r: [(ok, state)]}
D = {}
def add(task, series, r, ok, state):
    D.setdefault(task, {}).setdefault(series, {}).setdefault(r, []).append((ok, state))


for r_ in p2.recs:
    task = {"UU3": "letters +3", "UU5": "letters +5", "UU7": "letters +7",
            "UU11": "letters +11"}[r_["tag"]]
    stt = f"{r_['tag']}|{r_['s']}"
    add(task, "K=2", 1, r_["s1_jk"] > 0, stt)
    add(task, "K=2", 2, min(r_["s1_n2"], r_["s2_n2"]) > 0, stt)
GRP3 = {"mu x2": "letters ×2", "mu x3": "letters ×3", "reflect": "reflection", "num +3": "numbers +3"}
for r_ in p3.recs2:
    if r_["group"] not in GRP3:
        continue
    task = GRP3[r_["group"]]
    add(task, "K=2", 1, r_["s1_jk"] > 0, r_["state"])
    add(task, "K=2", 2, min(r_["s1_n2"], r_["s2_n2"]) > 0, r_["state"])
for r_ in p3.recs3:
    task = {"UU3": "letters +3", "UU5": "letters +5"}[r_["group"]]
    add(task, "K=3", 1, r_["s1_j2"] > 0, r_["state"])
    add(task, "K=3", 2, min(r_["s1_r2j"], r_["s2_r2j"]) > 0, r_["state"])
    add(task, "K=3", 3, min(r_["s1_r3"], r_["s2_r3"], r_["s3_r3"]) > 0, r_["state"])


def agg(pairs):
    by = {}
    for ok, s in pairs:
        by.setdefault(s, []).append(ok)
    sm = np.array([np.mean(v) for v in by.values()])
    se = sm.std(ddof=1) / np.sqrt(len(sm)) if len(sm) > 1 else 0.0
    return np.mean([ok for ok, _ in pairs]), 1.96 * se


ORDER = ["letters +3"]
SCOL = {"K=3": "tab:blue"}

fig, axes = plt.subplots(1, 1, figsize=(5.2, 4.4), sharex=True, sharey=True)
for ax, task in zip(np.atleast_1d(axes).ravel(), ORDER):
    rs_all = [1, 2, 3]
    ax.plot(rs_all, [0.5 ** r for r in rs_all], ls="--", color="k", lw=1.2,
            label=r"chance $0.5^{\,n}$")
    for series, dat in D.get(task, {}).items():
        if series != "K=3":
            continue
        rs = sorted(dat)
        pts = [agg(dat[r]) for r in rs]
        ax.errorbar(rs, [p[0] for p in pts], yerr=[p[1] for p in pts], marker="o",
                    capsize=3, color=SCOL[series], label="_nolegend_")
    ax.set_title(task)
    ax.set_xticks([1, 2, 3])
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7)
fig.supxlabel(r"operands $n$", fontsize=11)
for ax in [np.atleast_1d(axes).ravel()[0]]:
    ax.set_ylabel(r"frac of injection sets with $\min_i E(x_i) > 0$")
fig.suptitle(r"$E(x_i) = R(\mathrm{img}(x_i)) - \langle R(\mathrm{img}(y))\rangle_{y \in P}$"
             r",  $R(w) = \log_{10}\, \bar{P}_{\mathrm{pert}}(w) / \bar{P}_{\mathrm{base}}(w)$",
             fontsize=9.5)
fig.tight_layout()
fig.savefig(FIGS / "par_ladder_frac.png", dpi=150)
print(FIGS / "par_ladder_frac.png")
