"""Simple summary figure -> figs/par_frac.png
Top: frac(all injected members > 0) per task/arm, tick = chance 0.5^n, stars = clustered
t-test vs chance. Bottom: matshow, columns = letters tasks, rows = alphabet, color = mean
R(w) at the answer position over all joint cells (averaged over seeds/reps); black dots =
injection sites (size ~ frequency), red x = committed operand x_nat.
"""
import os
import importlib.util, json, string
from pathlib import Path

import numpy as np
from scipy import stats as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

PLAN = Path(__file__).resolve().parent
FIGS = Path(os.environ.get("DG_FIGS_DIR", str(Path(__file__).resolve().parent / "figs")))
EXP = Path(os.environ.get("DG_PLANNING_DIR", str(Path(__file__).resolve().parent / "exp")))
FLOOR = 1e-5
LOW = list(string.ascii_lowercase); UPP = list(string.ascii_uppercase); F52 = LOW + UPP


def load_mod(name):
    spec = importlib.util.spec_from_file_location(name, PLAN / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


p2 = load_mod("xtask_par2_read")
p3 = load_mod("xtask_par3_read")

# ---------- top panel: frac(all members > 0) --------------------------------------------
BARS = []   # (label, n, [(ok, state)])
for tag, lab in (("UU3", "+3"), ("UU5", "+5"), ("UU7", "+7"), ("UU11", "+11")):
    R = [r for r in p2.recs if r["tag"] == tag]
    BARS.append((lab, 2, [(min(r["s1_n2"], r["s2_n2"]) > 0, f"{r['tag']}|{r['s']}") for r in R]))
    R3 = [r for r in p3.recs3 if r["group"] == tag]
    if R3:
        BARS.append((lab, 3, [(min(r["s1_r3"], r["s2_r3"], r["s3_r3"]) > 0, r["state"]) for r in R3]))
for g, lab in (("reflect", "reflect"), ("mu x2", "×2"), ("mu x3", "×3"), ("num +3", "num +3")):
    R = [r for r in p3.recs2 if r["group"] == g]
    BARS.append((lab, 2, [(min(r["s1_n2"], r["s2_n2"]) > 0, r["state"]) for r in R]))

# ---------- bottom: per-task matshows, one column per injected-tuple config ------------
par2j = json.load(open(EXP / "xtask_par2.json"))
par3j = json.load(open(EXP / "xtask_par3.json"))
dns = json.load(open(EXP / "xtask_samecase_nsweep.json"))
dmun = json.load(open(EXP / "xtask_mult_nsweep.json"))
dopn = json.load(open(EXP / "xtask_ops_nsweep.json"))

def img_shift(k):
    return lambda w: (UPP[UPP.index(w) + k] if 0 <= UPP.index(w) + k < 26 else None)


def img_mu(k):
    return lambda w: (UPP[(UPP.index(w) + 1) * k - 1] if 1 <= (UPP.index(w) + 1) * k <= 26 else None)


dops = json.load(open(EXP / "xtask_ops.json"))
OPMETA = {k.split("|")[0]: v for k, v in dops.items() if k.endswith("|meta")}
RFIMG = OPMETA["RF"]["image"]

TASKS = [
    ("+3 n=2", "par2", ["UU3"], "n2", dns, img_shift(3), UPP[:23], 4),
    ("+3 n=3", "par3", ["UU3"], "r3", dns, img_shift(3), UPP[:23], 4),
    ("+5 n=2", "par2", ["UU5"], "n2", dns, img_shift(5), UPP[:21], 4),
    ("+7 n=2", "par2", ["UU7"], "n2", dns, img_shift(7), UPP[:19], 4),
    ("+11 n=2", "par2", ["UU11"], "n2", dns, img_shift(11), UPP[:15], 4),
    ("×2 n=2", "par3", ["MU2"], "n2", dmun, img_mu(2), UPP[:13], 4),
    ("×3 n=2", "par3", ["MU3"], "n2", dmun, img_mu(3), UPP[:8], 3),
    ("reflect n=2", "par3", ["RF", "RFB", "RFC", "RFD"], "n2", dopn,
     (lambda w: RFIMG.get(w)), list(OPMETA["RF"]["pool"]), 4),
]
ROWS = []   # (label, cfgs, M, marks per config)
for lab, battery, tags, arm, basesrc, img, pool, dstar in TASKS:
    src = par2j if battery == "par2" else par3j
    acc = {}
    for tag in tags:
        for s2 in range(10):
            bk3 = par3j.get(f"{tag}|s{s2}|par3|base")
            bsrc = bk3 or basesrc.get(f"{tag}|s{s2}|base")
            if not bsrc:
                continue
            bmap = {w: float(v) for w, v in zip(F52, np.array(bsrc["rows"]).mean(axis=0))}
            nat = bsrc["nat_op"]
            pairs = src.get(f"{tag}|s{s2}|{battery}|pairs", [])
            for r in range(len(pairs)):
                c = src.get(f"{tag}|s{s2}|{battery}|r{r}|{arm}")
                if not c or any(v == 0 for v in c["ranks"].values()):
                    continue
                cfg = tuple(sorted(c["subset"], key=UPP.index))
                amap = {w: float(v) for w, v in zip(F52, np.array(c["rows"]).mean(axis=0))}
                prof = np.array([np.log10(max(amap[w], FLOOR) / max(bmap[w], FLOOR)) for w in UPP])
                d = acc.setdefault(cfg, dict(prof=np.zeros(26), cnt=0, nats=set()))
                d["prof"] += prof; d["cnt"] += 1; d["nats"].add(nat)
    cfgs = sorted(acc)
    M = np.stack([acc[c]["prof"] / acc[c]["cnt"] for c in cfgs], axis=1) if cfgs else np.zeros((26, 0))
    marks = []
    for cfg in cfgs:
        nat = sorted(acc[cfg]["nats"])[0]
        ja = img(nat)
        tgt = [img(w) for w in cfg if img(w)]
        psrc = [y for y in pool if y not in cfg and y != nat
                and img(y) not in (ja, None) and img(y) not in cfg
                and all(abs(UPP.index(img(y)) - UPP.index(x)) >= dstar for x in cfg)]
        marks.append(dict(inj=list(cfg), tgt=tgt, psrc=psrc,
                          ptgt=[img(y) for y in psrc], nat=nat))
    ROWS.append((lab, cfgs, M, marks))

vmax = np.nanpercentile(np.abs(np.concatenate([Rw[2].ravel() for Rw in ROWS if Rw[2].size])), 98)
fig2 = plt.figure(figsize=(14, 30))
gs2 = fig2.add_gridspec(len(ROWS) + 1, 1, height_ratios=[0.9] + [1] * len(ROWS), hspace=0.5)
axb = fig2.add_subplot(gs2[0])
for i, (lab, n, rows) in enumerate(BARS):
    frac = np.mean([ok for ok, _ in rows]); chance = 0.5 ** n
    by = {}
    for ok, stt in rows:
        by.setdefault(stt, []).append(ok)
    sm = np.array([np.mean(v) for v in by.values()])
    pval = st.ttest_1samp(sm, chance).pvalue if len(sm) > 1 and sm.std() > 0 else 1.0
    se = sm.std(ddof=1) / np.sqrt(len(sm)) if len(sm) > 1 else 0.0
    axb.bar(i, frac, 0.7, color="tab:blue", alpha=0.75,
            yerr=1.96 * se, capsize=4, error_kw=dict(lw=1.2))
    axb.plot([i - 0.35, i + 0.35], [chance, chance], color="k", lw=1.4)
    star = "***" if pval < 1e-3 else "**" if pval < .01 else "*" if pval < .05 else ""
    if star:
        axb.text(i, frac + 1.96 * se + 0.02, star, ha="center", fontsize=11)
axb.set_xticks(range(len(BARS)))
axb.set_xticklabels([f"{lab}\nn={n}" for lab, n, _ in BARS], fontsize=8)
axb.set_ylabel("frac cells:\nALL members > 0")
axb.set_title("strict readout: frac of cells with every injected members' spec > 0 "
              "(bars: 95% CI, state-clustered); tick = chance $0.5^n$")

MK = dict(inj=dict(marker="o", color="tab:blue", filled=True, s=46, label="injected operand"),
          tgt=dict(marker="*", color="tab:red", filled=True, s=90, label="its target img(x)"),
          psrc=dict(marker="o", color="0.45", filled=False, s=18, label="placebo source (not injected)"),
          ptgt=dict(marker="D", color="tab:green", filled=False, s=26, label="placebo image (reference set P)"))
im = None
for ri, (lab, cfgs, M, marks) in enumerate(ROWS):
    ax = fig2.add_subplot(gs2[ri + 1])
    if not cfgs:
        ax.axis("off"); continue
    im = ax.imshow(M, aspect="auto", cmap="RdBu_r", norm=TwoSlopeNorm(0, -vmax, vmax), alpha=0.38)
    for ci, mk in enumerate(marks):
        for key in ("psrc", "ptgt", "inj", "tgt"):
            spec = MK[key]
            for w in mk[key]:
                ax.scatter(ci, UPP.index(w), marker=spec["marker"], s=spec["s"],
                           facecolors=(spec["color"] if spec["filled"] else "none"),
                           edgecolors=spec["color"], lw=1.2, zorder=3)
        ax.scatter(ci, UPP.index(mk["nat"]), marker="x", s=40, color="k", lw=1.6, zorder=4)
    ax.set_yticks(range(0, 26, 2)); ax.set_yticklabels(UPP[::2], fontsize=6.5)
    ax.set_xticks(range(len(cfgs)))
    ax.set_xticklabels(["".join(c) for c in cfgs], fontsize=6.5, rotation=90)
    ax.set_ylabel(lab, fontsize=9)
handles = [plt.scatter([], [], marker=v["marker"], s=v["s"],
                       facecolors=(v["color"] if v["filled"] else "none"),
                       edgecolors=v["color"], lw=1.2, label=v["label"]) for v in MK.values()]
handles.append(plt.scatter([], [], marker="x", s=40, color="k", lw=1.6, label="committed operand x_nat"))
fig2.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.900), ncol=5, fontsize=9)
fig2.colorbar(im, ax=fig2.axes[1:], pad=0.01, fraction=0.02,
              label="mean R at answer position (low-alpha background; seed-averaged per config)")
fig2.axes[1].set_title("rows = alphabet at answer position; each column = one injected config",
                       fontsize=10)
fig2.savefig(FIGS / "par_frac.png", dpi=130, bbox_inches="tight")
print(FIGS / "par_frac.png")
