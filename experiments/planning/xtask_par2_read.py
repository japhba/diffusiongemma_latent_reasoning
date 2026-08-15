"""Read out the par2 supra-threshold n=2 battery.

Per (state, rep): four arms (n1a, n1b, n2, jk) share canvases + the state's base rows, so
all contrasts are within-state paired. Shared reference across arms: placebo images
P = {img(y): y in pool \\ {x1,x2,nat}} minus {ja,jb} minus tokens within DSTAR of any
injected source (x1, x2, junk) minus the injected tokens — identical set for every arm.

spec(op|arm) = R_arm(img(op)) - <R_arm>_P.

Tables: per task, operand-1 spec under n1a / n2 / jk (+ operand-2 under n1b / n2);
paired diffs n2-n1a, jk-n1a, n2-jk with state-level clustering; release check dP(ja).
"""
import os
import json, string
from pathlib import Path

import numpy as np
from scipy import stats as st

EXP = Path(os.environ.get("DG_PLANNING_DIR", str(Path(__file__).resolve().parent / "exp")))
FLOOR = 1e-5
DSTAR = 4
LOW = list(string.ascii_lowercase); UPP = list(string.ascii_uppercase); F52 = LOW + UPP
TASKS = {"UU3": 3, "UU5": 5, "UU7": 7, "UU11": 11}

par = json.load(open(EXP / "xtask_par2.json"))
dns = json.load(open(EXP / "xtask_samecase_nsweep.json"))

HI = {"UU3": "W", "UU5": "U", "UU7": "S", "UU11": "O"}


def img(w, k):
    i = UPP.index(w) + k
    return UPP[i] if 0 <= i < 26 else None


recs = []   # one per (tag, s, rep): dict of specs per arm/op + meta
drop = 0
for tag, k in TASKS.items():
    hi_i = UPP.index(HI[tag])
    pool = UPP[:hi_i + 1]
    for s in range(10):
        bk = f"{tag}|s{s}|base"
        if bk not in dns:
            continue
        d = json.load(open(EXP / f"nego2/{tag}__s{s}.json"))
        nat, A, B = dns[bk]["nat_op"], dns[bk]["A"], dns[bk]["B"]
        bmap = {w: float(v) for w, v in zip(F52, np.array(dns[bk]["rows"]).mean(axis=0))}
        ja = img(nat, k)
        jb = d["id2str"].get(str(d["final_ids"][B]), "?").replace("▁", " ").strip()
        pairs = par.get(f"{tag}|s{s}|par2|pairs", [])
        for r, (x1, x2, junk) in enumerate(pairs):
            arms = {}
            for arm in ("n1a", "n1b", "n2", "jk"):
                c = par.get(f"{tag}|s{s}|par2|r{r}|{arm}")
                if not c:
                    continue
                if any(v == 0 for v in c["ranks"].values()):
                    drop += 1
                    continue
                amap = {w: float(v) for w, v in zip(F52, np.array(c["rows"]).mean(axis=0))}
                arms[arm] = {w: np.log10(max(amap[w], FLOOR) / max(bmap[w], FLOOR))
                             for w in UPP}   # ja/jb stay excluded from P and targets
            if len(arms) < 4:
                continue
            pi = {w: i for i, w in enumerate(UPP)}
            srcs = [x1, x2, junk]
            P = [img(y, k) for y in pool
                 if y not in (x1, x2, nat) and img(y, k) not in (ja, jb, None)]
            P = [w for w in set(P) - {x1, x2, junk}
                 if all(abs(pi[w] - pi[sx]) >= DSTAR for sx in srcs)
                 and w in arms["n1a"]]
            t1, t2 = img(x1, k), img(x2, k)
            if not P or t1 not in arms["n1a"] or t2 not in arms["n1a"]:
                continue
            spec = lambda arm, tgt: arms[arm][tgt] - np.mean([arms[arm][w] for w in P])
            recs.append(dict(
                tag=tag, s=s, rep=r, x1=x1, x2=x2, junk=junk, nP=len(P),
                s1_n1=spec("n1a", t1), s1_n2=spec("n2", t1), s1_jk=spec("jk", t1),
                s2_n1=spec("n1b", t2), s2_n2=spec("n2", t2),
                dja_n1=arms["n1a"].get(ja), dja_n2=arms["n2"].get(ja), dja_jk=arms["jk"].get(ja)))

print(f"{len(recs)} complete (state,rep) quadruples, {drop} rank-guard drops")


def agg(vals, states):
    """state-clustered mean ± SE + p (t-test over state means)"""
    by = {}
    for v, stt in zip(vals, states):
        by.setdefault(stt, []).append(v)
    sm = np.array([np.mean(v) for v in by.values()])
    se = sm.std(ddof=1) / np.sqrt(len(sm)) if len(sm) > 1 else float("nan")
    p = st.ttest_1samp(sm, 0).pvalue if len(sm) > 1 else float("nan")
    return sm.mean(), se, p, len(sm)


def fmt(a):
    m, se, p, n = a
    star = "***" if p < 1e-3 else "**" if p < .01 else "*" if p < .05 else "†" if p < .1 else ""
    return f"{m:+.3f}±{se:.3f}{star}"


print(f"\n=== operand-1 spec by arm (per task; state-clustered) ===")
print(f"{'task':>6} | {'solo n1':>16} | {'joint n2':>16} | {'junk-matched jk':>16} | "
      f"{'n2-n1 (paired)':>16} | {'jk-n1 (paired)':>16} | {'n2-jk (paired)':>16}")
for tag in list(TASKS) + ["ALL"]:
    R = [r for r in recs if tag == "ALL" or r["tag"] == tag]
    if not R:
        continue
    sts = [f"{r['tag']}|{r['s']}" for r in R]
    cols = [agg([r["s1_n1"] for r in R], sts), agg([r["s1_n2"] for r in R], sts),
            agg([r["s1_jk"] for r in R], sts),
            agg([r["s1_n2"] - r["s1_n1"] for r in R], sts),
            agg([r["s1_jk"] - r["s1_n1"] for r in R], sts),
            agg([r["s1_n2"] - r["s1_jk"] for r in R], sts)]
    print(f"{tag:>6} | " + " | ".join(f"{fmt(c):>16}" for c in cols))

print(f"\n=== operand-2 spec: solo (n1b) vs joint (n2) ===")
for tag in list(TASKS) + ["ALL"]:
    R = [r for r in recs if tag == "ALL" or r["tag"] == tag]
    if not R:
        continue
    sts = [f"{r['tag']}|{r['s']}" for r in R]
    cols = [agg([r["s2_n1"] for r in R], sts), agg([r["s2_n2"] for r in R], sts),
            agg([r["s2_n2"] - r["s2_n1"] for r in R], sts)]
    print(f"{tag:>6} | solo {fmt(cols[0]):>16} | joint {fmt(cols[1]):>16} | diff {fmt(cols[2]):>16}")

print(f"\n=== both operands pooled: joint - solo (the parallelism number) ===")
R = recs
sts = [f"{r['tag']}|{r['s']}" for r in R] * 2
vals = [r["s1_n2"] - r["s1_n1"] for r in R] + [r["s2_n2"] - r["s2_n1"] for r in R]
print(f"joint-solo: {fmt(agg(vals, sts))}   "
      f"(retention = joint/solo = "
      f"{np.mean([r['s1_n2'] for r in R] + [r['s2_n2'] for r in R]) / np.mean([r['s1_n1'] for r in R] + [r['s2_n1'] for r in R]):.2f})")

print(f"\n=== commitment-release check: R(ja) per arm (should match n2 vs jk) ===")
sts = [f"{r['tag']}|{r['s']}" for r in recs]
for key, lab in (("dja_n1", "solo"), ("dja_n2", "joint"), ("dja_jk", "junk")):
    vals = [r[key] for r in recs if r[key] is not None]
    ss = [f"{r['tag']}|{r['s']}" for r in recs if r[key] is not None]
    if vals:
        print(f"  {lab:>5}: R(committed answer) = {fmt(agg(vals, ss))}")
