"""Read out par3: (A) n=3 budget ladder UU3/UU5, (B/C) n=2 four-arm battery for
MU2/MU3, RF*, KB*, MN3, P3. Shared placebo reference per (state, rep) across arms:
images of pool operands not injected in ANY arm of the rep, >= DSTAR from every injected
token (sources + junks), excluding ja/jb and injected tokens themselves.
"""
import os
import json, string
from pathlib import Path

import numpy as np
from scipy import stats as st

EXP = Path(os.environ.get("DG_PLANNING_DIR", str(Path(__file__).resolve().parent / "exp")))
FLOOR = 1e-5
LOW = list(string.ascii_lowercase); UPP = list(string.ascii_uppercase); F52 = LOW + UPP
UNITS = ("two three four five six seven eight nine ten eleven twelve thirteen fourteen "
         "fifteen sixteen seventeen eighteen nineteen twenty").split()
TENS = "thirty forty fifty sixty seventy eighty ninety".split()
N26 = UNITS + TENS

par = json.load(open(EXP / "xtask_par3.json"))
dns = json.load(open(EXP / "xtask_samecase_nsweep.json"))
dmun = json.load(open(EXP / "xtask_mult_nsweep.json"))
dopn = json.load(open(EXP / "xtask_ops_nsweep.json"))
d12 = json.load(open(EXP / "xtask_compute12.json"))
dops = json.load(open(EXP / "xtask_ops.json"))
OPMETA = {k.split("|")[0]: v for k, v in dops.items() if k.endswith("|meta")}


def img_shift(field, k):
    return lambda w: (field[field.index(w) + k] if 0 <= field.index(w) + k < len(field) else None)


def img_mu(k):
    return lambda w: (UPP[(UPP.index(w) + 1) * k - 1] if 1 <= (UPP.index(w) + 1) * k <= 26 else None)


FAM = {
    "UU3": dict(kind="n3", field=F52, band=UPP, img=img_shift(UPP, 3), pool=UPP[:23],
                src=dns, seeds=range(10), dstar=4, group="UU3"),
    "UU5": dict(kind="n3", field=F52, band=UPP, img=img_shift(UPP, 5), pool=UPP[:21],
                src=dns, seeds=range(10), dstar=4, group="UU5"),
    "MU2": dict(kind="n2", field=F52, band=UPP, img=img_mu(2), pool=UPP[:13],
                src=dmun, seeds=[0, 1, 2, 5, 6, 7, 8, 9], dstar=4, group="mu x2"),
    "MU3": dict(kind="n2", field=F52, band=UPP, img=img_mu(3), pool=UPP[:8],
                src=dmun, seeds=range(10), dstar=3, group="mu x3"),
    **{tg: dict(kind="n2", field=F52, band=UPP, img=(lambda w, _im=OPMETA[tg]["image"]: _im.get(w)),
                pool=list(OPMETA[tg]["pool"]), src=dopn, seeds=sds, dstar=4, group=grp)
       for tg, sds, grp in (("RF", range(6), "reflect"), ("RFB", range(2), "reflect"),
                            ("RFC", range(2), "reflect"), ("RFD", [0], "reflect"),
                            ("KB", [0], "qwerty"), ("KBB", range(2), "qwerty"),
                            ("MN3", range(2), "minus3"))},
    "P3": dict(kind="n2", field=N26, band=UNITS, img=img_shift(UNITS, 3), pool=UNITS[:16],
               src=d12, seeds=range(4), dstar=4, group="num +3"),
}

recs2, recs3, drop = [], [], 0
for tag, cfg in FAM.items():
    field, band, img = cfg["field"], cfg["band"], cfg["img"]
    pi = {w: i for i, w in enumerate(band)}
    for s in cfg["seeds"]:
        bsrc = par.get(f"{tag}|s{s}|par3|base") or cfg["src"].get(f"{tag}|s{s}|base")
        pairs = par.get(f"{tag}|s{s}|par3|pairs")
        if not bsrc or not pairs:
            continue
        d = json.load(open(EXP / f"nego2/{tag}__s{s}.json"))
        nat, A, B = bsrc["nat_op"], bsrc.get("A", 6), bsrc.get("B", 8)
        bmap = {w: float(v) for w, v in zip(field, np.array(bsrc["rows"]).mean(axis=0))}
        ja = img(nat)
        jb = d["id2str"].get(str(d["final_ids"][B]), "?").replace("▁", " ").strip()
        armnames = ("n1", "j2", "r2j", "r3") if cfg["kind"] == "n3" else ("n1a", "n1b", "n2", "jk")
        for r, rp in enumerate(pairs):
            arms = {}
            for arm in armnames:
                c = par.get(f"{tag}|s{s}|par3|r{r}|{arm}")
                if not c:
                    continue
                if any(v == 0 for v in c["ranks"].values()):
                    drop += 1
                    continue
                amap = {w: float(v) for w, v in zip(field, np.array(c["rows"]).mean(axis=0))}
                arms[arm] = {w: np.log10(max(amap[w], FLOOR) / max(bmap[w], FLOOR)) for w in band}
            if len(arms) < 4:
                continue
            xs, junks = rp["x"], rp["junks"]
            allsrc = [w for w in xs + junks if w in pi]        # lowercase junk has no band pos
            P = [img(y) for y in cfg["pool"]
                 if y not in xs and y != nat and img(y) not in (ja, jb, None)]
            P = [w for w in set(P) - set(xs) - set(junks)
                 if all(abs(pi[w] - pi[sx]) >= cfg["dstar"] for sx in allsrc)]
            tgts = [img(x) for x in xs]
            if len(P) < 2 or any(t not in arms[armnames[0]] for t in tgts):
                continue
            spec = lambda arm, tgt: arms[arm][tgt] - np.mean([arms[arm][w] for w in P])
            state = f"{tag}|{s}"
            if cfg["kind"] == "n2":
                recs2.append(dict(group=cfg["group"], state=state, nP=len(P),
                                  s1_n1=spec("n1a", tgts[0]), s1_n2=spec("n2", tgts[0]),
                                  s1_jk=spec("jk", tgts[0]),
                                  s2_n1=spec("n1b", tgts[1]), s2_n2=spec("n2", tgts[1])))
            else:
                recs3.append(dict(group=cfg["group"], state=state, nP=len(P),
                                  s1_n1=spec("n1", tgts[0]), s1_j2=spec("j2", tgts[0]),
                                  s1_r2j=spec("r2j", tgts[0]), s1_r3=spec("r3", tgts[0]),
                                  s2_r2j=spec("r2j", tgts[1]), s2_r3=spec("r3", tgts[1]),
                                  s3_r3=spec("r3", tgts[2])))

print(f"n2 recs: {len(recs2)}, n3 recs: {len(recs3)}, rank-guard drops: {drop}")


def agg(vals, states):
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


print("\n=== n=2 four-arm battery, other task families (operand-1 spec; state-clustered) ===")
print(f"{'family':>8} | {'solo n1':>15} | {'joint n2':>15} | {'junk-matched':>15} | "
      f"{'n2-n1':>15} | {'jk-n1':>15} | {'n2-jk':>15}")
groups = []
for r in recs2:
    if r["group"] not in groups:
        groups.append(r["group"])
for g in groups + ["ALL"]:
    R = [r for r in recs2 if g == "ALL" or r["group"] == g]
    if not R:
        continue
    sts = [r["state"] for r in R]
    cols = [agg([r["s1_n1"] for r in R], sts), agg([r["s1_n2"] for r in R], sts),
            agg([r["s1_jk"] for r in R], sts),
            agg([r["s1_n2"] - r["s1_n1"] for r in R], sts),
            agg([r["s1_jk"] - r["s1_n1"] for r in R], sts),
            agg([r["s1_n2"] - r["s1_jk"] for r in R], sts)]
    print(f"{g:>8} | " + " | ".join(f"{fmt(c):>15}" for c in cols) + f"   [{len(R)} reps, {len(set(sts))} states]")

print("\n    operand-2: solo vs joint")
for g in groups + ["ALL"]:
    R = [r for r in recs2 if g == "ALL" or r["group"] == g]
    if not R:
        continue
    sts = [r["state"] for r in R]
    print(f"{g:>8} | solo {fmt(agg([r['s2_n1'] for r in R], sts)):>15} | "
          f"joint {fmt(agg([r['s2_n2'] for r in R], sts)):>15} | "
          f"diff {fmt(agg([r['s2_n2'] - r['s2_n1'] for r in R], sts)):>15}")

print("\n=== n=3 budget ladder (UU3/UU5, eps0=0.22, total mass 0.66 in j2/r2j/r3) ===")
print(f"{'task':>5} | {'solo n1@0.22':>15} | {'j2 (1 hyp)':>15} | {'r2j (2 hyp)':>15} | {'r3 (3 hyp)':>15}")
for g in ("UU3", "UU5", "ALL"):
    R = [r for r in recs3 if g == "ALL" or r["group"] == g]
    if not R:
        continue
    sts = [r["state"] for r in R]
    cols = [agg([r["s1_n1"] for r in R], sts), agg([r["s1_j2"] for r in R], sts),
            agg([r["s1_r2j"] for r in R], sts), agg([r["s1_r3"] for r in R], sts)]
    print(f"{g:>5} | " + " | ".join(f"{fmt(c):>15}" for c in cols) + f"   [{len(R)} reps, {len(set(sts))} states]")
R = recs3
sts = [r["state"] for r in R]
print("\nbudget-model ratio test (predicted r2j/j2 = 0.50, r3/j2 = 0.33):")
j2 = np.mean([r["s1_j2"] for r in R])
print(f"  observed r2j/j2 = {np.mean([r['s1_r2j'] for r in R]) / j2:.2f}, "
      f"r3/j2 = {np.mean([r['s1_r3'] for r in R]) / j2:.2f}")
print("  per-operand within r3 (op1/op2/op3): "
      f"{fmt(agg([r['s1_r3'] for r in R], sts))} / {fmt(agg([r['s2_r3'] for r in R], sts))} / "
      f"{fmt(agg([r['s3_r3'] for r in R], sts))}")
print("  sum over ops, r3 vs j2 single: "
      f"{np.mean([r['s1_r3'] + r['s2_r3'] + r['s3_r3'] for r in R]):+.3f} vs {j2:+.3f}")
print("  sum over ops, r2j vs j2 single: "
      f"{np.mean([r['s1_r2j'] + r['s2_r2j'] for r in R]):+.3f} vs {j2:+.3f}")
print("  paired r3-j2: " + fmt(agg([r["s1_r3"] - r["s1_j2"] for r in R], sts))
      + " | r2j-j2: " + fmt(agg([r["s1_r2j"] - r["s1_j2"] for r in R], sts))
      + " | r3-n1: " + fmt(agg([r["s1_r3"] - r["s1_n1"] for r in R], sts)))
