"""Per-cell payload for the interactive sina version of the controlled parallelism curves.
-> exp/dg_planning/par_sina.json : {families:[{label, note, series:[{name, col, eps0,
   pts:[{n, v, tag, s, r, tup, ranks, P, nat, ja}]}]}]}
v = operand-1 spec (placebo- & distance-matched, shared reference per rep). par2/par3 n=2
series include operand-2 dots as well (marked op=2).
"""
import os
import importlib.util, json, string
from pathlib import Path

import numpy as np

PLAN = Path(__file__).resolve().parent
EXP = Path(os.environ.get("DG_PLANNING_DIR", str(Path(__file__).resolve().parent / "exp")))
FLOOR = 1e-5
LOW = list(string.ascii_lowercase); UPP = list(string.ascii_uppercase); F52 = LOW + UPP
UNITS = ("two three four five six seven eight nine ten eleven twelve thirteen fourteen "
         "fifteen sixteen seventeen eighteen nineteen twenty").split()
N26 = UNITS + "thirty forty fifty sixty seventy eighty ninety".split()

par5 = json.load(open(EXP / "xtask_par5.json"))
dns = json.load(open(EXP / "xtask_samecase_nsweep.json"))
dmun = json.load(open(EXP / "xtask_mult_nsweep.json"))
dopn = json.load(open(EXP / "xtask_ops_nsweep.json"))
d12 = json.load(open(EXP / "xtask_compute12.json"))
dpar3 = json.load(open(EXP / "xtask_par3.json"))
dops = json.load(open(EXP / "xtask_ops.json"))
OPMETA = {k.split("|")[0]: v for k, v in dops.items() if k.endswith("|meta")}


def img_shift(field, k):
    return lambda w: (field[field.index(w) + k] if 0 <= field.index(w) + k < len(field) else None)


def img_mu(k):
    return lambda w: (UPP[(UPP.index(w) + 1) * k - 1] if 1 <= (UPP.index(w) + 1) * k <= 26 else None)


FAM5 = {
    "UU3": dict(field=F52, band=UPP, img=img_shift(UPP, 3), pool=UPP[:23], dstar=3,
                src=dns, seeds=range(10), eps0=0.19, group="letters +3"),
    "UU5": dict(field=F52, band=UPP, img=img_shift(UPP, 5), pool=UPP[:21], dstar=4,
                src=dns, seeds=range(10), eps0=0.22, group="letters +5"),
    "MU2": dict(field=F52, band=UPP, img=img_mu(2), pool=UPP[:13], dstar=3,
                src=dmun, seeds=range(10), eps0=0.22, group="letters ×2"),
    "MU3": dict(field=F52, band=UPP, img=img_mu(3), pool=UPP[:8], dstar=3,
                src=dmun, seeds=range(10), eps0=0.22, group="letters ×3"),
    **{tg: dict(field=F52, band=UPP, img=(lambda w, _im=OPMETA[tg]["image"]: _im.get(w)),
                pool=list(OPMETA[tg]["pool"]), dstar=4, src=dopn, seeds=sds, eps0=0.24,
                group="reflection")
       for tg, sds in (("RF", range(6)), ("RFB", range(2)), ("RFC", range(2)), ("RFD", [0]))},
    "P3": dict(field=N26, band=UNITS, img=img_shift(UNITS, 3), pool=UNITS[:16], dstar=3,
               src=d12, seeds=range(4), eps0=0.19, group="numbers +3"),
}

pts5 = {}
for tag, cfg in FAM5.items():
    field, band, img = cfg["field"], cfg["band"], cfg["img"]
    pi = {w: i for i, w in enumerate(band)}
    for s in cfg["seeds"]:
        bsrc = dpar3.get(f"{tag}|s{s}|par3|base") or cfg["src"].get(f"{tag}|s{s}|base")
        pairs = par5.get(f"{tag}|s{s}|par5|pairs")
        if not bsrc or not pairs:
            continue
        d = json.load(open(EXP / f"nego2/{tag}__s{s}.json"))
        nat, B = bsrc["nat_op"], bsrc.get("B", 8)
        bmap = {w: float(v) for w, v in zip(field, np.array(bsrc["rows"]).mean(axis=0))}
        ja = img(nat)
        jb = d["id2str"].get(str(d["final_ids"][B]), "?").replace("▁", " ").strip()
        for r, tup in enumerate(pairs):
            P = [img(y) for y in cfg["pool"] if y not in tup and y != nat
                 and img(y) not in (ja, jb, None)]
            P = sorted(set(P) - set(tup),
                       key=lambda w: pi[w])
            P = [w for w in P if all(abs(pi[w] - pi[sx]) >= cfg["dstar"] for sx in tup)]
            t1 = img(tup[0])
            if len(P) < 2 or t1 is None or t1 in (ja, jb):
                continue
            for m in range(1, len(tup) + 1):
                c = par5.get(f"{tag}|s{s}|par5|r{r}|n{m}")
                if not c or any(v == 0 for v in c["ranks"].values()):
                    continue
                amap = {w: float(v) for w, v in zip(field, np.array(c["rows"]).mean(axis=0))}
                L = {w: np.log10(max(amap[w], FLOOR) / max(bmap[w], FLOOR)) for w in band}
                spec = float(L[t1] - np.mean([L[w] for w in P]))
                pts5.setdefault(cfg["group"], []).append(dict(
                    n=m, v=round(spec, 4), tag=tag, s=s, r=r, op=1,
                    tup=list(c["subset"]), ranks=c["ranks"], P="".join(x[0] for x in P) if cfg["band"] is UPP else ",".join(P),
                    nat=nat, ja=ja))


def load_mod(name):
    spec = importlib.util.spec_from_file_location(name, PLAN / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


p2 = load_mod("xtask_par2_read")
p3 = load_mod("xtask_par3_read")

def mindot(base, n, pairs, junk=0):
    ops = [[w, round(float(v), 4)] for w, v in pairs]
    return dict(base, n=n, v=round(min(v for _, v in ops), 4), ops=ops, junk=junk)


pts2 = {}
for r in p2.recs:
    grp = {"UU3": "letters +3", "UU5": "letters +5", "UU7": "letters +7",
           "UU11": "letters +11"}[r["tag"]]
    base = dict(tag=r["tag"], s=r["s"], r=r["rep"], tup=[r["x1"], r["x2"]],
                ranks={}, P="", nat="", ja="", junk=0, op=0)
    pts2.setdefault(grp, []).append(mindot(base, 1, [(r["x1"], r["s1_n1"])]))
    pts2[grp].append(mindot(base, 1, [(r["x2"], r["s2_n1"])]))
    pts2[grp].append(mindot(base, 2, [(r["x1"], r["s1_n2"]), (r["x2"], r["s2_n2"])]))
    pts2[grp].append(mindot(base, 2, [(r["x1"], r["s1_jk"])], junk=1))

pts3l = {}
for r in p3.recs3:
    grp = {"UU3": "letters +3", "UU5": "letters +5"}[r["group"]]
    tag, s = r["state"].split("|")
    base = dict(tag=tag, s=int(s), r=0, tup=[], ranks={}, P="", nat="", ja="", junk=0, op=0)
    pts3l.setdefault(grp, []).append(mindot(base, 1, [("x1", r["s1_n1"])]))
    pts3l[grp].append(mindot(base, 2, [("x1", r["s1_r2j"]), ("x2", r["s2_r2j"])]))
    pts3l[grp].append(mindot(base, 3, [("x1", r["s1_r3"]), ("x2", r["s2_r3"]), ("x3", r["s3_r3"])]))
    pts3l[grp].append(mindot(base, 3, [("x1", r["s1_j2"])], junk=1))

pts3 = {}
GRP3 = {"mu x2": "letters ×2", "mu x3": "letters ×3", "reflect": "reflection", "num +3": "numbers +3"}
for r in p3.recs2:
    if r["group"] not in GRP3:
        continue
    grp = GRP3[r["group"]]
    tag, s = r["state"].split("|")
    base = dict(tag=tag, s=int(s), r=0, tup=[], ranks={}, P="", nat="", ja="", junk=0, op=0)
    pts3.setdefault(grp, []).append(mindot(base, 1, [(r["x1"] if "x1" in r else "x1", r["s1_n1"])]))
    pts3[grp].append(mindot(base, 1, [("x2", r["s2_n1"])]))
    pts3[grp].append(mindot(base, 2, [("x1", r["s1_n2"]), ("x2", r["s2_n2"])]))
    pts3[grp].append(mindot(base, 2, [("x1", r["s1_jk"])], junk=1))

EPS5 = {"letters +3": 0.19, "letters +5": 0.22, "letters ×2": 0.22, "letters ×3": 0.22,
        "reflection": 0.24, "numbers +3": 0.19}
NOTE = {"letters +3": "", "letters +5": "", "letters +7": "", "letters +11": "",
        "letters ×2": "", "letters ×3": "", "reflection": "", "numbers +3": ""}
EPS3 = {"letters ×2": 0.30, "letters ×3": 0.30, "reflection": 0.30, "numbers +3": 0.25}

fams = []
for grp in ("letters +3", "letters +5", "letters +7", "letters +11",
            "letters ×2", "letters ×3", "reflection", "numbers +3"):
    series = []
    if grp in pts2:
        series.append(dict(name="four-arm ε₀=0.30 · min over members", col="#7aa2ff",
                           eps0=0.30, pts=pts2[grp]))
    if grp in pts3:
        series.append(dict(name=f"four-arm ε₀={EPS3[grp]} · min over members", col="#7aa2ff",
                           eps0=EPS3[grp], pts=pts3[grp]))
    if grp in pts3l:
        series.append(dict(name="ladder ε₀=0.22 total-matched · min over members", col="#e8930c",
                           eps0=0.22, pts=pts3l[grp]))
    fams.append(dict(label=grp, note=NOTE[grp], series=series))

out = EXP / "par_sina.json"
json.dump(dict(families=fams), open(out, "w"))
print(out, f"{out.stat().st_size/1e3:.0f} KB,",
      sum(len(s['pts']) for f in fams for s in f['series']), "dots")
