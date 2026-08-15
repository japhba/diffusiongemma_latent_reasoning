"""Dedicated interactive report for THE superposition result (NE plateau):
reports/dg-planning/symbol_arithmetic.html (renamed from superpos.html 2026-07-28). Multi-k data model (numbers k in {1..11,-3,-8}; letters
k in {3,5,7,11}); pooled NE sinas with three composable controls: means+CI only, split by k
(rows), split by base operand (small-multiple panels). Click any dot -> full cell inspector
(2x2+ratio state-vector grid with gold injected-extra segments, per-draw E sina, provenance).
Data: xtask_compute{8,9,10,12,13,14}.json (merged when present)."""
import os
import json, string
from pathlib import Path

import numpy as np

EXP = Path(os.environ.get("DG_PLANNING_DIR", str(Path(__file__).resolve().parent / "exp")))
OUT = Path(os.environ.get("DG_REPORT_OUT", str(Path(__file__).resolve().parent / "out"))) / "symbol_arithmetic.html"

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("google/gemma-4-26b-a4b-it")

UNITS = ("two three four five six seven eight nine ten eleven twelve thirteen fourteen "
         "fifteen sixteen seventeen eighteen nineteen twenty").split()
TENS = "thirty forty fifty sixty seventy eighty ninety".split()
N26 = UNITS + TENS
LOW = list(string.ascii_lowercase)
UPP = list(string.ascii_uppercase)
F52 = LOW + UPP
FLOOR = 1e-5
T = 2


def load(name):
    p = EXP / f"xtask_{name}.json"
    return json.load(open(p)) if p.exists() else {}


d8, d9, d10, d12, d13 = load("compute8"), load("compute9"), load("compute10"), load("compute12"), load("compute13")
d14 = load("compute14")
dns = load("samecase_nsweep")
dmu = load("mult_nsweep")
# non-translation operation batteries (xtask_ops.py). Each op ships its own image map, so the
# report needs no per-family arithmetic: img_of and the transfer-map rows both look it up.
dops, dopn = load("ops"), load("ops_nsweep")
OPMETA = {k.split("|")[0]: v for k, v in dops.items() if k.endswith("|meta")}
OPLAB = {"RF": "letters reflected (A↔Z)", "KB": "letters QWERTY-successor",
         "CP": "letters identity (copy)", "MN3": "letters −3"}
OPNOTE = {"RF": "affine, slope −1 — not a translation, but full 26-letter pool and a dense image set",
          "KB": "a translation in a DIFFERENT ordering (keyboard, not alphabet)",
          "CP": "identity — ceiling anchor for the specificity scale",
          "MN3": "translation control for letters (subtraction)"}

NQ = ("Pick any number between two and {hi}, write it in words, then write the number "
      "{off} greater in words, separated by a comma. Begin your answer with 'Numbers:'.")
LQ = ("Pick any lowercase letter between a and {hi}, write it, then write the letter "
      "{off} positions later in the alphabet in uppercase, separated by a comma. "
      "Begin your answer with 'Letters:'.")
LLQ = ("Pick any lowercase letter between a and {hi}, write it, then write the letter "
       "{off} positions later in the alphabet, also in lowercase, separated by a comma. "
       "Begin your answer with 'Letters:'.")
UUQ = ("Pick any uppercase letter between A and {hi}, write it, then write the letter "
       "{off} positions later in the alphabet, also in uppercase, separated by a comma. "
       "Begin your answer with 'Letters:'.")
MUQ = ("Pick any uppercase letter from A to {hi}, write it, then multiply its alphabet index "
       "by {k} and write the uppercase letter at that index, separated by a comma. "
       "Begin your answer with 'Letters:'.")

TAGS = {
    "P2":  dict(dom="num", k=2, prompt=NQ.format(hi="eighteen", off="two"),
                srcs=[(d8, UNITS, "deep", "r", range(4)), (d9, N26, "deep", "r", (4, 5)),
                      (d13, N26, "ext", "e", range(3))]),
    "P3":  dict(dom="num", k=3, prompt=NQ.format(hi="seventeen", off="three"),
                srcs=[(d8, UNITS, "deep", "r", range(4)), (d9, N26, "deep", "r", (4, 5)),
                      (d12, N26, "ext", "e", range(8))]),
    "A5":  dict(dom="num", k=5, prompt=NQ.format(hi="fifteen", off="five"),
                srcs=[(d9, N26, "deep", "r", range(5)), (d13, N26, "ext", "e", range(3))]),
    "P6":  dict(dom="num", k=6, prompt=NQ.format(hi="fourteen", off="six"),
                srcs=[(d8, UNITS, "deep", "r", range(4)), (d9, N26, "deep", "r", (4, 5)),
                      (d13, N26, "ext", "e", range(3))]),
    "A7":  dict(dom="num", k=7, prompt=NQ.format(hi="thirteen", off="seven"),
                srcs=[(d9, N26, "deep", "r", range(5)), (d13, N26, "ext", "e", range(3))]),
    "P9":  dict(dom="num", k=9, prompt=NQ.format(hi="eleven", off="nine"),
                srcs=[(d8, UNITS, "deep", "r", range(4)), (d9, N26, "deep", "r", (4, 5)),
                      (d13, N26, "ext", "e", range(3))]),
    "L3":  dict(dom="let", k=3, prompt=LQ.format(hi="w", off="three"),
                srcs=[(d10, F52, "deep", "r", range(4)), (d12, F52, "ext", "e", range(8))]),
    "L7":  dict(dom="let", k=7, prompt=LQ.format(hi="s", off="seven"),
                srcs=[(d10, F52, "deep", "r", range(4)), (d13, F52, "ext", "e", range(3))]),
    "N1":  dict(dom="num", k=1, prompt=NQ.format(hi="nineteen", off="one"),
                srcs=[(d14, N26, "ext", "e", range(3))]),
    "N4":  dict(dom="num", k=4, prompt=NQ.format(hi="sixteen", off="four"),
                srcs=[(d14, N26, "ext", "e", range(3))]),
    "N8":  dict(dom="num", k=8, prompt=NQ.format(hi="twelve", off="eight"),
                srcs=[(d14, N26, "ext", "e", range(3))]),
    "N10": dict(dom="num", k=10, prompt=NQ.format(hi="ten", off="ten"),
                srcs=[(d14, N26, "ext", "e", range(3))]),
    "N11": dict(dom="num", k=11, prompt=NQ.format(hi="nine", off="eleven"),
                srcs=[(d14, N26, "ext", "e", range(3))]),
    "N12": dict(dom="num", k=12, prompt=NQ.format(hi="eight", off="twelve"),
                srcs=[(d14, N26, "ext", "e", range(3))]),
    "S3":  dict(dom="num", k=-3, prompt=("Pick any number between five and twenty, write it in words, "
                "then write the number three less in words, separated by a comma. Begin your answer "
                "with 'Numbers:'."),
                srcs=[(d9, N26, "deep", "r", range(5)), (d14, N26, "ext", "e", range(3))]),
    "S8":  dict(dom="num", k=-8, prompt=("Pick any number between ten and twenty, write it in words, "
                "then write the number eight less in words, separated by a comma. Begin your answer "
                "with 'Numbers:'."),
                srcs=[(d9, N26, "deep", "r", range(5)), (d14, N26, "ext", "e", range(3))]),
    "L5":  dict(dom="let", k=5, prompt=LQ.format(hi="u", off="five"),
                srcs=[(d14, F52, "ext", "e", range(3))]),
    "L11": dict(dom="let", k=11, prompt=LQ.format(hi="o", off="eleven"),
                srcs=[(d14, F52, "ext", "e", range(3))]),
    "LL3": dict(dom="ll", k=3, prompt=LLQ.format(hi="w", off="three"),
                srcs=[(dns, F52, "ext", "e", range(3))]),
    "LL7": dict(dom="ll", k=7, prompt=LLQ.format(hi="s", off="seven"),
                srcs=[(dns, F52, "ext", "e", range(3))]),
    "UU3": dict(dom="uu", k=3, prompt=UUQ.format(hi="W", off="three"),
                srcs=[(dns, F52, "ext", "e", range(3))]),
    "UU5": dict(dom="uu", k=5, prompt=UUQ.format(hi="U", off="five"),
                srcs=[(dns, F52, "ext", "e", range(3))]),
    "UU7": dict(dom="uu", k=7, prompt=UUQ.format(hi="S", off="seven"),
                srcs=[(dns, F52, "ext", "e", range(3))]),
    "UU11": dict(dom="uu", k=11, prompt=UUQ.format(hi="O", off="eleven"),
                srcs=[(dns, F52, "ext", "e", range(3))]),
    "MU2": dict(dom="mu", k=2, prompt=MUQ.format(hi="M", k=2),
                srcs=[(dmu, F52, "ext", "e", range(3))]),
    "MU3": dict(dom="mu", k=3, prompt=MUQ.format(hi="H", k=3),
                srcs=[(dmu, F52, "ext", "e", range(3))]),
    "MU4": dict(dom="mu", k=4, prompt=MUQ.format(hi="F", k=4),
                srcs=[(dmu, F52, "ext", "e", range(3))]),
}
DOMS = {
    "num": dict(label="numbers +k", eps0=0.05, field=N26, scope=UNITS,
                nlvl=[1, 2, 3, 4, 6, 8, 10, 12, 14, 16]),
    "let": dict(label="letters +k (case-flip)", eps0=0.04, field=F52, scope=F52,
                nlvl=[1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20]),
    "ll":  dict(label="letters +k (lower→lower)", eps0=0.04, field=F52, scope=F52,
                nlvl=[1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18]),
    "uu":  dict(label="letters +k (UPPER→UPPER)", eps0=0.04, field=F52, scope=F52,
                nlvl=[1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18]),
    "mu":  dict(label="letters ×k (UPPER→UPPER)", eps0=0.04, field=F52, scope=F52,
                nlvl=[1, 2, 3, 4, 6, 8, 10, 12]),
}
# one dom per non-translation op, driven entirely by its emitted image map
# phrasing variants share an op's map and pool into its dom as extra INDEPENDENT states
# (the sheet seed barely moves the T=2 draft; re-wording the prompt is what varies the canvas)
OPBASE = {"RF": "RF", "RFB": "RF", "RFC": "RF", "RFD": "RF", "RFE": "RF",
          "KB": "KB", "KBB": "KB", "CP": "CP", "MN3": "MN3"}
for _t, _m in OPMETA.items():
    _b = OPBASE.get(_t, _t)
    TAGS[_t] = dict(dom=_b.lower(), k=0, prompt=_m["prompt"],
                    srcs=[(dopn, F52, "ext", "e", range(3))])
    DOMS.setdefault(_b.lower(), dict(label=OPLAB.get(_b, _b), eps0=0.04, field=F52, scope=F52,
                                     nlvl=[1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18],
                                     image=_m["image"], op=_b, note=OPNOTE.get(_b, "")))


def img_of(dom, w, k):
    if dom in DOMS and "image" in DOMS[dom]:      # op doms: explicit map, k unused
        return DOMS[dom]["image"].get(w)
    if dom == "num":
        i = UNITS.index(w) + k
        return UNITS[i] if 0 <= i < len(UNITS) else None
    if dom == "mu":
        # multiplicative: image = letter at alphabet position k TIMES the operand's (A=1..Z=26)
        i = (UPP.index(w) + 1) * k
        return UPP[i - 1] if 1 <= i <= 26 else None
    if dom == "ll":
        i = LOW.index(w) + k
        return LOW[i] if 0 <= i < 26 else None
    if dom == "uu":
        i = UPP.index(w) + k
        return UPP[i] if 0 <= i < 26 else None
    i = LOW.index(w) + k
    return UPP[i] if 0 <= i < 26 else None


r4 = lambda x: round(float(x), 4)


def sheet_vec(d, pos, field, tokid, promos=None, eps=0.0):
    ids = list(d["s_rec"]["ids"][T][pos])
    p = np.exp(np.array(d["s_rec"]["lp"][T][pos], dtype=float))
    if promos:
        p = p * (1.0 - eps * len(promos))
        ids2, p2 = list(ids), p.copy()
        for tk_ in promos:
            if tk_ in ids2:
                p2[ids2.index(tk_)] += eps
            else:
                j = int(np.argmin(p2)); ids2[j] = tk_; p2[j] = eps
        ids, p = ids2, p2
    m = {i: float(v) for i, v in zip(ids, p)}
    return [round(m.get(tokid[w], 0.0), 5) for w in field]


def sheet_top(d, pos, n=10, promos=None, eps=0.0):
    ids = list(d["s_rec"]["ids"][T][pos])
    p = np.exp(np.array(d["s_rec"]["lp"][T][pos], dtype=float))
    injset = set()
    if promos:
        p = p * (1.0 - eps * len(promos))
        for tk_ in promos:
            if tk_ in ids:
                p[ids.index(tk_)] += eps
            else:
                j = int(np.argmin(p)); ids[j] = tk_; p[j] = eps
            injset.add(tk_)
    i2s = d["id2str"]
    dec = lambda i: (i2s.get(str(i)) or tok.decode([i])).replace("▁", " ")
    rows = sorted(zip(ids, p), key=lambda x: -x[1])[:n]
    return [[dec(i), r4(v), 1 if i in injset else 0] for i, v in rows]


def canvas_steps(d):
    """Decoded canvas argmax over all denoising steps, delta-encoded: element 0 = full
    row at t=0 over kept positions, then per-step [col, token] changes. Kept positions =
    0..max(live)+1 capped at 24."""
    i2s = d["id2str"]
    dec = lambda i: (i2s.get(str(i)) or tok.decode([i])).replace("▁", " ")
    dead = set(d["eos_token_ids"]) | {d["pad_token_id"]}
    live = [p for p, x in enumerate(d["final_ids"]) if x not in dead]
    keep = list(range(0, min((max(live) + 2) if live else 12, len(d["final_ids"]), 24)))
    rows = [[dec(r[p]) for p in keep] for r in d["steps_argmax"]]
    out = [rows[0]]
    for t_ in range(1, len(rows)):
        out.append([[j, rows[t_][j]] for j in range(len(keep)) if rows[t_][j] != rows[t_ - 1][j]])
    return keep, out


DATA = {"doms": {}}
for dom, dcfg in DOMS.items():
    field, scope = dcfg["field"], dcfg["scope"]
    tokid = {w: tok.encode(" " + w, add_special_tokens=False)[0] for w in field}
    states, cells, prompts = {}, [], {}
    for tag, tcfg in TAGS.items():
        if tcfg["dom"] != dom:
            continue
        k = tcfg["k"]
        prompts[tag] = tcfg["prompt"]
        caps = {}
        for s in range(10):
            fp = EXP / f"nego2/{tag}__s{s}.json"
            if fp.exists():
                caps[s] = json.load(open(fp))
        for srci, (src, srcfield, kind, rl, reps) in enumerate(tcfg["srcs"]):
            srcname = f"src{srci}"
            for s in range(10):
                bk = f"{tag}|s{s}|base"
                if bk not in src or s not in caps:
                    continue
                d = caps[s]
                nat = src[bk]["nat_op"]
                A = src[bk].get("A", 6); B = src[bk].get("B", 8)
                skey = f"{tag}|{srcname}|s{s}"
                if skey not in states:
                    base_mean = np.array(src[bk]["rows"]).mean(axis=0)
                    bmap = {w: round(float(base_mean[srcfield.index(w)]), 6) for w in srcfield}
                    # jb = the ACTUALLY committed answer at B — equals ja everywhere except
                    # LL3 s0 ("g, k", the model's own off-by-one); excluded like ja
                    jb = d["id2str"].get(str(d["final_ids"][B]), "?").replace("▁", " ").strip()
                    cvp, cv = canvas_steps(d)
                    states[skey] = dict(seed=s, tag=tag, k=k, src=srcname, nat=nat, A=A, B=B,
                                        final=d["final_text"].split("<channel|>")[-1].strip(),
                                        cvp=cvp, cv=cv,
                                        ja=img_of(dom, nat, k), jb=jb,
                                        arow=sheet_top(d, A), brow=sheet_top(d, B),
                                        stA=sheet_vec(d, A, field, tokid),
                                        base={w: bmap.get(w) for w in field})
                for n in dcfg["nlvl"]:
                    for rep in reps:
                        c = src.get(f"{tag}|s{s}|{kind}|n{n}|{rl}{rep}")
                        if not c or any(r == 0 for r in c["ranks"].values()):
                            continue
                        rows = np.array(c["rows"])
                        mm = rows.mean(axis=0)
                        amap = {w: round(float(mm[srcfield.index(w)]), 6) for w in srcfield}
                        ja, jb = states[skey]["ja"], states[skey]["jb"]
                        L = {w: np.log10(max(amap.get(w, FLOOR), FLOOR) /
                                         max(states[skey]["base"].get(w) or FLOOR, FLOOR))
                             for w in scope if w not in (ja, jb)}
                        tj = {img_of(dom, w, k) for w in c["subset"] if img_of(dom, w, k)} - {ja, jb, None}
                        tj = tj & set(L)
                        nt = [w for w in L if w not in tj]
                        if not (tj and nt):
                            continue
                        E = float(np.mean([L[w] for w in tj]) - np.mean([L[w] for w in nt]))
                        perdraw = []
                        for dd in range(rows.shape[0]):
                            am = {w: rows[dd][srcfield.index(w)] for w in srcfield}
                            Ld = {w: np.log10(max(am.get(w, FLOOR), FLOOR) /
                                              max(states[skey]["base"].get(w) or FLOOR, FLOOR))
                                  for w in scope if w not in (ja, jb)}
                            perdraw.append(r4(np.mean([Ld[w] for w in tj]) - np.mean([Ld[w] for w in nt])))
                        cells.append(dict(
                            state=skey, tag=tag, k=k, n=n, rep=f"{srcname}:{rl}{rep}",
                            subset=c["subset"], ranks=c["ranks"], E=r4(E),
                            Enorm=r4(E / (dcfg["eps0"] * n)), perdraw=perdraw,
                            arm={w: amap.get(w) for w in field},
                            stAp=sheet_vec(caps[s], A, field, tokid,
                                           promos=[tokid[w] for w in c["subset"]], eps=dcfg["eps0"])))
    ks = sorted({c["k"] for c in cells})
    DATA["doms"][dom] = dict(label=dcfg["label"], eps0=dcfg["eps0"], nlvl=dcfg["nlvl"],
                             field=field, scope=scope, ks=ks, prompts=prompts,
                             states=states, cells=cells,
                             **({"image": dcfg["image"], "op": dcfg["op"], "note": dcfg["note"]}
                                if "image" in dcfg else {}))
    print(dom, len(cells), "cells,", len(states), "states, ks:", ks)

# ---- natural-operand histogram: which letter the model commits at A, per capture
DATA["nathist"] = {}
for dom in ("uu", "mu", "let", "ll") + tuple(sorted({v.lower() for v in OPBASE.values()})):
    hist = {}
    for tag, tcfg in TAGS.items():
        if tcfg["dom"] != dom:
            continue
        for s_ in range(10):
            fp = EXP / f"nego2/{tag}__s{s_}.json"
            if not fp.exists():
                continue
            d_ = json.load(open(fp))
            i2s, fin = d_["id2str"], d_["final_ids"]
            dead = set(d_["eos_token_ids"]) | {d_["pad_token_id"]}
            live = [p for p, x in enumerate(fin) if x not in dead]
            txt_ = lambda x: i2s.get(str(x), "?").replace("▁", " ")
            pc = next((p for p in live if txt_(fin[p]).strip() == ","), None)
            if pc is None:
                continue
            natl = txt_(fin[max(p for p in live if p < pc)]).strip()
            hist.setdefault(tag, {})[natl] = hist.setdefault(tag, {}).get(natl, 0) + 1
    DATA["nathist"][dom] = hist
print("nathist:", {dm: {tg: dict(cs) for tg, cs in h.items()} for dm, h in DATA["nathist"].items()})

# ---- n=1 transfer-map payload: numbers = compute4 sub arm (eps=.3), letters = n=1 superpos cells
r6 = lambda x: round(float(x), 6)
d4 = load("compute4")
NUMS12 = "two three four five six seven eight nine ten eleven twelve thirteen".split()
K4 = {"C4": 2, "C2": 3, "C3": 4}
Q4 = ("Pick any number between two and nine, write it in words, then write the number "
      "{off} greater in words, separated by a comma.")
tm_num = dict(field=NUMS12, cols=NUMS12[:8], eps=0.3, draws=12, states={}, cells=[],
              prompts={"C4": Q4.format(off="two"), "C2": Q4.format(off="three"), "C3": Q4.format(off="four")})
for tag in ("C4", "C2", "C3"):
    for s, t_ in ((0, 1), (0, 2), (1, 2), (2, 2)):
        bk = f"{tag}|s{s}|t{t_}|base"
        if bk not in d4:
            continue
        nat = d4[bk]["nat_op"]
        stk = f"{tag}|s{s}|t{t_}"
        tm_num["states"][stk] = dict(k=K4[tag], nat=nat, ja=NUMS12.index(nat) + K4[tag],
                                     base=[r6(v) for v in np.array(d4[bk]["rows"]).mean(axis=0)])
        for xw in NUMS12[:8]:
            cell = d4.get(f"{tag}|s{s}|t{t_}|sub|{xw}")
            if not cell or cell.get("skipped") or xw == nat:
                continue
            tm_num["cells"].append(dict(st=stk, x=xw,
                                        arm=[r6(v) for v in np.array(cell["rows"]).mean(axis=0)]))
# ---- letters transfer maps: the 3 task variants at the largest eps that stays subleading
# (case-flip from xtask_eps_sweep, lower->lower + UPPER->UPPER from xtask_samecase; k in {3,7},
# seeds 0-1, every basis; the rank>=1 guard rejects injections that would become the leader,
# so eps_max = largest candidate with >=70% surviving cells)
dsw, dsc = load("eps_sweep"), load("samecase")
LVAR = [("flip", ("L3", "L7"), dsw), ("LL", ("LL3", "LL7"), dsc), ("UU", ("UU3", "UU5", "UU7", "UU11"), dsc)]
surv = {}
for e in (0.45, 0.316228, 0.177828):
    st_ = [c for src in (dsw, dsc) for ck, c in src.items() if f"|e{e}|b1|" in ck]
    surv[e] = (len(st_), sum(1 for c in st_ if not c.get("skipped")))
EPSL = next(e for e, (tot, ok) in surv.items() if tot and ok / tot >= 0.7)
print("letters eps survival:", {e: f"{ok}/{tot}" for e, (tot, ok) in surv.items()}, "-> eps_max", EPSL)

LLQ = ("Pick any lowercase letter between a and {hi}, write it, then write the letter "
       "{off} positions later in the alphabet, also in lowercase, separated by a comma. "
       "Begin your answer with 'Letters:'.")
UUQ = ("Pick any uppercase letter between A and {hi}, write it, then write the letter "
       "{off} positions later in the alphabet, also in uppercase, separated by a comma. "
       "Begin your answer with 'Letters:'.")
KV = {"L3": 3, "L7": 7, "LL3": 3, "LL7": 7, "UU3": 3, "UU5": 5, "UU7": 7, "UU11": 11}


def committed_B(tag, s):
    """The letter actually committed at answer position B (can differ from the arithmetic
    image when the model's own arithmetic slipped: LL3 s0 wrote 'g, k')."""
    d = json.load(open(EXP / f"nego2/{tag}__s{s}.json"))
    i2s, fin = d["id2str"], d["final_ids"]
    dead = set(d["eos_token_ids"]) | {d["pad_token_id"]}
    live = [p for p, x in enumerate(fin) if x not in dead]
    txt = lambda x: i2s.get(str(x), "?").replace("▁", " ")
    pc = next(p for p in live if txt(fin[p]).strip() == ",")
    return txt(fin[next(p for p in live if p > pc)]).strip()


tm_let = dict(field=F52, cols=LOW[:23], eps=EPSL, draws=8, states={}, cells=[],
              prompts={"L3": TAGS["L3"]["prompt"], "L7": TAGS["L7"]["prompt"],
                       "LL3": LLQ.format(hi="w", off="three"), "LL7": LLQ.format(hi="s", off="seven"),
                       "UU3": UUQ.format(hi="W", off="three"), "UU5": UUQ.format(hi="U", off="five"),
                       "UU7": UUQ.format(hi="S", off="seven"), "UU11": UUQ.format(hi="O", off="eleven")})
for v, vtags, src in LVAR:
    for tag in vtags:
        k_ = KV[tag]
        for s in range(10):
            bk = f"{tag}|s{s}|base"
            if bk not in src:
                continue
            nat = src[bk]["nat"]
            ja = (LOW.index(nat.lower()) + k_) if v == "LL" else (26 + LOW.index(nat.lower()) + k_)
            natb = committed_B(tag, s)
            stk = f"{tag}|hi|s{s}"
            tm_let["states"][stk] = dict(k=k_, v=v, nat=nat, ja=ja,
                                         jb=F52.index(natb) if natb in F52 else ja,
                                         base=[r6(x) for x in np.array(src[bk]["rows"]).mean(axis=0)])
            pref = f"{tag}|s{s}|e{EPSL}|b1|"
            for ck, cell in src.items():
                if not ck.startswith(pref) or cell.get("skipped"):
                    continue
                tm_let["cells"].append(dict(st=stk, x=cell["subset"][0],
                                            arm=[r6(x) for x in np.array(cell["rows"]).mean(axis=0)]))
# drop states whose cells were all rejected at eps_max (e.g. LL7: weak sheet leader, every
# injection >= 0.316 would take the lead) so no empty all-hatched map group renders
ref = {c["st"] for c in tm_let["cells"]}
tm_let["states"] = {sk: sv for sk, sv in tm_let["states"].items() if sk in ref}

# ---- MULTIPLICATIVE transfer maps: same eps_max protocol, but the image map is no longer a
# translation, so map rows are the PREIMAGE x'/k (the source whose image is x'); outputs whose
# position is not divisible by k (or out of pool) fall into 'other'.
dmm = load("mult")
survm = {}
for e in (0.45, 0.316228, 0.177828):
    st_ = [c for ck, c in dmm.items() if f"|e{e}|b1|" in ck]
    survm[e] = (len(st_), sum(1 for c in st_ if not c.get("skipped")))
EPSM = next((e for e, (tot, ok) in survm.items() if tot and ok / tot >= 0.7), 0.45)
print("mult eps survival:", {e: f"{ok}/{tot}" for e, (tot, ok) in survm.items()}, "-> eps_max", EPSM)
KVM = {"MU2": 2, "MU3": 3, "MU4": 4}
tm_mu = dict(field=F52, cols=LOW[:13], eps=EPSM, draws=8, states={}, cells=[],
             prompts={tg: TAGS[tg]["prompt"] for tg in KVM})
for tag, k_ in KVM.items():
    for s in range(10):
        bk = f"{tag}|s{s}|base"
        if bk not in dmm:
            continue
        nat = dmm[bk]["nat"]
        im = img_of("mu", nat, k_)
        ja = 26 + UPP.index(im) if im else -1
        natb = committed_B(tag, s)
        stk = f"{tag}|hi|s{s}"
        tm_mu["states"][stk] = dict(k=k_, nat=nat, ja=ja,
                                    jb=F52.index(natb) if natb in F52 else ja,
                                    base=[r6(x) for x in np.array(dmm[bk]["rows"]).mean(axis=0)])
        pref = f"{tag}|s{s}|e{EPSM}|b1|"
        for ck, cell in dmm.items():
            if not ck.startswith(pref) or cell.get("skipped"):
                continue
            tm_mu["cells"].append(dict(st=stk, x=cell["subset"][0],
                                       arm=[r6(x) for x in np.array(cell["rows"]).mean(axis=0)]))
refm = {c["st"] for c in tm_mu["cells"]}
tm_mu["states"] = {sk: sv for sk, sv in tm_mu["states"].items() if sk in refm}

# ---- non-translation op transfer maps. Rows are supplied as an explicit rowmap (output letter
# -> row index) built from each op's inverse map, so the renderer needs no per-family arithmetic.
tm_ops = {}
_BYBASE = {}
for _tg in OPMETA:
    _BYBASE.setdefault(OPBASE.get(_tg, _tg), []).append(_tg)
for tg, meta in OPMETA.items():
    if OPBASE.get(tg, tg) != tg:
        continue          # variant: folded into its base op's map below
    pool, image = meta["pool"], meta["image"]
    inv = {image[w]: w for w in pool if image.get(w)}
    other = len(pool)
    rowmap = [pool.index(inv[c]) if c in inv else other for c in F52]
    vtags_ = _BYBASE[tg]
    epso = 0.45
    for _e in (0.45, 0.316228):
        _st = [c for ck, c in dops.items()
               if any(ck.startswith(f"{vt}|s") for vt in vtags_) and f"|e{_e}|b1|" in ck]
        if _st and sum(1 for c in _st if not c.get("skipped")) / len(_st) >= 0.7:
            epso = _e
            break
    tm = dict(field=F52, cols=[c.lower() for c in pool], rowmap=rowmap,
              rlab=pool + ["other"], eps=epso, draws=8, states={}, cells=[],
              prompts={tg: meta["prompt"]}, note=OPNOTE.get(tg, ""), label=OPLAB.get(tg, tg))
    for vtag in _BYBASE[tg]:
      for s in range(10):
        bk = dops.get(f"{vtag}|s{s}|base")
        if not bk:
            continue
        im = image.get(bk["nat"])
        natb = committed_B(vtag, s)
        stk = f"{vtag}|hi|s{s}"
        tm["states"][stk] = dict(k=0, nat=bk["nat"], ja=F52.index(im) if im in F52 else -1,
                                 jb=F52.index(natb) if natb in F52 else -1,
                                 base=[r6(x) for x in np.array(bk["rows"]).mean(axis=0)])
        pref = f"{vtag}|s{s}|e{epso}|b1|"
        for ck, cell in dops.items():
            if not ck.startswith(pref) or cell.get("skipped"):
                continue
            tm["cells"].append(dict(st=stk, x=cell["subset"][0],
                                    arm=[r6(x) for x in np.array(cell["rows"]).mean(axis=0)]))
    refo = {c["st"] for c in tm["cells"]}
    tm["states"] = {sk: sv for sk, sv in tm["states"].items() if sk in refo}
    if tm["cells"]:
        tm_ops[tg.lower()] = tm
        print(f"tmap op {tg}: {len(tm['cells'])} cells, {len(tm['states'])} states, eps {epso}")

DATA["tmap"] = {"num": tm_num, "let": tm_let, "mu": tm_mu, **tm_ops}
print(f"tmap: num {len(tm_num['cells'])} cells, let {len(tm_let['cells'])} cells "
      f"({len(tm_let['states'])} states, eps {EPSL}), mu {len(tm_mu['cells'])} cells "
      f"({len(tm_mu['states'])} states, eps {EPSM})")

OPSFIND = r"""
<div class="capbox"><b>What the operation sweep shows</b> (&epsilon;=0.316, the dose at which
<i>no</i> family is filtered by the rank&ge;1 guard; placebo-corrected specificity, clustered by
(task, basis), 253 units).
<ul style="margin:6px 0 4px">
<li><b>Three tiers.</b> <b>identity</b> +1.375&plusmn;0.118 (90% argmax) &gt;
<b>subtraction &minus;3</b> +0.668&plusmn;0.133, <b>addition +k</b> +0.573&plusmn;0.151,
<b>reflection 27&minus;pos</b> +0.435&plusmn;0.057 &gt; <b>QWERTY successor</b> +0.228&plusmn;0.059,
<b>multiplication k&middot;x</b> +0.094&plusmn;0.075. Within the middle tier nothing separates:
+k vs &minus;3 p=0.64, +k vs reflection p=0.40, &minus;3 vs reflection p=0.12. Within the bottom
tier likewise: QWERTY vs &times;k p=0.17.</li>
<li><b>It is not a shift operator.</b> Reflection inverts the alphabet (slope &minus;1) and transfers
as well as a shift &mdash; so the channel is not restricted to translations.</li>
<li><b>But it is not "any simple map" either &mdash; the QWERTY result is the decisive one.</b>
Keyboard-successor <i>is</i> a translation, merely in a different coordinate, and it falls to the
bottom tier alongside multiplication (&minus;0.451 vs &minus;0.473 relative to +k in the ANCOVA
below; the two are indistinguishable from each other). So what the channel privileges is not
translation, and not affineness in the abstract, but <b>affine maps of the alphabet index in
particular</b>: the operand appears to be carried in an alphabet-ordered coordinate, and only
operations that are affine <i>in that coordinate</i> survive it.</li>
<li><b>Commitment is a real covariate, and controlling for it does not change the ordering.</b> At a
fixed &epsilon; a weaker incumbent means a relatively larger injection, and leader mass indeed enters
negatively (&minus;0.703&plusmn;0.258, p=0.007). Regressing specificity on leader mass <i>and</i>
operation class (baseline +k, n=253 units) leaves: &times;k &minus;0.473&plusmn;0.156 (p=0.003),
QWERTY &minus;0.451&plusmn;0.138 (p=0.001), reflection &minus;0.158&plusmn;0.114 (<b>p=0.17, n.s.</b>),
&minus;3 +0.118&plusmn;0.161 (p=0.46, n.s.), identity +0.589&plusmn;0.173 (p=8&times;10<sup>&minus;4</sup>).</li>
<li><b>&times;3 mod 26 was dropped as NO-GO</b> (2/8 correct): DG answers it with a stable but wrong
map (G&rarr;O where 7&times;3=21&rarr;U in 6/8 seeds), so a null there would measure its arithmetic,
not the channel. This is why every operation is capability-probed before it is run.</li>
</ul>
Companion figure: <span class="mono">figs/xtask_ops.png</span>.</div>
<img class="fig" src="figs/xtask_ops.png" alt="operations compared over the S^t channel">
"""

# ---- HTML fragment for the non-translation op families (built from whatever data exists)
OPSEC = ""
for _t in ("RF", "KB", "CP", "MN3"):
    _dm = _t.lower()
    _has_map, _has_ne = _dm in tm_ops, bool(DATA["doms"].get(_dm, {}).get("cells"))
    if not (_has_map or _has_ne):
        continue
    OPSEC += (f'<h3>{OPLAB[_t]} <span class="dim">&mdash; {OPNOTE[_t]}</span></h3>'
              f'<p class="dim mono" style="font-size:.78em">{OPMETA[_t]["prompt"]}</p>')
    if _has_map:
        OPSEC += (f'<div class="card"><div id="tmap_{_dm}"></div>'
                  f'<div id="tmap_{_dm}S" style="margin-top:16px"></div></div>')
    if _has_ne:
        OPSEC += (f'<details><summary class="dim">Deprecated NE(n) sweep, &epsilon;&#8320;=0.04 '
                  f'({len(DATA["doms"][_dm]["cells"])} cells, '
                  f'{len(DATA["doms"][_dm]["states"])} states) &mdash; superseded by the controlled '
                  f'curves in the n&gt;1 section (RF: see the per-family curve; CP/KB/MN3: no valid '
                  f'confound-free regime beyond n=2&ndash;3)</summary>'
                  f'<div class="card" id="{_dm}"></div></details>')
    OPSEC += (f'<h4 class="dim">natural operand choice</h4>'
              f'<div class="card" id="nathist_{_dm}"></div>')
if OPSEC:
    OPSEC = ('<h2 style="margin-top:30px">Non-translation operations</h2>'
             '<p class="dim">Addition and subtraction are both <b>translations</b> of the alphabet '
             'index, so "+k transfers, &times;k barely does" has two readings: S<sup>t</sup> carries '
             'the operand in a representation where <i>translation specifically</i> is cheap, or '
             'harder computations simply transfer worse. These operations separate them. Each was '
             'capability-probed first (8 seeds); <span class="mono">&times;3 mod 26</span> was '
             'dropped as NO-GO &mdash; DG answers it with a stable but <i>wrong</i> map '
             '(G&rarr;O where 7&times;3=21&rarr;U, 6/8 seeds), so a diagonal there would measure '
             'its arithmetic error rather than transfer. Maps use the same &epsilon;=0.45 dose and '
             'the same rank&ge;1 subleading guard as every family above; rows are the <b>operand '
             "whose image is that output</b> (the preimage under each op's own map).</p>") + OPSEC + OPSFIND

payload = json.dumps(DATA, separators=(",", ":"))
print(f"payload {len(payload)/1e6:.2f} MB")

STYLE = Path("build_seasonal.py").read_text().split('STYLE = """')[1].split('"""')[0]
EXTRA = """
img.fig{max-width:min(1080px,100%);display:block;margin:10px auto;border-radius:8px}
.card{background:#fff}
@media (prefers-color-scheme: dark){ .card{background:var(--card)} }
html[data-theme=light] .card{background:#fff}
html[data-theme=dark] .card{background:var(--card)}
.chwrap{display:inline-block;vertical-align:top}
.chttl{font-size:.75em;color:var(--dim);margin:0 0 2px 46px;max-width:540px}
.chylab{writing-mode:vertical-rl;transform:rotate(180deg);font-size:.75em;color:var(--dim);display:flex;align-items:center;justify-content:center}
.chxlab{font-size:.75em;color:var(--dim);text-align:center}
.chttl .katex,.chylab .katex,.chxlab .katex{font-size:1.02em}
.sinawrap{overflow-x:auto}
rect.tmc.tmhl{stroke:var(--fg);stroke-width:1.7}
svg text{fill:var(--fg);font:11px ui-monospace,Menlo,monospace}
svg .dimt{fill:var(--dim)}
.dot{cursor:pointer}
.dot.sel{stroke:var(--accent);stroke-width:2.5px}
.tokpill.inj{background:#e8590c33;box-shadow:inset 0 0 0 1.5px #e8590c}
.det{border-left:4px solid var(--accent)}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:.82em;color:var(--dim);margin:4px 0}
.sw{display:inline-block;width:11px;height:11px;border-radius:2px;vertical-align:-1px;margin-right:4px}
pre.algo{font:11.5px/1.45 ui-monospace,Menlo,monospace;}
.c-com{color:#868e96;font-style:italic}.c-kw{color:#d6336c}.c-str{color:#2f9e44}.c-num{color:#1971c2}
pre.algo{background:var(--codebg);border:1px solid var(--line);border-radius:8px;padding:10px 14px;overflow-x:auto;margin:8px 0 2px}
.capbox{font-size:.88em;background:var(--codebg);border:1px solid var(--line);border-radius:8px;padding:10px 14px;margin:8px 0}
.krow{margin:2px 0 10px}
.krow h3{margin:.4em 0 .2em}
"""

JS = r"""
const D = window.__DATA__;
const $ = id => document.getElementById(id);
function mulberry32(a){return function(){a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);
  t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;}}
function esc(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;');}
function vir(v){ const st=[[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]];
  const x=Math.max(0,Math.min(1,v))*(st.length-1), i=Math.min(st.length-2,Math.floor(x)), f=x-i;
  const c=st[i].map((a,j)=>Math.round(a+(st[i+1][j]-a)*f)); return `rgb(${c[0]},${c[1]},${c[2]})`;}
const thBtn=$('themeToggle');
function setTheme(m){ if(m) document.documentElement.setAttribute('data-theme',m);
  else document.documentElement.removeAttribute('data-theme');
  thBtn.textContent=(m||'auto')+' theme'; localStorage.setItem('spTheme',m||'');}
thBtn.onclick=()=>{const c=localStorage.getItem('spTheme')||'';setTheme(c===''?'dark':(c==='dark'?'light':''));};
setTheme(localStorage.getItem('spTheme')||'');
const SEEDCOL={0:'#1971c2',1:'#2f9e44',2:'#e8590c',3:'#9c36b5'};
function KCOL(dom,k){ const ks=D.doms[dom].ks; const i=ks.indexOf(k);
  return vir(ks.length>1? i/(ks.length-1) : 0.5); }
let sel=null;

function imgOf(dom,w,k){
  if(D.doms[dom]&&D.doms[dom].image) return D.doms[dom].image[w]||null;
  if(dom==='num'){ const U=D.doms.num.scope; const i=U.indexOf(w)+k; return (i>=0&&i<U.length)?U[i]:null; }
  const lo='abcdefghijklmnopqrstuvwxyz', up='ABCDEFGHIJKLMNOPQRSTUVWXYZ';
  if(dom==='ll'){ const i=lo.indexOf(w)+k; return (i>=0&&i<26)?lo[i]:null; }
  if(dom==='uu'){ const i=up.indexOf(w)+k; return (i>=0&&i<26)?up[i]:null; }
  if(dom==='mu'){ const i=(up.indexOf(w)+1)*k; return (i>=1&&i<=26)?up[i-1]:null; }
  const i=lo.indexOf(w)+k; return (i>=0&&i<26)?up[i]:null;
}
function kfmt(k,dom){ if(D.doms[dom]&&D.doms[dom].op) return D.doms[dom].op;
  return dom==='mu'?('×'+k):((k>=0?'+':'')+k); }
function nlevels(cells){ return [...new Set(cells.map(c=>c.n))].sort((a,b)=>a-b); }

// ---- switchable NE metric ----
function bandSets(dom){
  if(dom==='num'){ const sc=D.doms.num.scope;
    return {near:sc, far:D.doms.num.field.filter(w=>!sc.includes(w))}; }
  const f=D.doms[dom].field;
  const upr=f.filter(w=>w>='A'&&w<='Z'), lwr=f.filter(w=>w>='a'&&w<='z');
  return dom==='ll'?{near:lwr, far:upr}:{near:upr, far:lwr};
}
function termLbl(dom,which){
  return which==='T'?'⟨R⟩_T'
       : which==='mixed'?'⟨R⟩_N'
       : which==='near'?(dom==='num'?'⟨R⟩_unitNT':(dom==='ll'?'⟨R⟩_lowerNT':'⟨R⟩_upperNT'))
       : (dom==='num'?'⟨R⟩_tens':(dom==='ll'?'⟨R⟩_upper':'⟨R⟩_lower'));
}
function eDefNames(dom){ return {r1:termLbl(dom,$('eT').value), r2:termLbl(dom,$('eN').value)}; }
function eParts(dom,c){
  const w1=$('eT').value, w2=$('eN').value, t=D.doms[dom], st=t.states[c.state];
  const R={}; t.field.forEach(w=>{ R[w]=Math.log10(Math.max(c.arm[w]||0,1e-5)/Math.max(st.base[w]||0,1e-5)); });
  const tjs=new Set(c.subset.map(w=>imgOf(dom,w,c.k)).filter(Boolean));
  const {near,far}=bandSets(dom);
  const vals=which=>{
    if(which==='T') return [...tjs].filter(w=>w!==st.ja&&w!==st.jb&&R[w]!==undefined).map(w=>R[w]);
    if(which==='mixed') return t.scope.filter(w=>!tjs.has(w)&&w!==st.ja&&w!==st.jb).map(w=>R[w]);
    if(which==='near') return near.filter(w=>!tjs.has(w)&&w!==st.ja&&w!==st.jb).map(w=>R[w]);
    return far.filter(w=>!tjs.has(w)&&w!==st.jb).map(w=>R[w]);
  };
  const v1=vals(w1), v2=vals(w2);
  const mean=a=>a.length?a.reduce((x,y)=>x+y,0)/a.length:null;
  return {m1:mean(v1), m2:mean(v2),
          n1:v1.length?Math.min(...v1):null, x1:v1.length?Math.max(...v1):null,
          n2:v2.length?Math.min(...v2):null, x2:v2.length?Math.max(...v2):null};
}

function niceTicks(lo,hi){
  const span=Math.max(hi-lo,1e-9);
  const step=[0.01,0.02,0.05,0.1,0.2,0.25,0.5,1,2,2.5,5,10].find(s=>span/s<=7)||10;
  const out=[]; for(let v=Math.ceil(lo/step)*step; v<=hi+1e-9; v+=step) out.push(+v.toFixed(4));
  return out;
}
function chartSVG(dom, cells, title, series, ylab){
  const t=D.doms[dom], meansOnly=$('meansOnly').checked, multi=series.length>1;
  const nlv=nlevels(cells);
  const step=multi?58:48, W=Math.max(420, nlv.length*step+118), H=300, L=58, Bm=38, Tm=18;
  const byN={}; nlv.forEach(n=>byN[n]=[]);
  cells.forEach(c=>byN[c.n].push(c));
  const vals=series.map(sr=>{ const m={}; nlv.forEach(n=>{
    m[n]=byN[n].map(c=>({c,v:sr.val(c)})).filter(o=>o.v!=null); }); return m; });
  const stats=series.map((sr,si)=>{ const st={}; nlv.forEach(n=>{ const v=vals[si][n].map(o=>o.v);
    if(!v.length) return;
    const m=v.reduce((a,b)=>a+b,0)/v.length;
    const se=v.length>1?Math.sqrt(v.reduce((a,b)=>a+(b-m)*(b-m),0)/(v.length-1))/Math.sqrt(v.length):0;
    st[n]={m, ci:1.96*se, k:v.length}; }); return st; });
  let lo,hi;
  if(meansOnly){ const ms=[].concat(...stats.map(st=>Object.values(st)));
    lo=Math.min(0,...ms.map(s=>s.m-s.ci)); hi=Math.max(...ms.map(s=>s.m+s.ci));
  } else { const av=[].concat(...series.map((sr,si)=>[].concat(...nlv.map(n=>vals[si][n].map(o=>o.v)))));
    lo=Math.min(0,...av); hi=Math.max(...av); }
  const pad=(hi-lo)*0.06+1e-9, y=v=>Tm+(1-(v-(lo-pad))/((hi+pad)-(lo-pad)))*(H-Tm-Bm);
  const x=ni=>L+32+ni*step;
  let s=`<svg width="${W}" height="${H}">`;
  s+=`<line x1="${L}" x2="${W-8}" y1="${y(0)}" y2="${y(0)}" stroke="var(--dim)" stroke-dasharray="4 3"/>`;
  for(const g of niceTicks(lo-pad,hi+pad)) s+=`<line x1="${L}" x2="${W-8}" y1="${y(g)}" y2="${y(g)}" stroke="var(--dim)" opacity="0.15"/><text class="dimt" x="${L-6}" y="${y(g)+3.5}" text-anchor="end" font-size="9">${g}</text>`;
  series.forEach((sr,si)=>{
    const off=multi?(si===0?-13:13):0, halfw=multi?11:24, dotr=multi?2.6:3.4;
    nlv.forEach((n,ni)=>{
      const os=vals[si][n]; if(!os.length) return;
      if(si===0) s+=`<text class="dimt" x="${x(ni)}" y="${H-22}" text-anchor="middle">${n}</text>`;
      if(meansOnly){ const st_=stats[si][n]; const cx=x(ni)+off;
        const col=sr.col||'var(--accent)';
        s+=`<line x1="${cx}" x2="${cx}" y1="${y(st_.m-st_.ci)}" y2="${y(st_.m+st_.ci)}" stroke="${col}" stroke-width="1.6"/>`;
        for(const e of [-1,1]) s+=`<line x1="${cx-6}" x2="${cx+6}" y1="${y(st_.m+e*st_.ci)}" y2="${y(st_.m+e*st_.ci)}" stroke="${col}" stroke-width="1.6"/>`;
        return; }
      const v=os.map(o=>o.v), vmin=Math.min(...v), vmax=Math.max(...v);
      const srt=[...v].sort((a,b)=>a-b), q=p=>srt[Math.min(srt.length-1,Math.floor(p*(srt.length-1)))];
      const bw=0.05*(vmax-vmin+1e-9);
      const dens=u=>v.reduce((a,w)=>a+Math.exp(-0.5*((u-w)/bw)**2),0);
      const dmax=Math.max(...v.map(dens));
      s+=`<rect x="${x(ni)+off-halfw}" y="${y(vmax)}" width="${2*halfw}" height="${Math.max(1,y(vmin)-y(vmax))}" fill="none" stroke="var(--dim)" opacity="0.7"/>`;
      s+=`<line x1="${x(ni)+off-halfw}" x2="${x(ni)+off+halfw}" y1="${y(q(0.5))}" y2="${y(q(0.5))}" stroke="var(--fg)" stroke-width="1.5"/>`;
      for(const p of [0.25,0.75]) s+=`<line x1="${x(ni)+off-halfw*0.45}" x2="${x(ni)+off+halfw*0.45}" y1="${y(q(p))}" y2="${y(q(p))}" stroke="var(--dim)"/>`;
      const rnd=mulberry32(11+ni+97*si);
      os.forEach(o=>{ const jit=(rnd()-0.5)*2*(dens(o.v)/dmax)*(halfw-2);
        const st_=D.doms[dom].states[o.c.state], seed=st_.seed;
        const col=sr.col||KCOL(dom,o.c.k);
        const tt=`${sr.name} = ${o.v.toFixed(3)}\nk = ${kfmt(o.c.k,dom)} (${o.c.tag})   s = ${seed}   t = 2   subset draw ${o.c.rep}\nx_nat = '${st_.nat}' → x'_nat = '${st_.ja}'   positions A=${st_.A} B=${st_.B}\npert @A (n=${o.c.n}, ε₀=${t.eps0}, Σε=${(t.eps0*o.c.n).toFixed(2)}): ${o.c.subset.map(w=>w+'(r'+(o.c.ranks[w]+1)+')').join(' ')}\nclick for full cell inspector`;
        s+=`<circle class="dot${sel&&sel[0]===dom&&sel[1]===o.c._gi?' sel':''}" data-dom="${dom}" data-ci="${o.c._gi}" cx="${x(ni)+off+jit}" cy="${y(o.v)}" r="${dotr}" fill="${col}" fill-opacity="0.42"><title>${tt}</title></circle>`;});
    });
    const mpts=[];
    nlv.forEach((n,ni)=>{ const st_=stats[si][n]; if(st_) mpts.push([x(ni)+off,y(st_.m),n,st_.m]); });
    const mcol=sr.col||'var(--accent)';
    s+=`<polyline fill="none" stroke="${mcol}" stroke-width="2" points="${mpts.map(p=>p[0]+','+p[1]).join(' ')}"/>`;
    for(const p of mpts) s+=`<circle cx="${p[0]}" cy="${p[1]}" r="3.6" fill="${mcol}" stroke="var(--bg)" stroke-width="1.2"><title>mean ${sr.name} at n=${p[2]}: ${p[3].toFixed(3)}</title></circle>`;
  });
  s+=`</svg>`;
  return `<div class="chwrap">${title?`<div class="chttl">${title}</div>`:''}`+
    `<div style="display:flex;align-items:stretch"><div class="chylab"><span>${ylab}</span></div>${s}</div>`+
    `<div class="chxlab">⟦n⟧ simultaneous subleading injections (⟦\\varepsilon_0=${t.eps0}⟧ each)</div></div>`;
}
function triptych(dom, cs, gt, variant){
  const eT=$('eT').value, eN=$('eN').value, v=variant||'mean';
  const CFG={
    mean:  {a1:'mean',a2:'mean',f1:c=>c._R1, f2:c=>c._R2, fe:c=>c._E, fv:c=>c._V, sup:'', note:''},
    strict:{a1:'min', a2:'max', f1:c=>c._R1n,f2:c=>c._R2x,fe:c=>c._Es,fv:c=>c._Vs,sup:'^{\\min}',
            note:'STRICT duplicate — positive only if EVERY target beats EVERY reference letter'},
    loose: {a1:'max', a2:'max', f1:c=>c._R1x,f2:c=>c._R2x,fe:c=>c._El,fv:c=>c._Vl,sup:'^{\\max}',
            note:'MAX-vs-MAX duplicate — positive iff the top letter over T∪N is a target'},
  }[v];
  const t1=termTex(dom,eT,CFG.a1), t2=termTex(dom,eN,CFG.a2);
  const nm=eDefNames(dom);
  const asc=(agg,lbl)=>agg==='mean'?lbl:agg+' '+lbl.replace('⟨R⟩','R');
  const S1=[{name:asc(CFG.a1,nm.r1),col:'#e8590c',val:CFG.f1},{name:asc(CFG.a2,nm.r2),col:'#7aa2ff',val:CFG.f2}];
  const hdr=[gt,CFG.note].filter(Boolean).join(' — ');
  return `<div class="krow">${hdr?`<h3 class="dim">${hdr}</h3>`:''}<div class="sinawrap" style="display:flex;gap:16px;align-items:flex-start">`+
    chartSVG(dom, cs, `⟦${t1}⟧ (orange)  vs  ⟦${t2}⟧ (blue)`, S1, `⟦R_c\\ (\\log_{10})⟧`)+
    chartSVG(dom, cs, `⟦E${CFG.sup}_c = ${t1} - \\big[${t2}\\big]⟧`, [{name:'E'+(CFG.sup?'_'+CFG.a1:''),col:null,val:CFG.fe}], `⟦E${CFG.sup}_c\\ (\\log_{10})⟧`)+
    chartSVG(dom, cs, `⟦\\mathrm{NE}${CFG.sup}_c = E${CFG.sup}_c/(n\\varepsilon_0)⟧`, [{name:'NE'+(CFG.sup?'_'+CFG.a1:''),col:null,val:CFG.fv}], `⟦\\mathrm{NE}${CFG.sup}_c⟧`)+
  `</div></div>`;
}

function opPanelsHTML(dom, cells, title){
  const t=D.doms[dom], meansOnly=$('meansOnly').checked;
  const nlv=nlevels(cells);
  const acc={};
  const eT=$('eT').value, eN=$('eN').value, bs=bandSets(dom);
  const refpool = eN==='mixed'? t.scope : (eN==='near'? bs.near : bs.far);
  cells.forEach(c=>{ const st_=t.states[c.state];
    const L={}, tjs=new Set(c.subset.map(w=>imgOf(dom,w,c.k)).filter(Boolean));
    t.field.forEach(w=>{ if(w===st_.ja||w===st_.jb) return;
      L[w]=Math.log10(Math.max(c.arm[w]||0,1e-5)/Math.max(st_.base[w]||0,1e-5)); });
    const refv=refpool.filter(w=>!tjs.has(w)&&w!==st_.ja&&w!==st_.jb).map(w=>L[w]);
    const ref=refv.reduce((a,b)=>a+b,0)/refv.length;
    c.subset.forEach(w=>{ const im=imgOf(dom,w,c.k);
      if(!im || im===st_.ja || im===st_.jb || !(im in L)) return;
      const pv = eT==='T' ? (L[im]-ref)/(t.eps0*c.n) : c._V;
      if(pv==null) return;
      ((acc[w]=acc[w]||{})[c.n]=acc[w][c.n]||[]).push(pv); });});
  const ops=Object.keys(acc).sort((a,b)=>t.scope.indexOf(a)-t.scope.indexOf(b));
  if(!ops.length) return '';
  const pool={};
  nlv.forEach(n=>{ const v=cells.filter(c=>c.n===n).map(c=>c._V).filter(x=>x!=null);
    if(v.length) pool[n]=v.reduce((a,b)=>a+b,0)/v.length; });
  let vs=[];
  for(const w of ops) for(const n in acc[w])
    vs.push(...(meansOnly?[acc[w][n].reduce((a,b)=>a+b,0)/acc[w][n].length]:acc[w][n]));
  const lo=Math.min(0,...vs), hi=Math.max(...vs), pad=(hi-lo)*0.08+1e-9;
  const PW=170, PH=140, PL=30, PB=22, PT=16, COLS=Math.min(6,ops.length);
  const ROWS=Math.ceil(ops.length/COLS), W=COLS*(PW+14)+40, H=ROWS*(PH+26)+20;
  let s=`<svg width="${W}" height="${H}">`;
  ops.forEach((w,wi)=>{
    const px=30+(wi%COLS)*(PW+14), py=10+Math.floor(wi/COLS)*(PH+26);
    const x=ni=>px+PL+(nlv.length>1?ni/(nlv.length-1):0.5)*(PW-PL-8);
    const y=v=>py+PT+(1-(v-(lo-pad))/((hi+pad)-(lo-pad)))*(PH-PT-PB);
    const col=vir(wi/Math.max(1,ops.length-1));
    const ims=[...new Set(cells.filter(c=>c.subset.includes(w)).map(c=>imgOf(dom,w,c.k)).filter(Boolean))].join('/');
    s+=`<rect x="${px}" y="${py}" width="${PW}" height="${PH}" fill="none" stroke="var(--line)"/>`;
    s+=`<text x="${px+4}" y="${py+12}" font-size="10" fill="${col}" font-weight="bold">${w} → ${ims}</text>`;
    s+=`<line x1="${px+PL}" x2="${px+PW-8}" y1="${y(0)}" y2="${y(0)}" stroke="var(--dim)" stroke-dasharray="3 2" opacity="0.5"/>`;
    const ppts=nlv.map((n,ni)=>pool[n]!==undefined?[x(ni),y(pool[n])]:null).filter(Boolean);
    if(ppts.length>1) s+=`<polyline fill="none" stroke="var(--dim)" stroke-width="1" opacity="0.55" points="${ppts.map(p=>p.join(',')).join(' ')}"/>`;
    const rnd=mulberry32(41+wi);
    const mpts=[];
    nlv.forEach((n,ni)=>{ const raw=acc[w][n]; if(!raw) return;
      const m=raw.reduce((a,b)=>a+b,0)/raw.length; mpts.push([x(ni),y(m),n,m,raw.length]);
      if(!meansOnly) for(const v of raw)
        s+=`<circle cx="${x(ni)+(rnd()-0.5)*8}" cy="${y(v)}" r="1.9" fill="${col}" fill-opacity="0.5"><title>'${w}' n=${n}: ${v.toFixed(2)}</title></circle>`;});
    if(mpts.length>1) s+=`<polyline fill="none" stroke="${col}" stroke-width="1.6" points="${mpts.map(p=>p[0]+','+p[1]).join(' ')}"/>`;
    for(const p of mpts) s+=`<circle cx="${p[0]}" cy="${p[1]}" r="2.6" fill="${col}"><title>'${w}' mean NE at n=${p[2]}: ${p[3].toFixed(2)} (${p[4]} cells)</title></circle>`;
    if(wi%COLS===0) for(const g of [0,1,2]) if(g>lo-pad&&g<hi+pad)
      s+=`<text class="dimt" x="${px+PL-3}" y="${y(g)+2.5}" font-size="7.5" text-anchor="end">${g}</text>`;
    if(Math.floor(wi/COLS)===ROWS-1&&nlv.length>1) for(const ni of [0,nlv.length-1])
      s+=`<text class="dimt" x="${x(ni)}" y="${py+PH+11}" font-size="7.5" text-anchor="middle">${nlv[ni]}</text>`;
  });
  s+=`</svg>`;
  return `<div class="krow">${title?`<h3 class="dim">${title}</h3>`:''}
<div class="dim" style="margin:2px 0 4px">one panel per basis b (shared y; grey = pooled mean of this group)</div><div class="sinawrap">${s}</div></div>`;
}

function kBaseGridHTML(dom){
  // matrix layout: rows = k (always), columns = base operand (always), shared y & x.
  const t=D.doms[dom], meansOnly=$('meansOnly').checked;
  const nlv=nlevels(t.cells);
  // per (k, base): values by n
  const acc={};   // acc[k][w][n] = [vals]
  const eT=$('eT').value, eN=$('eN').value, bs=bandSets(dom);
  const refpool = eN==='mixed'? t.scope : (eN==='near'? bs.near : bs.far);
  t.cells.forEach(c=>{ const st_=t.states[c.state];
    const L={}, tjs=new Set(c.subset.map(w=>imgOf(dom,w,c.k)).filter(Boolean));
    t.field.forEach(w=>{ if(w===st_.ja||w===st_.jb) return;
      L[w]=Math.log10(Math.max(c.arm[w]||0,1e-5)/Math.max(st_.base[w]||0,1e-5)); });
    const refv=refpool.filter(w=>!tjs.has(w)&&w!==st_.ja&&w!==st_.jb).map(w=>L[w]);
    const ref=refv.reduce((a,b)=>a+b,0)/refv.length;
    c.subset.forEach(w=>{ const im=imgOf(dom,w,c.k);
      if(!im || im===st_.ja || im===st_.jb || !(im in L)) return;
      const pv = eT==='T' ? (L[im]-ref)/(t.eps0*c.n) : c._V;
      if(pv==null) return;
      (((acc[c.k]=acc[c.k]||{})[w]=acc[c.k][w]||{})[c.n]=acc[c.k][w][c.n]||[]).push(pv); });});
  const ks=t.ks;
  const bases=[...new Set([].concat(...ks.map(k=>Object.keys(acc[k]||{}))))]
    .sort((a,b)=>t.scope.indexOf(a)-t.scope.indexOf(b));
  let vs=[];
  for(const k of ks) for(const w in (acc[k]||{})) for(const n in acc[k][w])
    vs.push(...(meansOnly?[acc[k][w][n].reduce((a,b)=>a+b,0)/acc[k][w][n].length]:acc[k][w][n]));
  const lo=Math.min(0,...vs), hi=Math.max(...vs), pad=(hi-lo)*0.08+1e-9;
  const PW=104, PH=84, PL=16, PB=8, PT=4, GX=8, GY=8, HX=56, HY=26;
  const W=HX+bases.length*(PW+GX)+20, H=HY+ks.length*(PH+GY)+26;
  let s=`<svg width="${W}" height="${H}">`;
  bases.forEach((w,bi)=>{ s+=`<text x="${HX+bi*(PW+GX)+PW/2}" y="${HY-8}" text-anchor="middle" font-size="10" font-weight="bold" fill="${vir(bi/Math.max(1,bases.length-1))}">${w}</text>`; });
  ks.forEach((k,ki)=>{
    const py=HY+ki*(PH+GY);
    s+=`<text x="${HX-8}" y="${py+PH/2+4}" text-anchor="end" font-size="10" font-weight="bold" fill="var(--fg)">k=${kfmt(k,dom)}</text>`;
    bases.forEach((w,bi)=>{
      const px=HX+bi*(PW+GX);
      const cellD=(acc[k]||{})[w];
      s+=`<rect x="${px}" y="${py}" width="${PW}" height="${PH}" fill="none" stroke="var(--line)"${cellD?'':' stroke-dasharray="2 3" opacity="0.4"'}/>`;
      if(!cellD) return;
      const x=ni=>px+PL+(nlv.length>1?ni/(nlv.length-1):0.5)*(PW-PL-6);
      const y=v=>py+PT+(1-(v-(lo-pad))/((hi+pad)-(lo-pad)))*(PH-PT-PB);
      s+=`<line x1="${px+PL}" x2="${px+PW-6}" y1="${y(0)}" y2="${y(0)}" stroke="var(--dim)" stroke-dasharray="2 2" opacity="0.5"/>`;
      if(bi===0) s+=`<text class="dimt" x="${px+PL-2}" y="${y(0)+2.5}" font-size="7" text-anchor="end">0</text>`;
      const col=vir(bi/Math.max(1,bases.length-1));
      const rnd=mulberry32(53+ki*31+bi);
      const mpts=[];
      nlv.forEach((n,ni)=>{ const raw=cellD[n]; if(!raw) return;
        const m=raw.reduce((a,b)=>a+b,0)/raw.length; mpts.push([x(ni),y(m),n,m,raw.length]);
        if(!meansOnly) for(const v of raw)
          s+=`<circle cx="${x(ni)+(rnd()-0.5)*5}" cy="${y(v)}" r="1.5" fill="${col}" fill-opacity="0.45"><title>k=${kfmt(k,dom)} '${w}' n=${n}: ${v.toFixed(2)}</title></circle>`;});
      if(mpts.length>1) s+=`<polyline fill="none" stroke="${col}" stroke-width="1.3" points="${mpts.map(p=>p[0]+','+p[1]).join(' ')}"/>`;
      for(const p of mpts) s+=`<circle cx="${p[0]}" cy="${p[1]}" r="2" fill="${col}"><title>k=${kfmt(k,dom)} '${w}'→'${imgOf(dom,w,k)}' mean NE at n=${p[2]}: ${p[3].toFixed(2)} (${p[4]} cells)</title></circle>`;
    });});
  s+=`<text class="dimt" x="${HX}" y="${H-8}" font-size="9">rows = shift k, columns = basis b; y = per-basis NE (shared ${lo.toFixed(1)}…${hi.toFixed(1)}), x = n ∈ {${nlv[0]}…${nlv[nlv.length-1]}}; dashed empty cell = basis outside that k's pool</text>`;
  s+=`</svg>`;
  return `<div class="sinawrap">${s}</div>`;
}

function renderDom(dom){
  const t=D.doms[dom], el=$(dom);
  const splitK=$('splitK').checked, splitB=$('splitOp').checked;
  t.cells.forEach((c,i)=>c._gi=i);
  t.cells.forEach(c=>{ const pr=eParts(dom,c); c._R1=pr.m1; c._R2=pr.m2;
    c._E=(pr.m1!=null&&pr.m2!=null)?pr.m1-pr.m2:null;
    c._V=c._E!=null?c._E/(t.eps0*c.n):null;
    c._R1n=pr.n1; c._R1x=pr.x1; c._R2n=pr.n2; c._R2x=pr.x2;
    c._Es=(pr.n1!=null&&pr.x2!=null)?pr.n1-pr.x2:null; c._Vs=c._Es!=null?c._Es/(t.eps0*c.n):null;
    c._El=(pr.x1!=null&&pr.x2!=null)?pr.x1-pr.x2:null; c._Vl=c._El!=null?c._El/(t.eps0*c.n):null; });
  let h='';
  if(splitK && splitB){
    h = kBaseGridHTML(dom);
  } else {
    const groups = splitK ? t.ks.map(k=>[`k = ${kfmt(k,dom)} (${t.cells.filter(c=>c.k===k).length} cells)`, t.cells.filter(c=>c.k===k)])
                          : [[null, t.cells]];
    for(const [gt, cs] of groups){
      if(!cs.length) continue;
      if(splitB) h+=opPanelsHTML(dom, cs, gt);
      else { h+=triptych(dom, cs, gt); h+=triptych(dom, cs, gt, 'strict'); h+=triptych(dom, cs, gt, 'loose'); }
    }
  }
  el.innerHTML=h; kx(el);
  el.querySelectorAll('.dot').forEach(d=>{ d.onclick=()=>{ sel=[d.dataset.dom,parseInt(d.dataset.ci)];
    renderAll(); openDetail(d.dataset.dom,parseInt(d.dataset.ci)); };});
}
const OPDOMS=['rf','kb','cp','mn3'];
function renderAll(){ ['num','let','ll','uu','mu'].concat(OPDOMS).forEach(dm=>{
  if(D.doms[dm]&&D.doms[dm].cells.length) renderDom(dm); }); }

function vecGrid(dom, st, c){
  const t=D.doms[dom], field=t.field, nw=field.length;
  const ch=Math.max(10, Math.min(18, Math.floor(560/nw)));
  const fs=ch>=14?10:8.5;
  const cw=62, bw=110, gapx=40, gapy=30, Lx=46, Ty=26;
  const panW=cw+bw, H=Ty+2*(nw*ch)+gapy+30, W=Lx+3*(panW+gapx)+30;
  const shade=p=>Math.max(0,Math.min(1,(Math.log10(Math.max(p,1e-5))+5)/5));
  const LMAX=3;
  const divc=v=>{const x=Math.max(-1,Math.min(1,v/LMAX));
    const r=x>0?232:Math.round(232*(1+x)), b=x<0?232:Math.round(232*(1-x));
    const g=Math.round(232*(1-Math.abs(x)*0.8)); return `rgb(${r},${g},${b})`;};
  const tjs=new Set(c.subset.map(w=>imgOf(dom,w,c.k)).filter(Boolean));
  const injs=new Set(c.subset);
  const CL={target:'#e8590c', non:'#7aa2ff', attr:'#868e96', inj:'#9c36b5'};
  const vecs=[["base",0,0, field.map(w=>st.base[w]||0)],
              ["pert",1,0, field.map(w=>c.arm[w]||0)],
              ["base",0,1, st.stA],
              ["pert",1,1, c.stAp]];
  let s=`<svg width="${W}" height="${H}">`;
  for(const cx_ of [0,1,2]) s+=`<text class="dimt" x="${Lx+cx_*(panW+gapx)+panW/2}" y="14" text-anchor="middle">${['log10 P\u0304 base','log10 P\u0304 pert','response R(x\u2032) = log10(P\u0304^pert/P\u0304^base)'][cx_]}</text>`;
  s+=`<text class="dimt" transform="rotate(-90 12 ${Ty+nw*ch/2})" x="12" y="${Ty+nw*ch/2}" text-anchor="middle">S^{t+1} @ B</text>`;
  s+=`<text class="dimt" transform="rotate(-90 12 ${Ty+nw*ch+gapy+nw*ch/2})" x="12" y="${Ty+nw*ch+gapy+nw*ch/2}" text-anchor="middle">S^t @ A</text>`;
  for(const [lab,cx_,ry,vals] of vecs){
    const x0=Lx+cx_*(panW+gapx), y0=Ty+ry*(nw*ch+gapy);
    field.forEach((w,j)=>{ const v=vals[j]||0, yy=y0+j*ch;
      const cls = ry===0 ? (tjs.has(w)?'target':((w===st.ja||w===st.jb)?'attr':(injs.has(w)?'inj':'non')))
                         : (injs.has(w)?'inj':(w===st.nat?'attr':'non'));
      const lg=Math.log10(Math.max(v,1e-5));
      s+=`<rect x="${x0}" y="${yy}" width="${cw}" height="${ch-0.6}" fill="${vir(shade(v))}" fill-opacity="0.3"${cls==='target'||cls==='inj'?` stroke="${CL[cls]}" stroke-width="1.3"`:''}><title>${lab} ${ry===0?'S^{t+1}@B':'S^t@A'}  log10 P(${w}) = ${lg.toFixed(2)}   (P = ${v})</title></rect>`;
      s+=`<text x="${x0+3}" y="${yy+ch-2.2}" font-size="${fs}" fill="var(--fg)" pointer-events="none">${w}</text>`;
      const barX=x0+cw+2, barEnd=barX+Math.max(0.5,(lg+5)/5*(bw-6));
      let split=null;
      if(cx_===1 && ry===1 && injs.has(w)){
        const bvalsA=vecs.find(vv=>vv[1]===0&&vv[2]===1)[3];
        const lgB=Math.log10(Math.max(bvalsA[j]||0,1e-5));
        split=barX+((Math.max(lgB,-5)+5)/5)*(bw-6);
        if(split>=barEnd-0.5) split=null;
      }
      const tt=`<title>${lab} ${ry===0?'S^{t+1}@B':'S^t@A'}  log10 P(${w}) = ${lg.toFixed(2)}   (P = ${v})${split!==null?'  — light-purple segment = injected extra mass (ε='+t.eps0+')':''}</title>`;
      if(split===null){
        s+=`<rect x="${barX}" y="${yy+1}" width="${barEnd-barX}" height="${ch-2.5}" fill="${CL[cls]}" fill-opacity="0.85">${tt}</rect>`;
      } else {
        s+=`<rect x="${barX}" y="${yy+1}" width="${split-barX}" height="${ch-2.5}" fill="${CL[cls]}" fill-opacity="0.85">${tt}</rect>`;
        s+=`<rect x="${split}" y="${yy+1}" width="${barEnd-split}" height="${ch-2.5}" fill="#d0a6f0" fill-opacity="0.95">${tt}</rect>`;
      }
    });
    if(ry===1){ for(const tick of [-5,-2.5,0]){ const tx=x0+cw+2+((tick+5)/5)*(bw-6);
      s+=`<line x1="${tx}" x2="${tx}" y1="${y0+nw*ch}" y2="${y0+nw*ch+4}" stroke="var(--dim)"/>`+
         `<text class="dimt" x="${tx}" y="${y0+nw*ch+13}" text-anchor="middle" font-size="8">${tick}</text>`;}}
    if(cx_===1){
      const bvals=vecs.find(v=>v[1]===0&&v[2]===ry)[3];
      const xr=Lx+2*(panW+gapx), mid=xr+cw+2+(bw-6)/2;
      field.forEach((w,j)=>{ const yy=y0+j*ch;
        const l=Math.log10(Math.max(vals[j]||0,1e-5)/Math.max(bvals[j]||0,1e-5));
        const cls = ry===0 ? (tjs.has(w)?'target':((w===st.ja||w===st.jb)?'attr':(injs.has(w)?'inj':'non')))
                           : (injs.has(w)?'inj':(w===st.nat?'attr':'non'));
        s+=`<rect x="${xr}" y="${yy}" width="${cw}" height="${ch-0.6}" fill="${divc(l)}" fill-opacity="0.3"${cls==='target'||cls==='inj'?` stroke="${CL[cls]}" stroke-width="1.3"`:''}><title>${ry===0?'S^{t+1}@B':'S^t@A'}  R_${w} = ${l.toFixed(2)}</title></rect>`;
        s+=`<text x="${xr+3}" y="${yy+ch-2.2}" font-size="${fs}" fill="var(--fg)" pointer-events="none">${w}</text>`;
        const wlen=Math.min(1,Math.abs(l)/LMAX)*(bw-6)/2;
        s+=`<rect x="${l>=0?mid:mid-wlen}" y="${yy+1}" width="${Math.max(0.5,wlen)}" height="${ch-2.5}" fill="${CL[cls]}" fill-opacity="0.85"><title>R_${w} = ${l.toFixed(2)}</title></rect>`;
      });
      s+=`<line x1="${mid}" x2="${mid}" y1="${y0}" y2="${y0+nw*ch}" stroke="var(--dim)" stroke-width="0.8"/>`;
    }}
  s+=`<text class="dimt" x="${Lx}" y="${H-6}">cells: viridis α=0.3, log 1e-5…1; `+
     `<tspan fill="#e8590c">■ target images</tspan>  <tspan fill="#7aa2ff">■ non-targets</tspan>  `+
     `<tspan fill="#9c36b5">■ injected operands</tspan>  <tspan fill="#d0a6f0">■ injected EXTRA mass</tspan>  <tspan fill="#868e96">■ attractor</tspan>; hover for exact values</text>`;
  s+=`</svg>`;
  return `<div style="overflow-x:auto">${s}</div>`;
}
function sinaDraws(vals){
  const W=340,H=170,L=40,Tm=24,Bm=14, lo=Math.min(0,...vals), hi=Math.max(...vals);
  const pad=(hi-lo)*0.1+1e-6, y=v=>Tm+(1-(v-(lo-pad))/((hi+pad)-(lo-pad)))*(H-Tm-Bm);
  const cx=L+120, srt=[...vals].sort((a,b)=>a-b), q=p=>srt[Math.min(srt.length-1,Math.floor(p*(srt.length-1)))];
  const bw=0.05*(hi-lo+1e-9), dens=u=>vals.reduce((a,w)=>a+Math.exp(-0.5*((u-w)/bw)**2),0);
  const dmax=Math.max(...vals.map(dens)), rnd=mulberry32(5);
  let s=`<svg width="${W}" height="${H}"><text class="dimt" x="4" y="14">per-draw E_c (paired renoise draws d; R per draw)</text>`;
  s+=`<rect x="${cx-30}" y="${y(srt[srt.length-1])}" width="60" height="${Math.max(1,y(srt[0])-y(srt[srt.length-1]))}" fill="none" stroke="var(--dim)" opacity="0.7"/>`;
  s+=`<line x1="${cx-30}" x2="${cx+30}" y1="${y(q(0.5))}" y2="${y(q(0.5))}" stroke="var(--fg)" stroke-width="1.6"/>`;
  for(const p of [0.25,0.75]) s+=`<line x1="${cx-12}" x2="${cx+12}" y1="${y(q(p))}" y2="${y(q(p))}" stroke="var(--dim)"/>`;
  for(const v of vals){ const jit=(rnd()-0.5)*2*(dens(v)/dmax)*24;
    s+=`<circle cx="${cx+jit}" cy="${y(v)}" r="3.2" fill="#e8590c" opacity="0.85"><title>${v}</title></circle>`;}
  s+=`<line x1="${L}" x2="${W-8}" y1="${y(0)}" y2="${y(0)}" stroke="var(--dim)" stroke-dasharray="4 3"/></svg>`;
  return s;
}
function arowAfter(dom, c){
  const t=D.doms[dom], f=t.field;
  const pairs=f.map((w,j)=>[w, c.stAp[j], c.subset.includes(w)?1:0]).filter(p=>p[1]>0);
  pairs.sort((a,b)=>b[1]-a[1]); return pairs.slice(0,10);
}
function openDetail(dom, ci){
  const t=D.doms[dom], c=t.cells[ci], st=t.states[c.state];
  let h=`<div><span class="pill">${t.label}</span><span class="pill">k=${kfmt(c.k,dom)} (${c.tag})</span>`+
        `<span class="pill">sheet-seed s=${st.seed}</span><span class="pill">subset draw ${c.rep}</span>`+
        `<span class="pill">n=${c.n}</span><span class="pill">ε=${t.eps0} each → Σε=${(t.eps0*c.n).toFixed(2)}</span>`+
        `<span class="pill">E_c=${c.E}</span><span class="pill">NE=${c.Enorm}</span></div>`;
  h+=`<div class="dim" style="margin:4px 0 2px">task prompt:</div><div class="gen">${esc(t.prompts[c.tag])}</div>`;
  h+=`<div class="dim" style="margin:4px 0 2px">natural generation of this state (attractor; its image '${st.ja}' is excluded everywhere):</div><div class="gen">${esc(st.final)}</div>`;
  h+=`<div class="dim" style="margin-top:8px">injected operands (all strictly subleading; post-injection rank in the row):</div><div>`+
     c.subset.map(w=>`<span class="tokpill inj">${w} ε=${t.eps0} rank ${c.ranks[w]+1}</span>`).join('')+`</div>`;
  h+=`<div class="dim" style="margin-top:8px">state vectors (field-word projection; columns base | pert | response R_v; rows S^{t+1}@B over S^t@A; draw-averaged):</div>`;
  h+=vecGrid(dom, st, c);
  h+=sinaDraws(c.perdraw);
  h+=canvasTable(st);
  $('detail').innerHTML=h;
}
function canvasTable(st){
  if(!st.cv) return '';
  const P=st.cvp, cur=st.cv[0].slice(), rows=[cur.slice()], chg=[new Array(P.length).fill(false)];
  for(let t_=1;t_<st.cv.length;t_++){ const ch=new Array(P.length).fill(false);
    st.cv[t_].forEach(([j,w])=>{ cur[j]=w; ch[j]=true; });
    rows.push(cur.slice()); chg.push(ch); }
  const iA=P.indexOf(st.A), iB=P.indexOf(st.B);
  let h=`<div class="dim" style="margin-top:10px">canvas argmax over denoising steps t (row = step, top = t0; `+
        `<span style="outline:1.5px solid var(--accent)">A</span> = operand position, `+
        `<span style="outline:1.5px dashed var(--accent)">B</span> = answer position; shaded cell = token changed vs previous step):</div>`;
  h+=`<div style="overflow:auto;max-height:420px;margin-top:4px"><table class="mono" style="border-collapse:collapse;font-size:.72em">`;
  h+=`<tr><th class="dim" style="position:sticky;top:0;background:var(--bg)">t</th>`+P.map((p,j)=>
    `<th class="dim" style="position:sticky;top:0;background:var(--bg);padding:1px 5px${j===iA?';outline:1.5px solid var(--accent)':j===iB?';outline:1.5px dashed var(--accent)':''}">${p}</th>`).join('')+`</tr>`;
  rows.forEach((r,t_)=>{ h+=`<tr><td class="dim" style="padding:1px 5px">${t_}</td>`+r.map((w,j)=>
    `<td style="padding:1px 5px;white-space:pre${chg[t_][j]?';background:color-mix(in srgb, var(--accent) 22%, transparent)':''}${j===iA||j===iB?';font-weight:600':''}" title="t=${t_} pos=${P[j]}">${esc(w)}</td>`).join('')+`</tr>`; });
  h+=`</table></div>`;
  return h;
}
function nathistSVG(doms_, pool){
  const tags=[], counts={};
  doms_.forEach(dm=>{ const h=D.nathist&&D.nathist[dm]; if(!h) return;
    Object.keys(h).forEach(tg=>{ if(!Object.keys(h[tg]).length) return;
      tags.push([dm,tg]); counts[dm+'|'+tg]=h[tg]; }); });
  if(!tags.length) return '<span class="dim">no captures</span>';
  const ls=pool;
  const tot=ls.map(w=>tags.reduce((a,[dm,tg])=>a+(counts[dm+'|'+tg][w]||0),0));
  const ymax=Math.max(...tot), bw=34, H=170, L=34, W=L+ls.length*bw+14;
  let sv=`<svg width="${W}" height="${H+42}">`;
  tags.forEach(([dm,tg],ti)=>{ const k=(Object.values(D.doms[dm].states).find(st=>st.tag===tg)||{}).k;
    sv+=`<rect x="${W-64*(tags.length-ti)}" y="2" width="10" height="10" fill="${k!=null?KCOL(dm,k):'var(--accent)'}"/>`+
        `<text class="dimt" x="${W-64*(tags.length-ti)+13}" y="11" font-size="9">${tg}</text>`; });
  for(const yv of niceTicks(0,ymax)){ const y=H-yv/ymax*(H-24);
    sv+=`<line x1="${L}" x2="${W-8}" y1="${y}" y2="${y}" stroke="var(--dim)" stroke-opacity="0.25"/>`+
        `<text class="dimt" x="${L-4}" y="${y+3}" text-anchor="end" font-size="9">${yv}</text>`; }
  ls.forEach((w,i)=>{ let y0=H;
    tags.forEach(([dm,tg])=>{ const n=counts[dm+'|'+tg][w]||0; if(!n) return;
      const hh=n/ymax*(H-24), k=D.doms[dm]&&D.doms[dm].states?(Object.values(D.doms[dm].states).find(st=>st.tag===tg)||{}).k:null;
      sv+=`<rect x="${L+i*bw+4}" y="${y0-hh}" width="${bw-8}" height="${hh}" fill="${k!=null?KCOL(dm,k):'var(--accent)'}"><title>${tg}: ${n} of the captures commit '${w}'</title></rect>`;
      y0-=hh; });
    sv+=`<text class="dimt" x="${L+i*bw+bw/2}" y="${H+14}" text-anchor="middle" font-size="11" class="mono">${w}</text>`+
        `<text class="dimt" x="${L+i*bw+bw/2}" y="${H+28}" text-anchor="middle" font-size="8"${tot[i]?'':' opacity="0.35"'}>${tot[i]}</text>`; });
  sv+=`<text class="dimt" transform="rotate(-90 10 ${H/2})" x="10" y="${H/2}" text-anchor="middle" font-size="10"># captures</text></svg>`;
  return sv;
}
function renderNathist(){
  if($('nathist')) $('nathist').innerHTML=nathistSVG(['uu'],'ABCDEFGHIJKLMNOPQRSTUVW'.split(''));
  if($('nathistMu')) $('nathistMu').innerHTML=nathistSVG(['mu'],'ABCDEFGHIJKLM'.split(''));
  OPDOMS.forEach(o=>{ const el=$('nathist_'+o);
    if(el&&D.nathist&&D.nathist[o]) el.innerHTML=nathistSVG([o],'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('')); });
  if($('nathistArch')) $('nathistArch').innerHTML=nathistSVG(['let','ll'],'abcdefghijklmnopqrstuvw'.split(''));
}
// ---- interactive n=1 transfer maps ----
const TMDIV=new Set(['R','dP']);
const TMLBL={R:'R = log10 P\u0304^pert / P\u0304^base', dP:'ΔP = P\u0304^pert − P\u0304^base',
             logPp:'log10 P\u0304^pert', logPb:'log10 P\u0304^base', Pp:'P\u0304^pert', Pb:'P\u0304^base'};
const TMK={R:'⟦R_c=\\log_{10}\\,\\bar P^{\\mathrm{pert}}_c/\\bar P^{\\mathrm{base}}_c⟧',
 dP:'⟦\\Delta\\bar P=\\bar P^{\\mathrm{pert}}_c-\\bar P^{\\mathrm{base}}_c⟧',
 logPp:'⟦\\log_{10}\\bar P^{\\mathrm{pert}}_c⟧', logPb:'⟦\\log_{10}\\bar P^{\\mathrm{base}}_c⟧',
 Pp:'⟦\\bar P^{\\mathrm{pert}}_c⟧', Pb:'⟦\\bar P^{\\mathrm{base}}_c⟧'};
function kx(el){ if(window.renderMathInElement) renderMathInElement(el,{delimiters:[{left:'⟦',right:'⟧',display:false}]}); }
function termSub(dom,which){
  if(which==='T') return 'T_c';
  if(which==='mixed') return 'N_c';
  if(which==='near') return dom==='num'?'\\mathrm{unit\\,NT}':(dom==='ll'?'\\mathrm{lower\\,NT}':'\\mathrm{UPPER\\,NT}');
  return dom==='num'?'\\mathrm{tens}':(dom==='ll'?'\\mathrm{UPPER}':'\\mathrm{lower}');
}
function termTex(dom,which,agg){
  const s=termSub(dom,which);
  if(agg==='min') return `\\min_{${s}} R_c`;
  if(agg==='max') return `\\max_{${s}} R_c`;
  return `\\langle R_c\\rangle_{${s}}`;
}
function tmMetricVal(pp,pb){
  const m=$('tmMetric').value, fp=Math.max(pp||0,1e-5), fb=Math.max(pb||0,1e-5);
  if(m==='R') return Math.log10(fp/fb);
  if(m==='dP') return (pp||0)-(pb||0);
  if(m==='logPp') return Math.log10(fp);
  if(m==='logPb') return Math.log10(fb);
  if(m==='Pp') return pp||0;
  return pb||0;
}
function divCol(v,vm){ const t=Math.max(-1,Math.min(1,v/(vm||1)));
  const lerp=(a,b,f)=>a.map((x,i)=>Math.round(x+(b[i]-x)*f));
  const c=t<0?lerp([221,221,221],[59,76,192],-t):lerp([221,221,221],[180,4,38],t);
  return `rgb(${c[0]},${c[1]},${c[2]})`; }
function nsw1(dk, kf){
  // n=1 point of the means-vs-n triptych (same doms cells, same refset), current cell metric
  const td=D.doms[dk]; if(!td) return null;
  const aT=[], aN=[];
  td.cells.forEach(c=>{ if(c.n!==1||(kf!=null&&c.k!==kf)) return;
    const s=td.states[c.state], tj=imgOf(dk,c.subset[0],c.k);
    if(!tj||tj===s.ja||tj===s.jb) return;
    const val=w=>tmMetricVal(c.arm[w],s.base[w]);
    const rn=td.scope.filter(w=>w!==tj&&w!==s.ja&&w!==s.jb).map(val);
    if(!rn.length) return;
    aT.push(val(tj)); aN.push(rn.reduce((a,b)=>a+b,0)/rn.length); });
  if(!aT.length) return null;
  const m=a=>a.reduce((x,y)=>x+y,0)/a.length;
  return {t:m(aT), n:m(aN), c:aT.length, e0:td.eps0};
}
let TMSC=null;  // global z-scale over ALL transfer matshows (set by tmRenderAll's collect pass)
function tmRender(dom, vsel, elid, collect){
  const T=D.tmap && D.tmap[dom]; if(!T) return;
  const el=$(elid||(dom==='num'?'tmapNum':'tmapLet'));
  const field=T.field, cols=T.cols, m=$('tmMetric').value;
  const nr=T.rowmap?(T.rlab.length-1):(dom==='num'?8:(dom==='mu'?13:23)), other=nr, nrow=nr+1;
  const hasV=dom==='let';
  const VLAB={flip:'lower → UPPER (case-flip)',LL:'lower → lower',UU:'UPPER → UPPER'};
  // row = the source whose image is this output: x'−k additively, x'/k multiplicatively
  const rowOf=(j,k,v)=>{ if(T.rowmap) return T.rowmap[j];
    if(dom==='num'){ const rv=j+2-k; return (rv>=2&&rv<=9)?rv-2:other; }
    if(dom==='mu'){ if(j<26) return other; const p=j-26+1;
      return (p%k===0&&p/k>=1&&p/k<=nr)?p/k-1:other; }
    if(v==='LL'){ if(j>=26) return other; const iv=j-k; return (iv>=0&&iv<nr)?iv:other; }
    if(j<26) return other; const iv=j-26-k; return (iv>=0&&iv<nr)?iv:other; };
  const vcells=(hasV&&vsel)?T.cells.filter(c=>vsel.includes(T.states[c.st].v)):T.cells;
  const ks=[...new Set(vcells.map(c=>T.states[c.st].k))].sort((a,b)=>a-b);
  const pool=$('tmAgg').value==='pool';
  const vord=['UU','flip','LL'].filter(v=>(!vsel||vsel.includes(v))&&Object.values(T.states).some(st=>st.v===v));
  const gkey=st=>hasV?st.v+'|'+(pool?'all':st.k):(pool?'all':String(st.k));
  const groups=hasV
    ?(pool?vord.map(v=>v+'|all')
          :vord.flatMap(v=>ks.filter(k=>Object.values(T.states).some(st=>st.v===v&&st.k===k)).map(k=>v+'|'+k)))
    :(pool?['all']:ks.map(String));
  const mk=()=>{ const a=[]; for(let r=0;r<nrow;r++) a.push(new Array(cols.length).fill(0)); return a; };
  const z=()=>new Array(cols.length).fill(0);
  const acc={};
  for(const g of groups) acc[g]={cnt:mk(),sum:mk(),hit:mk(),ncol:z(),
    sT:z(),cT:z(),sN:z(),cN:z(),trow:new Array(cols.length).fill(-1)};
  const lcx=hasV||dom==='mu'||!!T.rowmap;  // cols are lowercase for every letter dom; x is UPPER
  // doms key for the n-sweep reference convention (scope + imgOf) matching this tmap group
  const domsKey=v=>dom==='let'?(v==='UU'?'uu':v==='LL'?'ll':'let'):dom;
  const scIdx={};
  vcells.forEach(c=>{ const st=T.states[c.st], k=st.k, A=acc[gkey(st)],
    ci=cols.indexOf(lcx?c.x.toLowerCase():c.x); if(ci<0) return;
    A.ncol[ci]++;
    let best=-1e9,bj=-1;
    field.forEach((w,j)=>{ if(j===st.ja||j===st.jb) return;
      const v=tmMetricVal(c.arm[j],st.base[j]), r=rowOf(j,k,st.v);
      A.sum[r][ci]+=v; A.cnt[r][ci]++;
      if(v>best){best=v;bj=j;} });
    if(bj>=0) A.hit[rowOf(bj,k,st.v)][ci]++;
    // debug companions: exact-image target value + n-sweep-convention reference mean
    const dk=domsKey(st.v);
    if(!(dk in scIdx)){ const SC=(D.doms[dk]&&D.doms[dk].scope)||field;
      scIdx[dk]=SC.map(w=>field.indexOf(w)).filter(j=>j>=0); }
    const img=imgOf(dk,c.x,k), jt=img==null?-1:field.indexOf(img);
    if(jt>=0&&jt!==st.ja&&jt!==st.jb){
      const rv=scIdx[dk].filter(j=>j!==jt&&j!==st.ja&&j!==st.jb).map(j=>tmMetricVal(c.arm[j],st.base[j]));
      if(rv.length){ A.sT[ci]+=tmMetricVal(c.arm[jt],st.base[jt]); A.cT[ci]++;
        A.sN[ci]+=rv.reduce((a,b)=>a+b,0)/rv.length; A.cN[ci]++; A.trow[ci]=rowOf(jt,k,st.v); }
    } });
  // z-scale values: per-row means + the companion target/reference means
  const mv=[];
  for(const g of groups){ const A=acc[g];
    for(let r=0;r<nrow;r++)for(let ci=0;ci<cols.length;ci++) if(A.cnt[r][ci]) mv.push(A.sum[r][ci]/A.cnt[r][ci]);
    for(let ci=0;ci<cols.length;ci++){ if(A.cT[ci]) mv.push(A.sT[ci]/A.cT[ci]); if(A.cN[ci]) mv.push(A.sN[ci]/A.cN[ci]); } }
  if(collect) return mv;
  const {vm,mn,mx}=TMSC;
  const div=TMDIV.has(m);
  const cs=dom==='num'?26:(dom==='mu'?22:16), LX=64, TY=20, fsz=(dom==='num'||dom==='mu')?9:7.5;
  function panel(A,getV,getTT,ttl,isFreq,rlab,clab){
    let s=`<svg width="${LX+cols.length*cs+12}" height="${TY+nrow*cs+34}">`;
    for(let r=0;r<nrow;r++){
      s+=`<text class="dimt" x="${LX-5}" y="${TY+(nrow-1-r)*cs+cs/2+3.5}" text-anchor="end" font-size="${fsz}"${r===other?' font-style="italic"':''}>${rlab[r]}</text>`;
      for(let ci=0;ci<cols.length;ci++){ const yy=TY+(nrow-1-r)*cs, xx=LX+ci*cs;
        const v=A.ncol[ci]?getV(r,ci):null, dp=(rlab[r]+'|'+clab[ci]).toLowerCase();
        if(v===null){ s+=`<rect class="tmc" data-p="${dp}" x="${xx}" y="${yy}" width="${cs-1}" height="${cs-1}" fill="var(--dim)" fill-opacity="0.08"/>`; continue; }
        s+=`<rect class="tmc" data-p="${dp}" x="${xx}" y="${yy}" width="${cs-1}" height="${cs-1}" fill="${isFreq?vir(v):(div?divCol(v,vm):vir((v-mn)/Math.max(1e-9,mx-mn)))}"><title>${getTT(r,ci)}</title></rect>`; } }
    for(let ci=0;ci<cols.length;ci++)
      s+=`<text class="dimt" x="${LX+ci*cs+cs/2}" y="${TY+nrow*cs+12}" text-anchor="middle" font-size="${fsz}">${clab[ci]}</text>`;
    s+=`</svg>`;
    return `<div class="chwrap"><div class="chttl" style="max-width:${cols.length*cs}px;margin-left:${LX}px">${ttl}</div>`+
      `<div style="display:flex;align-items:stretch"><div class="chylab"><span>${T.rowmap?"operand ⟦x⟧ with image ⟦x'⟧":"output token ⟦"+(dom==='mu'?"x'/k":"x'-k")+"⟧"}</span></div>${s}</div>`+
      `<div class="chxlab" style="margin-left:${LX}px">source token ⟦x⟧</div></div>`; }
  let h='';
  for(const g of groups){ const A=acc[g];
    const gv=hasV?g.split('|')[0]:null, gk=hasV?g.split('|')[1]:g;
    const rlab=T.rlab||((dom==='num'?[...Array(8)].map((_,i)=>String(i+2))
      :dom==='mu'?'ABCDEFGHIJKLM'.split('')
      :(gv==='UU'?'ABCDEFGHIJKLMNOPQRSTUVW'.split(''):'abcdefghijklmnopqrstuvw'.split(''))).concat(['other']));
    const clab=dom==='num'?cols.map((_,i)=>String(i+2))
      :(dom==='mu'||gv==='UU'||T.rowmap)?cols.map(c=>c.toUpperCase()):cols;
    const nc=A.ncol.reduce((a,b)=>a+b,0);
    const vks=[...new Set(vcells.filter(c=>!hasV||T.states[c.st].v===gv).map(c=>T.states[c.st].k))].sort((a,b)=>a-b);
    const kpart=T.rowmap?(T.label+(T.note?' — '+T.note:''))
      :((pool||gk==='all')?`pooled over k ∈ {${vks.map(k=>kfmt(k,dom)).join(', ')}}`:`k = ${kfmt(+gk,dom)}`);
    const glab=(hasV?VLAB[gv]+' — ':'')+kpart;
    const f1=panel(A,(r,ci)=>A.hit[r][ci]/A.ncol[ci],
      (r,ci)=>`share ${(A.hit[r][ci]/A.ncol[ci]).toFixed(2)} = ${A.hit[r][ci]}/${A.ncol[ci]} configs (row ${rlab[r]}, x=${clab[ci]})`,
      `argmax histogram: row wins ⟦\\operatorname{argmax}_{x'\\notin\\{x'_{\\mathrm{nat}},\\,x'_{\\mathrm{comm}}\\}}⟧ of ${TMK[m]}`, true, rlab, clab);
    const f2=panel(A,(r,ci)=>A.cnt[r][ci]?A.sum[r][ci]/A.cnt[r][ci]:null,
      (r,ci)=>`⟨${TMLBL[m]}⟩ = ${(A.sum[r][ci]/A.cnt[r][ci]).toFixed(4)} (row ${rlab[r]}, x=${clab[ci]}; ${A.cnt[r][ci]} contributions)`,
      `mean ⟦\\langle\\cdot\\rangle_c⟧ of ${TMK[m]} over configs; global color (all maps) ${div?('±'+vm.toFixed(2)):('viridis '+mn.toFixed(2)+' … '+mx.toFixed(2))}`, false, rlab, clab);
    // debug companions: same color scale as the mean map. f3 keeps columns resolved but
    // replaces every off-target cell by that column's n-sweep-convention reference mean
    // (mean over scope minus {image, x'_nat, x'_comm} — the triptych's ⟨R⟩_N, which pools the
    // unresponsive letters that this map hides in the 'other' row); f4 pools columns too, so
    // diag/bg are single numbers = the ⟨R⟩_T / ⟨R⟩_N pair at this map's dose.
    const colT=ci=>A.cT[ci]?A.sT[ci]/A.cT[ci]:null, colN=ci=>A.cN[ci]?A.sN[ci]/A.cN[ci]:null;
    const sum=a=>a.reduce((x,y)=>x+y,0);
    const gT=sum(A.cT)?sum(A.sT)/sum(A.cT):null, gN=sum(A.cN)?sum(A.sN)/sum(A.cN):null;
    const f3=panel(A,(r,ci)=>r===A.trow[ci]?colT(ci):colN(ci),
      (r,ci)=>r===A.trow[ci]
        ?`target ⟨${TMLBL[m]}⟩ = ${(colT(ci)??NaN).toFixed(4)} at the exact image (x=${clab[ci]}; ${A.cT[ci]} configs)`
        :`column reference mean ⟨${TMLBL[m]}⟩_N = ${(colN(ci)??NaN).toFixed(4)} over scope∖{image, x'_nat, x'_comm} (x=${clab[ci]}; ${A.cN[ci]} configs)`,
      `debug: off-target cells → column ⟦\\langle\\cdot\\rangle_N⟧ (n-sweep refset incl. the letters pooled into 'other'); diagonal = exact-image target mean`, false, rlab, clab);
    const n1=nsw1(hasV?domsKey(gv):domsKey(null), (pool||gk==='all')?null:+gk);
    const f4=panel(A,(r,ci)=>A.cT[ci]?(r===A.trow[ci]?gT:gN):(A.cN[ci]?gN:null),
      (r,ci)=>r===A.trow[ci]
        ?`pooled target mean = ${(gT??NaN).toFixed(4)}`
        :`pooled reference mean = ${(gN??NaN).toFixed(4)}`,
      `debug: + pooled over x — ⟦\\langle\\cdot\\rangle_T⟧ = ${gT==null?'–':gT.toFixed(3)} vs ⟦\\langle\\cdot\\rangle_N⟧ = ${gN==null?'–':gN.toFixed(3)} at this map's ε=${T.eps}`+
      (n1?`; means-vs-n triptych at n=1 (ε₀=${n1.e0}, ${n1.c} cells): ${n1.t.toFixed(3)} vs ${n1.n.toFixed(3)}`:''), false, rlab, clab);
    const ptags=[...new Set(Object.keys(T.states).filter(sk=>{ const st=T.states[sk];
      return (!hasV||st.v===gv)&&(pool||gk==='all'||st.k==gk); }).map(sk=>sk.split('|')[0]))];
    const plines=ptags.filter(tg=>T.prompts&&T.prompts[tg]).map(tg=>`<div class="dim mono" style="font-size:.74em;margin:1px 0">${(pool||gk==='all')?tg+': ':''}${esc(T.prompts[tg])}</div>`).join('');
    h+=`<div class="krow"><h3 class="dim">${glab} <span style="font-weight:normal">(${nc} configs)</span></h3>${plines}`+
       `<div style="display:flex;gap:26px;flex-wrap:wrap;align-items:flex-end" class="sinawrap">${f1}${f2}${f3}${f4}</div></div>`;
  }
  el.innerHTML=h; kx(el);
}
function tmRenderStrict(dom, vsel, elid){
  // STRICT ARGMAX DUPLICATE of the transfer map: each config contributes exactly ONE cell —
  // x = the perturbed letter, y = the aligned position of argmax_{x'} R_c (argmax(x')-k for the
  // additive families, the preimage x'/k for xk, the op's preimage otherwise). The argmax is
  // FIXED to R (independent of the cell-metric dropdown) over the 52-letter field minus
  // {x'_nat, committed answer}. Diagonal cells (argmax = exact image) are outlined.
  const T=D.tmap && D.tmap[dom]; if(!T) return;
  const el=$(elid); if(!el) return;
  const field=T.field, cols=T.cols;
  const nr=T.rowmap?(T.rlab.length-1):(dom==='num'?8:(dom==='mu'?13:23)), other=nr, nrow=nr+1;
  const hasV=dom==='let';
  const VLAB={flip:'lower → UPPER (case-flip)',LL:'lower → lower',UU:'UPPER → UPPER'};
  const rowOf=(j,k,v)=>{ if(T.rowmap) return T.rowmap[j];
    if(dom==='num'){ const rv=j+2-k; return (rv>=2&&rv<=9)?rv-2:other; }
    if(dom==='mu'){ if(j<26) return other; const p=j-26+1;
      return (p%k===0&&p/k>=1&&p/k<=nr)?p/k-1:other; }
    if(v==='LL'){ if(j>=26) return other; const iv=j-k; return (iv>=0&&iv<nr)?iv:other; }
    if(j<26) return other; const iv=j-26-k; return (iv>=0&&iv<nr)?iv:other; };
  const vcells=(hasV&&vsel)?T.cells.filter(c=>vsel.includes(T.states[c.st].v)):T.cells;
  const ks=[...new Set(vcells.map(c=>T.states[c.st].k))].sort((a,b)=>a-b);
  const pool=$('tmAgg').value==='pool';
  const vord=['UU','flip','LL'].filter(v=>(!vsel||vsel.includes(v))&&Object.values(T.states).some(st=>st.v===v));
  const gkey=st=>hasV?st.v+'|'+(pool?'all':st.k):(pool?'all':String(st.k));
  const groups=hasV
    ?(pool?vord.map(v=>v+'|all')
          :vord.flatMap(v=>ks.filter(k=>Object.values(T.states).some(st=>st.v===v&&st.k===k)).map(k=>v+'|'+k)))
    :(pool?['all']:ks.map(String));
  const mk=()=>{ const a=[]; for(let r=0;r<nrow;r++) a.push(new Array(cols.length).fill(0)); return a; };
  const acc={};
  for(const g of groups) acc[g]={hit:mk(),ncol:new Array(cols.length).fill(0)};
  const lcx=hasV||dom==='mu'||!!T.rowmap;
  vcells.forEach(c=>{ const st=T.states[c.st], k=st.k, A=acc[gkey(st)],
    ci=cols.indexOf(lcx?c.x.toLowerCase():c.x); if(ci<0) return;
    A.ncol[ci]++;
    let best=-1e9,bj=-1;
    field.forEach((w,j)=>{ if(j===st.ja||j===st.jb) return;
      const v=Math.log10(Math.max(c.arm[j]||0,1e-5)/Math.max(st.base[j]||0,1e-5));
      if(v>best){best=v;bj=j;} });
    if(bj>=0) A.hit[rowOf(bj,k,st.v)][ci]++; });
  const cs=dom==='num'?26:(dom==='mu'?22:16), LX=64, TY=20, fsz=(dom==='num'||dom==='mu')?9:7.5;
  const yl=T.rowmap?"preimage of ⟦\\operatorname{argmax}_{x'} R_c⟧"
        :dom==='mu'?"⟦(\\operatorname{argmax}_{x'} R_c)/k⟧"
        :"⟦\\operatorname{argmax}_{x'} R_c - k⟧";
  let h='';
  for(const g of groups){ const A=acc[g];
    const gv=hasV?g.split('|')[0]:null, gk=hasV?g.split('|')[1]:g;
    const rlab=T.rlab||((dom==='num'?[...Array(8)].map((_,i)=>String(i+2))
      :dom==='mu'?'ABCDEFGHIJKLM'.split('')
      :(gv==='UU'?'ABCDEFGHIJKLMNOPQRSTUVW'.split(''):'abcdefghijklmnopqrstuvw'.split(''))).concat(['other']));
    const clab=dom==='num'?cols.map((_,i)=>String(i+2))
      :(dom==='mu'||gv==='UU'||T.rowmap)?cols.map(c=>c.toUpperCase()):cols;
    const nc=A.ncol.reduce((a,b)=>a+b,0); if(!nc) continue;
    let diag=0; for(let ci=0;ci<Math.min(cols.length,nr);ci++) diag+=A.hit[ci][ci];
    let s=`<svg width="${LX+cols.length*cs+12}" height="${TY+nrow*cs+34}">`;
    for(let r=0;r<nrow;r++){
      s+=`<text class="dimt" x="${LX-5}" y="${TY+(nrow-1-r)*cs+cs/2+3.5}" text-anchor="end" font-size="${fsz}"${r===other?' font-style="italic"':''}>${rlab[r]}</text>`;
      for(let ci=0;ci<cols.length;ci++){ const yy=TY+(nrow-1-r)*cs, xx=LX+ci*cs;
        const dp=(rlab[r]+'|'+clab[ci]).toLowerCase();
        if(!A.ncol[ci]){ s+=`<rect class="tmc" data-p="${dp}" x="${xx}" y="${yy}" width="${cs-1}" height="${cs-1}" fill="var(--dim)" fill-opacity="0.08"/>`; continue; }
        const sh=A.hit[r][ci]/A.ncol[ci];
        s+=`<rect class="tmc" data-p="${dp}" x="${xx}" y="${yy}" width="${cs-1}" height="${cs-1}" fill="${vir(sh)}"><title>share ${sh.toFixed(2)} = ${A.hit[r][ci]}/${A.ncol[ci]} configs (row ${rlab[r]}, x=${clab[ci]})</title></rect>`;
        if(r===ci&&r<nr) s+=`<rect x="${xx}" y="${yy}" width="${cs-1}" height="${cs-1}" fill="none" stroke="var(--accent)" stroke-width="1.3"/>`; } }
    for(let ci=0;ci<cols.length;ci++)
      s+=`<text class="dimt" x="${LX+ci*cs+cs/2}" y="${TY+nrow*cs+12}" text-anchor="middle" font-size="${fsz}">${clab[ci]}</text>`;
    s+=`</svg>`;
    const vks=[...new Set(vcells.filter(c=>!hasV||T.states[c.st].v===gv).map(c=>T.states[c.st].k))].sort((a,b)=>a-b);
    const kpart=T.rowmap?T.label
      :((pool||gk==='all')?`pooled over k ∈ {${vks.map(k=>kfmt(k,dom)).join(', ')}}`:`k = ${kfmt(+gk,dom)}`);
    const glab=(hasV?VLAB[gv]+' — ':'')+kpart;
    const panel=`<div class="chwrap"><div class="chttl" style="max-width:${cols.length*cs}px;margin-left:${LX}px">one cell per config: aligned position of ⟦\\operatorname{argmax}_{x'\\notin\\{x'_{\\mathrm{nat}},\\,x'_{\\mathrm{comm}}\\}} R_c⟧ (color = share; diagonal outlined = exact image)</div>`+
      `<div style="display:flex;align-items:stretch"><div class="chylab"><span>${yl}</span></div>${s}</div>`+
      `<div class="chxlab" style="margin-left:${LX}px">perturbed source token ⟦x⟧</div></div>`;
    h+=`<div class="krow"><h3 class="dim">STRICT argmax duplicate — ${glab} <span style="font-weight:normal">(${nc} configs; argmax on the diagonal ${(100*diag/nc).toFixed(0)}%)</span></h3>`+
       `<div class="sinawrap">${panel}</div></div>`;
  }
  el.innerHTML=h; kx(el);
}
function tmRenderAll(){
  const trip=[['num',null,'tmapNum'],['let',['UU'],'tmapLet'],['mu',null,'tmapMul'],
              ['let',['flip','LL'],'tmapLetArch'],
              ...OPDOMS.filter(o=>D.tmap&&D.tmap[o]).map(o=>[o,null,'tmap_'+o])];
  // pass 1: collect every panel value across ALL maps -> one global z-scale
  let mv=[]; for(const [dm,vs,el] of trip) mv=mv.concat(tmRender(dm,vs,el,true)||[]);
  const srtA=mv.map(Math.abs).sort((a,b)=>a-b);
  TMSC={vm:srtA[Math.floor(0.98*(srtA.length-1))]||1, mn:Math.min(...mv), mx:Math.max(...mv)};
  for(const [dm,vs,el] of trip) tmRender(dm,vs,el);
  tmRenderStrict('num',null,'tmapNumS'); tmRenderStrict('let',['UU'],'tmapLetS');
  tmRenderStrict('mu',null,'tmapMulS'); tmRenderStrict('let',['flip','LL'],'tmapLetArchS');
  OPDOMS.forEach(o=>{ if(D.tmap&&D.tmap[o]) tmRenderStrict(o,null,'tmap_'+o+'S'); }); }
// cross-map hover: highlight the same (row,col) cell in every transfer matshow
document.addEventListener('mouseover',e=>{ const t=e.target.closest&&e.target.closest('rect.tmc');
  if(!t) return;
  document.querySelectorAll(`rect.tmc[data-p="${CSS.escape(t.dataset.p)}"]`).forEach(x=>x.classList.add('tmhl')); });
document.addEventListener('mouseout',e=>{ const t=e.target.closest&&e.target.closest('rect.tmc');
  if(!t) return;
  document.querySelectorAll('rect.tmhl').forEach(x=>x.classList.remove('tmhl')); });

$('meansOnly').onchange=renderAll; $('splitOp').onchange=renderAll; $('splitK').onchange=renderAll;
$('eT').onchange=renderAll; $('eN').onchange=renderAll; $('tmMetric').onchange=tmRenderAll; $('tmAgg').onchange=tmRenderAll;
renderAll(); tmRenderAll(); renderNathist();
"""

KATEX = r"""
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
 onload="renderMathInElement(document.body,{delimiters:[{left:'$$',right:'$$',display:true},{left:'\\(',right:'\\)',display:false},{left:'⟦',right:'⟧',display:false}]});"></script>
"""

NOTATION = r"""
<div class="capbox"><b>Canonical notation (used throughout this page).</b>
<ul style="margin:6px 0 4px">
<li>\(k\) — the task prompt's arithmetic parameter: an <b>additive shift</b> in the \(+k\) families
(image \(x' = x{+}k\), written \(+k\)) and a <b>multiplier</b> in the \(\times k\) family (image
\(x' =\) the letter at \(k\cdot\mathrm{pos}(x)\), written \(\times k\))</li>
<li>\(s\) — sheet seed. DG generates by iteratively denoising a 64-step canvas; at positions not
yet committed, the only information carried from step \(t\) to \(t{+}1\) is the self-conditioning sheet
\(S^t\) — the per-position token distribution the model produced at step \(t\), fed back as input to step
\(t{+}1\). \(s\) indexes which stochastic rollout we captured: it fixes the canvas draft, the committed
operand \(x_{\mathrm{nat}}\), and the recorded sheet \(S^t\) (top-32 per position) that every probe of
that state starts from.</li>
<li>\(t\) — diffusion step of the captured sheet (every probe reads the \(t \to t{+}1\) update)</li>
<li>\(d\) — paired renoise draw. Each probe re-runs ONE denoising step offline: the captured canvas
with the operand position and the answer position overwritten by random tokens (mimicking the sampler's renoise
of unaccepted positions — so neither answer can be read off the canvas and \(S^t\) is the sole carrier),
with the perturbed or the base sheet installed. \(d = 1 \ldots D\) indexes independent random fillings of
those two positions; the pert and base arms consume the IDENTICAL \(d\)-th canvas, so their difference
isolates the sheet edit. \(D=12\) for the numbers maps, \(D=8\) for the batteries.</li>
<li>\(b\) — basis token (an operand the task adds \(k\) to; its image is \(b{+}k\), case-flipped for
letters)</li>
<li>\(x\) — source token: the basis whose subleading mass we perturb in \(S^t\) at the operand position</li>
<li>\(x'\) — target token, read at the answer position</li>
<li>\(x_{\mathrm{nat}},\ x'_{\mathrm{nat}}\) — the state's committed operand and its answer image (both excluded from
scoring)</li>
<li>\(r\) — operand-subset draw: which random size-\(n\) subset of sources \(x\) (from the task pool minus
\(x_{\mathrm{nat}}\)) a cell injects; independent \(r\) draws share the state's \(d\)-canvases</li>
<li>\(c\) — experiment cell: \(c=(k,s,t)\) for the single-injection maps; \(c=(k,s,t,r)\) with
operand-subset draw \(r\) and \(n=|T_c|\) injected operands for the n&gt;1 batteries</li>
<li>\(\bar P_c(x') = \tfrac{1}{D}\textstyle\sum_{d=1}^{D} P_{c,d}(x')\) — draw-averaged state (always
averaged BEFORE any ratio)</li>
<li>\(R_c(x') = \log_{10}\big[\bar P^{\,\mathrm{pert}}_c(x') \,/\, \bar P^{\,\mathrm{base}}_c(x')\big]\)
— response (floor \(10^{-5}\) inside the log)</li>
<li>\(E_c = \langle R_c\rangle_{T_c} - \langle R_c\rangle_{N_c}\) — effect; \(T_c\) = images of the
injected operands under shift \(k\), \(N_c\) = reference non-targets (selectable: mixed / near-band / far-band)</li>
<li>\(\mathrm{NE}_c = E_c/\Sigma\varepsilon = E_c/(n\,\varepsilon_0)\) — normalized effect (per unit
injected mass)</li>
</ul></div>
"""

ALGO_SRC = r"""
# ---- capture (once per task k and sheet seed s) ----------------------------
rollout = DG.sample(prompt(k), seed=s,                 # s = stochastic rollout seed
                    T=64, C=128, t_max=1.3, t_min=0.8, # hot regime
                    entropy_bound=0.3, top_k=10,
                    s_topk_record=32)                  # record S^t (top-32/position) at EVERY step

t = 2          # <-- the actual choice: t=2 for all n>1 batteries and the letters maps;
               #     the numbers maps additionally pool (s,t) in {(0,1),(0,2),(1,2),(2,2)}
draft = rollout.steps_argmax[t]        # canvas draft as of step t
S_t   = rollout.s_rec[t]               # the sheet the model handed to step t+1
A, B  = locate_positions(rollout.final)    # operand position A, answer position B (found via the comma)
x_nat = rollout.final[A]               # committed operand of this state

# ---- canvas init: D paired renoise draws (shared by every cell of this state)
rng = default_rng(f(s))                # d-canvases are a function of s
canvases = []
for d in range(D):                     # D = 8 (batteries) / 12 (numbers maps)
    cv = list(draft)
    cv[A] = rng.integers(0, VOCAB)     # overwrite BOTH probed positions with random tokens,
    cv[B] = rng.integers(0, VOCAB)     # mimicking renoise: canvas carries neither operand
    canvases.append(cv)                # nor answer -- S^t is the sole carrier

# ---- one cell c = (k, s, t, r): flat-mass perturbation of S^t at position A ----
X = subset_draw(pool - {x_nat}, size=n, seed=r)   # n source tokens x  (n=1 for the maps)
S_pert = copy(S_t)
S_pert[A] = (1 - n*eps0) * S_t[A]                 # rescale the whole row, then
for x in X:
    S_pert[A][x] += eps0                          # add eps0 to each x (absent tokens evict
                                                  # the row's argmin top-32 entry)
assert all(rank(S_pert[A], x) >= 1 for x in X)    # every injection stays subleading, else drop cell

# ---- paired one-step probes: a single denoising step t -> t+1 --------------
P = {"pert": [None] * D, "base": [None] * D}
for arm, sheet in (("pert", S_pert), ("base", S_t)):
    for d in range(D):
        # one forward step conditioned on canvas d and the installed sheet,
        # at the schedule temperature of step t:
        P[arm][d] = DG.step(prompt(k), canvases[d], sheet,
                            temperature=1.3 - 0.5*t/63)[B]   # full next-step dist at position B

Pbar = {arm: mean(P[arm][d] for d in range(D)) for arm in ("pert", "base")}   # bar-P
R = lambda x1: log10(max(Pbar["pert"][x1], 1e-5) / max(Pbar["base"][x1], 1e-5))
# E, NE, argmax maps etc. are computed from R as defined above
"""

import re as _re
def _hlesc(x):
    return x.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
def _hl(src):
    KW = r"\b(for|in|if|else|elif|assert|lambda|range|list|copy|all|def|return|and|or|not)\b"
    out = []
    for line in src.strip("\n").split("\n"):
        if "#" in line:
            code, com = line.split("#", 1)
            com = '<span class="c-com">#' + _hlesc(com) + "</span>"
        else:
            code, com = line, ""
        code = _hlesc(code)
        strs = []
        def _st(m):
            strs.append(m.group(0)); return "\x00" + "@" * len(strs) + "\x00"
        code = _re.sub(r'"[^"]*"', _st, code)
        code = _re.sub(KW, r'<span class="c-kw">\1</span>', code)
        code = _re.sub(r"(?<![\w.])(\d+(?:\.\d+)?(?:e-?\d+)?)\b", r'<span class="c-num">\1</span>', code)
        for i, sv in enumerate(strs):
            code = code.replace("\x00" + "@" * (i + 1) + "\x00", '<span class="c-str">' + sv + "</span>")
        out.append(code + com)
    return "\n".join(out)

ALGO = (r"""
<div class="capbox"><b>The exact algorithm (pseudo-Python, at the abstraction level where \(s\), \(d\),
\(t\) enter).</b>
<pre class="algo">""" + _hl(ALGO_SRC) + "</pre></div>\n")


CAPTION = r"""
<div class="capbox"><b>n&gt;1 index sets.</b> Cell \(c = (\text{domain}, k, s, r)\): sheet seed
\(s \in \{0,1,2,3\}\), independent operand-subset draw \(r\), determining \(T_c\) and
\(n = |T_c|\) (batches compute8/9/10/12/13/14; every injection strictly subleading post-injection,
rank-verified per token). The curves estimate \(\mathrm{NE}(n) = \mathbb{E}[\mathrm{NE}_c \mid
|T_c| = n]\) by the mean over cells at that \(n\) (mass-normalized). Pooled charts additionally pool
over \(k\) (per-basis panels pool over \(k\) within a basis \(b\)) — split with the checkboxes; nothing
else is pooled. Error bars: 95% CI \(= 1.96\cdot\mathrm{SE}\) over the cells of the displayed
group.</div>
"""

TM_DEF = r"""
<p class="dim">Configs here are single-injection cells \(c=(k,s,t)\). For the selected cell metric
\(m_c(x')\) the left panel shows the argmax histogram
\(\big\langle \mathbf{1}\big[\, x' = \operatorname{argmax}_{\tilde x \neq x'_{\mathrm{nat}}} m_c(\tilde x)
\,\big]\big\rangle_{c\,:\,x \neq x_{\mathrm{nat}}}\) and the right panel the mean
\(\langle m_c(x')\rangle_{c\,:\, x \neq x_{\mathrm{nat}},\, x' \neq x'_{\mathrm{nat}}}\)
(for the letters variants the actually-committed answer letter is excluded alongside
\(x'_{\mathrm{nat}}\); they differ only for LL3 s0). The default metric
is the canonical response
$$ R_c(x' \mid x) \;=\; \log_{10}\,
\frac{\bar P^{\,\mathrm{pert}(x)}_c(x')}{\bar P^{\,\mathrm{base}}_c(x')} $$
— the same averaging as the n&gt;1 section below: \(\bar P\) is the mean over the paired renoise
draws \(d\), taken before the ratio. The other options expose the numerator and denominator separately,
in log or linear form (floors \(10^{-5}\) apply inside logs only).</p>
"""

MULTFIND = r"""
<div class="capbox"><b>What the &times;k family shows (&epsilon;=0.45 maps, 30 captures, 174 surviving
cells).</b>
<ul style="margin:6px 0 4px">
<li><b>The multiplicative map does transfer, ~4&times; more weakly than a shift.</b> Placebo-corrected
specificity (below), <b>clustered by (task, basis)</b> &mdash; see the replication caveat &mdash; is
<b>+0.238&plusmn;0.102</b> (p=0.028, 24 units) for &times;k versus <b>+1.000&plusmn;0.114</b>
(p=5&times;10<sup>&minus;13</sup>, 74 units) for +k at the same dose; the families differ at
p=4&times;10<sup>&minus;4</sup>. The argmax lands on the true image in <b>24.1%</b> of &times;k cells
vs <b>36.2%</b> of +k cells (chance &asymp;2% over the 50-letter field). Naive per-cell errors give
the same point estimates with far tighter intervals (+0.252&plusmn;0.042, +0.967&plusmn;0.037); those
p-values are inflated and should not be quoted.</li>
<li><b>Replication caveat (applies to every family on this page).</b> The sheet seed rarely moves the
t=2 draft: 40 UPPER&rarr;UPPER captures yield only <b>6</b> distinct drafts and 30 &times;k captures
only <b>3</b>, so seeds within a task are near-replicates and per-cell intervals pseudo-replicate. The
honest unit is <b>(task, basis)</b>: average over seeds, then count each injected basis once. Point
estimates barely move under this correction; confidence does &mdash; &times;k drops from
p=2&times;10<sup>&minus;8</sup> to a marginal p=0.028. State diversity comes from varying the
<i>prompt</i> (a different k, a different operand pool), not the seed.</li>
<li><b>Resolved per k, the families overlap.</b> &times;3 (+0.406, p=5&times;10<sup>&minus;6</sup>) and
&times;4 (+0.243, p=8&times;10<sup>&minus;4</sup>) sit within a factor ~2 of the weaker shifts +5 (+0.542)
and +11 (+0.594); the strong shifts are +3 (+1.520) and +7 (+1.095). <b>&times;2 alone is null</b>
(+0.073, p=0.18) &mdash; so "multiplication transfers worse" is really "&times;2 transfers nothing and
&times;3/&times;4 transfer like a weak-to-mid shift". Note k is not comparable <i>as a number</i> across
families (&times;2 moves an operand further than +3 does); what the split shows is that no single
multiplier reaches the strong additive shifts.</li>
<li><b>Why a placebo is required here.</b> Under &times;k the image set is a <i>lattice</i> (the
multiples of k), and the answer position prefers that lattice wholesale &mdash; injected or not. The raw
within-band edge therefore overstates &times;k: raw +0.413 against a placebo floor of +0.161 (images of
the operands that were <i>not</i> injected), versus +1.070 against +0.103 for +k. Every number above is
the difference. Without this control the &times;k signal at &epsilon;&#8320;=0.04 looks significant
(+0.050, p=0.001) but is <b>entirely non-specific</b> (placebo +0.043; specificity +0.007, p=0.68).</li>
<li><b>Superlinear in dose, for both families.</b> Placebo-corrected specificity across the ladder,
+k / &times;k: &epsilon;&#8320;=0.04 &minus;0.006 / +0.007 (both null) &rarr; 0.178 +0.111 / +0.030 (both
n.s.) &rarr; 0.316 <b>+0.573</b> / +0.087 (p=8&times;10<sup>&minus;4</sup>) &rarr; 0.45 <b>+1.070</b> /
<b>+0.252</b>. The &times;k channel needs a bigger push to become visible at all, and stays ~5&ndash;7&times;
below the additive one wherever both are detectable.</li>
<li><b>Not a commitment artifact.</b> Sheet commitment does matter <i>within</i> the &times;k family when
it varies widely &mdash; at &epsilon;=0.316 (guard floor 0.462) per-state leader mass predicts
specificity, r=+0.69, p=2&times;10<sup>&minus;5</sup>, over a 0.54&ndash;0.97 range. But the headline
&epsilon;=0.45 comparison is <b>commitment-matched by construction</b>: the rank&ge;1 guard admits only
states with leader mass &ge;0.818, and the two families are then indistinguishable on commitment
(0.951 vs 0.929, p=0.09). Regressing per-state specificity on family <i>and</i> leader mass over those
63 states gives family &minus;0.661&plusmn;0.092 (t=&minus;7.2, p=1&times;10<sup>&minus;9</sup>) with
leader mass contributing nothing (t=&minus;0.65, p=0.52). The gap is the map, not the sheet.</li>
</ul>
<b>Caveat on NE(n).</b> Applying the same placebo logic to the n-sweep &mdash; scoring injected images
against the images of <i>non</i>-injected operands rather than against the mixed field &mdash; flattens
NE to &asymp;0 at every n for <b>both</b> families at &epsilon;&#8320;=0.04 (additive
&minus;0.15&hellip;+0.36, multiplicative &minus;0.07&hellip;+0.26, almost all p&gt;0.1). That is
consistent with this page's existing n=1 null at &epsilon;&#8320;=0.04, and it means the NE plateau at
this dose should be read as a diffuse image-set/band response rather than per-operand routing. One
caveat in the other direction: for a translation the images of adjacent operands are adjacent letters,
so positional smearing could mask genuine additive transfer under this reference; the &times;k map,
whose images are k apart, is the cleaner test. Companion figure:
<span class="mono">figs/xtask_mult.png</span>.</div>
<img class="fig" src="figs/xtask_mult.png" alt="additive vs multiplicative transfer">
"""

BANDCAV = r"""
<p class="dim"><b>Band caveat (2026-07-28):</b> letters \(E_c\) uses the mixed 52-token \(N_c\) (both
cases), so the case-band susceptibility lift — the whole UPPERCASE band rising vs lowercase — is inside
these NE values. With token sets \(T_c = \{\mathrm{img}(x_i + k)\}\setminus\{x'_{\mathrm{nat}}\}\) (target images; img = case-flip or same-case per variant),
\(U_c = \mathrm{UPPERCASE}\setminus T_c\setminus\{x'_{\mathrm{nat}}\}\) (uppercase non-targets), \(\Lambda\) = all 26
lowercase (incl. the injected operands themselves = identity-leak channel), and
\(\langle R_c\rangle_A\) the unweighted mean of \(R_c(x')\) over \(x' \in A\):
$$ \mathrm{NE}^{\mathrm{mixed}}_c = \frac{\langle R_c\rangle_{T_c} - \langle R_c\rangle_{U_c \cup \Lambda}}{n\varepsilon_0},
\qquad \mathrm{NE}^{\mathrm{spec}}_c = \frac{\langle R_c\rangle_{T_c} - \langle R_c\rangle_{U_c}}{n\varepsilon_0},
\qquad \mathrm{NE}^{\mathrm{band}}_c = \frac{\langle R_c\rangle_{U_c} - \langle R_c\rangle_{\Lambda}}{n\varepsilon_0} $$
\(\mathrm{NE}^{\mathrm{mixed}}\) is the default plotted quantity (\(N_c = U_c \cup \Lambda\)); they obey
the exact identity
\(\mathrm{NE}^{\mathrm{mixed}}_c = \mathrm{NE}^{\mathrm{spec}}_c + \frac{26}{\,51-|T_c|\,}\,\mathrm{NE}^{\mathrm{band}}_c\)
(weight \(\approx 0.52\)–\(0.58\)). Measured (68 cells/n): \(\mathrm{NE}^{\mathrm{spec}}\) is
\(-0.08\pm0.22\) at \(n{=}1\) (null: no letter-specific transfer at a single \(\varepsilon_0{=}0.04\)
injection), rises to \(\approx +0.3\) by \(n{=}2\)–3 and stays flat through \(n{=}18\) (\(+0.31\pm0.03\)
at \(n{=}14\)); \(\mathrm{NE}^{\mathrm{band}}\) decays \(+1.74 \rightarrow \approx +0.5\). The plotted
letters plateau is therefore ≈ half band + half specific: the count-independence conclusion survives
within-band, but at level ≈0.3 rather than ≈0.5–0.7, and the cross-domain "collapse" with numbers is
partly band-inflated. Numbers curves are unaffected (single word-class field: any diffuse band response
hits \(T\) and \(N\) equally and cancels in \(E_c\)). Switch the E terms above to see the decomposition
directly. The dose-sweep appendix at the bottom of the page shows the \(n{=}1\) specific null is a
low-dose artifact: the specific channel is super-linear in \(\varepsilon_0\) and overtakes the band lift
by \(\varepsilon_0 \approx 0.25\)&ndash;\(0.3\).</p>
"""

APPENDIX = r"""
<details>
<summary style="cursor:pointer;font-size:1.4em;font-weight:bold;margin:22px 0 8px">Appendix: letters dose sweep + archived case-flip &amp; lower&rarr;lower runs</summary>
<p class="dim"><b>Dose-sweep update:</b> the \(n{=}1\) null at \(\varepsilon_0{=}0.04\) is a
dose/power artifact, not a missing channel. Sweeping the single-injection mass
\(\varepsilon_0 \in \{0.01 \ldots 0.45\}\) on L3/L7 (the rank\(\ge\)1 guard rejects injections that would
take the row lead; 0.45 is the largest dose that survives it in most states), the within-band specific
edge \(R_c(\mathrm{tgt}) - \langle R_c\rangle_{\mathrm{band}}\) grows <b>super-linearly</b> — \(+0.013\)
(\(p{=}.18\)) at 0.04, \(+0.053\) (\(p{=}.013\)) at 0.1, \(+0.74\) (\(p{=}7{\times}10^{-16}\)) at 0.45;
\(\mathrm{edge}/\varepsilon_0\) rises \(0.3 \to 1.6\) — while the case-band lift <b>saturates</b>
(\(\mathrm{band}/\varepsilon_0\) falls \(1.8 \to 0.7\)). The two cross at \(\varepsilon_0 \approx 0.25\)&ndash;\(0.3\),
which is why the \(\varepsilon{=}0.45\) maps above show a clean diagonal where the old 0.04 maps showed none.
The same-case control tasks (lower&rarr;lower, UPPER&rarr;UPPER &mdash; no case transform) roughly
<b>double</b> the specific channel at every dose (at 0.04: \(+0.029\) both, \(p{=}.007/.019\), vs
\(+0.013\); at 0.45: LL \(+1.28\), UU \(+1.49\) vs flip \(+0.74\)), so the case flip costs about half the
per-unit-mass specific transfer; the band lift itself appears in all variants (at 0.45: \(+0.31\) flip,
\(+0.58\) same-case). Batteries: <span class="mono">xtask_eps_sweep</span> /
<span class="mono">xtask_samecase</span>; note LL7 (and UU7 s1) drop out at
\(\varepsilon \ge 0.316\) &mdash; their sheet leaders are too weak to keep any injection subleading.</p>
<img class="fig" src="figs/xtask_eps_samecase.png" alt="dose sweep + same-case variants">
<h3>Archived transfer maps: case-flip &amp; lower&rarr;lower at the same \(\varepsilon\) <span class="dim">(same
metric/averaging controls as the main maps)</span></h3>
<div class="card"><div id="tmapLetArch"></div><div id="tmapLetArchS" style="margin-top:16px"></div></div>
<h3>Archived n&gt;1 runs: letters +k (case-flip) <span class="dim">(\(\varepsilon_0{=}0.04\), k &isin; {3,5,7,11},
seeds 0&ndash;3)</span></h3>
<div class="card" id="let"></div>
<h3>Archived n&gt;1 runs: letters +k (lower&rarr;lower) <span class="dim">(\(\varepsilon_0{=}0.04\), k &isin; {3,7},
seeds 0&ndash;1)</span></h3>
<div class="card" id="ll"></div>
<h3>Archived variants: natural operand choice <span class="dim">(case-flip &amp; lower&rarr;lower captures)</span></h3>
<div class="card" id="nathistArch"></div>
</details>
"""

PARTABLES = open(EXP / "par_tables.html").read()
PAR4TABLES = open(EXP / "par4_tables.html").read()
PARSINA = open(EXP / "par_sina.json").read()
PARLADDER = open(EXP / "par_ladder.html").read()
PARFREE = r"""
<h3>Controlled parallelism battery &mdash; the operative n&gt;1 result</h3>
<div class="capbox"><b>Design (the most-controlled setting; everything else lives in the collapsed
blocks).</b> Per (state, rep), arms sharing the state's canvases and base rows: <b>solo</b>
x&#8321;@&epsilon;&#8320; / solo x&#8322; / <b>joint</b> {x&#8321;,x&#8322;} / <b>junk-matched</b>
x&#8321;+junk (an imageless letter &mdash; same total mass, same incumbent suppression, one routable
hypothesis). For letters +3/+5 an <b>n=3 ladder</b> holds total mass fixed at 0.66 (3 tokens @0.22) and
varies ONLY the number of routable hypotheses (1/2/3, junk-padded). Estimator: per-operand spec =
R(img(x)) &minus; &lang;R&rang; over non-injected images &ge;4 alphabet positions from every injected
token; operand tuples separation-constrained (per-k D_MIN, images clear of the other source) &mdash;
copy-leak and proximity smear excluded <b>by construction</b>; strictly-subleading guard everywhere
(&epsilon;&#8320; &lt; L/(1+nL); 0.30/0.25 is the largest n=2-guarded dose). Zero-point validated by a
cross-state transplant null: +0.0005&plusmn;0.011 (p=.97) at 0.22, &minus;0.048&plusmn;0.006 at 0.30 (a
small conservative bias, identical across arms, cancelling in every paired contrast). The confound-free
regime ends at n&asymp;3&ndash;4; the max-n extension and the guard-dropped co-dominant regime are in
the collapsed block below.</div>
<p><label><input type="checkbox" id="pcMeans"> <b>means + 95% CI only</b></label>
<span class="dim"> &mdash; y = <b>min over injected members</b> of the per-operand spec,
E(x&#7522;) = R(img(x&#7522;)) &minus; &lang;R(img(non-injected))&rang;<sub>P</sub>: one filled dot per
JOINT cell, positive only if EVERY injected member's image is elevated &mdash; the strict all-members
parallelism readout (at n=1 the min is just the solo spec). Hollow dots (offset right) = the
junk-matched control (single routable member at the same total mass; note the min of n draws is
order-statistically biased low vs this single-member ceiling). Box = min&rarr;max extent, thick bar =
MEDIAN, ticks = quartiles, line = per-level mean. Hover for the per-member breakdown, click for
provenance.</span></p>
<div class="card" id="parCurves"></div>
<div class="card" id="pcDetail" style="display:none"></div>
<script>
const PC = """ + PARSINA + r""";
(function(){
 const $=id=>document.getElementById(id);
 function mb32(a){return function(){a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}
 function ticks(lo,hi){const sp=(hi-lo)/4,m=Math.pow(10,Math.floor(Math.log10(sp+1e-12)));
  const st=[1,2,5,10].map(k=>k*m).find(k=>k>=sp)||m; const out=[];
  for(let g=Math.ceil(lo/st)*st;g<=hi+1e-9;g+=st) out.push(Math.round(g*1e6)/1e6); return out;}
 function drawCol(sctx,os,cx,halfw,col,hollow,fi,si,ni,sr){
  const {s,y}=sctx; let svg='';
  const v=os.map(o=>o.v), vmin=Math.min(...v), vmax=Math.max(...v);
  const srt=[...v].sort((a,b)=>a-b), q=p=>srt[Math.min(srt.length-1,Math.floor(p*(srt.length-1)))];
  const bw=0.05*(vmax-vmin+1e-9), dens=u=>v.reduce((a,w)=>a+Math.exp(-0.5*((u-w)/bw)**2),0);
  const dmax=Math.max(...v.map(dens));
  svg+=`<rect x="${cx-halfw}" y="${y(vmax)}" width="${2*halfw}" height="${Math.max(1,y(vmin)-y(vmax))}" fill="none" stroke="var(--dim)" opacity="0.7"${hollow?' stroke-dasharray="3 2"':''}/>`;
  svg+=`<line x1="${cx-halfw}" x2="${cx+halfw}" y1="${y(q(0.5))}" y2="${y(q(0.5))}" stroke="var(--fg)" stroke-width="1.5"/>`;
  for(const p of [0.25,0.75]) svg+=`<line x1="${cx-halfw*0.45}" x2="${cx+halfw*0.45}" y1="${y(q(p))}" y2="${y(q(p))}" stroke="var(--dim)"/>`;
  const rnd=mb32(19+fi*7+ni+97*si+(hollow?411:0));
  os.forEach(o=>{ const jit=(rnd()-0.5)*2*(dens(o.v)/dmax)*(halfw-1.5);
   const tt=`min-spec = ${o.v.toFixed(3)}   (${sr.name}${hollow?' — junk-matched arm':''})\n${o.tag} s${o.s} rep ${o.r} n=${o.n}`+
     (o.ops&&o.ops.length>1?`\nper member: ${o.ops.map(q=>q[0]+': '+q[1].toFixed(3)).join('   ')}`:'')+
     (o.tup&&o.tup.length?`\ninjected: ${o.tup.map(w=>w+(o.ranks&&o.ranks[w]!=null?'(r'+(o.ranks[w]+1)+')':'')).join(' ')}`:'')+
     (o.P?`\nreference: ${o.P}`:'')+(o.nat?`\nx_nat='${o.nat}' → x'_nat='${o.ja}'`:'')+`\nclick for provenance card`;
   svg+=`<circle class="dot" data-f="${fi}" data-sr="${si}" data-i="${sr.pts.indexOf(o)}" cx="${cx+jit}" cy="${y(o.v)}" r="2.9" ${hollow?`fill="none" stroke="${col}" stroke-width="1.2"`:`fill="${col}" fill-opacity="0.42"`} style="cursor:pointer"><title>${tt}</title></circle>`;});
  return svg;}
 function render(){
  const meansOnly=$('pcMeans').checked; let html='<div style="display:flex;flex-wrap:wrap;gap:6px">';
  PC.families.forEach((F,fi)=>{
   const ns=[...new Set([].concat(...F.series.map(sr=>sr.pts.map(p=>p.n))))].sort((a,b)=>a-b);
   const W=360,H=252,Lm=44,Bm=34,Tm=24,Rm=10;
   const allv=[].concat(...F.series.map(sr=>sr.pts.map(p=>p.v)));
   const lo=Math.min(0,...allv), hi=Math.max(0,...allv), pad=(hi-lo)*0.08+1e-9;
   const x=ni=>Lm+18+(ns.length>1?ni/(ns.length-1):0.5)*(W-Lm-Rm-76);
   const y=v=>Tm+(1-(v-(lo-pad))/((hi+pad)-(lo-pad)))*(H-Tm-Bm);
   let s=`<svg width="${W}" height="${H}">`;
   s+=`<text x="${Lm}" y="13" font-size="11" font-weight="bold" fill="var(--fg)">${F.label}</text>`;
   for(const g of ticks(lo-pad,hi+pad)) s+=`<line x1="${Lm}" x2="${W-Rm}" y1="${y(g)}" y2="${y(g)}" stroke="var(--dim)" opacity="0.13"/><text x="${Lm-4}" y="${y(g)+3}" font-size="8.5" fill="var(--dim)" text-anchor="end">${g}</text>`;
   s+=`<line x1="${Lm}" x2="${W-Rm}" y1="${y(0)}" y2="${y(0)}" stroke="var(--dim)" stroke-dasharray="3 2" opacity="0.6"/>`;
   const K=F.series.length, spread=K>1?15:0, halfw=K>1?6.5:9;
   F.series.forEach((sr,si)=>{
    const off=(si-(K-1)/2)*spread, mpts=[];
    ns.forEach((n,ni)=>{
     const all=sr.pts.filter(p=>p.n===n); if(!all.length) return;
     if(si===0) s+=`<text x="${x(ni)}" y="${H-20}" font-size="10" fill="var(--dim)" text-anchor="middle">${n}</text>`;
     const real=all.filter(p=>!p.junk), jnk=all.filter(p=>p.junk);
     if(real.length){
      const v=real.map(o=>o.v), m=v.reduce((a,b)=>a+b,0)/v.length;
      mpts.push([x(ni)+off,y(m),n,m,v.length]);
      if(meansOnly){ const sd=Math.sqrt(v.reduce((a,b)=>a+(b-m)**2,0)/Math.max(1,v.length-1));
       const ci=1.96*sd/Math.sqrt(v.length), cx=x(ni)+off;
       s+=`<line x1="${cx}" x2="${cx}" y1="${y(m-ci)}" y2="${y(m+ci)}" stroke="${sr.col}" stroke-width="1.6"/>`;
       for(const e of [-1,1]) s+=`<line x1="${cx-5}" x2="${cx+5}" y1="${y(m+e*ci)}" y2="${y(m+e*ci)}" stroke="${sr.col}" stroke-width="1.6"/>`;
      } else s+=drawCol({s,y},real,x(ni)+off,halfw,sr.col,false,fi,si,ni,sr);
     }
     if(jnk.length&&!meansOnly) s+=drawCol({s,y},jnk,x(ni)+off+2.6*halfw+3,halfw*0.8,'#8a8a8a',true,fi,si,ni,sr);
     else if(jnk.length&&meansOnly){ const v=jnk.map(o=>o.v), m=v.reduce((a,b)=>a+b,0)/v.length;
      s+=`<circle cx="${x(ni)+off+2.6*halfw+3}" cy="${y(m)}" r="3.2" fill="none" stroke="#8a8a8a" stroke-width="1.4"><title>junk-matched mean = ${m.toFixed(3)} (${v.length} cells)</title></circle>`;}
    });
    if(mpts.length>1) s+=`<polyline fill="none" stroke="${sr.col}" stroke-width="2" points="${mpts.map(p=>p[0]+','+p[1]).join(' ')}"/>`;
    for(const p of mpts) s+=`<circle cx="${p[0]}" cy="${p[1]}" r="3.4" fill="${sr.col}" stroke="var(--bg)" stroke-width="1.1"><title>${sr.name}: mean at n=${p[2]} = ${p[3].toFixed(3)} (${p[4]} cells)</title></circle>`;
    s+=`<rect x="${Lm+4}" y="${Tm+2+si*11}" width="8" height="8" fill="${sr.col}" fill-opacity="0.7"/><text x="${Lm+15}" y="${Tm+9.5+si*11}" font-size="8.5" fill="var(--dim)">${sr.name}</text>`;
   });
   s+=`<text x="${(Lm+W-Rm)/2}" y="${H-6}" font-size="9" fill="var(--dim)" text-anchor="middle">n real operands &mdash; MIN over injected members of per-operand spec</text></svg>`;
   html+=s;
  });
  $('parCurves').innerHTML=html+'</div>';
  $('parCurves').querySelectorAll('circle.dot').forEach(c=>c.addEventListener('click',()=>{
   const F=PC.families[+c.dataset.f], sr=F.series[+c.dataset.sr], o=sr.pts[+c.dataset.i];
   const el=$('pcDetail'); el.style.display='block';
   el.innerHTML=`<b>${F.label}</b> &mdash; <span class="dim">${sr.name}${o.junk?' — junk-matched arm':''}</span><br>`+
    `<span class="mono">state ${o.tag}|s${o.s}, subset draw r${o.r}, n=${o.n}: min-spec = <b>${o.v.toFixed(4)}</b></span><br>`+
    (o.ops&&o.ops.length?`<span class="mono">per member: ${o.ops.map(q=>q[0]+' = '+q[1].toFixed(4)).join(', ')}</span><br>`:'')+
    (o.tup&&o.tup.length?`<span class="mono">injected tuple: ${o.tup.map(w=>w+(o.ranks&&o.ranks[w]!=null?' (rank '+(o.ranks[w]+1)+')':'')).join(', ')} @ ε₀=${sr.eps0} each</span><br>`:'')+
    (o.P?`<span class="mono">placebo reference: ${o.P}</span><br>`:'')+
    (o.nat?`<span class="mono">committed x_nat='${o.nat}' → x'_nat='${o.ja}' (excluded)</span>`:'');
  }));
 }
 $('pcMeans').addEventListener('change',render);
 render();
})();
</script>
<h3>Four-arm contrasts (state-clustered)</h3>
""" + PARTABLES + r"""
<div class="capbox"><b>Headline.</b> S<sup>t</sup> carries and maps 2&ndash;3 subleading operand
hypotheses in one denoising step, each at &asymp;solo strength or better (joint&minus;solo: UU +k
+0.01 n.s., reflection +0.12**, numbers +0.21; no testable family loses). The n=3 ladder refutes strict
budget-sharing: at fixed total mass, per-operand spec is flat as routable hypotheses go 1&rarr;3
(r3/j2 = 0.95 vs predicted 0.33; paired r3&minus;j2 = &minus;0.03 n.s.) and each operand at n=3 beats
its solo anchor (+0.11*). The junk arm reveals an in-band amplification (uppercase junk boosts a solo
operand +0.19***; reflection's lowercase junk does not, +0.02 n.s.) &mdash; so joint&minus;solo, not
joint&minus;junk, is the clean parallelism contrast; mild competition exists only at per-operand 0.30
vs the uppercase-junk comparator and is gone at 0.22 with three hypotheses. &times;k remains untestable
under the guard (needs ~0.45/operand): at 0.30 its solo spec is null/negative, and with the guard
dropped (2&times;0.45 co-dominant, collapsed below) there is no interference either &mdash; &times;k
parallelism is bounded by its weak single-operand transfer, not by competition.</div>
<details><summary><b>Additional regimes &amp; materials</b> <span class="dim">&mdash; summary figure,
n=3 ladder table, max-n extension curves, co-dominant (guard-dropped) &times;k @0.45,
provenance</span></summary>
<img src="figs/par_confoundfree.png" style="max-width:980px;width:100%">
""" + PARLADDER + r"""
<h4>Max-n extension <span class="dim">(dose forced down by the guard; reference tightens to d*=3 at
n<sub>max</sub> &mdash; less controlled than the battery above, kept for the n-extent question)</span></h4>
<img src="figs/par_curves.png" style="max-width:1100px;width:100%">
<h4>Co-dominant n=2 @0.45, subleading guard dropped <span class="dim">(2&times;0.45 = 0.9 squashes the
committed operand to &asymp;0.08 &mdash; superposed overwriting, not subleading hypotheses; solo anchors
= the &epsilon;=0.45 map cells; &times;4 pool cannot host a separation-clean pair)</span></h4>
<img src="figs/par4_noguard.png" style="max-width:760px;width:100%">
""" + PAR4TABLES + r"""
<p class="dim">Provenance: batteries <span class="mono">xtask_par2&ndash;par5</span> (2026-08-12/13, DG
worker), readers <span class="mono">xtask_par*_read.py</span>, assets
<span class="mono">build_par*.py</span>; raw JSONs in <span class="mono">exp/dg_planning/</span>.
Additional sanity checks: range-matched mirror on the &epsilon;=0.45 maps positive (UU3 +0.94,
p=5&times;10<sup>&minus;11</sup>) where the 0.04&ndash;0.05 mirror was null/negative.</p>
</details>
<div class="capbox"><b>Deprecation note.</b> The old interactive NE(n) sweeps (flat
&epsilon;&#8320;=0.04&ndash;0.05, mixed reference, n to 20; collapsed at the end of this section) are
superseded as evidence of parallelism: at that dose the specific channel is below threshold and the
plateau decomposes into answer-set/range preference + copy echo + commitment release. They remain valid
as the diffuse answer-set susceptibility result.</div>
"""


HTML = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta http-equiv="cache-control" content="no-cache, no-store, must-revalidate">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="icon" href="favicon.png">
<title>DG symbol arithmetic — interactive</title>
{KATEX}
<style>{STYLE}{EXTRA}</style></head><body>
<button id="themeToggle">auto theme</button>
<div class="wrap">
<h1>Symbol arithmetic over S<sup>t</sup>: parallel routing of subleading operand hypotheses</h1>
<p class="dim">Inject n simultaneous, strictly-subleading operand hypotheses (flat &epsilon;&#8320; each)
into S<sup>t</sup> at the operand position of one-step "+k" tasks; read the computed images at the answer position
in S<sup>t+1</sup>. Confound-free batteries (placebo- &amp; distance-matched per-operand scoring,
2026-08-13) show genuine parallel routing of n=2&ndash;3 supra-threshold hypotheses at &asymp;solo
strength each; the historical long-n NE plateau (&asymp;0.5 out to n=20 at &epsilon;&#8320;=0.04)
re-reads as a diffuse answer-set response &mdash; see the n&gt;1 section. The task's arithmetic
parameter k is a swept axis: <b>numbers +k</b>, k &isin; {{1&hellip;11}} &cup; {{&minus;3,&minus;8}}
(subtraction), <b>letters +k</b>, k &isin; {{3,5,7,11}} &mdash; headline variant <b>UPPER&rarr;UPPER</b>
(same-case; no case transform) &mdash; and <b>letters &times;k</b>, k &isin; {{&times;2,&times;3,&times;4}},
where the image is the letter at k times the operand's alphabet index, so the map is no longer a
translation. The original case-flip task (lowercase operand &rarr; UPPERCASE image) and the
lower&rarr;lower control are archived in the appendix. Companion
static figure: <span class="mono">figs/xtask_dosecount.png</span>; full arc in
<a href="seasonal.html">seasonal.html</a>.</p>
{NOTATION}
{ALGO}
<h2>Single-injection transfer maps (n=1) — interactive</h2>
<p class="dim"><b>Numbers</b>: dedicated single-injection battery (&epsilon;=0.3, 12 paired draws,
k &isin; {{2,3,4}}). <b>Letters</b>: the headline <b>UPPER&rarr;UPPER</b> variant at the <b>largest injection that
stays strictly subleading</b>, &epsilon;={EPSL:g} (8 paired draws, k &isin; {{3,5,7,11}}, seeds
0&ndash;9, every basis in the pool; the rank&ge;1 guard rejects any injection that would become the
row leader &mdash; at this dose the sheet's committed operand must hold &gt;{EPSL:g}/(1&minus;{EPSL:g})
of the row mass). The original <b>case-flip</b> task and the <b>lower&rarr;lower</b> control are
archived in the appendix, along with the dose sweep explaining why the &epsilon;&#8320;=0.04 maps
showed nothing. Maps default to pooling
over k within each variant (shared color scale); the averaging dropdown resolves them per (variant, k).
Rows = the source whose image is that output &mdash; the aligned output x&prime;&minus;k for the additive
families, the preimage x&prime;/k for the &times;k family below &mdash; shown as
its operand letter (row case follows the variant's source case; out-of-class and out-of-range outputs
pool into <i>other</i>), columns = perturbed subleading input x; per config the incumbent operand
x<sub>nat</sub>, its answer image x&prime;<sub>nat</sub> AND the actually-committed answer letter are
excluded (these differ once: LL3 s0 committed "g, k", an off-by-one of the model's own arithmetic).
Left = argmax histogram of the selected metric, right = its mean over configs. Hover any cell for
exact values and counts; hatched columns = no sampled config.
<b>Below each map, a STRICT ARGMAX duplicate</b>: each config contributes exactly <b>one</b> cell
&mdash; x = the perturbed letter, y = the aligned position of \(\operatorname{{argmax}}_{{x'}} R_c\)
(the argmax output minus k for the additive families, its preimage x&prime;/k for &times;k, the
op&rsquo;s preimage for the operation maps). The argmax is fixed to R (independent of the cell-metric
dropdown) over the 52-letter field minus {{x&prime;<sub>nat</sub>, committed answer}}; diagonal cells
(argmax = exact image) are outlined and the header reports the diagonal share.</p>
{TM_DEF}
<p><label><b>cell metric:</b> <select id="tmMetric">
<option value="R">R = log&#8321;&#8320; P&#772;^pert / P&#772;^base (log ratio)</option>
<option value="dP">&Delta;P = P&#772;^pert &minus; P&#772;^base (linear)</option>
<option value="logPp">log&#8321;&#8320; P&#772;^pert</option>
<option value="logPb">log&#8321;&#8320; P&#772;^base</option>
<option value="Pp">P&#772;^pert (linear)</option>
<option value="Pb">P&#772;^base (linear)</option>
</select></label>
&nbsp;&nbsp; <label><b>averaging:</b> <select id="tmAgg">
<option value="pool">pool over k (aligned rows: x&prime;&minus;k additive, x&prime;/k multiplicative)</option>
<option value="perk">resolve per k (one panel row per shift)</option>
</select></label> <span class="dim">(floors at 10<sup>&minus;5</sup> apply inside logs only)</span></p>
<h3>numbers +k</h3>
<div class="card"><div id="tmapNum"></div><div id="tmapNumS" style="margin-top:16px"></div></div>
<h3>letters +k (UPPER&rarr;UPPER) at &epsilon; = {EPSL:g} <span class="dim">&mdash; headline variant;
case-flip and lower&rarr;lower maps archived in the appendix</span></h3>
<div class="card"><div id="tmapLet"></div><div id="tmapLetS" style="margin-top:16px"></div></div>
<h3>letters &times;k (UPPER&rarr;UPPER) at &epsilon; = {EPSM:g} <span class="dim">&mdash; multiplicative
image, rows = preimage x&prime;/k</span></h3>
<p class="dim">Same protocol and same dose as the additive maps, but the task asks for the letter at
<b>k times</b> the operand's alphabet index (A=1&hellip;Z=26) rather than k positions later, so the
image map is no longer a translation. Rows are therefore the <b>preimage</b> x&prime;/k &mdash; the
source letter whose image is that output &mdash; and outputs whose position is not divisible by k (or
whose preimage falls outside the pool) collect in <i>other</i>, which is why the <i>other</i> row holds
most of the field here and grows with k. The diagonal still reads the same way: a lit (row=x, col=x)
cell means injecting x at the operand position raised exactly the letter at k&middot;pos(x). Pools shrink
with k since pos&middot;k &le; 26: k=&times;2 &rarr; A&hellip;M, &times;3 &rarr; A&hellip;H,
&times;4 &rarr; A&hellip;F. DG computes this map natively (36/36 correct in a 2-phrasing &times;
3-multiplier &times; 6-seed capability probe), so a null diagonal here would mean absent transfer, not
absent arithmetic.</p>
<div class="card"><div id="tmapMul"></div><div id="tmapMulS" style="margin-top:16px"></div></div>
{MULTFIND}

<h2 style="margin-top:30px">n &gt; 1 parallelism</h2>
{PARFREE}
<details style="margin-top:24px"><summary><b>Deprecated interactive NE(n) sweeps</b> <span class="dim">(mixed
reference, &epsilon;&#8320;=0.04&ndash;0.05, n to 20 &mdash; REPLACED by the controlled curves above;
superseded for parallelism, see deprecation note; kept collapsed for provenance and as the diffuse
answer-set susceptibility result)</span></summary>
{CAPTION}
<p>
<label><input type="checkbox" id="meansOnly"> <b>means + 95% CI only</b></label> &nbsp;&nbsp;
<label><input type="checkbox" id="splitK"> <b>split by k</b> (rows)</label> &nbsp;&nbsp;
<label><input type="checkbox" id="splitOp"> <b>split by basis b</b> (one panel per basis)</label>
&nbsp;&nbsp; <label><b>E =</b> <select id="eT">
<option value="T">⟨R⟩_T (targets)</option>
<option value="near">⟨R⟩_nearNT (near-band non-targets)</option>
</select> &minus; <select id="eN">
<option value="mixed">⟨R⟩_N (mixed non-targets)</option>
<option value="near">⟨R⟩_nearNT (near-band non-targets)</option>
<option value="far">⟨R⟩_far (far band)</option>
</select></label><br>
<span class="dim">Both terms of E are selectable: the first term is the target set T or the near-band
non-targets; the second term (the reference N) is the mixed non-target field, the near band, or the far
band. Presets: mixed = (T, N); band-free spec = (T, nearNT); pure band contrast = (nearNT, far). Each
group renders three panels side by side: the two &lang;R&rang; terms (orange = first, blue = second),
then E = R&#8321; &minus; R&#8322;, then NE = E/&Sigma;&epsilon;. Near band = UPPERCASE (letters) / unit
words (numbers; the numbers mixed N is already units-only); far band = lowercase / tens. Component-panel
dots colored by term; E/NE dots by shift k (lowered alpha). Per-basis splits show (R(img(b)) &minus;
&lang;R&rang;<sub>N</sub>)/&Sigma;&epsilon; with the selected N when the first term is T, else the
basis-independent cell value.</span><br>
<span class="dim">pooled view pools over k, base subsets and seeds; split-by-k renders one row per shift;
split-by-basis renders one small panel per basis b with per-basis value
(R(img(b)) &minus; &lang;R&rang;<sub>N_c</sub>)/&Sigma;&epsilon;, grey = group pooled mean; BOTH splits
together render the fixed matrix: k = rows, basis b = columns (aligned; dashed = basis outside that k's
pool). Dot color = sheet-seed. Click any dot for the cell inspector.</span><br>
<span class="dim"><b>Each group is duplicated twice with stricter criteria</b> (same T/N dropdown
selections): <b>STRICT</b> replaces the means with \(E^{{\min}}_c = \min_{{T}}R_c - \max_{{N}}R_c\)
&mdash; positive only when <i>every</i> target beats <i>every</i> reference letter &mdash; and
<b>MAX-vs-MAX</b> with \(E^{{\max}}_c = \max_{{T}}R_c - \max_{{N}}R_c\)
(positive exactly when the top letter over T&cup;N is a target &mdash; the triptych analogue of the
strict argmax maps); each duplicate renders its own
term, E and NE panels. (The duplicates are not repeated inside the split-by-basis views.)</span></p>
<div class="legend"><span>dot color = shift k (viridis: dark = lowest k &rarr; yellow = highest;
hover any dot for exact k / seed / draw)</span>
<span><span class="sw" style="background:var(--accent)"></span>connected line = per-level MEAN;
box = min&rarr;max extent, thick bar = MEDIAN, ticks = quartiles</span></div>
<h2>numbers +k <span class="dim">(&epsilon;&#8320;=0.05)</span></h2>
<div class="card" id="num"></div>
<h2>natural operand choice <span class="dim">(x<sub>nat</sub> across UPPER&rarr;UPPER captures;
stack segments = task shift k, viridis; number under each bar = total)</span></h2>
<p class="dim">The letter the model commits at the operand position A in each hot-regime rollout
&mdash; i.e. the x<sub>nat</sub> that every battery and map excludes. The model has a strong
favourite; operand diversity across seeds comes almost entirely from sheet stochasticity, not from
the choice itself.</p>
<div class="card" id="nathist"></div>
<h2>letters +k (UPPER&rarr;UPPER) <span class="dim">(&epsilon;&#8320;=0.04 &mdash; headline letters
variant, k &isin; {{3,5,7,11}}, seeds 0&ndash;9; case-flip and lower&rarr;lower archived in the
appendix)</span></h2>
{BANDCAV}
<div class="card" id="uu"></div>
<h2>letters &times;k (UPPER&rarr;UPPER) <span class="dim">(&epsilon;&#8320;=0.04 &mdash; multiplicative
image, k &isin; {{&times;2,&times;3,&times;4}}, seeds 0&ndash;9)</span></h2>
<p class="dim">The same NE(n) sweep with a <b>multiplicative</b> image map (x &rarr; letter at
k&middot;pos(x)) instead of an additive shift, at the identical &epsilon;&#8320;=0.04, so NE is directly
comparable to the +k families above. The target set T<sub>c</sub> is smaller and more scattered here
(images are the multiples of k, so at &times;4 only six sources have an in-range image), which shrinks
the number of injectable operands per task: 12 / 7 / 5 for &times;2 / &times;3 / &times;4 &mdash; hence
n runs to 12 rather than 18.</p>
<div class="card" id="mu"></div>
<h3 class="dim">natural operand choice, &times;k tasks</h3>
<div class="card" id="nathistMu"></div>
</details>
{OPSEC}
<h2>Cell inspector</h2>
<div class="card det" id="detail"><span class="dim">click a dot: provenance (incl. k), natural generation,
injected operands with masses and ranks, S^t before&rarr;after, state-vector grid with R_v column,
per-draw E sina</span></div>
<h2>Letters n=1 specificity (context for the transfer maps above)</h2>
<p class="dim">The missing letters diagonal at n=1 is REAL, not a display artifact: at
&epsilon;&#8320;=0.04 there is no detectable letter-specific transmission. Within the uppercase band the
target is indistinguishable from the other uppercase letters (paired edge &minus;0.003&plusmn;0.009,
p=0.62, 68 cells; aligned-offset profile flat: &delta;=0 gives +0.081 vs +0.084&plusmn;0.002 at
|&delta;|&ge;3). What the perturbation does move is the case band wholesale: uppercase non-targets lift
+0.084 vs lowercase +0.014 (p=9&times;10<sup>&minus;8</sup>) &mdash; a diffuse answer-band susceptibility
response. (An apparent +0.033 target edge vs the mixed 50-candidate field is entirely this band effect
&mdash; caveat also applies to n=1 points of letters curves referenced against lowercase/untouched-case.)
Letter-specific transmission emerges with total dose: at n=8 (&Sigma;&epsilon;=0.32) the within-uppercase
edge is +0.084&plusmn;0.015 (p=3&times;10<sup>&minus;6</sup>), and the numbers map at &epsilon;=0.3 shows
the diagonal outright. Static PNG versions: <span class="mono">figs/xtask_confusion_effect.png</span>,
<span class="mono">figs/xtask_confusion_effect_letters.png</span>.</p>
{APPENDIX}
</div>
<script>window.__DATA__ = {payload};</script>
<script>{JS}</script>
</body></html>"""

OUT.write_text(HTML)
print("wrote", OUT, f"{OUT.stat().st_size/1e6:.2f} MB")
