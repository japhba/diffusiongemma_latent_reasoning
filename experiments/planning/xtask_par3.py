"""par3: (A) n=3 budget ladder on UU3/UU5 and (B/C) n=2 four-arm battery on the other
task families (MU2/MU3, RF variants, KB variants, MN3, numbers P3).

A) eps0=0.22, arms at matched TOTAL mass 0.66 except the solo anchor:
     n1  = {x1}@0.22
     j2  = {x1, junk1, junk2}   (1 routable hypothesis, total 0.66)
     r2j = {x1, x2, junk1}      (2 routable, total 0.66)
     r3  = {x1, x2, x3}         (3 routable, total 0.66)
   Budget model predicts spec1(j2) : spec1(r2j) : spec1(r3) = 1 : 1/2 : 1/3.
   Canvases rng 73+s (pairs with xtask_samecase_nsweep base rows).
B) eps0=0.3, arms n1a/n1b/n2/jk exactly as par2; guard-passing states only
   (MU2 minus s3/s4; RF/RFB/RFC + RFD s0; KB s0 + KBB; MN3). MU4 skipped: pool too small
   for separation-clean pairs. Fresh base rows recorded per state (canvases rng 77+s mult,
   83+s ops). RF junk = lowercase (full 26-letter pool leaves no imageless uppercase).
C) numbers P3 (+3, pool two..seventeen), eps0=0.25 (weak leaders), junk in
   {eighteen, nineteen, twenty}, fresh bases, canvases rng 91+s.
Keys {tag}|s{s}|par3|r{r}|{arm} (+ |pairs, |base) -> exp/dg_planning/xtask_par3.json.
"""
import os
import json, string, time, urllib.request
from pathlib import Path

import numpy as np

W = os.environ.get("DG_WORKER", "http://localhost:18711")
EXP = Path(os.environ.get("DG_PLANNING_DIR", str(Path(__file__).resolve().parent / "exp")))
FRAME = "{q}\n\nOutput ONLY the text itself — no preamble, no quotes, no explanation."
VOCAB = 262144
DRAWS = 8
T = 2
NREP = 4

from tokenizers import Tokenizer
tok = Tokenizer.from_pretrained("google/gemma-4-26b-a4b-it")
def tid(s):
    ids = tok.encode(s, add_special_tokens=False).ids
    assert len(ids) == 1, (s, ids)
    return ids[0]

LOW = list(string.ascii_lowercase); UPP = list(string.ascii_uppercase)
F52 = LOW + UPP
UNITS = ("two three four five six seven eight nine ten eleven twelve thirteen fourteen "
         "fifteen sixteen seventeen eighteen nineteen twenty").split()
TENS = "thirty forty fifty sixty seventy eighty ninety".split()
N26 = UNITS + TENS

dns = json.load(open(EXP / "xtask_samecase_nsweep.json"))
dmun = json.load(open(EXP / "xtask_mult_nsweep.json"))
dopn = json.load(open(EXP / "xtask_ops_nsweep.json"))
d12 = json.load(open(EXP / "xtask_compute12.json"))
dops = json.load(open(EXP / "xtask_ops.json"))
OPMETA = {k.split("|")[0]: v for k, v in dops.items() if k.endswith("|meta")}

UUQ = ("Pick any uppercase letter between A and {hi}, write it, then write the letter "
       "{off} positions later in the alphabet, also in uppercase, separated by a comma. "
       "Begin your answer with 'Letters:'.")
MUQ = ("Pick any uppercase letter from A to {hi}, write it, then multiply its alphabet index "
       "by {k} and write the uppercase letter at that index, separated by a comma. "
       "Begin your answer with 'Letters:'.")
P3Q = ("Pick any number between two and seventeen, write it in words, then write the number "
       "three greater in words, separated by a comma. Begin your answer with 'Numbers:'.")


def img_shift(field, k):
    def f(w):
        i = field.index(w) + k
        return field[i] if 0 <= i < len(field) else None
    return f


def img_mu(k):
    def f(w):
        i = (UPP.index(w) + 1) * k
        return UPP[i - 1] if 1 <= i <= 26 else None
    return f


# family configs. kind: "n3" -> ladder arms; "n2" -> par2 arms.
FAM = {}
FAM["UU3"] = dict(kind="n3", prompt=UUQ.format(hi="W", off="three"), field=F52, band=UPP,
                  img=img_shift(UPP, 3), pool=UPP[:23], dmin=7, sep=4, eps0=0.22,
                  junkpool=UPP[23:], src=dns, rngb=73, seeds=range(10), fresh_base=False)
FAM["UU5"] = dict(kind="n3", prompt=UUQ.format(hi="U", off="five"), field=F52, band=UPP,
                  img=img_shift(UPP, 5), pool=UPP[:21], dmin=9, sep=4, eps0=0.22,
                  junkpool=UPP[21:], src=dns, rngb=73, seeds=range(10), fresh_base=False)
_mu2img = img_mu(2)
FAM["MU2"] = dict(kind="n2", prompt=MUQ.format(hi="M", k=2), field=F52, band=UPP,
                  img=_mu2img, pool=UPP[:13], dmin=5, sep=4, eps0=0.3,
                  junkpool=[w for w in UPP[13:] if w not in {_mu2img(x) for x in UPP[:13]}],
                  src=dmun, rngb=77, seeds=[0, 1, 2, 5, 6, 7, 8, 9], fresh_base=True)
_mu3img = img_mu(3)
FAM["MU3"] = dict(kind="n2", prompt=MUQ.format(hi="H", k=3), field=F52, band=UPP,
                  img=_mu3img, pool=UPP[:8], dmin=4, sep=3, eps0=0.3,
                  junkpool=[w for w in UPP[8:] if w not in {_mu3img(x) for x in UPP[:8]}],
                  src=dmun, rngb=77, seeds=range(10), fresh_base=True)
for tg, sds in (("RF", range(6)), ("RFB", range(2)), ("RFC", range(2)), ("RFD", [0])):
    FAM[tg] = dict(kind="n2", prompt=OPMETA[tg]["prompt"], field=F52, band=UPP,
                   img=lambda w, _im=OPMETA[tg]["image"]: _im.get(w), pool=list(OPMETA[tg]["pool"]),
                   dmin=5, sep=4, eps0=0.3, junkpool="LOWER", src=dopn, rngb=83,
                   seeds=sds, fresh_base=True)
for tg, sds in (("KB", [0]), ("KBB", range(2))):
    FAM[tg] = dict(kind="n2", prompt=OPMETA[tg]["prompt"], field=F52, band=UPP,
                   img=lambda w, _im=OPMETA[tg]["image"]: _im.get(w), pool=list(OPMETA[tg]["pool"]),
                   dmin=5, sep=4, eps0=0.3, junkpool=["P", "L", "M"], src=dopn, rngb=83,
                   seeds=sds, fresh_base=True)
FAM["MN3"] = dict(kind="n2", prompt=OPMETA["MN3"]["prompt"], field=F52, band=UPP,
                  img=lambda w, _im=OPMETA["MN3"]["image"]: _im.get(w), pool=list(OPMETA["MN3"]["pool"]),
                  dmin=7, sep=4, eps0=0.3, junkpool=["A", "B", "C"], src=dopn, rngb=83,
                  seeds=range(2), fresh_base=True)
FAM["P3"] = dict(kind="n2", prompt=P3Q, field=N26, band=UNITS,
                 img=img_shift(UNITS, 3), pool=UNITS[:16], dmin=7, sep=4, eps0=0.25,
                 junkpool=UNITS[16:], src=d12, rngb=91, seeds=range(4), fresh_base=True)


def post(path, req, timeout=1800):
    for a in range(6):
        try:
            r = urllib.request.urlopen(urllib.request.Request(
                f"{W}/{path}", json.dumps(req).encode(), {"Content-Type": "application/json"}), timeout=timeout)
            return json.loads(r.read())
        except Exception as e:
            print(f"  retry {a}: {type(e).__name__}", flush=True)
            time.sleep(15 * (a + 1))
    raise RuntimeError("worker unreachable")


def ok_pair(cfg, pi, ja, jb, x1, x2):
    t1, t2 = cfg["img"](x1), cfg["img"](x2)
    if not t1 or not t2 or t1 == t2 or t1 in (x2, ja, jb) or t2 in (x1, ja, jb):
        return False
    if abs(pi[x1] - pi[x2]) < cfg["dmin"]:
        return False
    return abs(pi[t1] - pi[x2]) >= cfg["sep"] and abs(pi[t2] - pi[x1]) >= cfg["sep"]


def junks_for(cfg, pi, ja, jb, x1, need):
    if cfg["junkpool"] == "LOWER":
        return [LOW[(pi[x1] + 13 + 3 * i) % 26] for i in range(need)]
    t1 = cfg["img"](x1)
    js = [j for j in cfg["junkpool"] if j not in (ja, jb)
          and abs(pi[j] - pi[x1]) >= 5 and abs(pi[j] - pi[t1]) >= 4]
    js.sort(key=lambda j: -min(abs(pi[j] - pi[x1]), abs(pi[j] - pi[t1])))
    return js[:need] if len(js) >= need else None


def main():
    out, f = {}, EXP / "xtask_par3.json"
    if f.exists():
        out = json.load(open(f))
    temp = 1.3 + (0.8 - 1.3) * T / 63
    t0 = time.time()
    for tag, cfg in FAM.items():
        field = cfg["field"]
        ids = {w: tid(" " + w) for w in field}
        probe = sorted(set(ids.values()))
        pix = {v: i for i, v in enumerate(probe)}
        pi = {w: i for i, w in enumerate(cfg["band"])}
        for s in cfg["seeds"]:
            bk = f"{tag}|s{s}|base"
            if bk not in cfg["src"]:
                continue
            d = json.load(open(EXP / f"nego2/{tag}__s{s}.json"))
            nat, A, B = cfg["src"][bk]["nat_op"], cfg["src"][bk].get("A", 6), cfg["src"][bk].get("B", 8)
            ja = cfg["img"](nat)
            jb = d["id2str"].get(str(d["final_ids"][B]), "?").replace("▁", " ").strip()
            ops = [w for w in cfg["pool"] if w != nat]
            rng = np.random.default_rng(sum(ord(c) * (i + 7) for i, c in enumerate(f"{tag}|{s}|par3")))
            # ---- draw reps
            reps = []
            if cfg["kind"] == "n2":
                cand = [(a, b) for a in ops for b in ops
                        if a != b and ok_pair(cfg, pi, ja, jb, a, b) and junks_for(cfg, pi, ja, jb, a, 1)]
                rng.shuffle(cand)
                used = set()
                for a, b in cand:
                    if a in used:
                        continue
                    reps.append(dict(x=[a, b], junks=junks_for(cfg, pi, ja, jb, a, 1)))
                    used.add(a)
                    if len(reps) == NREP:
                        break
                for a, b in cand:
                    if len(reps) == NREP:
                        break
                    if not any(r["x"] == [a, b] for r in reps):
                        reps.append(dict(x=[a, b], junks=junks_for(cfg, pi, ja, jb, a, 1)))
            else:
                cand = [(a, b, c) for a in ops for b in ops for c in ops
                        if len({a, b, c}) == 3
                        and ok_pair(cfg, pi, ja, jb, a, b) and ok_pair(cfg, pi, ja, jb, a, c)
                        and ok_pair(cfg, pi, ja, jb, b, c) and junks_for(cfg, pi, ja, jb, a, 2)]
                rng.shuffle(cand)
                used = set()
                for a, b, c in cand:
                    if a in used:
                        continue
                    reps.append(dict(x=[a, b, c], junks=junks_for(cfg, pi, ja, jb, a, 2)))
                    used.add(a)
                    if len(reps) == NREP:
                        break
                for a, b, c in cand:
                    if len(reps) == NREP:
                        break
                    if not any(r["x"] == [a, b, c] for r in reps):
                        reps.append(dict(x=[a, b, c], junks=junks_for(cfg, pi, ja, jb, a, 2)))
            if not reps:
                print(f"{tag} s{s}: no valid reps, skipped", flush=True)
                continue
            pk = f"{tag}|s{s}|par3|pairs"
            if pk not in out:
                out[pk] = [dict(x=r["x"], junks=r["junks"]) for r in reps]
                json.dump(out, open(f, "w"))
            # ---- canvases + sheets
            draft = d["steps_argmax"][T]
            sheet0 = {"ids": d["s_rec"]["ids"][T], "lp": d["s_rec"]["lp"][T]}
            rng0 = np.random.default_rng(cfg["rngb"] + s)
            noise = rng0.integers(0, VOCAB, size=(DRAWS, 2))
            canvases = []
            for dd in range(DRAWS):
                cv = list(draft)
                cv[A] = int(noise[dd][0]); cv[B] = int(noise[dd][1])
                canvases.append(cv)

            def brows(sheet):
                rows = []
                for i in range(0, DRAWS, 2):
                    rows += post("energy", {"prompt": FRAME.format(q=cfg["prompt"]),
                                            "canvases": canvases[i:i + 2], "probe_ids": probe,
                                            "s_sparse": sheet, "temperature": temp})["probe"]
                return [[round(float(np.exp(r[B][pix[ids[w]]])), 6) for w in field] for r in rows]

            bk3 = f"{tag}|s{s}|par3|base"
            if cfg["fresh_base"] and bk3 not in out:
                out[bk3] = dict(nat_op=nat, A=A, B=B, rows=brows(sheet0))
                json.dump(out, open(f, "w"))

            for r, rp in enumerate(reps):
                if cfg["kind"] == "n2":
                    x1, x2 = rp["x"]; ju = rp["junks"][0]
                    arms = dict(n1a=[x1], n1b=[x2], n2=[x1, x2], jk=[x1, ju])
                else:
                    x1, x2, x3 = rp["x"]; j1, j2_ = rp["junks"]
                    arms = dict(n1=[x1], j2=[x1, j1, j2_], r2j=[x1, x2, j1], r3=[x1, x2, x3])
                for arm, subset in arms.items():
                    key = f"{tag}|s{s}|par3|r{r}|{arm}"
                    if key in out:
                        continue
                    ids2 = [list(rr) for rr in sheet0["ids"]]
                    lp2 = [list(rr) for rr in sheet0["lp"]]
                    p = np.exp(np.array(lp2[A], dtype=float)) * (1.0 - cfg["eps0"] * len(subset))
                    row = list(ids2[A])
                    for w in subset:
                        tk_ = ids[w]
                        if tk_ in row:
                            p[row.index(tk_)] += cfg["eps0"]
                        else:
                            j = int(np.argmin(p)); row[j] = tk_; p[j] = cfg["eps0"]
                    ranks = {w: int(np.sum(p > p[row.index(ids[w])])) for w in subset}
                    ids2[A] = row
                    lp2[A] = [float(x) for x in np.log(np.maximum(p, 1e-12))]
                    out[key] = dict(subset=subset, eps0=cfg["eps0"], ranks=ranks,
                                    rows=brows({"ids": ids2, "lp": lp2}))
                    json.dump(out, open(f, "w"))
            ncells = sum(1 for kk in out if "|par3|r" in kk)
            print(f"{tag} s{s}: done ({ncells} cells, {time.time()-t0:.0f}s)", flush=True)
    print("PAR3 DONE", flush=True)


if __name__ == "__main__":
    main()
