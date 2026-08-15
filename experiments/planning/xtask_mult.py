"""MULTIPLICATIVE uppercase-letter battery: image = letter at alphabet position k TIMES the
operand's (A=1 ... Z=26), instead of the additive shift used by every task so far. Same paired
one-step probe as xtask_uu_10x (capture -> base rows -> n-sweep at eps0 -> n=1 every-basis map
at eps_max), so NE units are directly comparable across task families.

Pools shrink with k (need pos*k <= 26): k=2 -> A..M (13), k=3 -> A..H (8), k=4 -> A..F (6).
Phrasing = the 'mul' variant validated by xtask_mult_probe (36/36 parsed, 36/36 correct); it
keeps a worked 'A=1,B=2,...' key OUT of the prompt so no extra letter tokens contaminate the
answer position. -> exp/dg_planning/xtask_mult{,_nsweep}.json"""
import os
import json, string, time, urllib.request
from pathlib import Path

import numpy as np

W = os.environ.get("DG_WORKER", "http://localhost:18711")
EXP = Path(os.environ.get("DG_PLANNING_DIR", str(Path(__file__).resolve().parent / "exp")))
FRAME = "{q}\n\nOutput ONLY the text itself — no preamble, no quotes, no explanation."
HOTR = dict(T=64, C=128, t_max=1.3, t_min=0.8, entropy_bound=0.3, early_stop=False, top_k=10)
VOCAB = 262144
DRAWS = 8
T = 2
NREP = 3
NLVL = [1, 2, 3, 4, 6, 8, 10, 12]
EPS0 = 0.04
EPSM = 0.45
SEEDS = list(range(10))

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("google/gemma-4-26b-a4b-it")
def tid(s):
    t = tok.encode(s, add_special_tokens=False)
    assert len(t) == 1
    return t[0]

LOW = list(string.ascii_lowercase); UPP = list(string.ascii_uppercase)
F52 = LOW + UPP

MUQ = ("Pick any uppercase letter from A to {hi}, write it, then multiply its alphabet index "
       "by {k} and write the uppercase letter at that index, separated by a comma. "
       "Begin your answer with 'Letters:'.")
TASKS = {
    "MU2": dict(k=2, hi="M", prompt=MUQ.format(hi="M", k=2)),
    "MU3": dict(k=3, hi="H", prompt=MUQ.format(hi="H", k=3)),
    "MU4": dict(k=4, hi="F", prompt=MUQ.format(hi="F", k=4)),
}


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


def main():
    fmu, fns = EXP / "xtask_mult.json", EXP / "xtask_mult_nsweep.json"
    dmu = json.load(open(fmu)) if fmu.exists() else {}
    dns = json.load(open(fns)) if fns.exists() else {}
    temp = 1.3 + (0.8 - 1.3) * T / 63
    ids = {w: tid(" " + w) for w in F52}
    probe = sorted(set(ids.values()))
    pix = {v: i for i, v in enumerate(probe)}
    ncap = nsw = nmp = nskip = 0
    for tag, cfg in TASKS.items():
        pool = [c for c in UPP if c <= cfg["hi"]]
        for s in SEEDS:
            fp = EXP / f"nego2/{tag}__s{s}.json"
            if not fp.exists():
                d = post("sample", dict(prompt=FRAME.format(q=cfg["prompt"]), seed=s, **HOTR, s_topk_record=32))
                slim = {kk: d[kk] for kk in ("final_ids", "final_text", "pad_token_id", "eos_token_ids",
                                             "id2str", "canvas_length", "s_rec")}
                slim["steps_argmax"] = [st["argmax"] for st in d["steps"]]
                slim["tag"] = tag; slim["seed"] = s; slim["q"] = cfg["prompt"]
                json.dump(slim, open(fp, "w")); ncap += 1
                print(f"cap {tag} s{s}: {d['final_text'].splitlines()[-1][:70]!r}", flush=True)
            d = json.load(open(fp))
            i2s, fin = d["id2str"], d["final_ids"]
            dead = set(d["eos_token_ids"]) | {d["pad_token_id"]}
            live = [p for p, x in enumerate(fin) if x not in dead]
            txt = lambda x: i2s.get(str(x), "?").replace("▁", " ")
            pc = next((p for p in live if txt(fin[p]).strip() == ","), None)
            if pc is None:
                print(f"{tag} s{s}: no comma — SKIP", flush=True); continue
            A = max(p for p in live if p < pc)
            B = next(p for p in live if p > pc)
            nat = txt(fin[A]).strip()
            if nat not in pool:
                print(f"{tag} s{s}: nat {nat!r} outside pool — SKIP", flush=True); continue
            draft = d["steps_argmax"][T]
            sheet0 = {"ids": d["s_rec"]["ids"][T], "lp": d["s_rec"]["lp"][T]}
            rng0 = np.random.default_rng(77 + s)
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
                return [[round(float(np.exp(r[B][pix[ids[w]]])), 6) for w in F52] for r in rows]

            bk = f"{tag}|s{s}|base"
            if bk not in dmu:
                base_rows = brows(sheet0)
                dmu[bk] = dict(nat=nat, A=A, B=B, rows=base_rows)
                json.dump(dmu, open(fmu, "w"))
            if bk not in dns:
                dns[bk] = dict(nat_op=nat, A=A, B=B, rows=dmu[bk]["rows"])
                json.dump(dns, open(fns, "w"))
            ops = [o for o in pool if o != nat]

            def inject(subset, eps):
                ids2 = [list(r) for r in sheet0["ids"]]
                lp2 = [list(r) for r in sheet0["lp"]]
                p = np.exp(np.array(lp2[A], dtype=float)) * (1.0 - eps * len(subset))
                row = list(ids2[A])
                for w in subset:
                    tk_ = ids[w]
                    if tk_ in row:
                        p[row.index(tk_)] += eps
                    else:
                        j = int(np.argmin(p)); row[j] = tk_; p[j] = eps
                ranks = {w: int(np.sum(p > p[row.index(ids[w])])) for w in subset}
                ids2[A] = row
                lp2[A] = [float(x) for x in np.log(np.maximum(p, 1e-12))]
                return {"ids": ids2, "lp": lp2}, ranks

            for n in [x for x in NLVL if x <= len(ops)]:
                seedstr = f"{tag}|{s}|mun|{n}"
                rng = np.random.default_rng(sum(ord(c) * (i + 7) for i, c in enumerate(seedstr)))
                for rep in range(NREP):
                    key = f"{tag}|s{s}|ext|n{n}|e{rep}"
                    if key in dns:
                        continue
                    subset = list(rng.choice(ops, size=n, replace=False))
                    sheet, ranks = inject(subset, EPS0)
                    dns[key] = dict(subset=subset, ranks=ranks, rows=brows(sheet))
                    json.dump(dns, open(fns, "w")); nsw += 1
            for b in ops:
                key = f"{tag}|s{s}|e{EPSM}|b1|{b}"
                if key in dmu:
                    continue
                sheet, ranks = inject([b], EPSM)
                if ranks[b] == 0:
                    dmu[key] = dict(skipped=True); json.dump(dmu, open(fmu, "w")); nskip += 1; continue
                dmu[key] = dict(eps0=EPSM, subset=[b], ranks=ranks, rows=brows(sheet))
                json.dump(dmu, open(fmu, "w")); nmp += 1
            print(f"{tag} s{s}: done (nat {nat}, {len(ops)} ops) [nsweep {nsw}, map {nmp}, map-skip {nskip}]",
                  flush=True)
    print(f"MULT DONE: {ncap} captures, {nsw} nsweep cells, {nmp} map cells, {nskip} rank-0", flush=True)


if __name__ == "__main__":
    main()
