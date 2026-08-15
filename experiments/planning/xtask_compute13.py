"""k-sweep power extension: bring ALL additive shifts to 4 sheet-seeds with fresh subset
draws (numbers k in {2,5,6,7,9} via P2/A5/P6/A7/P9; letters k=7 via L7). Captures missing
seeds s2/s3, then flat-eps deep series, 3 independent subset draws per (seed, n), keys
{tag}|s{s}|ext|n{n}|e{rep}. -> exp/dg_planning/xtask_compute13.json"""
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
NLVL = [1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18]

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("google/gemma-4-26b-a4b-it")
def tid(s):
    t = tok.encode(s, add_special_tokens=False)
    assert len(t) == 1
    return t[0]

UNITS = ("two three four five six seven eight nine ten eleven twelve thirteen fourteen "
         "fifteen sixteen seventeen eighteen nineteen twenty").split()
TENS = "thirty forty fifty sixty seventy eighty ninety".split()
N26 = UNITS + TENS
LOW = list(string.ascii_lowercase); UPP = list(string.ascii_uppercase)
F52 = LOW + UPP

NQ = ("Pick any number between two and {hi}, write it in words, then write the number "
     "{off} greater in words, separated by a comma. Begin your answer with 'Numbers:'.")
LQ = ("Pick any lowercase letter between a and {hi}, write it, then write the letter "
      "{off} positions later in the alphabet in uppercase, separated by a comma. "
      "Begin your answer with 'Letters:'.")
TASKS = {
    "P2": dict(prompt=NQ.format(hi="eighteen", off="two"), dom="num", k=2, hi=18, eps0=0.05),
    "A5": dict(prompt=NQ.format(hi="fifteen", off="five"), dom="num", k=5, hi=15, eps0=0.05),
    "P6": dict(prompt=NQ.format(hi="fourteen", off="six"), dom="num", k=6, hi=14, eps0=0.05),
    "A7": dict(prompt=NQ.format(hi="thirteen", off="seven"), dom="num", k=7, hi=13, eps0=0.05),
    "P9": dict(prompt=NQ.format(hi="eleven", off="nine"), dom="num", k=9, hi=11, eps0=0.05),
    "L7": dict(prompt=LQ.format(hi="s", off="seven"), dom="let", k=7, hi="s", eps0=0.04),
}


def post(path, req, timeout=1800):
    for a in range(5):
        try:
            r = urllib.request.urlopen(urllib.request.Request(
                f"{W}/{path}", json.dumps(req).encode(), {"Content-Type": "application/json"}), timeout=timeout)
            return json.loads(r.read())
        except Exception as e:
            print(f"  retry {a}: {type(e).__name__}", flush=True)
            time.sleep(8 * (a + 1))
    raise RuntimeError("worker unreachable")


def main():
    out, f = {}, EXP / "xtask_compute13.json"
    if f.exists():
        out = json.load(open(f))
    temp = 1.3 + (0.8 - 1.3) * T / 63
    for tag, cfg in TASKS.items():
        num = cfg["dom"] == "num"
        field = N26 if num else F52
        ids = {w: tid(" " + w) for w in field}
        probe = sorted(set(ids.values()))
        pix = {v: i for i, v in enumerate(probe)}
        if num:
            pool = [w for w in UNITS if 2 <= UNITS.index(w) + 2 <= cfg["hi"]]
        else:
            pool = [c for c in LOW if c <= cfg["hi"]]
        for s in range(4):
            fp = EXP / f"nego2/{tag}__s{s}.json"
            if not fp.exists():
                d = post("sample", dict(prompt=FRAME.format(q=cfg["prompt"]), seed=s, **HOTR,
                                        s_topk_record=32))
                slim = {kk: d[kk] for kk in ("final_ids", "final_text", "pad_token_id",
                                             "eos_token_ids", "id2str", "canvas_length", "s_rec")}
                slim["steps_argmax"] = [st["argmax"] for st in d["steps"]]
                json.dump(slim, open(fp, "w"))
                print(f"cap {tag} s{s}: {d['final_text'].split(chr(10))[-1][:60]!r}", flush=True)
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
            nat = nat.lower() if num else nat
            if nat not in pool:
                print(f"{tag} s{s}: nat {nat!r} outside pool — SKIP", flush=True); continue
            print(f"{tag} s{s}: natural {nat}, {txt(fin[B]).strip()}", flush=True)
            ops = [o for o in pool if o != nat]
            draft = d["steps_argmax"][T]
            sheet0 = {"ids": d["s_rec"]["ids"][T], "lp": d["s_rec"]["lp"][T]}
            rng0 = np.random.default_rng((67 if num else 71) + s)
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
                return [[round(float(np.exp(r[B][pix[ids[w]]])), 5) for w in field] for r in rows]

            bk = f"{tag}|s{s}|base"
            if bk not in out:
                out[bk] = dict(nat_op=nat, A=A, B=B, rows=brows(sheet0))
                json.dump(out, open(f, "w"))
            for n in [x for x in NLVL if x <= len(ops)]:
                seedstr = f"{tag}|{s}|ext13|{n}"
                rng = np.random.default_rng(sum(ord(c) * (i + 7) for i, c in enumerate(seedstr)))
                for rep in range(NREP):
                    key = f"{tag}|s{s}|ext|n{n}|e{rep}"
                    if key in out:
                        continue
                    subset = list(rng.choice(ops, size=n, replace=False))
                    ids2 = [list(r) for r in sheet0["ids"]]
                    lp2 = [list(r) for r in sheet0["lp"]]
                    p = np.exp(np.array(lp2[A], dtype=float)) * (1.0 - cfg["eps0"] * n)
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
                    out[key] = dict(subset=subset, ranks=ranks, rows=brows({"ids": ids2, "lp": lp2}))
                    json.dump(out, open(f, "w"))
            print(f"{tag} s{s}: done (pool {len(ops)})", flush=True)
    print("COMPUTE13 DONE", flush=True)


if __name__ == "__main__":
    main()
