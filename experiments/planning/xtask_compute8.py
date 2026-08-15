"""Deep-n saturation probe: more ks AND more operand base values. Format prefix
("Begin your answer with 'Numbers:'") puts the operand in SPACED token form, so operands
range past nine (spaced number words are single-token through twenty). Prompts:
  P2: two..eighteen, k=2 (pool 17)   P3: two..seventeen, k=3 (pool 16)
  P6: two..fourteen, k=6 (pool 13)   P9: two..eleven,  k=9 (pool 10)
Flat eps=0.05 per hypothesis (Sigma<1 even at n=14), n in {1,2,3,4,6,8,10,12,14} capped by
pool size. Stage 1 captures 2 seeds/prompt and verifies format + correctness; battery only
on verified states. -> exp/dg_planning/xtask_compute8.json"""
import os
import json, time, urllib.request
from itertools import combinations
from pathlib import Path

import numpy as np

W = os.environ.get("DG_WORKER", "http://localhost:18711")
EXP = Path(os.environ.get("DG_PLANNING_DIR", str(Path(__file__).resolve().parent / "exp")))
FRAME = "{q}\n\nOutput ONLY the text itself — no preamble, no quotes, no explanation."
HOTR = dict(T=64, C=128, t_max=1.3, t_min=0.8, entropy_bound=0.3, early_stop=False, top_k=10)
VOCAB = 262144
DRAWS = 8
EPS = 0.05
NLVL = [1, 2, 3, 4, 6, 8, 10, 12, 14]
NREP = 4

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("google/gemma-4-26b-a4b-it")
def tid(s):
    t = tok.encode(s, add_special_tokens=False)
    assert len(t) == 1, (s, t)
    return t[0]

Q = ("Pick any number between two and {hi}, write it in words, then write the number "
     "{off} greater in words, separated by a comma. Begin your answer with 'Numbers:'.")
CFG = {"P2": ("eighteen", "two", 2), "P3": ("seventeen", "three", 3),
       "P6": ("fourteen", "six", 6), "P9": ("eleven", "nine", 9)}
PROMPTS = {k: Q.format(hi=hi, off=off) for k, (hi, off, _) in CFG.items()}
K = {k: v[2] for k, v in CFG.items()}
NUMS = ("two three four five six seven eight nine ten eleven twelve thirteen fourteen "
        "fifteen sixteen seventeen eighteen nineteen twenty").split()
N2I = {w: i + 2 for i, w in enumerate(NUMS)}
NUMID = {w: tid(" " + w) for w in NUMS}
HI = {"P2": 18, "P3": 17, "P6": 14, "P9": 11}
POOL = {k: [w for w in NUMS if N2I[w] <= HI[k]] for k in CFG}
T = 2
PROBE = sorted(set(NUMID.values()))
PIX = {v: i for i, v in enumerate(PROBE)}


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
    out, f = {}, EXP / "xtask_compute8.json"
    if f.exists():
        out = json.load(open(f))
    temp = 1.3 + (0.8 - 1.3) * T / 63

    for tag, q in PROMPTS.items():
        for s in range(2):
            fp = EXP / f"nego2/{tag}__s{s}.json"
            if not fp.exists():
                d = post("sample", dict(prompt=FRAME.format(q=q), seed=s, **HOTR, s_topk_record=32))
                slim = {k: d[k] for k in ("final_ids", "final_text", "pad_token_id", "eos_token_ids",
                                          "id2str", "canvas_length", "s_rec")}
                slim["steps_argmax"] = [st["argmax"] for st in d["steps"]]
                json.dump(slim, open(fp, "w"))
                print(f"cap {tag} s{s}: {d['final_text'].split(chr(10))[-1][:70]!r}", flush=True)

    for tag in PROMPTS:
        for s in range(2):
            d = json.load(open(EXP / f"nego2/{tag}__s{s}.json"))
            i2s, fin = d["id2str"], d["final_ids"]
            dead = set(d["eos_token_ids"]) | {d["pad_token_id"]}
            live = [p for p, x in enumerate(fin) if x not in dead]
            txt = lambda x: i2s.get(str(x), "?").replace("▁", " ")
            pc = next((p for p in live if txt(fin[p]).strip() == ","), None)
            if pc is None:
                print(f"{tag} s{s}: no comma — SKIP", flush=True)
                continue
            A = max(p for p in live if p < pc)
            B = next(p for p in live if p > pc)
            a_w, b_w = txt(fin[A]).strip().lower(), txt(fin[B]).strip().lower()
            if a_w not in N2I or b_w not in N2I:
                print(f"{tag} s{s}: non-number layout A={a_w!r} B={b_w!r} — SKIP", flush=True)
                continue
            ok = N2I[b_w] == N2I[a_w] + K[tag]
            print(f"{tag} s{s}: A@{A} B@{B} natural {a_w}, {b_w} ({'correct' if ok else 'WRONG'})", flush=True)
            nat_op = a_w
            ops = [w for w in POOL[tag] if w != nat_op]
            draft = d["steps_argmax"][T]
            sheet0 = {"ids": d["s_rec"]["ids"][T], "lp": d["s_rec"]["lp"][T]}
            rng0 = np.random.default_rng(61 + s)
            noise = rng0.integers(0, VOCAB, size=(DRAWS, 2))
            canvases = []
            for dd in range(DRAWS):
                cv = list(draft)
                cv[A] = int(noise[dd][0]); cv[B] = int(noise[dd][1])
                canvases.append(cv)

            def brows(sheet):
                rows = []
                for i in range(0, DRAWS, 2):
                    rows += post("energy", {"prompt": FRAME.format(q=PROMPTS[tag]),
                                            "canvases": canvases[i:i + 2], "probe_ids": PROBE,
                                            "s_sparse": sheet, "temperature": temp})["probe"]
                return [[round(float(np.exp(r[B][PIX[NUMID[w]]])), 5) for w in NUMS] for r in rows]

            bk = f"{tag}|s{s}|base"
            if bk not in out:
                out[bk] = dict(nat_op=nat_op, A=A, B=B, ok=ok, rows=brows(sheet0))
                json.dump(out, open(f, "w"))
            for n in [x for x in NLVL if x <= len(ops)]:
                seedstr = f"{tag}|{s}|deep|{n}"
                rng = np.random.default_rng(sum(ord(c) * (i + 7) for i, c in enumerate(seedstr)))
                for rep in range(NREP):
                    key = f"{tag}|s{s}|deep|n{n}|r{rep}"
                    if key in out:
                        continue
                    subset = list(rng.choice(ops, size=n, replace=False))
                    ids2 = [list(r) for r in sheet0["ids"]]
                    lp2 = [list(r) for r in sheet0["lp"]]
                    p = np.exp(np.array(lp2[A], dtype=float)) * (1.0 - EPS * n)
                    row = list(ids2[A])
                    ranks = {}
                    for w in subset:
                        tk = NUMID[w]                      # SPACED form at A here
                        if tk in row:
                            p[row.index(tk)] += EPS
                        else:
                            j = int(np.argmin(p)); row[j] = tk; p[j] = EPS
                    for w in subset:
                        ranks[w] = int(np.sum(p > p[row.index(NUMID[w])]))
                    ids2[A] = row
                    lp2[A] = [float(x) for x in np.log(np.maximum(p, 1e-12))]
                    rows = brows({"ids": ids2, "lp": lp2})
                    out[key] = dict(subset=subset, ranks=ranks, rows=rows)
                    json.dump(out, open(f, "w"))
            print(f"{tag} s{s}: battery done (pool {len(ops)})", flush=True)
    print("COMPUTE8 DONE", flush=True)


if __name__ == "__main__":
    main()
