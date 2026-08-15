"""Letter-domain deep-n series (52-token field; reviewer requirement: non-target class must
stay LARGE — here non-targets >= 31 even at n=20). Case-flip variants put the image band
fully disjoint from the operand band:
  L3:  'Pick any lowercase letter between a and w, write it, then write the letter three
        positions later in the alphabet in uppercase, separated by a comma.'  (a..w -> D..Z)
  L7:  same with seven positions, a..s -> H..Z
Flat eps=0.04 per hypothesis (Sigma<=0.8 at n=20), n in {1,2,3,4,6,8,10,12,14,16,18,20},
4 subsets per (state, n), 2 seeds/prompt. Field = 52 letters (both cases); edge measured vs
ALL non-targets AND vs the untouched-case reference (lowercase non-operand letters — typical
slots, never targets). -> exp/dg_planning/xtask_compute10.json"""
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
EPS = 0.04
NLVL = [1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20]
NREP = 4

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("google/gemma-4-26b-a4b-it")
def tid(s):
    t = tok.encode(s, add_special_tokens=False)
    assert len(t) == 1, (s, t)
    return t[0]

QB = ("Pick any lowercase letter between a and {hi}, write it, then write the letter "
      "{off} positions later in the alphabet in uppercase, separated by a comma. "
      "Begin your answer with 'Letters:'.")
CFG = {"L3": ("w", "three", 3), "L7": ("s", "seven", 7)}
PROMPTS = {t: QB.format(hi=h, off=o) for t, (h, o, _) in CFG.items()}
K = {t: v[2] for t, v in CFG.items()}
LOW = list(string.ascii_lowercase)
UPP = list(string.ascii_uppercase)
FIELD = LOW + UPP
TIDS = {c: tid(" " + c) for c in FIELD}
PROBE = sorted(set(TIDS.values()))
PIX = {v: i for i, v in enumerate(PROBE)}
T = 2


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
    out, f = {}, EXP / "xtask_compute10.json"
    if f.exists():
        out = json.load(open(f))
    temp = 1.3 + (0.8 - 1.3) * T / 63
    for tag, q in PROMPTS.items():
        hi, k = CFG[tag][0], K[tag]
        pool_all = [c for c in LOW if c <= hi]
        for s in range(2):
            fp = EXP / f"nego2/{tag}__s{s}.json"
            if not fp.exists():
                d = post("sample", dict(prompt=FRAME.format(q=q), seed=s, **HOTR, s_topk_record=32))
                slim = {kk: d[kk] for kk in ("final_ids", "final_text", "pad_token_id", "eos_token_ids",
                                             "id2str", "canvas_length", "s_rec")}
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
            a_c, b_c = txt(fin[A]).strip(), txt(fin[B]).strip()
            if a_c not in LOW or b_c not in UPP:
                print(f"{tag} s{s}: layout A={a_c!r} B={b_c!r} — SKIP", flush=True); continue
            ok = UPP[LOW.index(a_c) + k] == b_c if LOW.index(a_c) + k < 26 else False
            print(f"{tag} s{s}: A@{A} B@{B} natural {a_c}, {b_c} ({'correct' if ok else 'WRONG'})", flush=True)
            nat = a_c
            ops = [c for c in pool_all if c != nat]
            draft = d["steps_argmax"][T]
            sheet0 = {"ids": d["s_rec"]["ids"][T], "lp": d["s_rec"]["lp"][T]}
            rng0 = np.random.default_rng(71 + s)
            noise = rng0.integers(0, VOCAB, size=(DRAWS, 2))
            canvases = []
            for dd in range(DRAWS):
                cv = list(draft)
                cv[A] = int(noise[dd][0]); cv[B] = int(noise[dd][1])
                canvases.append(cv)

            def brows(sheet):
                rows = []
                for i in range(0, DRAWS, 2):
                    rows += post("energy", {"prompt": FRAME.format(q=q), "canvases": canvases[i:i + 2],
                                            "probe_ids": PROBE, "s_sparse": sheet,
                                            "temperature": temp})["probe"]
                return [[round(float(np.exp(r[B][PIX[TIDS[c]]])), 5) for c in FIELD] for r in rows]

            bk = f"{tag}|s{s}|base"
            if bk not in out:
                out[bk] = dict(nat_op=nat, A=A, B=B, ok=ok, rows=brows(sheet0))
                json.dump(out, open(f, "w"))
            for n in [x for x in NLVL if x <= len(ops)]:
                seedstr = f"{tag}|{s}|let|{n}"
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
                    for c in subset:
                        tk = TIDS[c]
                        if tk in row:
                            p[row.index(tk)] += EPS
                        else:
                            j = int(np.argmin(p)); row[j] = tk; p[j] = EPS
                    ranks = {c: int(np.sum(p > p[row.index(TIDS[c])])) for c in subset}
                    ids2[A] = row
                    lp2[A] = [float(x) for x in np.log(np.maximum(p, 1e-12))]
                    out[key] = dict(subset=subset, ranks=ranks, rows=brows({"ids": ids2, "lp": lp2}))
                    json.dump(out, open(f, "w"))
            print(f"{tag} s{s}: battery done (pool {len(ops)})", flush=True)
    print("COMPUTE10 DONE", flush=True)


if __name__ == "__main__":
    main()
