"""Smoothing + operation-diversity extension of the deep-n series (user: more data, more
ks, subtraction/multiplication since the contiguous token field ends at twenty).
New prompts (spaced-operand 'Numbers:' format, flat eps=0.05):
  A5  '+5'  two..fifteen   (pool 13)      A7  '+7'  two..thirteen (pool 11)
  S3  '-3'  five..twenty   (pool <=15)    S8  '-8'  ten..twenty   (pool <=10)
  M10 'x10' two..nine      (pool <=7; images twenty..ninety — tens words, single-token,
                            image field fully DISJOINT from operand cluster)
plus 2 extra reps (r4, r5) for the existing P2/P3/P6/P9 states (bases reused from
xtask_compute8.json). Probe field = two..twenty + thirty..ninety (26 words).
-> exp/dg_planning/xtask_compute9.json"""
import os
import json, time, urllib.request
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
NREP = 5

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("google/gemma-4-26b-a4b-it")
def tid(s):
    t = tok.encode(s, add_special_tokens=False)
    assert len(t) == 1, (s, t)
    return t[0]

QB = ("Pick any number between {lo} and {hi}, write it in words, then write the number "
      "{rel} in words, separated by a comma. Begin your answer with 'Numbers:'.")
CFG = {  # tag: (lo, hi, relation phrase, k as signed int or 'x10')
    "A5": ("two", "fifteen", "five greater", 5),
    "A7": ("two", "thirteen", "seven greater", 7),
    "S3": ("five", "twenty", "three less", -3),
    "S8": ("ten", "twenty", "eight less", -8),
    "M10": ("two", "nine", "ten times as large", "x10"),
}
PROMPTS = {t: QB.format(lo=a, hi=b, rel=r) for t, (a, b, r, _) in CFG.items()}
UNITS = ("two three four five six seven eight nine ten eleven twelve thirteen fourteen "
         "fifteen sixteen seventeen eighteen nineteen twenty").split()
TENS = "thirty forty fifty sixty seventy eighty ninety".split()
NUMS = UNITS + TENS
VAL = {w: i + 2 for i, w in enumerate(UNITS)} | {w: (i + 3) * 10 for i, w in enumerate(TENS)}
V2W = {v: w for w, v in VAL.items()}
NUMID = {w: tid(" " + w) for w in NUMS}
PROBE = sorted(set(NUMID.values()))
PIX = {v: i for i, v in enumerate(PROBE)}
T = 2


def img(x, k):
    v = x * 10 if k == "x10" else x + k
    return V2W.get(v)


def rng_lo_hi(tag):
    lo, hi = CFG[tag][0], CFG[tag][1]
    return VAL[lo], VAL[hi]


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


def run_state(tag, s, d, out, f, prompt, reps=range(NREP)):
    """generic per-state battery on capture d (comma layout), writes into out."""
    i2s, fin = d["id2str"], d["final_ids"]
    dead = set(d["eos_token_ids"]) | {d["pad_token_id"]}
    live = [p for p, x in enumerate(fin) if x not in dead]
    txt = lambda x: i2s.get(str(x), "?").replace("▁", " ")
    pc = next((p for p in live if txt(fin[p]).strip() == ","), None)
    if pc is None:
        print(f"{tag} s{s}: no comma — SKIP", flush=True); return
    A = max(p for p in live if p < pc)
    B = next(p for p in live if p > pc)
    a_w, b_w = txt(fin[A]).strip().lower(), txt(fin[B]).strip().lower()
    k = CFG[tag][3]
    if a_w not in VAL or b_w not in VAL:
        print(f"{tag} s{s}: layout A={a_w!r} B={b_w!r} — SKIP", flush=True); return
    ok = img(VAL[a_w], k) == b_w
    print(f"{tag} s{s}: A@{A} B@{B} natural {a_w}, {b_w} ({'correct' if ok else 'WRONG'})", flush=True)
    lo, hi = rng_lo_hi(tag)
    ops = [w for w in UNITS if lo <= VAL[w] <= hi and w != a_w and img(VAL[w], k)]
    temp = 1.3 + (0.8 - 1.3) * T / 63
    draft = d["steps_argmax"][T]
    sheet0 = {"ids": d["s_rec"]["ids"][T], "lp": d["s_rec"]["lp"][T]}
    rng0 = np.random.default_rng(67 + s)
    noise = rng0.integers(0, VOCAB, size=(DRAWS, 2))
    canvases = []
    for dd in range(DRAWS):
        cv = list(draft)
        cv[A] = int(noise[dd][0]); cv[B] = int(noise[dd][1])
        canvases.append(cv)

    def brows(sheet):
        rows = []
        for i in range(0, DRAWS, 2):
            rows += post("energy", {"prompt": prompt, "canvases": canvases[i:i + 2],
                                    "probe_ids": PROBE, "s_sparse": sheet, "temperature": temp})["probe"]
        return [[round(float(np.exp(r[B][PIX[NUMID[w]]])), 5) for w in NUMS] for r in rows]

    bk = f"{tag}|s{s}|base"
    if bk not in out:
        out[bk] = dict(nat_op=a_w, A=A, B=B, ok=ok, rows=brows(sheet0))
        json.dump(out, open(f, "w"))
    for n in [x for x in NLVL if x <= len(ops)]:
        seedstr = f"{tag}|{s}|deep9|{n}"
        rng = np.random.default_rng(sum(ord(c) * (i + 7) for i, c in enumerate(seedstr)))
        drawn = {}
        for rep in range(max(reps) + 1):
            drawn[rep] = list(rng.choice(ops, size=n, replace=False))  # keep rng stream aligned
        for rep in reps:
            key = f"{tag}|s{s}|deep|n{n}|r{rep}"
            if key in out:
                continue
            subset = drawn[rep]
            ids2 = [list(r) for r in sheet0["ids"]]
            lp2 = [list(r) for r in sheet0["lp"]]
            p = np.exp(np.array(lp2[A], dtype=float)) * (1.0 - EPS * n)
            row = list(ids2[A])
            for w in subset:
                tk = NUMID[w]
                if tk in row:
                    p[row.index(tk)] += EPS
                else:
                    j = int(np.argmin(p)); row[j] = tk; p[j] = EPS
            ranks = {w: int(np.sum(p > p[row.index(NUMID[w])])) for w in subset}
            ids2[A] = row
            lp2[A] = [float(x) for x in np.log(np.maximum(p, 1e-12))]
            out[key] = dict(subset=subset, ranks=ranks, rows=brows({"ids": ids2, "lp": lp2}))
            json.dump(out, open(f, "w"))
    print(f"{tag} s{s}: battery done (pool {len(ops)})", flush=True)


def main():
    out, f = {}, EXP / "xtask_compute9.json"
    if f.exists():
        out = json.load(open(f))
    # new prompts: capture + battery
    for tag, q in PROMPTS.items():
        for s in range(2):
            fp = EXP / f"nego2/{tag}__s{s}.json"
            if not fp.exists():
                d = post("sample", dict(prompt=FRAME.format(q=q), seed=s, **HOTR, s_topk_record=32))
                slim = {kk: d[kk] for kk in ("final_ids", "final_text", "pad_token_id", "eos_token_ids",
                                             "id2str", "canvas_length", "s_rec")}
                slim["steps_argmax"] = [st["argmax"] for st in d["steps"]]
                json.dump(slim, open(fp, "w"))
                print(f"cap {tag} s{s}: {d['final_text'].split(chr(10))[-1][:70]!r}", flush=True)
            run_state(tag, s, json.load(open(fp)), out, f, FRAME.format(q=q))
    # extra reps (r4, r5) for existing P-prompts — fresh 26-col base + cells in THIS file
    # (d8 rows are 19-col; keep sources separate, merge at word level in the fig script).
    PQ = ("Pick any number between two and {hi}, write it in words, then write the number "
          "{off} greater in words, separated by a comma. Begin your answer with 'Numbers:'.")
    P_CFG = {"P2": ("eighteen", "two", 2), "P3": ("seventeen", "three", 3),
             "P6": ("fourteen", "six", 6), "P9": ("eleven", "nine", 9)}
    for tag, (hi, off, k) in P_CFG.items():
        CFG[tag] = ("two", hi, f"{off} greater", k)
        for s in range(2):
            run_state(tag, s, json.load(open(EXP / f"nego2/{tag}__s{s}.json")), out, f,
                      FRAME.format(q=PQ.format(hi=hi, off=off)), reps=(4, 5))
    print("COMPUTE9 DONE", flush=True)


if __name__ == "__main__":
    main()
