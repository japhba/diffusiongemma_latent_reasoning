"""n-sweep superposition batteries for the same-case letter variants (LL3/UU3/LL7/UU7,
seeds 0-1): exact replica of the compute13 ext design — flat eps0=0.04, NLVL up to 18,
3 subset draws per (seed, n), 8 paired draws, t=2, canvases rng 73+s (identical to
xtask_samecase so its base rows pair; they are copied in as {tag}|s{s}|base with nat_op).
Keys {tag}|s{s}|ext|n{n}|e{rep}. -> exp/dg_planning/xtask_samecase_nsweep.json"""
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
NREP = 3
NLVL = [1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18]
EPS0 = 0.04
SEEDS = [0, 1]

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("google/gemma-4-26b-a4b-it")
def tid(s):
    t = tok.encode(s, add_special_tokens=False)
    assert len(t) == 1
    return t[0]

LOW = list(string.ascii_lowercase); UPP = list(string.ascii_uppercase)
F52 = LOW + UPP

LLQ = ("Pick any lowercase letter between a and {hi}, write it, then write the letter "
       "{off} positions later in the alphabet, also in lowercase, separated by a comma. "
       "Begin your answer with 'Letters:'.")
UUQ = ("Pick any uppercase letter between A and {hi}, write it, then write the letter "
       "{off} positions later in the alphabet, also in uppercase, separated by a comma. "
       "Begin your answer with 'Letters:'.")
TASKS = {
    "LL3": dict(case="low", hi="w", prompt=LLQ.format(hi="w", off="three")),
    "UU3": dict(case="upp", hi="W", prompt=UUQ.format(hi="W", off="three")),
    "LL7": dict(case="low", hi="s", prompt=LLQ.format(hi="s", off="seven")),
    "UU7": dict(case="upp", hi="S", prompt=UUQ.format(hi="S", off="seven")),
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
    out, f = {}, EXP / "xtask_samecase_nsweep.json"
    if f.exists():
        out = json.load(open(f))
    dsc = json.load(open(EXP / "xtask_samecase.json"))
    temp = 1.3 + (0.8 - 1.3) * T / 63
    ids = {w: tid(" " + w) for w in F52}
    probe = sorted(set(ids.values()))
    pix = {v: i for i, v in enumerate(probe)}
    for tag, cfg in TASKS.items():
        pool = ([c for c in LOW if c <= cfg["hi"]] if cfg["case"] == "low"
                else [c for c in UPP if c <= cfg["hi"]])
        for s in SEEDS:
            d = json.load(open(EXP / f"nego2/{tag}__s{s}.json"))
            scb = dsc[f"{tag}|s{s}|base"]
            nat, A, B = scb["nat"], scb["A"], scb["B"]
            bk = f"{tag}|s{s}|base"
            if bk not in out:
                out[bk] = dict(nat_op=nat, A=A, B=B, rows=scb["rows"])
                json.dump(out, open(f, "w"))
            ops = [o for o in pool if o != nat]
            draft = d["steps_argmax"][T]
            sheet0 = {"ids": d["s_rec"]["ids"][T], "lp": d["s_rec"]["lp"][T]}
            rng0 = np.random.default_rng(73 + s)
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

            for n in [x for x in NLVL if x <= len(ops)]:
                seedstr = f"{tag}|{s}|scn|{n}"
                rng = np.random.default_rng(sum(ord(c) * (i + 7) for i, c in enumerate(seedstr)))
                for rep in range(NREP):
                    key = f"{tag}|s{s}|ext|n{n}|e{rep}"
                    if key in out:
                        continue
                    subset = list(rng.choice(ops, size=n, replace=False))
                    ids2 = [list(r) for r in sheet0["ids"]]
                    lp2 = [list(r) for r in sheet0["lp"]]
                    p = np.exp(np.array(lp2[A], dtype=float)) * (1.0 - EPS0 * n)
                    row = list(ids2[A])
                    for w in subset:
                        tk_ = ids[w]
                        if tk_ in row:
                            p[row.index(tk_)] += EPS0
                        else:
                            j = int(np.argmin(p)); row[j] = tk_; p[j] = EPS0
                    ranks = {w: int(np.sum(p > p[row.index(ids[w])])) for w in subset}
                    ids2[A] = row
                    lp2[A] = [float(x) for x in np.log(np.maximum(p, 1e-12))]
                    out[key] = dict(subset=subset, ranks=ranks, rows=brows({"ids": ids2, "lp": lp2}))
                    json.dump(out, open(f, "w"))
            print(f"{tag} s{s}: done (nat {nat}, {len(ops)} ops)", flush=True)
    print("SAMECASE NSWEEP DONE", flush=True)


if __name__ == "__main__":
    main()
