"""Fill the MULTIPLICATIVE n=1 every-basis map at an ADDITIONAL dose (argv[1], default 0.316228),
for every MU state that xtask_mult.py already captured. Needed when eps=0.45 fails the >=70%
cell-survival rule that picks the map dose (the rank>=1 guard rejects injections that would
become the sheet row leader, and that ceiling is state-dependent). Resume-safe; reuses the
existing captures and base rows, so it only adds map cells. -> exp/dg_planning/xtask_mult.json"""
import os
import json, string, sys, time, urllib.request
from pathlib import Path

import numpy as np

W = os.environ.get("DG_WORKER", "http://localhost:18711")
EXP = Path(os.environ.get("DG_PLANNING_DIR", str(Path(__file__).resolve().parent / "exp")))
FRAME = "{q}\n\nOutput ONLY the text itself — no preamble, no quotes, no explanation."
VOCAB = 262144
DRAWS = 8
T = 2
EPS = float(sys.argv[1]) if len(sys.argv) > 1 else 0.316228
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
    fmu = EXP / "xtask_mult.json"
    dmu = json.load(open(fmu))
    temp = 1.3 + (0.8 - 1.3) * T / 63
    ids = {w: tid(" " + w) for w in F52}
    probe = sorted(set(ids.values()))
    pix = {v: i for i, v in enumerate(probe)}
    nmp = nskip = 0
    for tag, cfg in TASKS.items():
        pool = [c for c in UPP if c <= cfg["hi"]]
        for s in SEEDS:
            fp = EXP / f"nego2/{tag}__s{s}.json"
            bk = f"{tag}|s{s}|base"
            if not fp.exists() or bk not in dmu:
                continue
            d = json.load(open(fp))
            A, B, nat = dmu[bk]["A"], dmu[bk]["B"], dmu[bk]["nat"]
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

            for b in [o for o in pool if o != nat]:
                key = f"{tag}|s{s}|e{EPS}|b1|{b}"
                if key in dmu:
                    continue
                sheet, ranks = inject([b], EPS)
                if ranks[b] == 0:
                    dmu[key] = dict(skipped=True); json.dump(dmu, open(fmu, "w")); nskip += 1; continue
                dmu[key] = dict(eps0=EPS, subset=[b], ranks=ranks, rows=brows(sheet))
                json.dump(dmu, open(fmu, "w")); nmp += 1
            print(f"{tag} s{s}: eps={EPS} done [map {nmp}, rank-0 {nskip}]", flush=True)
    print(f"MULT DOSE {EPS} DONE: {nmp} map cells, {nskip} rank-0", flush=True)


if __name__ == "__main__":
    main()
