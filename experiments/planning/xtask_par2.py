"""par2: supra-threshold n=2 parallelism battery (UPPER->UPPER, eps0=0.3 per operand).

Per state (UU3/5/7/11 x seeds 0-9) and rep r=0..3, four arms sharing the state's canvases
(rng 73+s, identical to xtask_samecase -> base rows in xtask_samecase_nsweep.json pair):
  n1a  promote {x1} @0.3          (solo anchor, operand 1)
  n1b  promote {x2} @0.3          (solo anchor, operand 2)
  n2   promote {x1, x2} @0.3 each (joint arm - the parallelism probe)
  jk   promote {x1, junk} @0.3    (junk = imageless uppercase letter > hi; total-mass +
                                   incumbent-release matched control for n1a vs n2)
Pair constraints (alphabet idx): |x1-x2| >= D_MIN[k] and images >= 4 from the other source
(kills copy-leak + proximity cross-talk by construction); img(xi) not in {ja, jb};
junk >= 5 from x1 and >= 4 from img(x1). Guard: every promoted token strictly subleading
post-injection (leader*(1-Seps) > 0.3 -- all 40 states pass at n=2, min leader 0.803).
Keys {tag}|s{s}|par2|r{r}|{arm} -> exp/dg_planning/xtask_par2.json (resume-safe).
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
EPS0 = 0.3
SEEDS = list(range(10))
D_MIN = {3: 7, 5: 9, 7: 11, 11: 5}

from tokenizers import Tokenizer
tok = Tokenizer.from_pretrained("google/gemma-4-26b-a4b-it")
def tid(s):
    ids = tok.encode(s, add_special_tokens=False).ids
    assert len(ids) == 1, (s, ids)
    return ids[0]

LOW = list(string.ascii_lowercase); UPP = list(string.ascii_uppercase)
F52 = LOW + UPP

UUQ = ("Pick any uppercase letter between A and {hi}, write it, then write the letter "
       "{off} positions later in the alphabet, also in uppercase, separated by a comma. "
       "Begin your answer with 'Letters:'.")
TASKS = {
    "UU3":  dict(k=3, hi="W", prompt=UUQ.format(hi="W", off="three")),
    "UU5":  dict(k=5, hi="U", prompt=UUQ.format(hi="U", off="five")),
    "UU7":  dict(k=7, hi="S", prompt=UUQ.format(hi="S", off="seven")),
    "UU11": dict(k=11, hi="O", prompt=UUQ.format(hi="O", off="eleven")),
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


def pick_pairs(k, hi_i, nat_i, ja_i, jb_i, rng):
    """-> up to NREP tuples (x1_i, x2_i, junk_i), alphabet 0-based idx"""
    pool = [i for i in range(hi_i + 1) if i != nat_i]
    junk_pool = [j for j in range(hi_i + 1, 26) if j not in (ja_i, jb_i)]
    cand = []
    for x1 in pool:
        if x1 + k in (ja_i, jb_i):
            continue
        junks = [j for j in junk_pool if abs(j - x1) >= 5 and abs(j - (x1 + k)) >= 4]
        if not junks:
            continue
        junk = max(junks, key=lambda j: min(abs(j - x1), abs(j - (x1 + k))))
        for x2 in pool:
            if x2 == x1 or x2 + k in (ja_i, jb_i):
                continue
            if abs(x1 - x2) < D_MIN[k] or abs(x1 + k - x2) < 4 or abs(x2 + k - x1) < 4:
                continue
            cand.append((x1, x2, junk))
    rng.shuffle(cand)
    picked, used1 = [], set()
    for x1, x2, junk in cand:                       # prefer distinct x1 across reps
        if x1 in used1:
            continue
        picked.append((x1, x2, junk)); used1.add(x1)
        if len(picked) == NREP:
            return picked
    for c in cand:                                   # fill up allowing repeats
        if c not in picked:
            picked.append(c)
            if len(picked) == NREP:
                break
    return picked


def main():
    out, f = {}, EXP / "xtask_par2.json"
    if f.exists():
        out = json.load(open(f))
    dns = json.load(open(EXP / "xtask_samecase_nsweep.json"))
    temp = 1.3 + (0.8 - 1.3) * T / 63
    ids = {w: tid(" " + w) for w in F52}
    probe = sorted(set(ids.values()))
    pix = {v: i for i, v in enumerate(probe)}
    t0 = time.time()
    for tag, cfg in TASKS.items():
        k = cfg["k"]; hi_i = UPP.index(cfg["hi"])
        for s in SEEDS:
            bk = f"{tag}|s{s}|base"
            if bk not in dns:
                continue
            d = json.load(open(EXP / f"nego2/{tag}__s{s}.json"))
            nat, A, B = dns[bk]["nat_op"], dns[bk]["A"], dns[bk]["B"]
            nat_i = UPP.index(nat); ja_i = nat_i + k
            jb = d["id2str"].get(str(d["final_ids"][B]), "?").replace("▁", " ").strip()
            jb_i = UPP.index(jb) if jb in UPP else -1
            seedstr = f"{tag}|{s}|par2"
            rng = np.random.default_rng(sum(ord(c) * (i + 7) for i, c in enumerate(seedstr)))
            pairs = pick_pairs(k, hi_i, nat_i, ja_i, jb_i, rng)
            pk = f"{tag}|s{s}|par2|pairs"
            if pk not in out:
                out[pk] = [[UPP[a], UPP[b], UPP[j]] for a, b, j in pairs]
                json.dump(out, open(f, "w"))
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

            for r, (x1, x2, junk) in enumerate(pairs):
                arms = dict(n1a=[UPP[x1]], n1b=[UPP[x2]],
                            n2=[UPP[x1], UPP[x2]], jk=[UPP[x1], UPP[junk]])
                for arm, subset in arms.items():
                    key = f"{tag}|s{s}|par2|r{r}|{arm}"
                    if key in out:
                        continue
                    ids2 = [list(rr) for rr in sheet0["ids"]]
                    lp2 = [list(rr) for rr in sheet0["lp"]]
                    p = np.exp(np.array(lp2[A], dtype=float)) * (1.0 - EPS0 * len(subset))
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
                    out[key] = dict(subset=subset, junk=(UPP[junk] if arm == "jk" else None),
                                    eps0=EPS0, ranks=ranks, rows=brows({"ids": ids2, "lp": lp2}))
                    json.dump(out, open(f, "w"))
            ncells = sum(1 for kk in out if "|par2|r" in kk)
            print(f"{tag} s{s}: done ({ncells} cells, {time.time()-t0:.0f}s elapsed)", flush=True)
    print("PAR2 DONE", flush=True)


if __name__ == "__main__":
    main()
