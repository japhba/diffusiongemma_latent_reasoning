"""Same-case letters variant: lowercase->lowercase (LL3/LL7) and uppercase->uppercase
(UU3/UU7) shift tasks — tests whether the case transform is what suppresses the n=1
letter-specific channel. Phase 1: hot-regime captures (seeds 0-1) -> nego2/{tag}__s{s}.json.
Phase 2: n=1 flat injection at eps0=0.04 (matched to the case-flip battery), every basis b
in pool minus x_nat, t=2, 8 paired draws (rng 73+s). Keys {tag}|s{s}|base and
{tag}|s{s}|b1|{b}. -> exp/dg_planning/xtask_samecase.json"""
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
EPSB = [0.04, 0.177828, 0.316228, 0.45]  # 0.04 keeps legacy keys {tag}|s{s}|b1|{b}
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
    "LL3": dict(k=3, case="low", hi="w", prompt=LLQ.format(hi="w", off="three")),
    "UU3": dict(k=3, case="upp", hi="W", prompt=UUQ.format(hi="W", off="three")),
    "LL7": dict(k=7, case="low", hi="s", prompt=LLQ.format(hi="s", off="seven")),
    "UU7": dict(k=7, case="upp", hi="S", prompt=UUQ.format(hi="S", off="seven")),
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


def capture():
    for tag, cfg in TASKS.items():
        for s in SEEDS:
            f = EXP / f"nego2/{tag}__s{s}.json"
            if f.exists():
                continue
            d = post("sample", dict(prompt=FRAME.format(q=cfg["prompt"]), seed=s, **HOTR, s_topk_record=32))
            slim = {k: d[k] for k in ("final_ids", "final_text", "pad_token_id", "eos_token_ids",
                                      "id2str", "canvas_length", "s_rec")}
            slim["steps_argmax"] = [st["argmax"] for st in d["steps"]]
            slim["tag"] = tag; slim["seed"] = s; slim["q"] = cfg["prompt"]
            json.dump(slim, open(f, "w"))
            print(f"{tag} s{s}: {d['final_text'].splitlines()[-1][:80]!r}", flush=True)
    print("SAMECASE CAPTURE DONE", flush=True)


def battery():
    out, f = {}, EXP / "xtask_samecase.json"
    if f.exists():
        out = json.load(open(f))
    temp = 1.3 + (0.8 - 1.3) * T / 63
    ids = {w: tid(" " + w) for w in F52}
    probe = sorted(set(ids.values()))
    pix = {v: i for i, v in enumerate(probe)}
    ncells = nskip = 0
    for tag, cfg in TASKS.items():
        pool = ([c for c in LOW if c <= cfg["hi"]] if cfg["case"] == "low"
                else [c for c in UPP if c <= cfg["hi"]])
        for s in SEEDS:
            fp = EXP / f"nego2/{tag}__s{s}.json"
            if not fp.exists():
                print(f"{tag} s{s}: no capture — SKIP", flush=True); continue
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

            bkey = f"{tag}|s{s}|base"
            if bkey not in out:
                out[bkey] = dict(nat=nat, A=A, B=B, rows=brows(sheet0))
                json.dump(out, open(f, "w"))
            for eps0 in EPSB:
                for b in pool:
                    if b == nat:
                        continue
                    key = (f"{tag}|s{s}|b1|{b}" if eps0 == 0.04 else f"{tag}|s{s}|e{eps0}|b1|{b}")
                    if key in out:
                        continue
                    ids2 = [list(r) for r in sheet0["ids"]]
                    lp2 = [list(r) for r in sheet0["lp"]]
                    p = np.exp(np.array(lp2[A], dtype=float)) * (1.0 - eps0)
                    row = list(ids2[A])
                    tk_ = ids[b]
                    if tk_ in row:
                        p[row.index(tk_)] += eps0
                    else:
                        j = int(np.argmin(p)); row[j] = tk_; p[j] = eps0
                    rank = int(np.sum(p > p[row.index(tk_)]))
                    if rank == 0:
                        out[key] = dict(skipped=True); json.dump(out, open(f, "w")); nskip += 1; continue
                    ids2[A] = row
                    lp2[A] = [float(x) for x in np.log(np.maximum(p, 1e-12))]
                    out[key] = dict(eps0=eps0, subset=[b], ranks={b: rank}, rows=brows({"ids": ids2, "lp": lp2}))
                    json.dump(out, open(f, "w")); ncells += 1
                print(f"{tag} s{s} e{eps0}: done (pool {len(pool)}, nat {nat}) [total {ncells} cells, {nskip} rank-0]", flush=True)
    print(f"SAMECASE BATTERY DONE: {ncells} new cells, {nskip} rank-0 skips", flush=True)


if __name__ == "__main__":
    capture()
    battery()
