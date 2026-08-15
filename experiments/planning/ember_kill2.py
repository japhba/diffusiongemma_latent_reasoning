"""Ember kill, round 2. Two questions:

(1) IS THE KILL REAL, OR A FLUCTUATION? The round-1 single-step kill succeeded in only 4/36
    (seeds 3,4,7 x t=1..12): s3 flipped at t=3..6, s4 once at t=3, s7 never. That is far too
    thin to call a window. Here every idiom seed gets the full sweep, plus fresh captures
    (s8..s23) to enlarge both arms -- and, since seasonal is the rare outcome (1/8 in round 1),
    to find more seeds for the rescue arm.

(2) DOES CANVAS COMMITMENT BLOCK LATE KILLS? Round 1's asymmetry is suggestive: a persistent
    kill works later than a single-step one (t~6-8 vs t~3-6), as if the canvas re-commits the
    idiom as soon as you stop pushing. If commitment is the blocker, then holding the five DIFF
    slots FLUID at the kill step should restore intervenability at a late t where the kill
    otherwise fails.

    The worker's new `no_commit` op has two separable mechanisms, and they are NOT equivalent:
      canvas -- overwrite the fed-forward canvas token with a fresh random id. This is what
                actually un-commits, because at accepted positions the CANVAS is the carrier
                (S^t carries only the fluid ones).
      logits -- flatten the returned self-conditioning logits at that position. Does not
                un-commit on its own; included as the mechanism-isolating control.
    Arms per (seed, t): no-commit alone (does the manipulation flip it by itself?), no-commit
    + kill, and a window-matched kill-only arm so the 2-step window is not the explanation.
-> exp/dg_planning/ember_kill2.json
"""
import os
import json, sys, time, urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from constrained_common import CHOSEN

W = os.environ.get("DG_WORKER", "http://localhost:18711")
EXP = Path(os.environ.get("DG_PLANNING_DIR", str(Path(__file__).resolve().parent / "exp")))
FRAME = "{q}\n\nOutput ONLY the text itself — no preamble, no quotes, no explanation."
HOTR = dict(T=64, C=128, t_max=1.3, t_min=0.8, entropy_bound=0.3, early_stop=False, top_k=10)
Q = next(p["q"] for p in CHOSEN if p["id"] == "palindrome_words__3")

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("google/gemma-4-26b-a4b-it")
IDT = tok.encode("All for one and one for all.", add_special_tokens=False)
SET = tok.encode("All leaves fall when leaves fall all.", add_special_tokens=False)
L = 8
DIFF = [j for j in range(L) if IDT[j] != SET[j]]

NEW_SEEDS = list(range(8, 24))          # phase A: fresh captures
SINGLE_T = list(range(1, 13))
PERSIST_T = [2, 4, 6, 8, 10]
LATE_T = [7, 8, 10, 12]                 # phase D: where the single-step kill failed in round 1
NC_MODES = ["canvas", "logits"]
NC_SEEDS_MAX = 5                        # phase D seeds (idiom seeds, lowest first)
KILL_SEEDS_MAX = 8                      # phase B cap (idiom is the common outcome; keep runtime sane)


def post(req, timeout=1800):
    for a in range(6):
        try:
            r = urllib.request.urlopen(urllib.request.Request(
                f"{W}/sample", json.dumps(req).encode(), {"Content-Type": "application/json"}), timeout=timeout)
            return json.loads(r.read())
        except Exception as e:
            print(f"  retry {a}: {type(e).__name__}", flush=True)
            time.sleep(15 * (a + 1))
    raise RuntimeError("worker unreachable")


def outcome(text):
    t = text.lower()
    if "for one and one for" in t:
        return "idiom"
    if "leaves fall" in t or "fall all" in t:
        return "seasonal"
    return "other"


def slots_from(d):
    i2s, fin = d["id2str"], d["final_ids"]
    dead = set(d["eos_token_ids"]) | {d["pad_token_id"]}
    live = [p for p, x in enumerate(fin) if x not in dead]
    txt = lambda x: i2s.get(str(x), "?").replace("▁", " ")
    start = next((p for p in live if txt(fin[p]).strip().lower() == "all"), None)
    return None if start is None else [start + j for j in DIFF]


def reduce_run(d, slots):
    T = len(d["s_rec"]["ids"])
    def mass(t, p, tid):
        ids = d["s_rec"]["ids"][t][p]
        return float(np.exp(d["s_rec"]["lp"][t][p][ids.index(tid)])) if tid in ids else 0.0
    m_idt = [float(np.mean([mass(t, p, IDT[j]) for j, p in zip(DIFF, slots)])) for t in range(T)]
    m_set = [float(np.mean([mass(t, p, SET[j]) for j, p in zip(DIFF, slots)])) for t in range(T)]
    sa = [st["argmax"] for st in d["steps"]]
    f_idt = [float(np.mean([sa[t][p] == IDT[j] for j, p in zip(DIFF, slots)])) for t in range(T)]
    flip = next((t for t in range(T) if f_idt[t] > 0.5), None)
    return dict(final=d["final_text"], outcome=outcome(d["final_text"]), flip=flip,
                m_idt=m_idt[:20], m_set=m_set[:20], f_idt=f_idt[:20])


def main():
    f = EXP / "ember_kill2.json"
    out = json.load(open(f)) if f.exists() else {}

    def run(seed, tag, slots, bump=None, nocommit=None):
        key = f"s{seed}|{tag}"
        if key in out:
            return out[key]
        req = dict(prompt=FRAME.format(q=Q), seed=seed, **HOTR, s_topk_record=32)
        if bump:
            req["s_bump"] = bump
        if nocommit:
            req["no_commit"] = nocommit
            req["no_commit_seed"] = 1000 + seed
        d = post(req)
        r = reduce_run(d, slots) if slots else dict(final=d["final_text"],
                                                    outcome=outcome(d["final_text"]), flip=None)
        r.update(seed=seed, tag=tag)
        out[key] = r
        json.dump(out, open(f, "w"))
        print(f"  s{seed} {tag:18s} -> {r['outcome']:8s} flip={r['flip']} "
              f"{d['final_text'].split(chr(10))[-1][:52]!r}", flush=True)
        return r

    # ---- phase A: base outcome for every seed (old captures replayed, new ones sampled) ----
    base, slots_of = {}, {}
    for seed in list(range(8)) + NEW_SEEDS:
        cap = EXP / f"nego/palindrome_words__3__s{seed}.json"
        if cap.exists():
            d = json.load(open(cap))
            slots_of[seed] = slots_from(d)
            base[seed] = outcome(d["final_text"])
            if f"s{seed}|base" not in out:
                out[f"s{seed}|base"] = dict(seed=seed, tag="base", outcome=base[seed],
                                            final=d["final_text"], flip=None)
                json.dump(out, open(f, "w"))
            continue
        key = f"s{seed}|base"
        if key in out:
            base[seed] = out[key]["outcome"]
            slots_of[seed] = out[key].get("slots")
            continue
        d = post(dict(prompt=FRAME.format(q=Q), seed=seed, **HOTR, s_topk_record=32))
        sl = slots_from(d)
        r = dict(seed=seed, tag="base", outcome=outcome(d["final_text"]),
                 final=d["final_text"], flip=None, slots=sl)
        out[key] = r; json.dump(out, open(f, "w"))
        base[seed], slots_of[seed] = r["outcome"], sl
        print(f"  cap s{seed}: {r['outcome']:8s} {d['final_text'].split(chr(10))[-1][:52]!r}", flush=True)
    idiom = [s for s in base if base[s] == "idiom" and slots_of.get(s)]
    seasonal = [s for s in base if base[s] == "seasonal" and slots_of.get(s)]
    print(f"PHASE A: {len(idiom)} idiom seeds {idiom}, {len(seasonal)} seasonal seeds {seasonal}, "
          f"{len(base) - len(idiom) - len(seasonal)} other", flush=True)

    # ---- phases B/C: full kill sweep on every idiom seed, rescue sweep on every seasonal seed ----
    for arm, seeds, ids_ in (("kill", sorted(idiom)[:KILL_SEEDS_MAX], IDT), ("rescue", sorted(seasonal), SET)):
        for seed in seeds:
            sl = slots_of[seed]
            for t in SINGLE_T:
                run(seed, f"{arm}@t{t}", sl,
                    bump=[dict(pos=p, id=ids_[j], delta=-30000.0, steps=[t, t + 1])
                          for j, p in zip(DIFF, sl)])
            for t in PERSIST_T:
                run(seed, f"{arm}@t{t}+", sl,
                    bump=[dict(pos=p, id=ids_[j], delta=-30000.0, steps=[t, 10 ** 9])
                          for j, p in zip(DIFF, sl)])

    # ---- phase D: does holding the slots fluid restore late intervenability? ----
    for seed in sorted(idiom)[:NC_SEEDS_MAX]:
        sl = slots_of[seed]
        for t in LATE_T:
            kb = [dict(pos=p, id=IDT[j], delta=-30000.0, steps=[t, t + 2])
                  for j, p in zip(DIFF, sl)]
            run(seed, f"k2@t{t}", sl, bump=kb)                 # window-matched kill-only control
            for mode in NC_MODES:
                nc = [dict(pos=p, mode=mode, steps=[t, t + 2]) for p in sl]
                run(seed, f"nc{mode[:3]}@t{t}", sl, nocommit=nc)          # manipulation alone
                run(seed, f"nc{mode[:3]}+k@t{t}", sl, bump=kb, nocommit=nc)
    print("EMBER KILL 2 DONE", flush=True)


if __name__ == "__main__":
    main()
