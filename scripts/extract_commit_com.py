"""Extract commitment center-of-mass curves from the raw constrained-battery rollouts.

NOT rerunnable from a bare clone: reads the raw cruns archive (7.8 GB, not vendored;
/workspace-vast/jbauer/diffusiongemma/exp/dg_planning/cruns/) and writes the small derived
src_data/planning/commit_com.json consumed by figA11_causality.py.

Per run and denoising step t: content = canvas positions whose FINAL token is not pad/eos;
com = center of mass (mean canvas index) of the currently-committed content positions,
normalized to [0,1] over the content span; ref = the same for a matched left-to-right filler
(same committed COUNT, but the leftmost content positions); frac = committed fraction.
"""
import json
import multiprocessing as mp
import os
from pathlib import Path

CRUNS = Path("/workspace-vast/jbauer/diffusiongemma/exp/dg_planning/cruns")
OUT = Path(__file__).resolve().parent.parent / "src_data" / "planning" / "commit_com.json"


def one(fn):
    d = json.load(open(CRUNS / fn))
    dead = set(d["eos_token_ids"]) | {d["pad_token_id"]}
    content = [p for p, x in enumerate(d["final_ids"]) if x not in dead]
    if len(content) < 4:
        return None
    c0, c1 = content[0], content[-1]
    span = max(c1 - c0, 1)
    com, ref, frac = [], [], []
    for st in d["steps"]:
        cm = st["committed"]
        sel = [p for p in content if cm[p]]
        n = len(sel)
        frac.append(round(n / len(content), 3))
        com.append(round((sum(sel) / n - c0) / span, 4) if n else None)
        ref.append(round((sum(content[:n]) / n - c0) / span, 4) if n else None)
    cond = d["_cond"]
    ttype = cond.rsplit("__", 2)[0]
    regime = cond.rsplit("__", 1)[1].split("_")[0]
    return dict(cond=cond, ttype=ttype, regime=regime, T=d["num_steps"],
                n_content=len(content), com=com, ref=ref, frac=frac)


if __name__ == "__main__":
    os.nice(10)
    files = sorted(f for f in os.listdir(CRUNS) if f.endswith(".json"))
    with mp.Pool(4) as pool:
        rows = [r for r in pool.map(one, files) if r]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(OUT, "w"))
    print(OUT, len(rows), "runs")
