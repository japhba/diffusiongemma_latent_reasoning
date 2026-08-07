"""Extract per-position FINAL-commit steps from the raw constrained-battery rollouts.

NOT bare-clone-rerunnable (reads the un-vendored cruns archive); writes the small derived
src_data/planning/commit_order.json for the commit-order analysis (figA11 rework).

Per run: content positions (final token not pad/eos), and for each content position the
final-commit step = first step from which the sampler's accepted mask stays True through the
end (robust to early accept/renoise flicker).
"""
import json
import multiprocessing as mp
import os
from pathlib import Path

CRUNS = Path("/workspace-vast/jbauer/diffusiongemma/exp/dg_planning/cruns")
OUT = Path(__file__).resolve().parent.parent / "src_data" / "planning" / "commit_order.json"


def one(fn):
    d = json.load(open(CRUNS / fn))
    dead = set(d["eos_token_ids"]) | {d["pad_token_id"]}
    content = [p for p, x in enumerate(d["final_ids"]) if x not in dead]
    if len(content) < 6:
        return None
    T = d["num_steps"]
    masks = [d["steps"][t]["committed"] for t in range(T)]
    def final_commit(p):
        t = T
        while t > 0 and masks[t - 1][p]:
            t -= 1
        return t  # first step of the terminal all-True suffix
    fc = [final_commit(p) for p in content]
    cond = d["_cond"]
    return dict(cond=cond, ttype=cond.rsplit("__", 2)[0],
                regime=cond.rsplit("__", 1)[1].split("_")[0], T=T,
                content=content, first_commit=fc)


if __name__ == "__main__":
    os.nice(10)
    files = sorted(f for f in os.listdir(CRUNS) if f.endswith(".json"))
    with mp.Pool(4) as pool:
        rows = [r for r in pool.map(one, files) if r]
    json.dump(rows, open(OUT, "w"))
    print(OUT, len(rows), "runs")
