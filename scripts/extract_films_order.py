"""Extract commit order (argmax lock-in) for benchmark-style thinkfast tasks from the
denoising films. NOT bare-clone-rerunnable (reads exp/dg_lockin/thinkfast/films/, un-vendored);
writes src_data/planning/films_order.json. lock = start of the film's terminal constant suffix
at that position; content = final canvas positions that are not pad/eos/turn.
"""
import json
import os
from pathlib import Path

F = Path("/workspace-vast/jbauer/exp/dg_lockin/thinkfast/films")
OUT = Path(__file__).resolve().parent.parent / "src_data" / "planning" / "films_order.json"
TASKS = {"arithmetic", "square_count", "collatz", "tower_of_london", "reverse_chain"}
DEAD = {0, 1, 106}

rows = []
for fn in sorted(os.listdir(F)):
    task = fn.split("__")[0]
    if task not in TASKS:
        continue
    d = json.load(open(F / fn))
    if d["T"] < 16:
        continue
    for r in d["rolls"]:
        content = [p for p, x in enumerate(r["ids"]) if x not in DEAD]
        if len(content) < 6:
            continue
        film = r["film"]
        T = len(film)
        def lock(p):
            t = T - 1
            while t > 0 and film[t][p] == film[t - 1][p]:
                t -= 1
            return t
        rows.append(dict(task=task, depth=d["depth"], T=d["T"], seed=r["seed"], ok=r["ok"],
                         content=content, lock=[lock(p) for p in content]))
json.dump(rows, open(OUT, "w"))
import collections
print(OUT, len(rows), collections.Counter(r["task"] for r in rows))
