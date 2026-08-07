"""Extract digit-level commit order for the reverse_chain probe task from the thinkfast
denoising films (transparency-paper replication battery).

NOT bare-clone-rerunnable: reads /workspace-vast/jbauer/exp/dg_lockin/thinkfast/films/
(builder diffusiongemma/thinkfast/grid_films.py) and writes
src_data/planning/reverse_chain_order.json. Per roll: the DIGIT positions of the final canvas
(the chain elements x1..xn, in canvas order) and each one's argmax lock-in step (start of the
film's terminal constant suffix at that position).
"""
import json
import os
from pathlib import Path

F = Path("/workspace-vast/jbauer/exp/dg_lockin/thinkfast/films")
OUT = Path(__file__).resolve().parent.parent / "src_data" / "planning" / "reverse_chain_order.json"

rows = []
for fn in sorted(os.listdir(F)):
    if not fn.startswith("reverse_chain__"):
        continue
    d = json.load(open(F / fn))
    i2s = d["id2str"]
    for r in d["rolls"]:
        digits = [p for p, x in enumerate(r["ids"])
                  if i2s[str(x)].replace("▁", "").strip().isdigit()]
        if len(digits) < 3:
            continue
        film = r["film"]
        T = len(film)
        def lock(p):
            t = T - 1
            while t > 0 and film[t][p] == film[t - 1][p]:
                t -= 1
            return t
        rows.append(dict(depth=d["depth"], T=d["T"], seed=r["seed"], ok=r["ok"],
                         digit_pos=digits, lock=[lock(p) for p in digits], text=r["text"]))
json.dump(rows, open(OUT, "w"))
print(OUT, len(rows), "rolls")
