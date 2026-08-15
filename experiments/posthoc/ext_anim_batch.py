"""Entropy-resolution curves for the WHOLE n=40 battery (figA9 regime averages): one rollout
per problem (seed 0, same GRID/framing as the suscept study), storing per-step mean token
entropy over the answer span and over the CoT region (positions localized from the final
canvas, as in lockin_and_localize). Runs on the DG pod. Writes ext_anim_curves.json.
Original 20 problems come from engels battery (staged as battery20.py), extension from battery.py.
"""
import json
import numpy as np
from suscept import post, first_answer, lockin_and_localize, ANSWER_FIRST, GRID
import battery, battery20

PROBLEMS = battery20.PROBLEMS + battery.PROBLEMS
W = 14
out = {}
try:
    out = json.load(open("ext_anim_curves.json"))
except Exception:
    pass
for p in PROBLEMS:
    if p["id"] in out:
        continue
    s = post("/sample", dict(prompt=ANSWER_FIRST.format(q=p["q"]), seed=0, **GRID))
    info = lockin_and_localize(s)
    if not info["ans_pos"] or not info["cot_pos"]:
        print(f"{p['id']}: SKIP (no answer/cot span)", flush=True)
        out[p["id"]] = None
        json.dump(out, open("ext_anim_curves.json", "w")); continue
    E = np.array([st["entropy"] for st in s["steps"][:W]])   # [W, C]
    A = first_answer(s["final_text"])
    out[p["id"]] = dict(
        ans_curve=[round(float(E[k][info["ans_pos"]].mean()), 4) for k in range(E.shape[0])],
        cot_curve=[round(float(E[k][info["cot_pos"]].mean()), 4) for k in range(E.shape[0])],
        ans=A, ok=(A == p["correct"]), n_ans=len(info["ans_pos"]), n_cot=len(info["cot_pos"]))
    json.dump(out, open("ext_anim_curves.json", "w"))
    print(f"{p['id']}: ans={A} ok={A == p['correct']}", flush=True)
print("BATCH_DONE", flush=True)
