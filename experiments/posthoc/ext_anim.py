"""Entropy-resolution captures for the figA9 extension: one post-hoc and one load-bearing
exemplar from the 2026-08-08 extension battery, same GRID as the original anim capture.
Run from this directory (cwd = script dir) AFTER the anchor chain releases the worker.
Writes ext_anim.json with the com_posthoc_anim.json 'cases' record fields figA9 needs.
"""
import json
import numpy as np
from suscept import post, first_answer, lockin_and_localize, ANSWER_FIRST, GRID
from battery import PROBLEMS

PROB = {p["id"]: p for p in PROBLEMS}
CASES = [("post-hoc", "prod_then_digitsum", 0), ("true-checking", "cubes_10_1000", 2)]

out = []
for regime, pid, seed in CASES:
    p = PROB[pid]
    s = post("/sample", dict(prompt=ANSWER_FIRST.format(q=p["q"]), seed=seed, **GRID))
    info = lockin_and_localize(s)
    lastC = max(info["ans_pos"] + info["cot_pos"])
    ent = [[round(float(st["entropy"][q]), 3) for q in range(lastC + 1)] for st in s["steps"]]
    A = first_answer(s["final_text"])
    out.append(dict(regime=regime, pid=pid, seed=seed,
                    ans_pos=info["ans_pos"], cot_pos=info["cot_pos"], entropy=ent,
                    answer_path=info["answer_traj"], n_flips=len(set(info["answer_traj"])) - 1,
                    ans_lock=float(np.median(info["ans_lock"])),
                    cot_lock=float(np.median(info["cot_lock"])),
                    final=A, correct=p["correct"], ok=(A == p["correct"])))
    print(f"{regime} {pid}__{seed}: ans={A} ok={A == p['correct']} path={info['answer_traj']}", flush=True)
json.dump(dict(cases=out), open("ext_anim.json", "w"))
print("EXT_ANIM_DONE", flush=True)
