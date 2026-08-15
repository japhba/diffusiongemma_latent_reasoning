"""Stage 32 IMO problems for the paper Fig-2 'AMC/AIME/IMO variants' panel, from
OpenEvals/IMO-AnswerBench (DeepMind IMO-Bench short-answer split, ungated; 400 rows,
IMO + IMO-Shortlist problems rewritten to have a short final answer). We keep only
rows whose answer is a plain integer (auto-gradable with the same boxed-int grader as
AMC/AIME) and draw 32 seeded, stratified across the four categories.
-> commit_ds/imo_problems.json  [{pid, problem, answer, category, src}]
"""
import os
import json
import re
from pathlib import Path

import numpy as np
from datasets import load_dataset

CD = Path(os.environ.get("DG_LOCKIN_DIR", str(Path(__file__).resolve().parent / "data")))
N = 32

ds = load_dataset("OpenEvals/IMO-AnswerBench", split="train")
rows = [r for r in ds if re.fullmatch(r"-?\d{1,9}", str(r["Short Answer"]).strip())]
print(f"{len(rows)}/{len(ds)} rows have plain-integer answers")
by_cat = {}
for r in rows:
    by_cat.setdefault(r["Category"], []).append(r)
rng = np.random.default_rng(0)
per = {c: max(1, round(N * len(v) / len(rows))) for c, v in by_cat.items()}
while sum(per.values()) != N:
    c = max(per, key=lambda c: per[c]) if sum(per.values()) > N else min(per, key=lambda c: per[c])
    per[c] += -1 if sum(per.values()) > N else 1
picked = []
for c, v in sorted(by_cat.items()):
    idx = rng.choice(len(v), size=per[c], replace=False)
    picked += [v[i] for i in sorted(idx)]
out = [dict(pid=f"imo_{i:03d}", problem=r["Problem"].strip(), answer=int(str(r["Short Answer"]).strip()),
            category=r["Category"], src=r["Source"]) for i, r in enumerate(picked)]
(CD / "imo_problems.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(f"staged {len(out)}: " + ", ".join(f"{c}={per[c]}" for c in sorted(per)))
print("example:", out[0]["pid"], out[0]["src"], "ans", out[0]["answer"])
