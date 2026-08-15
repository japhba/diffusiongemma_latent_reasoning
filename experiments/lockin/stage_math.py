"""Stage the 72 AMC/AIME problems for the paper-regime math ladder (ds_ablate_math.py /
ds_paper_sweep.py) from the public AI-MO validation sets. Selection rule (recovered by
matching the original file against the datasets, exact): the FIRST 48 rows of
AI-MO/aimo-validation-amc and the FIRST 24 rows of AI-MO/aimo-validation-aime, dataset
order preserved (all 72 have integer answers). pids are globally numbered in file order:
amc_000..amc_047 then aime_048..aime_071 — seeds in the capture scripts key off these.
Verified byte-identical to the original commit_ds/math_problems.json (2026-08-15).
-> $DG_LOCKIN_DIR/math_problems.json  [{src, problem, answer, pid}]
"""
import json
import os
from pathlib import Path

from datasets import load_dataset

CD = Path(os.environ.get("DG_LOCKIN_DIR", str(Path(__file__).resolve().parent / "data")))

out = []
for src, repo, n in [("amc", "AI-MO/aimo-validation-amc", 48), ("aime", "AI-MO/aimo-validation-aime", 24)]:
    ds = load_dataset(repo, split="train")
    for r in list(ds)[:n]:
        assert float(r["answer"]) == int(float(r["answer"])), f"non-integer answer in {repo}"
        out.append(dict(src=src, problem=r["problem"], answer=int(float(r["answer"])),
                        pid=f"{src}_{len(out):03d}"))

CD.mkdir(parents=True, exist_ok=True)
(CD / "math_problems.json").write_text(json.dumps(out))
print(f"staged {len(out)} -> {CD / 'math_problems.json'}")
print("example:", out[0]["pid"], "ans", out[0]["answer"])
