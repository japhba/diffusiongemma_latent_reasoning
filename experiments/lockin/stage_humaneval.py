"""Stage the 40 HumanEval problems for the code ladder (ds_ablate_bench.py, task
`humaneval`) from the public openai/openai_humaneval dataset. The original 40-of-164
subsample order could not be tied to a simple seeded RNG, so the exact ordered selection
is pinned here as the task-id list below (order matters: capture seeds key off list
index). Content (prompt/test/entry_point) is rebuilt from HF. Verified byte-identical to
the original commit_ds/humaneval_problems.json (2026-08-15).
-> $DG_LOCKIN_DIR/humaneval_problems.json  [{pid, task_id, prompt, test, entry_point}]
"""
import json
import os
from pathlib import Path

from datasets import load_dataset

CD = Path(os.environ.get("DG_LOCKIN_DIR", str(Path(__file__).resolve().parent / "data")))

# exact ordered selection used in the study (pid = he_<idx> from the HumanEval task id)
IDX = [146, 67, 68, 94, 54, 50, 97, 71, 6, 14, 27, 9, 148, 108, 104, 121, 2, 82, 29, 88,
       119, 83, 32, 48, 99, 39, 152, 131, 59, 123, 151, 47, 1, 139, 12, 21, 132, 7, 20, 61]

ds = load_dataset("openai/openai_humaneval", split="test")
by_id = {r["task_id"]: r for r in ds}
out = [dict(pid=f"he_{i:03d}", task_id=f"HumanEval/{i}",
            prompt=by_id[f"HumanEval/{i}"]["prompt"], test=by_id[f"HumanEval/{i}"]["test"],
            entry_point=by_id[f"HumanEval/{i}"]["entry_point"]) for i in IDX]

CD.mkdir(parents=True, exist_ok=True)
(CD / "humaneval_problems.json").write_text(json.dumps(out, ensure_ascii=False))
print(f"staged {len(out)} -> {CD / 'humaneval_problems.json'}")
print("example:", out[0]["pid"], out[0]["entry_point"])
