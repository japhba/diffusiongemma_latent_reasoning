"""Stage the 32 LiveCodeBench problems for the pacing-closure ladder (ds_ablate_lcb.py /
ds_paper_sweep.py) from livecodebench/code_generation_lite, version_tag release_v1
(public stdin/stdout problems, all AtCoder). BEST-EFFORT RESTAGE, NOT RUN/DIFFED here:
the original 168MB commit_ds/lcb_problems.json was staged ad hoc on a pod; this stager
pins the exact ordered 32-qid selection (below) and rebuilds each record from HF, keeping
the first 8 of public+private tests per problem (the original per-record test cap).

NOTE: code_generation_lite ships a datasets loading script; `datasets>=3` no longer runs
those, so use datasets<3 for the plain load_dataset path, or port the private-test decode
(base64 -> zlib -> pickle, handled below) onto the raw parquet if your datasets is newer.
-> $DG_LOCKIN_DIR/lcb_problems.json  [{qid, title, difficulty, platform, problem, tests, pid}]
"""
import base64
import json
import os
import pickle
import zlib
from pathlib import Path

from datasets import load_dataset

CD = Path(os.environ.get("DG_LOCKIN_DIR", str(Path(__file__).resolve().parent / "data")))
MAX_TESTS = 8

# exact ordered selection used in the study (pid = lcb_<list index :03d>)
QIDS = ["abc341_a", "abc310_b", "abc312_a", "abc333_a", "abc318_e", "abc319_b", "abc342_b",
        "abc322_b", "abc315_f", "abc329_f", "abc302_c", "abc333_b", "abc341_b", "abc341_d",
        "abc332_d", "abc304_a", "abc319_e", "abc308_a", "abc339_c", "abc308_d", "abc324_d",
        "abc330_c", "abc339_b", "abc305_b", "abc307_d", "abc324_f", "abc304_d", "abc324_a",
        "abc338_e", "abc327_c", "abc319_c", "abc306_e"]


def load_tests(raw):
    """public_test_cases is a JSON string; private_test_cases may be base64+zlib+pickle."""
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return json.loads(pickle.loads(zlib.decompress(base64.b64decode(raw.encode()))))


ds = load_dataset("livecodebench/code_generation_lite", version_tag="release_v1",
                  split="test", trust_remote_code=True)
by_qid = {r["question_id"]: r for r in ds}

out = []
for i, qid in enumerate(QIDS):
    r = by_qid[qid]
    tests = load_tests(r["public_test_cases"]) + load_tests(r["private_test_cases"])
    out.append(dict(qid=qid, title=r["question_title"], difficulty=str(r["difficulty"]).lower(),
                    platform=str(r["platform"]).lower(), problem=r["question_content"],
                    tests=[dict(input=t["input"], output=t["output"]) for t in tests[:MAX_TESTS]],
                    pid=f"lcb_{i:03d}"))

CD.mkdir(parents=True, exist_ok=True)
(CD / "lcb_problems.json").write_text(json.dumps(out))
print(f"staged {len(out)} -> {CD / 'lcb_problems.json'}")
