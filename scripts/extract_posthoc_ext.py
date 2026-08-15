"""Merge the 2026-08-08 EXTENSION battery (20 new problems) into the vendored posthoc data,
growing the figA8 correlation scatter from n=20 to n=40 problems.

Raw inputs (NOT bare-clone-rerunnable):
  exp/dg_lockin/posthoc/ext_clean.json / ext_suscept.json — captured on the DG pod
    (the pod capture dir, now vendored as experiments/posthoc/ — suscept.py with the channel-scaffold parsing fix; same GRID
    C=256 T=128 t 0.9->0.5 eb 0.15, ANSWER_FIRST framing, clean 5 seeds,
    suscept rhos {0,.25,.5,.75,1} x corr_seeds {0..4} @ k=0 as the original run).
  diffusiongemma/posthoc/ext_difficulty.json — 3 fresh blind subagent raters (same protocol).

NOTE the extension rollouts carry the '<|channel>thought\\n<channel|>' canvas scaffold (pod
state since 2026-08-04; the original July run predates it). The capture script excludes the
scaffold positions from both the answer slot and the clamped CoT set, so commit/S semantics
are unchanged.

Writes (in-place merge): src_data/posthoc/{clean,suscept,difficulty}.json.
Original cells are never modified; a second run fails loudly on the key-overlap assert.
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SP = ROOT / "src_data" / "posthoc"
EXP = Path(os.environ.get("DG_POSTHOC_DIR", Path(__file__).resolve().parent.parent / "experiments" / "posthoc" / "out"))
DIFF_EXT = Path(os.environ.get("DG_DIFFICULTY_JSON", Path(__file__).resolve().parent.parent / "experiments" / "posthoc" / "ext_difficulty.json"))

for name, ext in [("clean.json", EXP / "ext_clean.json"), ("suscept.json", EXP / "ext_suscept.json")]:
    base = json.load(open(SP / name))
    new = json.load(open(ext))
    overlap = set(base) & set(new)
    assert not overlap, f"{name}: unexpected key overlap {sorted(overlap)[:5]}"
    base.update(new)
    json.dump(base, open(SP / name, "w"))
    print(f"{name}: {len(base) - len(new)} + {len(new)} = {len(base)} cells")

diff = json.load(open(SP / "difficulty.json"))
dext = json.load(open(DIFF_EXT))
for r in ("r1", "r2", "r3"):
    diff["raters"][r].update(dext["raters"][r])
diff["mean"].update(dext["mean"])
diff["note"] += " | extended 2026-08-08 with 20 new problems, same protocol"
json.dump(diff, open(SP / "difficulty.json", "w"), indent=1)
print(f"difficulty.json: {len(diff['mean'])} pids")
