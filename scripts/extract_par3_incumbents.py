"""Extract the incumbent image letter jb per (tag, seed) for the par3 ladder readout.

jb = the letter already sitting at canvas slot B of the base capture (the incumbent the
injection must displace); the reader excludes its image from the placebo pool P. Reading it
needs the heavy per-state denoising films exp/dg_planning/nego2/{tag}__s{s}.json (~38 MB/tag,
un-vendored) — NOT bare-clone-rerunnable; rerun after regenerating nego2 via the
experiments/planning/ captures. Output vendored as src_data/planning/par3_incumbents.json.
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXP = Path(os.environ.get("DG_PLANNING_DIR", "/workspace-vast/jbauer/diffusiongemma/exp/dg_planning"))

par = json.load(open(ROOT / "src_data" / "planning" / "xtask_par3.json"))
dns = json.load(open(ROOT / "src_data" / "planning" / "xtask_samecase_nsweep.json"))

out = {}
for tag in ("UU3", "UU5"):
    for s in range(10):
        if f"{tag}|s{s}|par3|pairs" not in par:
            continue
        bsrc = par.get(f"{tag}|s{s}|par3|base") or dns[f"{tag}|s{s}|base"]
        d = json.load(open(EXP / f"nego2/{tag}__s{s}.json"))
        B = bsrc.get("B", 8)
        out[f"{tag}|s{s}"] = d["id2str"][str(d["final_ids"][B])].replace("▁", " ").strip()

json.dump(out, open(ROOT / "src_data" / "planning" / "par3_incumbents.json", "w"), indent=1)
print(json.dumps(out, indent=1))
