"""A3: steering retention — blind-pair accuracy matrix (direction source x steered model) + example pair.

Left:  pooled blind-pair judge accuracy per cell (pr80 protocol), RepE tasks, gemini judge.
Right: +steer vs -steer generations for re_0_happiness, cell gd (gemma direction -> DG), with the
       judge's justification (italic).
"""
import json
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path("/workspace-vast/jbauer/dg_blog/figs")
SP = Path("/workspace-vast/jbauer/activation_oracles_dev/concept_probes/out/saeprobes")
jp = json.load(open(SP / "judged_dom_gens_paired_gemini.json"))
dgens = json.load(open(SP / "dom_gens.json"))
carriers = json.load(open(SP / "dom_carriers.json"))

acc = defaultdict(list)
for k, v in jp["pairs"].items():
    tag, pi, cell, prot, _ = k.split("|")
    if prot == "pr80" and "correct" in v:
        acc[cell].append(bool(v["correct"]))
ROWS = [("g", "gemma-4 causal"), ("e", "DG causal"), ("c", "DG denoising")]
COLS = [("g", "gemma-4"), ("d", "DG")]
M = np.array([[np.mean(acc[r + c]) for c, _ in COLS] for r, _ in ROWS])

KEY = "re_0_happiness|8|gd|pr80|pos"
v = jp["pairs"][KEY]
pos, neg = dgens[v["positive_key"]], dgens[v["negative_key"]]

fig, axL = plt.subplots(layout="constrained")
im_ = axL.imshow(M, cmap="viridis", vmin=0.5, vmax=1.0, aspect="auto")
axL.set_box_aspect(1)
for (i, j), val in np.ndenumerate(M):
    axL.text(j, i, f"{val:.2f}", ha="center", va="center", color="white" if val < 0.85 else "black")
axL.set_xticks([0, 1], [lab for _, lab in COLS])
axL.set_yticks([0, 1, 2], [lab for _, lab in ROWS])
axL.set_xlabel("steered model")
fig.colorbar(im_, ax=axL, shrink=0.7)
fig.savefig(OUT / "parts" / "figA3_matrix.png", dpi=200)
print(OUT / "parts" / "figA3_matrix.png")

# example-pair data consumed by build_html.py: same gemma-fit direction on BOTH target models
# pi=8 = 6 RepE originals + carrier_additions_sonnet5 EMOTIONS[2]
CARRIER = "My dog passed away this morning and I don't know what to do."
assert CARRIER in json.load(open(SP / "carrier_additions_sonnet5.json"))["EMOTIONS"]
DATA = OUT.parent / "data"; DATA.mkdir(exist_ok=True)
cells = {}
for cell, lab in (("gg", "gemma-4"), ("gd", "DiffusionGemma")):
    vv = jp["pairs"][f"re_0_happiness|8|{cell}|pr80|pos"]
    cells[cell] = {"model": lab,
                   "pos": dgens[f"re_0_happiness|8|{cell}|pr80|pos"][:600],
                   "neg": dgens[f"re_0_happiness|8|{cell}|pr80|neg"][:600],
                   "judge": vv["justification"], "confidence": vv.get("confidence")}
json.dump({"task": "happiness (RepE)", "direction": "fit on gemma-4 causal stream",
           "carrier": CARRIER, "cells": cells},
          open(DATA / "steer_pair.json", "w"), indent=1)
print(DATA / "steer_pair.json")
print("matrix:", {r + c: round(float(np.mean(acc[r + c])), 3) for r, _ in ROWS for c, _ in COLS})
