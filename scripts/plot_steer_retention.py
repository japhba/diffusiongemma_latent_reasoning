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

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figs"
SP = ROOT / "src_data" / "saeprobes"
jp = json.load(open(SP / "judged_dom_gens_paired_gemini.json"))
dgens = json.load(open(SP / "dom_gens.json"))
carriers = json.load(open(SP / "dom_carriers.json"))

acc = defaultdict(list)
for k, v in jp["pairs"].items():
    tag, pi, cell, prot, _ = k.split("|")
    if prot == "pr80" and "correct" in v:
        acc[cell].append(bool(v["correct"]))
# report-style model/mode/read tick hierarchy (source_label convention); targets carry the
# pr80 write protocol
ROWS = [("g", "G · causal · last"), ("e", "DG · causal · last"), ("c", "DG · bidirectional · last")]
COLS = [("g", "gemma-4\npr80 write"), ("d", "DiffusionGemma\npr80 write")]
M = np.array([[np.mean(acc[r + c]) for c, _ in COLS] for r, _ in ROWS])

# example: happiness on the dog carrier (both cells correct; pi=8 = 6 RepE originals +
# carrier_additions_sonnet5 EMOTIONS[2], hence the hardcoded carrier + assert)
EX_TAG, EX_PI = "re_0_happiness", 8

fig, axL = plt.subplots(layout="constrained")
im_ = axL.imshow(M, cmap="viridis", vmin=0.5, vmax=1.0, aspect="auto")
axL.set_box_aspect(1)
for (i, j), val in np.ndenumerate(M):
    axL.text(j, i, f"{val:.2f}", ha="center", va="center", color="white" if val < 0.85 else "black")
axL.set_xticks([0, 1], [lab for _, lab in COLS])
axL.set_yticks([0, 1, 2], [lab for _, lab in ROWS])
fig.supxlabel("target", fontweight="bold")
fig.supylabel("source", fontweight="bold")
fig.colorbar(im_, ax=axL, shrink=0.7)
fig.savefig(OUT / "parts" / "steer_matrix.png", dpi=200)
print(OUT / "parts" / "steer_matrix.png")

# example-pair data consumed by build_html.py: same gemma-fit direction on BOTH target models
CARRIER = "My dog passed away this morning and I don't know what to do."
assert CARRIER in json.load(open(SP / "carrier_additions_sonnet5.json"))["EMOTIONS"]
DATA = OUT.parent / "data"; DATA.mkdir(exist_ok=True)
cells = {}
for cell, lab in (("gg", "gemma-4"), ("gd", "DiffusionGemma")):
    vv = jp["pairs"][f"{EX_TAG}|{EX_PI}|{cell}|pr80|pos"]
    cells[cell] = {"model": lab,
                   "pos": dgens[f"{EX_TAG}|{EX_PI}|{cell}|pr80|pos"][:600],
                   "neg": dgens[f"{EX_TAG}|{EX_PI}|{cell}|pr80|neg"][:600],
                   "judge": vv["justification"], "confidence": vv.get("confidence")}
json.dump({"task": "happiness (RepE)", "concept": "happiness",
           "direction": "fit on gemma-4 causal stream",
           "carrier": CARRIER, "cells": cells},
          open(DATA / "steer_pair.json", "w"), indent=1)
print(DATA / "steer_pair.json")
print("matrix:", {r + c: round(float(np.mean(acc[r + c])), 3) for r, _ in ROWS for c, _ in COLS})
