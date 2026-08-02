"""A4: J-Lens retention — mean A-score matrix (Jacobian fit-stream x read stream) + example pair.

A = 1 - exp(-n), n = top-20 token appearances (all layers) matching a paper ground-truth regex.
Left:  mean A over 551 eval items per cell, six paper sets.
Right: a transfer hit (gemma-fit Jacobian read on DG residuals) and a miss with LLM-inspection
       rungs firing (italic).
"""
import json
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path("/workspace-vast/jbauer/dg_blog/figs")
SP = Path("/workspace-vast/jbauer/activation_oracles_dev/concept_probes/out/saeprobes")
jj = json.load(open(SP / "jlens/judged_jlens_percepts.json"))

agg = defaultdict(list)
for k, v in jj["scores"].items():
    st, idx, cfg, test = k.split("|")
    if "A" in v:
        agg[(cfg, test)].append(v["A"]["score"])
ROWS = [("g_shared", "gemma-4"), ("dgc_shared", "DG causal"), ("dgb_shared", "DG denoising")]
COLS = [("g", "gemma-4 residuals"), ("dg", "DG residuals")]
M = np.array([[np.mean(agg[(r, c)]) for c, _ in COLS] for r, _ in ROWS])

HIT = "multilingual|34|g_shared|dg"
MISS = "association|61|dgb_shared|g"

fig, axL = plt.subplots(layout="constrained")
im_ = axL.imshow(M, cmap="viridis", vmin=0.0, vmax=0.6, aspect="auto")
axL.set_box_aspect(1)
for (i, j), val in np.ndenumerate(M):
    axL.text(j, i, f"{val:.2f}", ha="center", va="center", color="white" if val < 0.4 else "black")
axL.set_xticks([0, 1], [lab for _, lab in COLS])
axL.set_yticks([0, 1, 2], [lab for _, lab in ROWS])
axL.set_xlabel("read on")
fig.colorbar(im_, ax=axL, shrink=0.7)
fig.savefig(OUT / "parts" / "figA4_matrix.png", dpi=200)
print(OUT / "parts" / "figA4_matrix.png")

# per-layer top-5 example data consumed by build_html.py (gemma-fit Jacobian, both read streams)
DATA = OUT.parent / "data"; DATA.mkdir(exist_ok=True)
ev = json.load(open(SP / "jlens/eval_2x2.json"))
it = next(x for x in ev["examples"] if x["name"] == "carnival-ocean")
LAYERS = [27, 20, 12, 4]
out = {"name": it["name"], "set": it["set"], "tail": it["prompt_tail"],
       "intermediates": it["intermediates"], "config": "g_shared", "layers": LAYERS,
       "tops": {test: {str(L): it["tops"][f"{test}|g_shared|L{L}"][:5] for L in LAYERS}
                for test in ("g", "dg")}}
json.dump(out, open(DATA / "jlens_layers.json", "w"), indent=1)
print(DATA / "jlens_layers.json")
print("matrix:", {f"{r}|{c}": round(float(np.mean(agg[(r, c)])), 3) for r, _ in ROWS for c, _ in COLS})
