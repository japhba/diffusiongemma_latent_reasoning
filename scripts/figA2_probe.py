"""A2: probe retention — 2x2 transfer AUC (trained-on x applied-to) + illustrating example pair.

Left:  mean held-out AUC over 56 concepts, source-CV-selected layer (probe_matrix.json headline).
Right: one positive + one negative held-out text for 161_agnews_0 (world news), scored by the
       gemma-trained probe reading gemma activations vs the SAME probe reading DG activations
       (probe_example_scores.json, cell gd).
"""
import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path("/workspace-vast/jbauer/dg_blog/figs")
SP = Path("/workspace-vast/jbauer/activation_oracles_dev/concept_probes/out/saeprobes")

pm = json.load(open(SP / "probe_matrix.json"))
h = pm["concept_auc"]["headline"]
M = np.array([[np.mean([r["auc"] for r in h["gg"]]), np.mean([r["auc"] for r in h["gd"]])],
              [np.mean([r["auc"] for r in h["dg"]]), np.mean([r["auc"] for r in h["dd"]])]])

pe = json.load(open(SP / "probe_example_scores.json"))
TAG = "161_agnews_0"
r = pe[TAG]
gd = r["cells"]["gd"]
x, ysc, y = np.array(gd["x"]), np.array(gd["y"]), np.array(r["y"])
ip = next(i for i in np.argsort(-np.minimum(x, ysc)) if y[i] == 1 and 40 < len(r["texts"][i]) < 150)
im = next(iter(np.argsort(np.maximum(x, ysc))[y[np.argsort(np.maximum(x, ysc))] == 0]))
shorten = lambda t: t if len(t) <= 145 else t[:145] + " …"
print("pos:", x[ip], ysc[ip], r["texts"][ip])
print("neg:", x[im], ysc[im], r["texts"][im][:160])

fig, axL = plt.subplots(layout="constrained")
im_ = axL.imshow(M, cmap="viridis", vmin=0.5, vmax=1.0, aspect="auto")
axL.set_box_aspect(1)
for (i, j), v in np.ndenumerate(M):
    axL.text(j, i, f"{v:.3f}", ha="center", va="center",
             color="white" if v < 0.85 else "black")
axL.set_xticks([0, 1], ["gemma-4", "DG"])
axL.set_yticks([0, 1], ["gemma-4", "DG"])
axL.set_xlabel("applied to")
axL.set_ylabel("trained on")
fig.colorbar(im_, ax=axL, shrink=0.7)
fig.savefig(OUT / "parts" / "figA2_matrix.png", dpi=200)
print(OUT / "parts" / "figA2_matrix.png")

# example-pair data consumed by build_html.py
DATA = OUT.parent / "data"; DATA.mkdir(exist_ok=True)
json.dump({"tag": TAG, "layer": r["layers"]["gd"],
           "pos": {"text": r["texts"][ip], "g": float(x[ip]), "d": float(ysc[ip])},
           "neg": {"text": r["texts"][im][:400], "g": float(x[im]), "d": float(ysc[im])}},
          open(DATA / "probe_pair.json", "w"), indent=1)
print(DATA / "probe_pair.json")
