"""A2: probe retention — 3x3 transfer AUC with DG's modes split + illustrating example pair.

Top:   mean held-out AUC over 56 concepts, source-CV-selected layer; rows/cols gemma-4 /
       DG causal / DG bidirectional, last-position read everywhere (probe_matrix.json modes:
       headline = DG causal, declast = DG bidirectional; causal<->bidirectional cross cells
       were not measured -> masked grey).
Below: one positive + one negative held-out text for 161_agnews_0 (world news), scored by the
       gemma-trained probe reading gemma activations vs the SAME probe reading DG activations
       (probe_example_scores.json, cell gd).
"""
import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figs"
SP = ROOT / "src_data" / "saeprobes"

pm = json.load(open(SP / "probe_matrix.json"))
mean = lambda mode, cell: float(np.mean([r["auc"] for r in pm["concept_auc"][mode][cell]]))
M = np.ma.masked_invalid([
    [mean("headline", "gg"), mean("headline", "gd"), mean("declast", "gd")],
    [mean("headline", "dg"), mean("headline", "dd"), np.nan],
    [mean("declast", "dg"),  np.nan,                 mean("declast", "dd")],
])
# report-style model/mode/read tick hierarchy (source_label convention)
LABS = ["G\ncausal\nlast", "DG\ncausal\nlast", "DG\nbidirectional\nlast"]

pe = json.load(open(SP / "probe_example_scores.json"))
# example concept chosen by scan over all 56: clean prose pos+neg, both reads agreeing at
# saturation (clickbait: pos 1.00/1.00, neg 0.00/0.01)
TAG, CONCEPT, POS_LAB, NEG_LAB = "105_click_bait", "clickbait headline", "clickbait", "regular headline"
r = pe[TAG]
gd = r["cells"]["gd"]
x, ysc, y = np.array(gd["x"]), np.array(gd["y"]), np.array(r["y"])
ok = lambda i: 60 < len(r["texts"][i]) < 220 and all(ord(c) < 0x2500 for c in r["texts"][i])
ip = max((i for i in range(len(y)) if y[i] == 1 and ok(i)), key=lambda i: min(x[i], ysc[i]))
im = min((i for i in range(len(y)) if y[i] == 0 and ok(i)), key=lambda i: max(x[i], ysc[i]))
print("pos:", x[ip], ysc[ip], r["texts"][ip])
print("neg:", x[im], ysc[im], r["texts"][im])

fig, axL = plt.subplots(layout="constrained")
cmap = plt.get_cmap("viridis").copy()
cmap.set_bad("#e9ecef")
im_ = axL.imshow(M, cmap=cmap, vmin=0.5, vmax=1.0, aspect="auto")
axL.set_box_aspect(1)
for (i, j), v in np.ndenumerate(M):
    axL.text(j, i, "—" if M.mask[i, j] else f"{v:.2f}", ha="center", va="center",
             color="0.4" if M.mask[i, j] else ("white" if v < 0.85 else "black"))
axL.set_xticks(range(3), LABS)
axL.set_yticks(range(3), [l.replace("\n", " · ") for l in LABS])
fig.supxlabel("target", fontweight="bold")
fig.supylabel("source", fontweight="bold")
fig.colorbar(im_, ax=axL, shrink=0.7)
fig.savefig(OUT / "parts" / "figA2_matrix.png", dpi=200)
print(OUT / "parts" / "figA2_matrix.png")

# example-pair data consumed by build_html.py
DATA = OUT.parent / "data"; DATA.mkdir(exist_ok=True)
json.dump({"tag": TAG, "concept": CONCEPT, "pos_label": POS_LAB, "neg_label": NEG_LAB,
           "layer": r["layers"]["gd"],
           "pos": {"text": r["texts"][ip], "g": float(x[ip]), "d": float(ysc[ip])},
           "neg": {"text": r["texts"][im][:400], "g": float(x[im]), "d": float(ysc[im])}},
          open(DATA / "probe_pair.json", "w"), indent=1)
print(DATA / "probe_pair.json")
