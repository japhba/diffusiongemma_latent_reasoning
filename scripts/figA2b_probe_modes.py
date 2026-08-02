"""A2b: probe transfer with DG's modes split — 3x3 trained-on x applied-to AUC matrix.

Rows/cols: gemma-4 (causal), DG causal, DG bidirectional; last-position read everywhere
(probe_matrix.json modes: headline = DG causal, declast = DG bidirectional last-token;
decmean printed for the caption). causal<->bidirectional cross cells were not measured.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path("/workspace-vast/jbauer/dg_blog/figs")
SP = Path("/workspace-vast/jbauer/activation_oracles_dev/concept_probes/out/saeprobes")

pm = json.load(open(SP / "probe_matrix.json"))
mean = lambda mode, cell: float(np.mean([r["auc"] for r in pm["concept_auc"][mode][cell]]))

gg = mean("headline", "gg")
M = np.ma.masked_invalid([
    [gg,                     mean("headline", "gd"), mean("declast", "gd")],
    [mean("headline", "dg"), mean("headline", "dd"), np.nan],
    [mean("declast", "dg"),  np.nan,                 mean("declast", "dd")],
])
LABS = ["gemma-4", "DG causal", "DG bidirectional"]

fig, ax = plt.subplots(layout="constrained")
cmap = plt.get_cmap("viridis").copy()
cmap.set_bad("#e9ecef")
im = ax.imshow(M, cmap=cmap, vmin=0.5, vmax=1.0, aspect="auto")
ax.set_box_aspect(1)
for (i, j), v in np.ndenumerate(M):
    ax.text(j, i, "—" if M.mask[i, j] else f"{v:.3f}", ha="center", va="center",
            color="0.4" if M.mask[i, j] else ("white" if v < 0.85 else "black"))
ax.set_xticks(range(3), [l.replace(" ", "\n") for l in LABS])
ax.set_yticks(range(3), LABS)
ax.set_xlabel("applied to")
ax.set_ylabel("trained on")
fig.colorbar(im, ax=ax, shrink=0.7)
fig.savefig(OUT / "figA2b_probe_modes.png", dpi=200)
print(OUT / "figA2b_probe_modes.png")
print({"gg": gg, "dd_causal": mean("headline", "dd"), "dd_bidir_last": mean("declast", "dd"),
       "dd_bidir_mean": mean("decmean", "dd"), "bidir_to_g": mean("declast", "dg")})
