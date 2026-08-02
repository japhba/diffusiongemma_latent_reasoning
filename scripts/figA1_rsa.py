"""A1: representation similarity between gemma-4 and DiffusionGemma — matched cosine + CKA by layer.

Data: concept_probes/out/saeprobes/dg_rsa_cka_curves.json (3584 texts, 30 layers).
Pairs: g_enc = gemma-4 x DG causal (model gap); g_dec = gemma-4 x DG denoising (both gaps);
       enc_dec = DG causal x DG denoising (mode gap).
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path("/workspace-vast/jbauer/dg_blog/figs")
SP = Path("/workspace-vast/jbauer/activation_oracles_dev/concept_probes/out/saeprobes")
c = json.load(open(SP / "dg_rsa_cka_curves.json"))
L = np.arange(c["n_layers"])
PAIRS = [("g_enc", "gemma-4 × DG causal (model gap)", "#1971c2"),
         ("g_dec", "gemma-4 × DG bidirectional (model + mode gap)", "#e8590c"),
         ("enc_dec", "DG causal × DG bidirectional (mode gap)", "#2f9e44")]

fig, axes = plt.subplots(1, 2, sharex=True, layout="constrained",
                         figsize=(plt.rcParams["figure.figsize"][0] * 1.6,
                                  plt.rcParams["figure.figsize"][1] * 0.85))
for ax, meas, title in ((axes[0], "cos", "matched cosine"), (axes[1], "cka", "linear CKA")):
    for pair, lab, col in PAIRS:
        ax.plot(L, c["pairs"][pair][meas], color=col, label=lab)
    ax.set_xlabel("layer $\\ell$")
    ax.set_ylabel("matched cosine" if meas == "cos" else "linear CKA")
    ax.set_ylim(0, 1.02)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].legend(loc="lower left", frameon=False)
fig.savefig(OUT / "figA1_rsa_cosine_cka.png", dpi=200)
print(OUT / "figA1_rsa_cosine_cka.png")
for pair, lab, _ in PAIRS:
    print(pair, "cos mean", round(float(np.mean(c["pairs"][pair]["cos"])), 3),
          "cka mean", round(float(np.mean(c["pairs"][pair]["cka"])), 3))
