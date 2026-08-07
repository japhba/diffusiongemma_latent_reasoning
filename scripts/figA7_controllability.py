"""A7 (self-correction): constraint-margin trajectories for the word-palindrome task, hot
sampler, on a linearly time-warped axis (tau = t / each run's last margin change, so the active
phase of every run spans [0,1]). Escapes bold green. Also emits data/selfcorr_steps.json:
three decoded canvases (early / mid-violating / late) of the flagship escape run
palindrome_words__3__hot_s3, for the steps card.

Data: src_data/planning/canalysis.json (margins) + src_data/planning/gallery.json (frames).
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figs"
RECS = json.load(open(ROOT / "src_data" / "planning" / "canalysis.json"))
OC = {"clean": ("#7aa2ff", 0.5, 1.0), "escape": ("#2f9e44", 1.0, 2.4), "trapped": ("#e8590c", 0.55, 1.0)}

rs = [r for r in RECS if r["ttype"] == "palindrome_words" and r["regime"] == "hot"]
fig, ax = plt.subplots(layout="constrained",
                       figsize=(plt.rcParams["figure.figsize"][0] * 1.15,
                                plt.rcParams["figure.figsize"][1] * 0.85))
for r in sorted(rs, key=lambda r: r["outcome"] == "escape"):
    m = np.array([np.nan if x is None else x for x in r["m_out"]], dtype=float)
    chg = [t for t in range(1, len(m)) if m[t] != m[t - 1] and not (np.isnan(m[t]) and np.isnan(m[t - 1]))]
    settle = max(chg) if chg else 1
    tau = np.arange(len(m)) / settle
    col, al, lw = OC[r["outcome"]]
    keep = tau <= 1.3
    ax.plot(tau[keep], m[keep], color=col, alpha=al, linewidth=lw)
ax.set_xlabel(r"warped diffusion time $\tau$  ($\tau{=}1$: last margin change)")
ax.set_ylabel("constraint margin (violations)")
ax.set_xlim(0, 1.3)
ax.spines[["top", "right"]].set_visible(False)
handles = [plt.Line2D([], [], color=OC[k][0], linewidth=2, label=k) for k in ("clean", "escape", "trapped")]
ax.legend(handles=handles, frameon=False, fontsize="small")
(OUT / "parts").mkdir(exist_ok=True)
fig.savefig(OUT / "parts" / "figA7_margins.png", dpi=200)
print(OUT / "parts" / "figA7_margins.png")
print({o: sum(1 for r in rs if r["outcome"] == o) for o in ("clean", "escape", "trapped")}, "n =", len(rs))

# ---- steps card data: early / mid / late canvases of the flagship escape ----
g = next(e for e in json.load(open(ROOT / "src_data" / "planning" / "gallery.json"))
         if e["name"] == "palindrome_words__3__hot_s3")
pad = set(g["pad_ids"])
def dec(t):
    s = "".join(g["id2str"][str(x)] for x in g["frames"][t] if x not in pad).replace("▁", " ")
    return s.rsplit("<channel|>", 1)[-1].strip()  # drop the channel markers, keep the visible text
STEPS = [(1, "early"), (5, "mid"), (10, "late")]
card = {"prompt_task": "Write a sentence of at least 6 words whose sequence of words reads the "
                       "same forwards and backwards.",
        "steps": [{"t": t, "phase": ph, "text": dec(t), "margin": g["margin"][t]} for t, ph in STEPS]}
DATA = ROOT / "data"; DATA.mkdir(exist_ok=True)
json.dump(card, open(DATA / "selfcorr_steps.json", "w"), indent=1)
print(DATA / "selfcorr_steps.json")
for s in card["steps"]:
    print(f"t={s['t']} ({s['phase']}) margin {s['margin']}: {s['text']}")
