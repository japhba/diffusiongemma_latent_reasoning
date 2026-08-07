"""A10: self-repair on the clock-strike problem ("A clock takes 6 seconds to strike 4 o'clock
(4 chimes) — how long for 9 o'clock?"; correct 16, fencepost attractor 18).

Left:  natural cold runs transiently visit wrong answers (12/18) on the canvas and patch them
       to 16 within a few steps (answer value per denoising step, decoded from the delta-encoded
       canvases in com_clock_anim.json).
Right: escaping a HARVESTED confident-wrong state (planted at init_step, re-denoised): cold
       recipients stay stuck at 18 at every depth; very-hot recipients escape to 16 only when
       planted into enough noise (sharp threshold ~step 48/128). Data: com_escape_minimum.json.
"""
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figs"
ANS_RX = re.compile(r"(-?\d[\d,]*|\bYes\b|\bNo\b)", re.I)

def first_answer(full):
    line1 = full.split("\n", 1)[0]
    m = ANS_RX.search(line1)
    return m.group(0).lstrip("0") or "0" if m else None

# ---- left: answer value per step, natural cold runs ----
anim = json.load(open(ROOT / "src_data" / "lockin" / "com_clock_anim.json"))["dg"]
cold = [c for c in anim if c["temp"] == "cold"]
W = 10
VALS = ["1", "12", "18", "16"]  # bottom -> top; 16 = correct on top
YPOS = {v: i for i, v in enumerate(VALS)}

fig, (axL, axR) = plt.subplots(1, 2, layout="constrained",
                               figsize=(plt.rcParams["figure.figsize"][0] * 1.7,
                                        plt.rcParams["figure.figsize"][1] * 0.9))
C = plt.rcParams["axes.prop_cycle"].by_key()["color"]
for ci, c in enumerate(cold):
    frames = [list(c["steps"][0])]
    for delta in c["steps"][1:W]:
        f = list(frames[-1])
        for pos, tok in delta:
            f[pos] = tok
        frames.append(f)
    ys = [YPOS.get(first_answer("".join(f)), np.nan) for f in frames]
    off = (ci - 1) * 0.06
    axL.step(range(len(ys)), [y + off for y in ys], where="post", color=C[ci],
             marker="o", markersize=3.5, label=f'seed {c["seed"]}')
axL.set_yticks(range(len(VALS)), VALS)
axL.axhline(YPOS["16"] - 0.5, color="0.85", linewidth=0.6, zorder=0)
axL.set_xlabel("denoising step $t$")
axL.set_ylabel("answer on the canvas")
axL.spines[["top", "right"]].set_visible(False)
axL.legend(frameon=False, fontsize="small", loc="lower right")

# ---- right: escape fractions vs plant depth ----
em = json.load(open(ROOT / "src_data" / "lockin" / "com_escape_minimum.json"))
rows, meta = em["rows"], em["meta"]
INIT = meta["init_steps"]
COLT = {"cold": "#1971c2", "vhot": "#c2255c"}
for tname in meta["recip_temps"]:
    esc = [np.mean([r["escaped"] for r in rows if r["recip_temp"] == tname and r["init_step"] == k]) for k in INIT]
    stk = [np.mean([r["stuck"] for r in rows if r["recip_temp"] == tname and r["init_step"] == k]) for k in INIT]
    axR.plot(INIT, esc, "-o", color=COLT[tname], markersize=4, label=f"{tname} — escape → 16")
    axR.plot(INIT, stk, "--o", color=COLT[tname], markersize=4, alpha=0.6, label=f"{tname} — stuck @ 18")
axR.set_ylim(-0.04, 1.06)
axR.set_xticks(INIT)
axR.set_xlabel(r"plant step of the wrong state (of $T{=}128$)")
axR.set_ylabel("fraction of recipients")
axR.spines[["top", "right"]].set_visible(False)
axR.legend(frameon=False, fontsize="small", loc="center right")
fig.savefig(OUT / "figA10_selfrepair_clock.png", dpi=200)
print(OUT / "figA10_selfrepair_clock.png")
for c in cold:
    print("cold seed", c["seed"], "path", "→".join(c["answer_path"]), "final", c["final"], "ok", c["ok"])
