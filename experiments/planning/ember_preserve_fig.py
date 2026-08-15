"""Multipanel figure for the preservation case (ember_kill_fig style): per exemplar seed, top
panel = base autonomous dynamics (idiom black, seasonal purple), bottom panels = persistent
idiom-kill started at t_abl in {2,4,6,8,10} (shaded from onset). Columns: s5 (strongest native
contest, preserved at early onsets) vs s1 (no contest -> third basin). Purple border = native
seasonal preserved, red = third basin. Base trajectories sampled once into ember_base_traj.json
(worker on :18711 needed only for that first run). -> figs/ember_preserve.png"""
import os
import json, sys, urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from constrained_common import CHOSEN

EXP = Path(os.environ.get("DG_PLANNING_DIR", str(Path(__file__).resolve().parent / "exp")))
FIGS = Path(os.environ.get("DG_FIGS_DIR", str(Path(__file__).resolve().parent / "figs")))
PERSIST_T = [2, 4, 6, 8, 10]
COLS = [(5, "s5 — native contest 0.75"), (1, "s1 — native contest 0.09")]
PUR = "#9c36b5"


def base_trajs():
    f = EXP / "ember_base_traj.json"
    if f.exists():
        return json.load(open(f))
    W = os.environ.get("DG_WORKER", "http://localhost:18711")
    FRAME = "{q}\n\nOutput ONLY the text itself — no preamble, no quotes, no explanation."
    HOTR = dict(T=64, C=128, t_max=1.3, t_min=0.8, entropy_bound=0.3, early_stop=False, top_k=10)
    Q = next(p["q"] for p in CHOSEN if p["id"] == "palindrome_words__3")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("google/gemma-4-26b-a4b-it")
    IDT = tok.encode("All for one and one for all.", add_special_tokens=False)
    SET = tok.encode("All leaves fall when leaves fall all.", add_special_tokens=False)
    DIFF = [j for j in range(8) if IDT[j] != SET[j]]
    out = {}
    for seed in (1, 2, 3, 4, 5, 6):
        cap = json.load(open(EXP / f"nego/palindrome_words__3__s{seed}.json"))
        i2s, fin = cap["id2str"], cap["final_ids"]
        dead = set(cap["eos_token_ids"]) | {cap["pad_token_id"]}
        live = [p for p, x in enumerate(fin) if x not in dead]
        txt = lambda x: i2s.get(str(x), "?").replace("▁", " ")
        start = next(p for p in live if txt(fin[p]).strip().lower() == "all")
        sl = [start + j for j in DIFF]
        r = urllib.request.urlopen(urllib.request.Request(
            f"{W}/sample", json.dumps(dict(prompt=FRAME.format(q=Q), seed=seed, **HOTR,
                                           s_topk_record=32)).encode(),
            {"Content-Type": "application/json"}), timeout=1800)
        d = json.loads(r.read())
        rec = d["s_rec"]
        def mass(t, p, tid):
            ids = rec["ids"][t][p]
            return float(np.exp(rec["lp"][t][p][ids.index(tid)])) if tid in ids else 0.0
        out[str(seed)] = dict(
            m_idt=[float(np.mean([mass(t, p, IDT[j]) for j, p in zip(DIFF, sl)])) for t in range(20)],
            m_set=[float(np.mean([mass(t, p, SET[j]) for j, p in zip(DIFF, sl)])) for t in range(20)],
            outcome=("idiom" if "for one and one for" in d["final_text"].lower() else "other"))
        print(f"  base s{seed} captured", flush=True)
    json.dump(out, open(f, "w"))
    return out


def main():
    k2 = json.load(open(EXP / "ember_kill2.json"))
    bases = base_trajs()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    nrow = len(PERSIST_T) + 1
    fig, axes = plt.subplots(nrow, len(COLS), figsize=(11, 9), sharex=True, sharey=True)
    for c, (seed, coltitle) in enumerate(COLS):
        axB = axes[0, c]
        b = bases[str(seed)]
        axB.plot(b["m_idt"], color="k", label="idiom")
        axB.plot(b["m_set"], color=PUR, alpha=0.9, label="seasonal")
        axB.text(0.99, 0.5, f"base → {b['outcome']}", transform=axB.transAxes,
                 ha="right", va="center", color="0.3", fontsize="x-small")
        axB.set_title(coltitle, fontsize=10)
        if c == 0:
            axB.legend(fontsize="x-small", ncols=2, loc="center left")
        for i, t_abl in enumerate(PERSIST_T):
            ax = axes[i + 1, c]
            r = k2[f"s{seed}|kill@t{t_abl}+"]
            ax.plot(r["m_idt"], color="k")
            ax.plot(r["m_set"], color=PUR, alpha=0.9)
            ax.axvspan(t_abl, 19, color="#d32f2f", alpha=0.08, lw=0)
            ax.axvline(t_abl, color="#d32f2f", lw=1.0, ls=":")
            ax.text(0.99, 0.5, f"kill t{t_abl}+ → {r['outcome']}", transform=ax.transAxes,
                    ha="right", va="center", color="0.3", fontsize="x-small")
            if r["outcome"] == "seasonal":
                for sp in ax.spines.values():
                    sp.set_edgecolor(PUR); sp.set_linewidth(1.8)
            elif r["outcome"] == "other":
                for sp in ax.spines.values():
                    sp.set_edgecolor("#d32f2f"); sp.set_linewidth(1.4)
    for ax in axes.flat:
        ax.set_ylim(-0.08, 1.08); ax.set_yticks([0, 1])
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[nrow // 2, 0].set_ylabel("final state S-mass at DIFF slots")
    for c in range(len(COLS)):
        axes[-1, c].set_xlabel("denoising step t")
    fig.suptitle("Preservation, not flipping: persistent idiom-kill from t$_{abl}$ (shaded) keeps the NATIVE seasonal\n"
                 "black = idiom mass, purple = seasonal mass; purple border = seasonal preserved, red = third basin",
                 fontsize=11)
    fig.tight_layout()
    p = FIGS / "ember_preserve.png"
    fig.savefig(p, dpi=150)
    print(p)


if __name__ == "__main__":
    main()
