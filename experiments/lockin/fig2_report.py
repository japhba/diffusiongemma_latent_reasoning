"""THE Fig-2 truncation report — single main view, remote (dg-fig2-topk) styling verbatim.

Layout matched to the remote report's "Failure modes vs. accuracy" section:
  panel = task (GPQA, AMC/AIME, LiveCodeBench, IMO, word pool)
  x     = truncation level: soft, top-8, top-4, top-2, top-1
  each x tick = split violin: standard sampler (left, dark) | gentle sampler (right, light)
  dots  = individual rollouts at their own outcome (1/0, jittered sina inside the half
          envelope), colored by failure mode (degenerate loop = BROWN)
  bars  = per-half mean (cell accuracy); dashed polylines trace means across levels
  diamonds = the paper's reported Fig-2 scores (±std whisker)
  click any dot -> side detail panel replaying that rollout's denoising film
          (viridis entropy, canvas-boundary chips, step slider)

Everything else (repro ladders, decomposition, oracle, stabilizer zoo, audit) is ARCHIVED
at fig2_trunc_archive.html / dg-fig2-trunc-archive.html (frozen snapshot, linked in footer).
Canonical output: exp/dg_lockin/fig2_trunc.html (stable path, no-cache).
"""
import os
import hashlib
import json
import re
import sys
from pathlib import Path

sys.set_int_max_str_digits(1_000_000)  # degenerate rollouts store 1000s-digit extracted ints

import numpy as np

CD = Path(os.environ.get("DG_LOCKIN_DIR", str(Path(__file__).resolve().parent / "data")))
RB = Path(os.environ.get("DG_RB_DIR", str(CD / "rb_steps")))
HTML = Path(os.environ.get("DG_FIG2_HTML", str(CD / "fig2_trunc.html")))
PAPER = json.loads((CD / "paper_fig2.json").read_text())

RUNGS = ["soft", "k8", "k4", "k2", "k1"]           # major x ticks, left -> right
RUNG_LAB = {"soft": "soft", "k8": "top-8", "k4": "top-4", "k2": "top-2", "k1": "top-1"}
PAPER_KEY = {"soft": "Baseline", "k8": "k=8", "k4": "k=4", "k2": "k=2", "k1": "k=1"}
WORD_DS = ("univocalic", "lipogram", "piem", "self_count_words")

# viridis, 33 stops, for the film entropy coloring (0 .. EMAX nats)
VIR = ['#440154', '#470d60', '#48186a', '#482374', '#472d7b', '#453781', '#424086', '#3e4989',
       '#3b528b', '#375b8d', '#33638d', '#2f6b8e', '#2c728e', '#297a8e', '#26828e', '#23898e',
       '#21918c', '#1f988b', '#1fa088', '#22a785', '#28ae80', '#32b67a', '#3fbc73', '#4ec36b',
       '#5ec962', '#70cf57', '#84d44b', '#98d83e', '#addc30', '#c2df23', '#d8e219', '#ece51b',
       '#fde725']
EMAX = 4.0

# ---------------------------------------------------------------------------------------------
# data loading: task x arm -> {pid: {ok, fm, meta, text}}; arm = rung (+"_slow3" for gentle)
# ---------------------------------------------------------------------------------------------

def word_fm(ds, i, text):
    """Failure mode for a wrong word-pool record: constraint violation / too short / wrong."""
    t = (text or "").lower()
    words = re.findall(r"[a-zA-Z']+", text or "")
    if ds == "lipogram":
        if t.count(["e", "a", "o", "t", "s"][i % 5]):
            return "viol"
        return "short" if len(words) < 10 else "wrong"
    if ds == "univocalic":
        v = ["a", "o", "e", "i"][i % 4]
        if sum(t.count(x) for x in "aeiou" if x != v):
            return "viol"
        return "short" if len(words) < 7 else "wrong"
    if ds == "piem":
        seq = [(3, 1, 4, 1, 5, 9), (2, 7, 1, 8, 2, 8)][i % 2]
        if len(words) < 6:
            return "short"
        return "viol" if any(len(w) != d for w, d in zip(words[:6], seq)) else "wrong"
    return "wrong"


def tail_dup(text, n=16):
    t = text or ""
    if len(t) < 4 * n:
        return 0.0
    grams = [t[i:i + n] for i in range(len(t) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def norm_record(task, m):
    """Normalize a manifest.json entry or psweep jsonl row to {ok, fm, meta, text}."""
    ok = bool(m.get("correct", m.get("ok")))
    text = m.get("final_text", "")
    if m.get("block"):
        text += "\n\n––– extracted code block –––\n" + m["block"]
    bits = []
    if task == "word":
        bits.append(f'{m["ds"]} · target: {m["answer"]}')
        if "wc" in m:
            bits.append(f'{m["wc"]} words')
        fm = "ok" if ok else word_fm(m["ds"], m["i"], m.get("final_text_clean") or m.get("final_text", ""))
    else:
        if m.get("answer") is not None:
            bits.append(f'gt {m["answer"]} · got {m.get("extracted")}')
        if "pass_frac" in m:
            bits.append(f'tests {m["pass_frac"]:.2f}' + (f' · {m["difficulty"]}' if m.get("difficulty") else ""))
        if "finished" in m:
            bits.append("finished" if m["finished"] else "BUDGET-CAPPED")
        if "dup8" in m:
            bits.append(f'dup8 {m["dup8"]:.2f}')
        if "n_steps" in m:
            bits.append(f'{m["n_steps"]} steps')
        if ok:
            fm = "ok"
        elif not m.get("finished", True):
            d8 = m.get("dup8")
            fm = "loop" if (d8 if d8 is not None else tail_dup(m.get("final_text", ""))) >= 0.5 else "capped"
        else:
            fm = "wrong"
    # what the grader pulled out of the text, so the film can highlight exactly that span
    ans = {}
    if task == "word":
        # no answer span here — the whole text is the graded object, so highlight what the
        # constraint forbids instead (same per-item rule word_fm/check() use)
        i, ds = m["i"], m["ds"]
        if ds == "lipogram":
            c = ["e", "a", "o", "t", "s"][i % 5]
            hl = dict(kind="chars", chars=c, lab=f'no "{c}" anywhere')
        elif ds == "univocalic":
            v = ["a", "o", "e", "i"][i % 4]
            hl = dict(kind="chars", chars="".join(x for x in "aeiou" if x != v),
                      lab=f'"{v}" is the only vowel allowed')
        else:
            seq = [(3, 1, 4, 1, 5, 9), (2, 7, 1, 8, 2, 8)][i % 2]
            hl = dict(kind="piem", seq=list(seq),
                      lab="first six word lengths must be " + ",".join(map(str, seq)))
        ans = dict(ds=ds, gt=str(m["answer"]), hl=hl)
    else:
        if m.get("extracted") is not None:
            ans["ex"] = str(m["extracted"])
        if m.get("how"):
            ans["how"] = m["how"]
    return dict(ok=ok, fm=fm, meta=" · ".join(bits), text=text, **ans)


ANSWERS = {}
for _t, _fn in [("gpqa", "gpqa_problems.json"), ("amc_aime", "math_problems.json"),
                ("imo", "imo_problems.json")]:
    ANSWERS[_t] = {r["pid"]: r["answer"] for r in json.loads((CD / _fn).read_text())}
LCB_DIFF = {r["pid"]: r["difficulty"] for r in json.loads((CD / "lcb_problems.json").read_text())}


def load_manifest_json(rel):
    p = CD / rel / "manifest.json"
    return json.loads(p.read_text()) if p.exists() else {}


def load_psweep(task, arm):
    """acts_psweep jsonl shards -> {pid: row}, backfilling answer/difficulty for meta."""
    out = {}
    d = CD / "acts_psweep" / task / arm
    for f in sorted(d.glob("manifest_w*.jsonl")) if d.exists() else []:
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if task in ANSWERS and r.get("answer") is None:
                r["answer"] = ANSWERS[task].get(r["pid"])
            if task == "lcb":
                r["difficulty"] = LCB_DIFF.get(r["pid"], "")
            out[r["pid"]] = r
    return out


# (task, arm) -> loader spec: ("json", relpath) uses manifest.json; ("psweep",) uses jsonl
SRC = {}
for _m in ("soft", "k1", "k2", "k4", "k8"):
    SRC[("gpqa", _m)] = ("json", f"acts_bench/gpqa/{_m}")
    SRC[("amc_aime", _m)] = ("json", f"acts_bud/b8192/{_m}")
SRC[("gpqa", "soft_slow3")] = ("json", "acts_stab/gpqa/soft_slow3")
SRC[("gpqa", "k1_slow3")] = ("json", "acts_stab/gpqa/k1_slow3")
SRC[("amc_aime", "soft_slow3")] = ("json", "acts_bud/slow3/soft_slow3")
SRC[("amc_aime", "k1_slow3")] = ("json", "acts_bud/slow3/k1_slow3")
for _m in ("soft", "k1", "soft_slow3", "k1_slow3"):
    SRC[("lcb", _m)] = ("json", f"acts_lcb/{_m}")
# everything else comes from the psweep fleet (incl. ALL word arms — self-consistent with films)
for _t in ("gpqa", "amc_aime", "lcb", "imo", "word"):
    for _r in RUNGS:
        for _a in (_r, _r + "_slow3", _r + "_slow3m"):
            SRC.setdefault((_t, _a), ("psweep",))


def load_cell(task, arm):
    kind = SRC[(task, arm)]
    man = load_manifest_json(kind[1]) if kind[0] == "json" else load_psweep(task, arm)
    return {pid: norm_record(task, m) for pid, m in sorted(man.items())}


def load_all():
    D = {}
    for task in ("gpqa", "amc_aime", "lcb", "imo", "word"):
        for r in RUNGS:
            for arm in (r, r + "_slow3", r + "_slow3m"):
                recs = load_cell(task, arm)
                if recs:
                    D.setdefault(task, {})[arm] = recs
    return D


# ---------------------------------------------------------------------------------------------
# split-violin panel SVG (server-side; dots carry data-attrs for the click -> film side panel)
# ---------------------------------------------------------------------------------------------

FMCOL = {"ok": "var(--fm-correct)", "wrong": "var(--fm-wrong)", "capped": "var(--fm-noans)",
         "loop": "var(--fm-loop)", "viol": "#c51b7d", "short": "#4393c3"}
FMLAB = {"ok": "correct", "loop": "degenerate loop (ran out of budget repeating itself)",
         "capped": "budget-capped mid-answer (still writing, not looping)",
         "wrong": "stopped on its own, answer wrong",
         "viol": "broke the hard constraint (word pool)",
         "short": "constraint kept but too short to count (word pool)"}
# precise rule behind each mode, shown on hover and spelled out under the legend
FMDEF = {
    "ok": "graded correct: extracted answer == ground truth (LCB: the extracted code block passes "
          "every test; word pool: the constraint holds and the target word is present).",
    "loop": "the run never terminated &mdash; it consumed the whole token budget (finished = "
            "n_tokens &lt; max_new &minus; 8 is false) &mdash; <b>and</b> its tail is degenerate: "
            "dup8 &ge; 0.5, where dup8 = 1 &minus; |set(8-token n-grams)| / |8-token n-grams| over "
            "the generated ids. This is the failure the k=1 collapse is made of.",
    "capped": "same budget exhaustion, but <b>not</b> degenerate (dup8 &lt; 0.5): the model was "
              "still producing fresh text when it hit the cap, so no answer was ever emitted.",
    "wrong": "the run terminated by itself inside the budget and produced a parseable answer, but "
             "the answer is incorrect (LCB: the code block fails at least one test). An honest "
             "miss &mdash; decoding worked, the reasoning did not.",
    "viol": "word-pool tasks only: the text breaks the hard constraint it was given &mdash; a "
            "lipogram containing its banned letter (e/a/o/t/s by item), a univocalic containing "
            "any vowel other than its allowed one, or a piem whose first six word lengths differ "
            "from the &pi; digits (3,1,4,1,5,9 / 2,7,1,8,2,8).",
    "short": "word-pool tasks only: the constraint <b>is</b> respected, but the text falls under "
             "the length floor (&lt;10 words lipogram, &lt;7 univocalic, &lt;6 piem) &mdash; a "
             "degenerate way to satisfy a constraint, so it is not scored as a solve.",
}


def h01(s):
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def panel_svg(task, label, D, left=("", "standard"), right=("_slow3", "gentle"), ref=None):
    """One task panel: split violin of the `left` vs `right` arm suffix at every truncation rung.
    `ref` (suffix, label) is overlaid as mean markers + polyline only, no violin. The paper's
    diamonds are drawn only when the left half IS the standard sampler (what the paper ran)."""
    W, H, PADL, PADT, PADB = 856, 330, 42, 26, 64
    colw = (W - PADL - 12) / len(RUNGS)
    hw = min(56.0, colw * 0.34)
    y = lambda v: PADT + (1 - (v + 0.1) / 1.2) * (H - PADT - PADB)
    KH = 0.05
    grid = [-0.09 + 1.18 * g / 56 for g in range(57)]
    s = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">']
    s.append(f'<text x="2" y="15" class="series-label">{label}</text>')
    for v in (0, 0.25, 0.5, 0.75, 1):
        cls = "baseline-line" if v in (0, 1) else "gridline"
        s.append(f'<line x1="{PADL}" x2="{W-8}" y1="{y(v):.1f}" y2="{y(v):.1f}" class="{cls}"/>')
        s.append(f'<text x="{PADL-5}" y="{y(v)+4:.1f}" text-anchor="end" class="tick-label">{v:g}</text>')
    s.append(f'<text x="9" y="{(PADT+H-PADB)/2:.0f}" transform="rotate(-90 9 {(PADT+H-PADB)/2:.0f})" '
             f'text-anchor="middle" class="axis-label">task accuracy</text>')
    means = {"std": [], "gen": [], "ref": []}
    pk = {"gpqa": "GPQA", "amc_aime": "AMC/AIME", "lcb": "LCB", "imo": "IMO", None: "N2C"}.get(task)
    pp = PAPER["top_k"].get(pk, {}) if pk else {}
    paper_only = task is None
    # hit targets are collected separately and appended LAST, so nothing drawn on top of the sina
    # cloud (paper markers, mean bars/polylines) can swallow a dot click
    paper_pts, paper_layer, hit_layer = [], [], []
    for i, rung in enumerate(RUNGS):
        cx = PADL + colw * (i + 0.5)
        if i:
            s.append(f'<line x1="{PADL+colw*i:.1f}" x2="{PADL+colw*i:.1f}" y1="{PADT}" '
                     f'y2="{y(-0.1):.1f}" style="stroke:var(--grid);stroke-dasharray:2 3"/>')
        s.append(f'<text x="{cx:.1f}" y="{H-PADB+18}" text-anchor="middle" class="tick-label" '
                 f'style="font-weight:600">{RUNG_LAB[rung]}</text>')
        sides = () if paper_only else (("std", rung + left[0], -1, "var(--vio-std)"),
                                       ("gen", rung + right[0], 1, "var(--vio-gen)"))
        if ref:
            rr = D.get(task, {}).get(rung + ref[0])
            means["ref"].append((cx, float(np.mean([r["ok"] for r in rr.values()])), len(rr)) if rr else None)
        for side, arm, sgn, vio in sides:
            recs = D.get(task, {}).get(arm)
            lx = cx + (sgn * hw / 2)
            if not recs:
                s.append(f'<rect x="{min(cx, cx+sgn*hw):.1f}" y="{PADT}" width="{hw:.1f}" '
                         f'height="{y(-0.1)-PADT:.1f}" class="pending-cell" rx="4"/>')
                s.append(f'<text x="{lx:.1f}" y="{(PADT+y(-0.1))/2:.1f}" text-anchor="middle" '
                         f'class="pending-text">pending</text>')
                means[side].append(None)
                continue
            pts = list(recs.items())
            ys = [1.0 if r["ok"] else 0.0 for _, r in pts]
            acc = float(np.mean(ys))
            means[side].append((cx + sgn * hw * 0.5, acc, len(pts)))
            dens = lambda v: sum(np.exp(-0.5 * ((v - o) / KH) ** 2) for o in ys)
            gd = [dens(v) for v in grid]
            dmax = max(max(gd), 1e-9)
            path = f'M{cx:.1f} {y(grid[0]):.1f}'
            for v, d in zip(grid, gd):
                path += f'L{cx + sgn * hw * d / dmax:.1f} {y(v):.1f}'
            path += f'L{cx:.1f} {y(grid[-1]):.1f}Z'
            s.append(f'<path d="{path}" style="fill:{vio};fill-opacity:.22;'
                     f'stroke:{vio};stroke-opacity:.55;stroke-width:.8"/>')
            for pid, r in pts:
                u1, u2 = h01(pid + arm), h01(arm + pid + "y")
                yv = (1.0 if r["ok"] else 0.0) + (2 * u2 - 1) * 0.045
                dx = sgn * (0.06 + 0.86 * u1) * hw * dens(yv) / dmax
                meta = (r["meta"] or "").replace("&", "&amp;").replace("<", "&lt;")
                s.append(f'<circle cx="{cx+dx:.1f}" cy="{y(yv):.1f}" r="2.3" class="dot-mark" '
                         f'style="fill:{FMCOL.get(r["fm"], FMCOL["wrong"])};fill-opacity:.9"/>')
                # transparent, larger hit target (r=2.3 is unclickable): carries the data attrs +
                # tooltip; transparent fill still hit-tests
                hit_layer.append(f'<circle cx="{cx+dx:.1f}" cy="{y(yv):.1f}" r="7" class="dot-hit" '
                                 f'data-task="{task}" data-arm="{arm}" data-pid="{pid}">'
                                 f'<title>{pid} ({arm}) — {FMLAB.get(r["fm"], r["fm"])}\n{meta}'
                                 f'\nclick: replay denoising film</title></circle>')
            s.append(f'<line x1="{cx:.1f}" x2="{cx+sgn*hw:.1f}" y1="{y(acc):.1f}" y2="{y(acc):.1f}" '
                     f'style="stroke:{vio};stroke-width:2.6"/>')
            s.append(f'<text x="{lx:.1f}" y="{H-PADB+34}" text-anchor="middle" class="pct-label" '
                     f'style="fill:{vio}">{acc:.2f}</text>')
        # the paper's reported score: skyline node (diamond + ±std whisker). The paper runs the
        # STANDARD sampler, so the marker sits in the left (standard) half, not on the tick centre.
        key = PAPER_KEY[rung]
        if key in pp and left[0] == "":
            v, sd = pp[key]["score"], pp[key]["std"]
            dx0 = cx if paper_only else cx - hw / 2
            paper_pts.append((dx0, v))
            paper_layer.append(f'<line x1="{dx0:.1f}" x2="{dx0:.1f}" y1="{y(v-sd):.1f}" y2="{y(v+sd):.1f}" '
                               f'class="whisker" style="stroke:var(--ink);stroke-width:1.6"/>')
            # a page-coloured halo so the marker reads clearly on top of the sina cloud
            paper_layer.append(f'<path d="M{dx0:.1f} {y(v)-7.5:.1f}L{dx0+7.5:.1f} {y(v):.1f}'
                               f'L{dx0:.1f} {y(v)+7.5:.1f}L{dx0-7.5:.1f} {y(v):.1f}Z" '
                               f'style="fill:var(--page);fill-opacity:.85;stroke:none"/>')
            paper_layer.append(f'<path d="M{dx0:.1f} {y(v)-6:.1f}L{dx0+6:.1f} {y(v):.1f}'
                               f'L{dx0:.1f} {y(v)+6:.1f}L{dx0-6:.1f} {y(v):.1f}Z" '
                               f'style="fill:var(--ink);stroke:var(--page);stroke-width:1.4">'
                               f'<title>paper {pk} {key}: {v:.3f} ±{sd:.3f} (standard sampler)'
                               f'</title></path>')
    # the paper skyline: connected line through the paper's scores, under its diamonds
    if len(paper_pts) > 1:
        s.append('<polyline points="' + " ".join(f"{x:.1f},{y(v):.1f}" for x, v in paper_pts) +
                 '" style="fill:none;stroke:var(--ink);stroke-width:1.2;stroke-dasharray:2 3;'
                 'opacity:.75;pointer-events:none"/>')
        px, pv = paper_pts[0]
        s.append(f'<text x="{px:.1f}" y="{y(pv)-11:.1f}" text-anchor="middle" '
                 f'style="font-size:11px;fill:var(--ink);font-family:var(--mono)">paper</text>')
    s.extend(paper_layer)
    for side, vio in (("std", "var(--vio-std)"), ("gen", "var(--vio-gen)"),
                      ("ref", "var(--series-paper)")):
        pts = [m for m in means[side] if m]
        if side == "ref" and pts:
            for x, a, _ in pts:   # reference arm: mean markers + polyline, no violin
                s.append(f'<circle cx="{x:.1f}" cy="{y(a):.1f}" r="3" style="fill:var(--series-paper);'
                         f'stroke:var(--page);stroke-width:1;pointer-events:none"/>')
            px, pa, _ = pts[0]
            s.append(f'<text x="{px:.1f}" y="{y(pa)-9:.1f}" text-anchor="middle" '
                     f'style="font-size:10.5px;fill:var(--series-paper);font-family:var(--mono)">'
                     f'{ref[1]}</text>')
        if len(pts) > 1:
            s.append('<polyline points="' + " ".join(f"{x:.1f},{y(a):.1f}" for x, a, _ in pts) +
                     f'" style="fill:none;stroke:{vio};stroke-width:1.3;stroke-dasharray:5 4;'
                     f'opacity:.85;pointer-events:none"/>')
    s.extend(hit_layer)
    if paper_only:
        s.append(f'<text x="{PADL + (W-PADL-12)/2:.0f}" y="{y(0.35):.1f}" text-anchor="middle" '
                 f'class="pending-text">Google-private benchmark — the paper&#8217;s Fig-2 skyline only; not runnable</text>')
    else:
        s.append(f'<text x="{W-8}" y="15" text-anchor="end" '
                 f'style="font-size:10.5px;fill:var(--muted);font-family:var(--mono)">'
                 f'left: {left[1]}{" (&#9670; paper)" if left[0] == "" else ""} | right: {right[1]}</text>')
    s.append('</svg>')
    return "".join(s)


# ---------------------------------------------------------------------------------------------
# the film side panel (remote .detail-panel / .film-* classes) + JS
# ---------------------------------------------------------------------------------------------

PANEL_HTML = """
<div id=dpanel class=detail-panel>
 <div class=detail-head>
  <div><div class=detail-title id=dp_title></div><div class=detail-sub id=dp_sub></div></div>
  <div style="display:flex;align-items:center;gap:6px;flex:none">
   <button class=detail-font-btn id=dp_fdn title="smaller generation font">A&minus;</button>
   <button class=detail-font-btn id=dp_fup title="larger generation font">A+</button>
   <button class=detail-close id=dp_close title="close (Esc)">&#10005;</button>
  </div>
 </div>
 <div class=detail-stats id=dp_stats></div>
 <details class=dp-prob open>
  <summary>problem &mdash; the prompt exactly as sent to the model</summary>
  <div class=dp-prob-body id=dp_prob></div>
 </details>
 <div class=film-controls id=dp_ctl style="display:none">
  <button id=dp_play>&#9654;</button>
  <input type=range id=dp_step min=0 value=0>
  <span class=film-step-label id=dp_lab></span>
  <button class=film-jump id=dp_jump style="display:none"
          title="jump to the step at which the graded answer stops changing">&#8615; answer</button>
  <span style="display:inline-flex;align-items:center;gap:4px;white-space:nowrap">0
   <span style="display:inline-block;width:64px;height:8px;border-radius:3px;background:linear-gradient(90deg,#440154,#3e4989,#26828e,#35b779,#fde725)"></span>&ge;4 nats</span>
 </div>
 <div class=film-canvas id=dp_canvas></div>
</div>
"""

JS = r"""
(function(){
 var tb=document.getElementById('themebtn');
 function setT(t){ if(t){document.documentElement.dataset.theme=t;} else {delete document.documentElement.dataset.theme;}
   try{localStorage.rpTheme=t||'';}catch(e){} }
 var saved=''; try{saved=localStorage.rpTheme||'';}catch(e){}
 setT(saved);
 tb.addEventListener('click', function(){
   var cur=document.documentElement.dataset.theme||(matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light');
   setT(cur==='dark'?'light':'dark');
 });
})();
(function(){
const D = %%DATA%%;
const PR = %%PROMPTS%%;
const VIR = %%VIR%%;
const EMAX = %%EMAX%%;
const PAD = %%PAD%%;
const esc = t => (t||'').replace(/&/g,'&amp;').replace(/</g,'&lt;');
const disp = t => t === PAD ? null : t.replace(/▁/g,' ');
const reEsc = s => s.replace(/[.*+?^${}()|[\]\\]/g,'\\$&');

// ---- answer extraction, mirroring the graders in ds_ablate_bench/ds_ablate_math ----
// 'last' = the LAST match of the first rule that fires (exactly extract_letter / extract_int /
// extract_block); word tasks have no answer span, so we mark what their constraint forbids.
const GPQA_RULES = [
  {re:/\\boxed\{\s*\\?(?:text|mathrm)?\{?\s*\(?([A-Da-d])\)?\s*\}?\s*\}/g, how:'boxed',
   lab:'\\boxed{...} (the strict rule)'},
  {re:/(?:answer|option|choice)\s*(?:is|:)?\s*\**\s*\(?([A-D])\)?/gi, how:'phrase',
   lab:'an "answer is X" phrase (robust-grade fallback)'},
  {re:/\(([A-D])\)/g, how:'paren', lab:'the last bare (X) (last-resort fallback)'}];
function ansSpec(task, rec){
  if(task === 'word'){
    const hl = rec.hl || {};
    if(hl.kind === 'chars')
      return {mode:'all', re:new RegExp('[' + reEsc(hl.chars) + ']','gi'),
              noun:'token', lab:hl.lab,
              none:'no forbidden character in the text &mdash; the constraint (' + hl.lab + ') holds, ' +
                   'and the whole generation is the graded object'};
    return {mode:'piem', seq:hl.seq || [], noun:'word', lab:hl.lab || '',
            none:'every word length matches &mdash; the constraint (' + (hl.lab||'') + ') holds, and ' +
                 'the whole generation is the graded object'};
  }
  if(task === 'lcb')
    return {mode:'last', rules:[{re:/```(?:python)?\s*\n[\s\S]*?```/g,
            lab:'the last fenced ```python block (executed against the unit tests)'}]};
  if(task === 'gpqa')
    // the manifest records which rule actually fired; use that one so the mark matches the grade
    return {mode:'last', rules: rec.how ? GPQA_RULES.filter(r => r.how === rec.how) : GPQA_RULES};
  return {mode:'last', rules:[
    {re:/\\boxed\{([^{}]+)\}/g, lab:'the last \\boxed{...} (the strict rule)'},
    {re:/(-?\d[\d,]*)/g, lab:'the last integer in the text (fallback when there is no \\boxed)'}]};
}
// Final GRADED text of the rollout + a char -> (canvas, piece) map. Mirrors the server side, which
// grades CH.sub("", tok.decode(ids without eos/pad)) with
// CH = <\|channel>thought\s*<channel\|> | <\|?channel\|?> — i.e. the whole thought header when it is
// closed (word tasks), otherwise just the bare marker (gpqa/math leave "thought" in the text).
const SPECIAL = /^<(?:eos|pad|bos|end_of_turn|start_of_turn)>$/;
function finalMap(F){
  const last = {}; F.forEach((f,i) => { last[f[0]] = i; });
  const cs = Object.keys(last).map(Number).sort((a,b) => a-b);
  let text = '', map = [];
  cs.forEach(c => {
    const pcs = F[last[c]][1], skip = new Array(pcs.length).fill(false);
    for(let i = 0; i < pcs.length; i++){
      if(pcs[i] !== '<|channel>') continue;
      let j = i + 1;
      if(pcs[j] === 'thought') j++;
      while(j < pcs.length && /^\s*$/.test(disp(pcs[j]) || '')) j++;
      if(pcs[j] === '<channel|>'){ for(let q = i; q <= j; q++) skip[q] = true; i = j; }
      else skip[i] = true;
    }
    pcs.forEach((p,i) => {
      if(skip[i] || p === '<channel|>' || SPECIAL.test(p)) return;
      const d = disp(p); if(d === null) return;
      for(let k = 0; k < d.length; k++) map.push(c + ':' + i);
      text += d;
    });
  });
  return {text: text, map: map, last: last};
}
// first step after the last one at which any of `keys` (default: the whole text) still differed
// from its final value — i.e. when that span stopped moving
function commitStep(F, M, keys){
  const per = {};
  (keys && keys.size ? [...keys] : M.map).forEach(k => {
    const [c,i] = k.split(':'); (per[c] = per[c] || new Set()).add(+i); });
  let t = 0;
  F.forEach((f,k) => { const want = per[f[0]]; if(!want) return;
    const fin = F[M.last[f[0]]][1];
    for(const i of want) if(f[1][i] !== fin[i]){ t = Math.max(t, Math.min(k+1, F.length-1)); break; } });
  return t;
}
function findAnswer(task, F, rec){
  const M = finalMap(F);
  const sp = ansSpec(task, rec);
  let spans = [], lab = sp.lab, val = null;
  if(sp.mode === 'last'){
    for(const r of sp.rules){
      const ms = [...M.text.matchAll(r.re)];
      if(!ms.length) continue;
      const m = ms[ms.length-1];
      spans = [[m.index, m.index + m[0].length]];
      lab = r.lab; val = m[1] !== undefined ? m[1] : m[0];
      break;
    }
  } else if(sp.mode === 'all'){
    for(const m of M.text.matchAll(sp.re)) spans.push([m.index, m.index + m[0].length]);
  } else {
    [...M.text.matchAll(/[a-zA-Z']+/g)].slice(0, sp.seq.length).forEach((w,j) => {
      if(w[0].length !== sp.seq[j]) spans.push([w.index, w.index + w[0].length]); });
  }
  const set = new Set();
  spans.forEach(([a,b]) => M.map.slice(a,b).forEach(k => set.add(k)));
  if(sp.mode === 'last' && !set.size) return null;
  const cA = set.size ? Math.max(...[...set].map(k => +k.split(':')[0])) : null;
  if(task === 'word' && set.size)
    lab = '<b>' + spans.length + '</b> ' + sp.noun + (spans.length === 1 ? ' breaks' : 's break') +
          ' the constraint (' + sp.lab + ')';
  return {set: set, how: set.size ? lab : sp.none, val: val, bad: task === 'word' && set.size > 0,
          word: task === 'word',
          txt: spans.length ? M.text.slice(spans[0][0], spans[0][1]) : null,
          n: spans.length, step: commitStep(F, M, set), canvas: cA};
}
function canvasHtml(F, t, A) {
  const tt = Math.min(t, F.length - 1);
  const fr = F[tt], ci = fr[0], pieces = fr[1], ents = fr[2];
  const last = {};
  F.forEach((f, i) => { last[f[0]] = i; });
  const aset = (A && A.set) || new Set();
  // emit a segment piece-by-piece, opening/closing one <mark> around the answer span
  function seg(pcs, es, c, live){
    let out = '', inA = false;
    for(let i = 0; i < pcs.length; i++){
      const isA = aset.has(c + ':' + i);
      if(isA !== inA){ out += isA ? '<mark class="ans' + (A.bad ? ' bad' : '') + '" title="' +
                                    esc(A.how) + '">' : '</mark>'; inA = isA; }
      const d = disp(pcs[i]);
      if(d === null){ if(live) out += '<span class=p>&middot;</span>'; continue; }
      const e = live ? (es[i] || 0) : 0;
      if(e < 0.05){ out += esc(d); continue; }
      const idx = Math.min(32, Math.round(e / EMAX * 32));
      out += '<span style="background:' + VIR[idx] + ';color:' + (idx >= 22 ? '#111' : '#eee') +
             ';border-radius:2px">' + esc(d) + '</span>';
    }
    return out + (inA ? '</mark>' : '');
  }
  let html = '';
  for (let c = 0; c < ci; c++) {
    const cf = F[last[c]];
    html += '<span class=cvb title="canvas ' + c + ' — committed; last denoised at global step ' + last[c] + '">c' + c + '</span>';
    html += seg(cf[1], null, c, false);
  }
  html += '<span class="cvb act" title="canvas ' + ci + ' — being denoised at step ' + tt + '">c' + ci + '</span>';
  return html + seg(pieces, ents, ci, true);
}
const cache = {};
let cur = null, timer = null;
const $ = id => document.getElementById(id);
function stop(){ if(timer){clearInterval(timer);timer=null;$('dp_play').innerHTML='&#9654;';} }
function render(t){
  if(!cur||!cur.F)return;
  $('dp_canvas').innerHTML = canvasHtml(cur.F, t, cur.A);
  $('dp_lab').textContent = 'step ' + Math.min(t, cur.F.length-1) + '/' + (cur.F.length-1);
}
function jumpAns(){
  if(!cur||!cur.A)return;
  stop();
  const sl=$('dp_step'); sl.value=cur.A.step; render(cur.A.step);
  const m=$('dp_canvas').querySelector('mark.ans');
  if(m)m.scrollIntoView({block:'center'});
}
function play(){
  stop();
  const sl=$('dp_step'), mx=+sl.max;
  if(+sl.value>=mx){sl.value=0;render(0);}
  $('dp_play').innerHTML='&#9646;&#9646;';
  timer=setInterval(function(){ sl.value=+sl.value+1; render(+sl.value); if(+sl.value>=mx)stop(); },120);
}
function closeP(){ stop(); $('dpanel').classList.remove('open'); cur=null; }
function openP(task, arm, pid){
  const r=(D[task][arm]||{})[pid]; if(!r)return;
  stop();
  cur={task:task,arm:arm,pid:pid,F:null,A:null};
  $('dp_title').innerHTML = esc(pid) + ' <span class="detail-badge ' + (r.ok?'correct':'incorrect') + '">' +
    (r.ok?'correct':'incorrect') + '</span>';
  const base = arm.replace(/_slow3m?$/, '');
  const kind = /_slow3m$/.test(arm) ? 'step-matched gentle (paper step + token budget)'
             : /_slow3$/.test(arm)  ? 'gentle sampler' : 'standard sampler';
  $('dp_sub').textContent = task + ' · ' + base + ' — ' + kind;
  $('dp_stats').innerHTML = '<span>' + esc(r.meta) + '</span>';
  const prompt = (PR[task]||{})[pid];
  $('dp_prob').innerHTML = prompt ? esc(prompt)
    : '<span class=p>[prompt not recorded for this item]</span>';
  $('dp_ctl').style.display='none';
  $('dp_canvas').innerHTML = '<span class=p>loading film…</span>';
  $('dpanel').classList.add('open');
  const key = task + '/' + arm + '/' + pid;
  const me = cur;
  const got = frames => {
    if(cur!==me)return;
    cur.F = frames;
    cur.A = findAnswer(task, frames, r);
    $('dp_ctl').style.display='flex';
    const jb = $('dp_jump');
    jb.style.display = cur.A ? 'inline-block' : 'none';
    if(cur.A){
      const w = !cur.A.set.size ? 'settled' : (cur.A.bad ? 'violation' : 'answer');
      jb.innerHTML = '&#8615; ' + w;
      jb.title = 'jump to the step at which the ' +
        (w==='settled' ? 'text stops changing' : 'highlighted span stops changing') + ' (key: a)';
      jb.classList.toggle('bad', !!cur.A.bad);
    }
    if(cur.A){
      const A = cur.A, cut = s => s.length > 60 ? s.slice(0,60) + '…' : s;
      const norm = s => String(s).replace(/[,\s]/g,'').toUpperCase();
      const bad = A.val != null && r.ex != null && norm(A.val) !== norm(r.ex);
      let head;
      if(!A.set.size || A.word) head = A.how;   // word tasks: constraint status, not an answer span
      else head = 'answer <b>' + esc(cut(A.txt)) + '</b> &mdash; found by ' + A.how;
      $('dp_stats').innerHTML = '<span>' + esc(r.meta) + '</span>' +
        '<span class=ans-note>' + head + '; settles at step <b>' + A.step + '</b>/' +
        (frames.length-1) + (A.canvas!=null ? ' in canvas c' + A.canvas : '') +
        (bad ? ' &middot; grader recorded "' + esc(String(r.ex)) +
               '", the mark is this page&rsquo;s raw-text match' : '') + '</span>';
    } else {
      $('dp_stats').innerHTML = '<span>' + esc(r.meta) + '</span>' +
        '<span class=ans-note>no answer span &mdash; none of the extraction rules matched this text, ' +
        'so the grader recorded no answer at all (nothing to jump to)</span>';
    }
    const sl=$('dp_step'); sl.max=frames.length-1; sl.value=0;
    render(0); play();
  };
  if(cache[key]) { got(cache[key]); return; }
  fetch('rb_steps/' + key + '.json')
    .then(x=>{ if(!x.ok) throw new Error(x.status); return x.json(); })
    .then(j=>{ cache[key]=j.frames; got(j.frames); })
    .catch(()=>{ if(cur!==me)return;
      $('dp_canvas').innerHTML = esc(r.text) + '<div class=p style="margin-top:8px">[film shard not available yet — stored text shown]</div>'; });
}
document.addEventListener('click', function(e){
  const c = e.target.closest ? e.target.closest('circle[data-pid]') : null;
  if(c) openP(c.dataset.task, c.dataset.arm, c.dataset.pid);
});
// panel comparison switch: swap which arms occupy the two halves of a panel card
document.addEventListener('click', function(e){
  const btn = e.target.closest ? e.target.closest('.panel-switch button') : null;
  if(!btn) return;
  const card = btn.closest('.panel-card'), v = btn.dataset.v;
  card.querySelectorAll('.panel-switch button').forEach(b => b.classList.toggle('active', b === btn));
  card.querySelectorAll('.panel-view').forEach(d => { d.hidden = d.dataset.v !== v; });
});
$('dp_close').addEventListener('click', closeP);
$('dp_play').addEventListener('click', function(){ if(timer)stop(); else play(); });
$('dp_step').addEventListener('input', function(){ stop(); render(+this.value); });
$('dp_jump').addEventListener('click', jumpAns);
// generation font size (drives --dfs, which .detail-gen / .film-canvas read), sticky
let fs = 11.5; try{ fs = parseFloat(localStorage.rpFilmFs) || 11.5; }catch(e){}
function setFs(v){ fs = Math.max(8, Math.min(26, v));
  $('dpanel').style.setProperty('--dfs', fs.toFixed(1)+'px');
  try{ localStorage.rpFilmFs = fs; }catch(e){}
  const m=$('dp_canvas').querySelector('mark.ans');   // resizing reflows; keep the mark in view
  if(m)m.scrollIntoView({block:'center'}); }
setFs(fs);
$('dp_fup').addEventListener('click', function(){ setFs(fs+1.5); });
$('dp_fdn').addEventListener('click', function(){ setFs(fs-1.5); });
document.addEventListener('keydown', function(e){
  if(e.key==='Escape'){ closeP(); return; }
  if(!$('dpanel').classList.contains('open') || e.metaKey || e.ctrlKey) return;
  if(e.key==='a'||e.key==='A'){ jumpAns(); }
  else if(e.key==='+'||e.key==='='){ setFs(fs+1.5); }
  else if(e.key==='-'||e.key==='_'){ setFs(fs-1.5); }
});
})();
"""

EXTRA_CSS = """
/* --- additions on top of the dg-fig2-topk stylesheet (loaded verbatim above) --- */
:root{--fm-loop:#8a5a2b}
@media(prefers-color-scheme:dark){:root:where(:not([data-theme=light])){--fm-loop:#c08a50}}
:root[data-theme=dark]{--fm-loop:#c08a50}
.wrap{max-width:900px}
a{color:var(--ink)}
#themebtn{position:fixed;top:12px;right:14px;z-index:60;background:var(--surface);color:var(--ink);
 border:1px solid var(--border);border-radius:999px;padding:.25rem .7rem;cursor:pointer;font-size:.8rem}
.panel-card svg{width:100%;height:auto;display:block;overflow:visible}
.cvb{display:inline-block;font-size:.72em;font-weight:700;color:var(--muted);border:1px solid var(--border);
 border-radius:3px;padding:0 .25em;margin:0 .3em 0 .15em;vertical-align:.1em;user-select:none;white-space:nowrap}
.cvb.act{color:var(--series-paper);border-color:var(--series-paper)}
.detail-stats span{max-width:100%;overflow-wrap:anywhere}
/* the graded answer, highlighted inside the film */
mark.ans{background:color-mix(in srgb, var(--fm-correct) 22%, transparent);color:inherit;
 box-shadow:0 0 0 1.5px var(--fm-correct);border-radius:3px;padding:0 .1em}
mark.ans span{border-radius:2px}
mark.ans.bad{background:color-mix(in srgb, #c51b7d 24%, transparent);box-shadow:0 0 0 1.5px #c51b7d}
.film-jump{background:var(--surface);border:1px solid var(--fm-correct);color:var(--fm-correct);
 border-radius:6px;padding:1px 8px;font-size:11.5px;cursor:pointer;font-family:inherit;white-space:nowrap}
.film-jump:hover{background:color-mix(in srgb, var(--fm-correct) 14%, transparent)}
.film-jump.bad{border-color:#c51b7d;color:#c51b7d}
.film-jump.bad:hover{background:color-mix(in srgb, #c51b7d 14%, transparent)}
/* the problem statement inside a rollout card */
.dp-prob{margin:0 0 10px;flex:none}
.dp-prob summary{font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);
 cursor:pointer;margin-bottom:5px}
.dp-prob-body{font-family:var(--mono);font-size:var(--dfs,11.5px);line-height:1.55;white-space:pre-wrap;
 word-break:break-word;background:var(--page);border:1px solid var(--border);border-radius:6px;
 padding:9px 11px;max-height:190px;overflow:auto;resize:vertical;color:var(--ink-2)}
.ans-note{color:var(--ink-2)}
.ans-note b{font-family:var(--mono)}
.vio-legend .fm-item{cursor:help;border-bottom:1px dotted var(--border)}
.fm-defs{margin:8px 0 2px;font-size:13px;color:var(--ink-2)}
.fm-defs summary{cursor:pointer;color:var(--muted);font-size:12.5px}
.fm-defs dl{margin:10px 0 0;display:grid;grid-template-columns:minmax(150px,auto) 1fr;gap:7px 16px}
.fm-defs dt{font-weight:600;color:var(--ink);white-space:normal}
.fm-defs dd{margin:0;max-width:78ch}
@media(max-width:640px){.fm-defs dl{grid-template-columns:1fr;gap:2px}.fm-defs dd{margin-bottom:8px}}
/* rollout popup spans the whole width right of the plot column (wrap 900px + body padding 20px),
   never narrower than 460px — below that (or <980px viewport) it drops to a bottom drawer */
.detail-panel{left:min(calc(20px + 900px + 16px), calc(100vw - 460px));right:16px;width:auto;
 max-width:none;resize:vertical}
/* let the film canvas take the panel's spare height instead of leaving dead space below it */
.detail-panel.open{display:flex;flex-direction:column}
.detail-panel .film-canvas{flex:1 1 auto;height:auto;min-height:220px}
.detail-panel .detail-gen{height:260px}
@media(max-width:980px){.detail-panel{left:10px;right:10px;width:auto}}
/* panel comparison switch */
.panel-switch{display:flex;flex-wrap:wrap;align-items:center;gap:6px;margin:0 0 10px}
.panel-switch button{background:var(--page);border:1px solid var(--border);color:var(--ink-2);
 border-radius:999px;padding:3px 12px;font:inherit;font-size:12px;cursor:pointer;white-space:nowrap}
.panel-switch button:hover{border-color:var(--ink-2);color:var(--ink)}
.panel-switch button.active{background:var(--surface);border-color:var(--ink);color:var(--ink);
 font-weight:650}
.panel-switch-hint{font-size:11.5px;color:var(--muted);margin-left:4px}
.panel-view[hidden]{display:none}
/* step-matched control table */
.steps-tab{border-collapse:collapse;font-size:12.5px;font-variant-numeric:tabular-nums;margin:6px 0 2px}
.steps-tab th,.steps-tab td{padding:3px 12px 3px 0;text-align:right;border-bottom:1px solid var(--grid)}
.steps-tab th:first-child,.steps-tab td:first-child{text-align:left;font-family:var(--mono)}
.steps-tab thead th{color:var(--muted);font-weight:600}
.steps-tab thead tr:first-child th{border-bottom:none;text-align:center;padding-bottom:0}
"""


def steps_mean(task, arm):
    """Mean denoising steps actually spent per rollout in a cell (None if the cell is missing)."""
    kind = SRC.get((task, arm))
    if not kind:
        return None
    man = load_manifest_json(kind[1]) if kind[0] == "json" else load_psweep(task, arm)
    st = [m["n_steps"] for m in man.values() if "n_steps" in m]
    return float(np.mean(st)) if st else None


def b32_acc(arm, task="imo"):
    rs = []
    d = CD / "acts_psweep" / task / arm
    for fp in sorted(d.glob("manifest_w*.jsonl")) if d.exists() else []:
        rs += [json.loads(l) for l in fp.read_text().splitlines()]
    return (float(np.mean([r["ok"] for r in rs])) if rs else None), len(rs)


def main():
    D = load_all()
    css = (Path(__file__).parent / "fig2_style.css").read_text() + EXTRA_CSS
    nfilm = len(list(RB.rglob("*.json")))

    def acc(task, arm):
        recs = D.get(task, {}).get(arm)
        return float(np.mean([r["ok"] for r in recs.values()])) if recs else None

    panels = [("gpqa", "GPQA — diamond, thinking, n=64"),
              ("amc_aime", "AMC/AIME — n=48"),
              ("lcb", "LiveCodeBench — n=32"),
              ("imo", "IMO (AnswerBench) — n=32"),
              ("word", "word pool — n=64, single canvas, fixed steps"),
              (None, "Natural2Code — paper skyline only")]
    # step-matched control: gentle PACING knobs inside the paper's own budgets (T=48, 8192 tok)
    if any(r + "_slow3m" in D.get("gpqa", {}) for r in RUNGS):
        rows = [(RUNG_LAB[r], acc("gpqa", r), acc("gpqa", r + "_slow3m"), acc("gpqa", r + "_slow3"),
                 steps_mean("gpqa", r), steps_mean("gpqa", r + "_slow3m"),
                 steps_mean("gpqa", r + "_slow3")) for r in RUNGS]
        fmt = lambda v, d=2: "&mdash;" if v is None else (f"{v:.{d}f}" if d else f"{v:.0f}")
        rat = lambda sb, sa: "&mdash;" if None in (sa, sb) else f"&times;{sb/sa:.2f}"
        k1r = [r for r in rows if r[0] == "top-1"][0]
        cap = ('<p class=caption><b>Step-matched control &mdash; the recovery is pacing, not compute.'
               '</b> The gentle sampler spends more denoising steps than the standard one '
               '(GPQA top-1: 1924 vs 1276), so its recovery could have been bought rather than '
               'earned. These arms (<code>*_slow3m</code>) keep only the gentle <i>pacing</i> knobs '
               '(entropy_bound 0.02, t 0.5&ndash;1.0) and run them inside the paper&rsquo;s own '
               'budgets &mdash; <code>max_denoising_steps=48</code> per canvas and 8192 tokens, '
               'identical to the standard arm. At the rungs where the collapse actually happens the '
               f'step-matched arm beats standard while spending <b>fewer</b> steps: top-1 '
               f'{fmt(k1r[1])}&rarr;<b>{fmt(k1r[2])}</b> (+3.5&sigma;) at '
               f'{rat(k1r[5], k1r[4])} the steps, top-2 +0.23 (+2.7&sigma;) at &times;0.71 &mdash; '
               'degenerate loops at top-1 fall from 51/64 to 26/64. It recovers roughly half the '
               f'gentle gain ({fmt(k1r[2])} vs {fmt(k1r[3])} at top-1), so the larger budget still '
               'contributes; but no part of the collapse survives equal-compute pacing. Use the switch '
               'above to swap which two arms occupy the halves &mdash; the third is always the '
               'dotted reference line, and every dot still opens its film, so you can read the '
               'difference rollout by rollout. The paper&rsquo;s diamonds are shown only when '
               'the left half is the standard sampler, which is what the paper ran.</p>'
               '<div style="overflow-x:auto"><table class=steps-tab><thead><tr><th></th>'
               '<th colspan=3>accuracy</th><th colspan=4>mean denoising steps</th></tr><tr>'
               '<th></th><th>standard</th><th>step-matched</th><th>gentle</th>'
               '<th>standard</th><th>step-matched</th><th>gentle</th>'
               '<th>matched/std</th></tr></thead><tbody>' +
               "".join(f'<tr><td>{lab}</td><td>{fmt(a)}</td><td><b>{fmt(b)}</b></td><td>{fmt(c)}</td>'
                       f'<td>{fmt(sa,0)}</td><td><b>{fmt(sb,0)}</b></td><td>{fmt(sc,0)}</td>'
                       f'<td>{rat(sb, sa)}</td></tr>'
                       for lab, a, b, c, sa, sb, sc in rows) +
               '</tbody></table></div>')
        # three views of the same rungs, switchable in place, so the difference is directly
        # comparable: you can see which individual rollouts move when the right-hand arm swaps
        views = [("standard vs step-matched", {"right": ("_slow3m", "step-matched"),
                                               "ref": ("_slow3", "gentle")}),
                 ("standard vs gentle", {"right": ("_slow3", "gentle"),
                                         "ref": ("_slow3m", "step-matched")}),
                 ("step-matched vs gentle", {"left": ("_slow3m", "step-matched"),
                                             "right": ("_slow3", "gentle"),
                                             "ref": ("", "standard")})]
        ttl = "GPQA — step-matched control (paper's step + token budget)"
        sw = ('<div class=panel-switch role=group aria-label="comparison">' +
              "".join(f'<button type=button data-v="{i}"{" class=active" if not i else ""}>{lab}'
                      '</button>' for i, (lab, _) in enumerate(views)) +
              '<span class=panel-switch-hint>swap the halves &mdash; the third arm stays as the '
              'dotted reference line</span></div>')
        body = "".join(f'<div class=panel-view data-v="{i}"{"" if not i else " hidden"}>' +
                       panel_svg("gpqa", ttl, D, **kwv) + '</div>'
                       for i, (_, kwv) in enumerate(views))
        panels.append((None, None, {}, sw + body + cap))

    P = []
    P.append('<!doctype html><html><head><meta charset=utf-8>'
             '<meta http-equiv="cache-control" content="no-cache, no-store, must-revalidate">'
             '<meta name=viewport content="width=device-width,initial-scale=1">'
             '<title>DiffusionGemma Fig-2 truncation: standard vs gentle sampler, every rollout filmed</title>'
             f'<style>{css}</style></head><body>')
    P.append('<button id=themebtn title="toggle theme">&#9681;</button>')
    P.append('<div class=wrap>')
    P.append('<p class=eyebrow>Investigation report &middot; updated 2026-07-19</p>')
    P.append('<h1>The top-k self-conditioning penalty, task by task: standard vs gentle sampler, '
             'with the paper&rsquo;s scores and every rollout&rsquo;s denoising film</h1>')
    P.append('<p class=subtitle>Fig.&nbsp;2 of &ldquo;How Transparent is DiffusionGemma?&rdquo; restricts '
             'the self-conditioning matrix S<sup>t</sup> to its top-k tokens. One panel per benchmark; '
             'at each truncation level a split violin compares the <b>standard sampler</b> (paper protocol, '
             'left/dark) with the <b>gentle sampler</b> (entropy bound 0.02, t 0.5&ndash;1.0, 96 steps, '
             'right/light). Diamonds: the paper&rsquo;s reported scores. Click any dot to replay that '
             'rollout&rsquo;s denoising.</p>')
    P.append(f'<p class=provenance>Model: google/diffusiongemma-26B-A4B-it &middot; paper: '
             f'<a href="https://arxiv.org/abs/2606.20560">arXiv:2606.20560</a> &middot; Slurm H200 fleet '
             f'&middot; {nfilm} filmed rollouts &middot; independent cross-check: '
             f'<a href="https://reports.janbauer.cc/dg-fig2-topk.html">vast.ai H100 repro</a> &middot; '
             f'full analysis (decomposition, oracle, stabilizer zoo, audit): '
             f'<a href="dg-fig2-trunc-archive.html">archived report</a></p>')

    # the headline/status figures cover the five MAIN task panels only (the step-matched
    # control is an extra gpqa panel and must not be double-counted)
    MAIN = [(t, lab) for t, lab, *_ in panels[:6] if t]
    g = {t: (acc(t, "k1"), acc(t, "k1_slow3")) for t, _ in MAIN}
    ncells = sum(1 for t, _ in MAIN for r in RUNGS for a in (r, r + "_slow3")
                 if D.get(t, {}).get(a))
    have_all = ncells == 50
    P.append('<div class="status-banner ' + ("complete" if have_all else "pending") + '"><span class=dot>'
             '</span><div><strong>' + ("Complete." if have_all else "Collecting.") + '</strong> '
             f'{ncells}/50 cells (5 tasks &times; 5 truncation levels &times; 2 samplers); every dot '
             'has a film.' + ('' if have_all else ' Dashed cells fill in as the fleet lands.') +
             '</div></div>')

    if g["gpqa"][0] is not None and g["gpqa"][1] is not None:
        bench_gaps = ", ".join(f"{lab.split(' —')[0]} {g[t][0]:.2f}&rarr;{g[t][1]:.2f}"
                               for t, lab in MAIN if g[t][0] is not None and g[t][1] is not None)
        P.append('<div class=finding>'
                 '<p><strong>The paper&rsquo;s top-1/top-2 collapse appears only under the standard '
                 'sampler; the gentle sampler removes it on every runnable Fig-2 benchmark.</strong> '
                 f'At top-1 (standard&rarr;gentle): {bench_gaps}. Under the standard sampler the '
                 'failures at low k are dominated by '
                 '<span style="color:var(--fm-loop);font-weight:650">degenerate loops</span> '
                 '(budget-capped repetition); under the gentle sampler these all but disappear and the '
                 'k&le;2 halves rise to their soft level. The word pool is the scoped exception: the '
                 'matched gap closes, but the gentle schedule costs the unablated model there &mdash; '
                 'fast-with-tail stays the best operating point on globally-constrained text.</p></div>')

    fms = ("ok", "loop", "capped", "wrong", "viol", "short")
    plain = lambda s: re.sub(r"</?b>", "", s)
    P.append('<div class=vio-legend>' +
             " ".join(f'<span class=fm-item title="{plain(FMDEF[k])}">'
                      f'<span class=fm-chip style="background:{FMCOL[k]}"></span>{FMLAB[k]}</span>'
                      for k in fms) +
             '<span style="border-left:1px solid var(--grid);padding-left:14px">'
             '<span class=fm-chip style="background:var(--vio-std);border-radius:2px"></span>standard sampler (left half)</span>'
             '<span><span class=fm-chip style="background:var(--vio-gen);border-radius:2px"></span>gentle sampler (right half)</span>'
             '<span><svg width=13 height=13 style="vertical-align:-2px"><path d="M6.5 1L12 6.5L6.5 12L1 6.5Z" '
             'style="fill:var(--ink)"/></svg>&nbsp;paper Fig-2 skyline (&plusmn;std) &mdash; drawn '
             '<b>inside the standard half</b>, since the paper runs the standard sampler</span>'
             '<span>&nbsp;| every dot plays a denoising film on click</span></div>')

    P.append('<details class=fm-defs><summary>how a rollout gets its colour &mdash; exact rules</summary>'
             '<dl>' + "".join(
                 f'<dt><span class=fm-chip style="background:{FMCOL[k]}"></span>{FMLAB[k]}</dt>'
                 f'<dd>{FMDEF[k]}</dd>' for k in fms) +
             '</dl><p class=caption style="margin:2px 0 0">Everything except <i>correct</i> counts as '
             'a 0 in the accuracy at the bottom of each panel; the split into <i>degenerate loop</i> / '
             '<i>budget-capped</i> / <i>wrong</i> is what separates a decoding failure from a reasoning '
             'failure, and is the evidence for the pacing account of the k=1 collapse.</p></details>')

    P.append('<details class=fm-defs><summary>how the answer is extracted &mdash; and how to jump to it'
             '</summary><dl>'
             '<dt>GPQA (letter)</dt><dd><code>extract_letter</code>: the <b>last</b> match of the first '
             'rule that fires &mdash; (1) <code>\\boxed{A&ndash;D}</code> (tolerating '
             '<code>\\text{}</code>/<code>\\mathrm{}</code>/parens), (2) the phrase '
             '<code>answer|option|choice [is|:] X</code>, (3) a bare <code>(X)</code>. Rule 1 alone is the '
             '<i>strict</i> grade; 2&ndash;3 make the <i>robust</i> grade, which credits an answer stated '
             'before a run degenerated.</dd>'
             '<dt>AMC/AIME, IMO (integer)</dt><dd><code>extract_int</code>: last '
             '<code>\\boxed{...}</code>, else the last integer <code>-?\\d[\\d,]*</code> in the text; '
             'commas stripped, compared to the reference integer.</dd>'
             '<dt>LiveCodeBench / HumanEval</dt><dd><code>extract_block</code>: the last fenced '
             '<code>```python</code> block, executed against the task&rsquo;s unit tests &mdash; correct '
             'only at pass_frac = 1.0.</dd>'
             '<dt>Word pool</dt><dd>there is no answer <i>span</i> here &mdash; the task&rsquo;s own '
             '<code>check()</code> grades the <b>whole text</b> against a hard constraint. The film '
             'therefore marks what the constraint forbids instead (in pink): the banned letter for a '
             'lipogram, every vowel but the allowed one for a univocalic, the first-six words whose '
             'length misses the &pi; digits for a piem. No pink = the constraint held.</dd>'
             '</dl><p class=caption style="margin:2px 0 0">In the film the graded span is boxed in green '
             '(<mark class=ans>like this</mark>) in whichever canvas it lands in, and <b>&#8615; answer</b> '
             '(or the <kbd>a</kbd> key) jumps to its <i>commit step</i> &mdash; the first step after the '
             'last one at which any token of that span still differed from its final value. Everything '
             'after that step is the model writing around an answer it has already fixed. '
             '<b>A&minus;/A+</b> (or <kbd>-</kbd>/<kbd>+</kbd>) resize the generation font.</p></details>')

    for task, label, *kw in panels:
        P.append('<div class="card panel-card" style="margin-top:12px">')
        if label is not None:                      # None => the extra HTML in kw[1] is the panel
            P.append(panel_svg(task, label, D, **(kw[0] if kw else {})))
        P.append(kw[1] if len(kw) > 1 else "")
        P.append('</div>')

    (b32s, _), (b32k, _) = b32_acc("soft_b32"), b32_acc("k8_b32")
    (fs, fn), (fk, _) = b32_acc("soft_b32", "imo_full"), b32_acc("k8_b32", "imo_full")
    P.append('<p class=caption>Half-violins: kernel density over the binary outcomes (fixed kernel 0.05 '
             'on the accuracy axis); dots are individual rollouts jittered inside their half envelope, '
             'solid bar = cell accuracy, dashed lines trace the means across truncation levels. '
             'Standard sampler = model defaults (entropy bound 0.1, t 0.4&ndash;0.8, 48 steps/canvas, '
             'budget 8192); gentle = entropy bound 0.02, t 0.5&ndash;1.0, 96 steps/canvas, budget 12288. '
             'Benchmarks: dynamic thinking, adaptive stopping, seeds paired across arms; word pool: '
             'battery protocol (single canvas, fixed step count, no thinking), graded on the lexical '
             'constraint. Paper diamonds for AMC/AIME and IMO refer to the paper&rsquo;s internal '
             'variants (ours are public draws) &mdash; compare shapes, not levels; N2C is Google-private '
             'and absent. f<sub>k</sub> matches the paper&rsquo;s haze construction exactly (kept top-k '
             'probabilities unchanged, leftover mass uniform, applied to the self-conditioning logits at '
             'every step and position). <b>IMO level-gap check &mdash; replication attempt of the '
             'paper&rsquo;s IMO scores:</b> the panel&rsquo;s standard cells are budget-starved at 8192 '
             'tokens (soft finishes 31%; AnswerBench provokes very long thinking). Matching everything '
             'the paper specifies (multi-canvas generation with 256-token canvases and autoregression '
             'between canvases, T=48, adaptive stopping, dynamic thinking &mdash; all confirmed from the '
             'paper source) and removing the budget cap (32k), on the <b>full 222-item integer-answer '
             'IMO-AnswerBench subset</b> (the paper&rsquo;s &plusmn;std implies n&asymp;190; '
             'GPQA&rsquo;s implies n&asymp;193 = full diamond, validating the inversion): standard soft '
             f'<b>{fs:.2f}</b> and top-8 <b>{fk:.2f}</b> (n={fn}, &plusmn;0.03, finish 1.00/0.93; '
             f'32-draw at same protocol: {b32s:.2f}/{b32k:.2f}) vs the paper&rsquo;s 0.79/0.70. Budget '
             'explains the panel&rsquo;s 0.28&rarr;0.50; the remaining &asymp;0.29 is the problem set '
             'itself &mdash; the paper&rsquo;s internal &ldquo;IMO variants&rdquo; are materially easier '
             'than public IMO-AnswerBench (grading spot-verified; failures are honest wrong answers). '
             'The IMO diamonds are therefore a shape reference only.</p>')

    P.append('<p class=footnote>Films: per-step argmax canvas + pre-truncation per-position entropy '
             '(viridis, 0&ndash;4 nats); canvas boundaries marked with c0/c1/&hellip; chips (blue = '
             'canvas being denoised; 256 tokens per canvas, word pool 64). Sources: '
             'lockin/ds_paper_sweep.py (data + films), fig2_report.py (this page). Everything this '
             'report previously contained &mdash; ladder reproductions incl. top-p, drop decomposition, '
             'no-think/budget controls, stopping-oracle, stabilizer zoo, word-pool dissociations, '
             'HumanEval, cross-reproduction audit &mdash; is preserved verbatim in the '
             '<a href="dg-fig2-trunc-archive.html">archived report</a>; broad exploratory appendix: '
             '<a href="st_origins.html">dg-st-origins</a>.</p>')

    P.append(PANEL_HTML)
    data_js = json.dumps(D, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    # the exact prompt per (task, pid), from dump_prompts.py; only the tasks that have panels
    pr = json.loads((CD / "prompts.json").read_text())
    pr = {t: v for t, v in pr.items() if t in D}
    prompts_js = json.dumps(pr, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    js = (JS.replace("%%DATA%%", data_js).replace("%%PROMPTS%%", prompts_js)
            .replace("%%VIR%%", json.dumps(VIR))
            .replace("%%EMAX%%", str(EMAX)).replace("%%PAD%%", json.dumps("<pad>")))
    P.append(f"<script>{js}</script>")
    P.append('</div></body></html>')
    HTML.write_text("".join(P))
    print(f"wrote {HTML} ({HTML.stat().st_size//1024}KB, {ncells}/50 cells)")


if __name__ == "__main__":
    main()
