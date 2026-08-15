"""Render post.md -> post.html (minimal md subset) and inject native-HTML illustration blocks
at <!--ILLUST:name--> markers (data from data/*.json).

House conventions: system light/dark + toggle, no-cache meta, ~900px column, generations in
code font, judge digests italic, log-scaled probability bars.
"""
import html
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
md = (ROOT / "post.md").read_text().replace("&nbsp;", " ")

MATH, ILL = [], []
md = re.sub(r"\$\$.*?\$\$|\$[^$\n]+\$|\\\(.*?\\\)", lambda m: (MATH.append(m.group(0)) or f"\x00M{len(MATH)-1}\x00"), md, flags=re.S)
md = re.sub(r"<!--ILLUST:(\w+)-->", lambda m: (ILL.append(m.group(1)) or f"\x02I{len(ILL)-1}\x02"), md)


def inline(s):
    s = html.escape(s, quote=False)
    s = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r'<img src="\2" alt="\1">', s)
    s = re.sub(r"\[([^\]]+)\]\((https?://[^)]+|[^)]+\.html)\)", r'<a href="\2">\1</a>', s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"__([^_]+)__", r"<strong>\1</strong>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<em>\1</em>", s)
    s = re.sub(r"(?<![\w./\"])(reports\.janbauer\.cc[\w./-]*)", r'<a href="https://\1">\1</a>', s)
    return s


out, i, lines = [], 0, md.splitlines()
while i < len(lines):
    ln = lines[i]
    if ln.startswith("### "): out.append(f"<h3>{inline(ln[4:])}</h3>")
    elif ln.startswith("## "): out.append(f"<h2>{inline(ln[3:])}</h2>")
    elif ln.startswith("# "): out.append(f"<h1>{inline(ln[2:])}</h1>")
    elif ln.strip() == "---": out.append("<hr>")
    elif re.fullmatch(r"\x02I\d+\x02", ln.strip()): out.append(ln.strip())
    elif re.match(r"^\s*[-*] ", ln) or re.match(r"^\s*\d+\. ", ln):
        ordered = bool(re.match(r"^\s*\d+\. ", ln))
        items = []
        while i < len(lines) and (re.match(r"^\s*[-*] ", lines[i]) or re.match(r"^\s*\d+\. ", lines[i])):
            items.append(re.sub(r"^\s*([-*]|\d+\.) ", "", lines[i])); i += 1
        i -= 1
        tag = "ol" if ordered else "ul"
        out.append(f"<{tag}>" + "".join(f"<li>{inline(it)}</li>" for it in items) + f"</{tag}>")
    elif ln.strip():
        para = [ln]
        while i + 1 < len(lines) and lines[i + 1].strip() and not re.match(r"^(#{1,3} |\s*[-*] |\s*\d+\. |---$|\x02)", lines[i + 1]):
            i += 1; para.append(lines[i])
        body = inline(" ".join(para))
        out.append(f"<figure>{body}</figure>" if re.fullmatch(r"\s*<img [^>]+>\s*", body) else f"<p>{body}</p>")
    i += 1
body = "\n".join(out)
body = re.sub("\x00M(\\d+)\x00", lambda m: MATH[int(m.group(1))], body)

# ---------------- illustration blocks ----------------
E = lambda s: html.escape(str(s), quote=False)
CL = {"inj": "#9c36b5", "tgt": "#e8590c", "att": "#868e96", "oth": "#7aa2ff"}


LOGFLOOR = 1e-5
lgw = lambda p: max(0.0, (math.log10(max(p, LOGFLOOR)) - math.log10(LOGFLOOR)) / -math.log10(LOGFLOOR)) * 100


def bar(p, col):
    return (f'<span class="btrack"><i style="width:{lgw(p):.1f}%;background:{col}"></i>'
            f'<b class="bv1">{p:.3g}</b></span>')


def logaxis():
    """Decade-gridded axis matching lgw()'s log scale (1e-5 -> 0%, one decade per 20%)."""
    ticks = "".join(f'<em style="left:{w}%"></em>' for w in range(0, 101, 20))
    def lab(w, txt):
        tr = "none" if w == 0 else ("translateX(-100%)" if w == 100 else "translateX(-50%)")
        return f'<u style="left:{w}%;transform:{tr}">{txt}</u>'
    return ('<span class="laxis">' + ticks + lab(0, "10<sup>&minus;5</sup>")
            + lab(40, "10<sup>&minus;3</sup>") + lab(80, "10<sup>&minus;1</sup>") + lab(100, "1") + "</span>")


def pills(*ps):
    return '<div class="pills">' + "".join(f'<span class="pill">{E(p)}</span>' for p in ps) + "</div>"


def ill_letters_example():
    d = json.load(open(DATA / "letters_example.json"))
    cls = lambda tok: ("inj" if tok == d["x"] else "tgt" if tok == d["img"] else
                       "att" if tok in (d["ja"], d["nat"]) else "oth")
    sub_t = "operand distribution $\\mathbf{s}^t_i$"
    sub_t1 = "answer distribution $\\mathbf{s}^{t+1}_{i+1}$"
    head = (f'<div class="brow b4c ghead"><span></span><span class="gh s2">baseline</span>'
            f'<span></span><span class="gh s2">intervention (inject mass '
            f'$\\varepsilon={d["eps"]:g}$ on \'{E(d["x"])}\')</span></div>'
            f'<div class="brow b4c bhead"><span></span><span class="bh">{sub_t}</span><span class="bh">{sub_t1}</span>'
            f'<span></span><span class="bh">{sub_t}</span><span class="bh">{sub_t1}</span></div>')
    body = []
    for tok, ab, ap, bb, bp in d["rows"]:
        col = CL[cls(tok)]
        # intervention @ operand: the injected increment beyond the carried-over mass, low alpha
        if tok == d["x"]:
            carried = ab * (1 - d["eps"])
            ap_bar = (f'<span class="btrack"><i style="width:{lgw(ap):.1f}%;background:{col};opacity:.3"></i>'
                      f'<i style="width:{lgw(carried):.1f}%;background:{col}"></i>'
                      f'<b class="bv1">{ap:.3g}</b></span>')
        else:
            ap_bar = bar(ap, col)
        # intervention @ answer for the image: mass gained over baseline as a light-alpha
        # extension (same grammar as the injected-operand bar: solid = baseline, faint = added)
        if tok == d["img"]:
            bp_bar = (f'<span class="btrack"><i style="width:{lgw(bp):.1f}%;background:{col};opacity:.3"></i>'
                      f'<i style="width:{lgw(bb):.1f}%;background:{col}"></i>'
                      f'<b class="bv1">{bp:.3g}</b></span>')
        else:
            bp_bar = bar(bp, col)
        body.append(f'<div class="brow b4c"><span class="btok" style="color:{col}">{E(tok)}</span>'
                    + bar(ab, col) + bar(bb, col) + "<span></span>" + ap_bar + bp_bar + "</div>")
    kword = {3: "three", 5: "five", 7: "seven", 11: "eleven"}[d["k"]]
    before, after = E(d["prompt"]).split(f" {kword} positions later", 1)
    prompt_html = (f'<code class="gen">{before} <u>{kword} positions later</u></code>'
                   f'<span class="kn"> (this sets the shift $k{{=}}{d["k"]}$) </span>'
                   f'<code class="gen">{after.lstrip()}</code>')
    ann = lambda tok, lab: (f'<span class="anntok"><code class="gen">{E(tok)}</code>'
                            f'<span class="annlab">{lab}</span></span>')
    gen_html = (f'<code class="gen">Letters:</code> {ann(d["nat"], "$x_i$")}'
                f'<code class="gen">,</code> {ann(d["ja"], "$x_{i+1}$")}')
    return f"""<div class="card">
<p class="small">prompt: {prompt_html}</p>
<p class="small">natural generation: &nbsp;{gen_html}</p>
{head}{''.join(body)}
<div class="brow b4c"><span></span>{logaxis()}{logaxis()}<span></span>{logaxis()}{logaxis()}</div>
<span class="leg"><i style="background:{CL['inj']}"></i> injected source token
<i style="background:{CL['tgt']}"></i> its arithmetic image
<i style="background:{CL['att']}"></i> committed operand / natural answer
<i style="background:{CL['oth']}"></i> other &nbsp;·&nbsp; all bars share one log scale ($10^{{-5}}$ … $1$)
&nbsp;·&nbsp; faint segments: mass added by the intervention</span>
</div>"""


def ill_probe_pair():
    d = json.load(open(DATA / "probe_pair.json"))
    def side(rec, lab):
        bars = "".join(
            f'<div class="brow wide"><span class="btok sm">{who}</span><span class="btrack">'
            f'<i style="width:{100*rec[k]:.1f}%;background:{col}"></i><b class="bv2">P={rec[k]:.2f}</b></span></div>'
            for who, k, col in (("read on gemma-4 acts", "g", "#1971c2"), ("read on DG acts", "d", "#e8590c")))
        return f'<div><h4>{lab}</h4><code class="gen block">{E(rec["text"][:260])}{" …" if len(rec["text"])>260 else ""}</code>{bars}</div>'
    return f"""<div class="card">
<div class="cols2">{side(d['pos'], f"positive ({d['pos_label']})")}{side(d['neg'], f"negative ({d['neg_label']})")}</div>
</div>"""


def ill_steer_pair():
    d = json.load(open(DATA / "steer_pair.json"))
    strip = lambda t: "".join(ch for ch in t if ord(ch) < 0x2500 or ch in "💖✨🌸")
    trunc = lambda t, n=380: (lambda s: s if len(s) <= n else s[:n] + " …")(strip(t))
    def gen(txt, lab, col):
        return (f'<div class="steergen" style="border-left:3px solid {col}"><b style="color:{col}">{lab}</b>'
                f'<code class="gen block">{E(trunc(txt))}</code></div>')
    g, dg = d["cells"]["gg"], d["cells"]["gd"]
    pos_lab = f"+steer (toward {d['concept']})"
    return f"""<div class="card">
<p class="small">carrier: <code class="gen">{E(d['carrier'])}</code></p>
<div class="cols2">
<h4>steered model: {E(g["model"])}</h4><h4>steered model: {E(dg["model"])}</h4>
{gen(g["pos"], pos_lab, "#2f9e44")}{gen(dg["pos"], pos_lab, "#2f9e44")}
{gen(g["neg"], "−steer (away)", "#c2255c")}{gen(dg["neg"], "−steer (away)", "#c2255c")}
</div>
</div>"""


def ill_jlens_pair():
    d = json.load(open(DATA / "jlens_layers.json"))
    GT = "#e8590c"
    hl = [h.lower() for h in d["highlight"]]
    def chip(tk):
        col = GT if any(h in tk.strip().lower() for h in hl) else None
        style = f'style="color:#fff;background:{col}"' if col else ""
        return f'<code class="gen chip" {style}>{E(tk)}</code>'
    def side(test, lab):
        rows = "".join(
            f'<div class="lrow"><span class="llab">L{L}</span>'
            + "".join(chip(tk) for tk in d["tops"][test][str(L)]) + "</div>"
            for L in d["layers"])
        return f'<div><h4>{lab}</h4>{rows}</div>'
    READ = "#1971c2"
    tail = d["tail"].rstrip()
    # poetry reads at the token containing the FINAL newline (cue-line ending), per paper_readout
    pre, post = tail.rsplit("\n", 1)
    br = lambda s: E(s).replace("\n", "<br>")
    tail_html = (f'<code class="gen">{br(pre)}</code>'
                 f'<span class="readtok" style="background:{READ}">↵</span> '
                 f'<code class="gen">{br(post)}</code>')
    return f"""<div class="card">
<p class="small">prompt tail: {tail_html}</p>
<div class="cols2">{side('g', 'read on gemma-4 residuals')}{side('dg', 'read on DG residuals')}</div>
<span class="leg"><i style="background:{GT}"></i> ground-truth intermediate
('{E(d['intermediates'][0])}', incl. variants)
<i style="background:{READ}"></i> lens read position — the newline token closing the cue line</span>
</div>"""


def ill_jlens_future():
    d = json.load(open(DATA / "jlens_future.json"))
    GT, READ = "#e8590c", "#1971c2"
    def side(v):
        op_w, read_w = v["op"].strip(), d["read_token"].strip()
        def chip(tk):
            style = f'style="color:#fff;background:{GT}"' if tk.strip() == op_w else ""
            return f'<code class="gen chip" {style}>{E(tk)}</code>'
        words = "".join(
            f'<span class="readtok" style="background:{READ}">{E(w)}</span> ' if w == read_w
            else (f'<b style="color:{GT}">{E(w)}</b> ' if w == op_w else E(w) + " ")
            for w in v["prompt"].split())
        rows = "".join(
            f'<div class="lrow"><span class="llab">L{L}</span>'
            + "".join(chip(tk) for tk in v["tops"][str(L)]) + "</div>"
            for L in d["layers"])
        return f'<div><h4>{E(v["label"])}</h4><p class="small"><code class="gen">{words}</code></p>{rows}</div>'
    return f"""<div class="card">
<div class="cols2">{side(d['variants']['sub'])}{side(d['variants']['add'])}</div>
<span class="leg"><i style="background:{GT}"></i> the variant's operation token
<i style="background:{READ}"></i> lens read position</span>
</div>"""


def ill_posthoc_case():
    d = json.load(open(DATA / "posthoc_case.json"))
    OK, BAD, IV = "#2f9e44", "#c2255c", "#9c36b5"
    def hl_answer(text, ans, col):
        # highlight the answer token inside the FIRST line (answer-first framing)
        nl = text.find("\n"); l1 = text[:nl] if nl >= 0 else text
        m = re.search(rf"(?<![\d]){re.escape(ans)}(?![\d])", l1)
        assert m, f"answer {ans!r} not on first line: {l1!r}"
        bg = "rgba(47,158,68,.12)" if col == OK else "rgba(194,37,92,.12)"
        return (E(l1[:m.start()]) + f'<b style="color:{col};background:{bg}">{E(ans)}</b>'
                + E(l1[m.end():]) + E(text[nl:] if nl >= 0 else ""))
    # interleave the two problems' cells so the grid rows (header/clean/suscept/verdict)
    # align vertically across columns
    cells = {r: [] for r in ("head", "cleanlab", "clean", "susclab", "susc", "verdict")}
    for tag, title in (("easy", "easy"), ("hard", "hard")):
        v = d[tag]; cl, sc = v["clean"], v["suscept"]
        nl = sc["text"].find("\n")
        sc_html = hl_answer(sc["text"][:nl], sc["answer"], OK if sc["match"] else BAD) \
            + f'<span style="color:{IV}">{E(sc["text"][nl:])} …</span>'
        commit = f"commits at step {cl['commit']:g}" + \
            (f" (path {' → '.join(cl['traj'])})" if len(cl["traj"]) > 1 else "")
        out = "unchanged" if sc["match"] else f"flips ({'/'.join(sorted(set(sc['answers'])))} over seeds)"
        cells["head"].append(f'<div><h4>{E(title)}</h4><p class="small">prompt: <code class="gen">{E(v["q"])}</code></p></div>')
        cells["cleanlab"].append(f'<p class="small"><b>clean</b> &nbsp;·&nbsp; {E(commit)}</p>')
        cells["clean"].append(f'<code class="gen block">{hl_answer(cl["text"], cl["answer"], OK)} …</code>')
        cells["susclab"].append(f'<p class="small"><b>susceptibility</b> &nbsp;·&nbsp; all {sc["n_cot"]} CoT positions (<span style="color:{IV}">purple</span>) pinned to random tokens every step</p>')
        cells["susc"].append(f'<code class="gen block">{sc_html}</code>')
        cells["verdict"].append(f'<p class="small"><b style="color:{OK if sc["match"] else BAD}">answer: {E(sc["answer"])}</b> &nbsp;·&nbsp; {E(out)} — S = {v["S"]:.2f}</p>')
    grid = "".join(cells[r][i] for r in ("head", "cleanlab", "clean", "susclab", "susc", "verdict") for i in (0, 1))
    return f"""<div class="card">
<div class="cols2" style="row-gap:6px">{grid}</div>
</div>"""


def ill_selfcorr_steps():
    d = json.load(open(DATA / "selfcorr_steps.json"))
    PH = {"early": "#7aa2ff", "mid": "#c2255c", "late": "#2f9e44"}
    rows = "".join(
        f'<div class="steergen" style="border-left:3px solid {PH[s["phase"]]}">'
        f'<b style="color:{PH[s["phase"]]}">step t={s["t"]} ({s["phase"]}) · margin {s["margin"]}'
        f'{" — violates the constraint" if s["phase"] == "mid" else (" — satisfied" if s["margin"] == 0 else "")}</b>'
        f'<code class="gen block">{E(s["text"])}</code></div>'
        for s in d["steps"])
    return f"""<div class="card">
<p class="small">task: <code class="gen">{E(d['prompt_task'])}</code></p>
{rows}
</div>"""


GEN = {"letters_example": ill_letters_example, "probe_pair": ill_probe_pair,
       "steer_pair": ill_steer_pair, "jlens_pair": ill_jlens_pair,
       "jlens_future": ill_jlens_future, "posthoc_case": ill_posthoc_case,
       "selfcorr_steps": ill_selfcorr_steps}
for j, name in enumerate(ILL):
    blk = GEN[name]()
    body = re.sub(rf"(<p>)?\x02I{j}\x02(</p>)?", lambda m: blk, body)

HEAD = """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<meta http-equiv="cache-control" content="no-cache, no-store, must-revalidate">
<meta http-equiv="pragma" content="no-cache"><meta http-equiv="expires" content="0">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Does DiffusionGemma have Latent Reasoning?</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>&#127744;</text></svg>">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
 onload="renderMathInElement(document.body,{delimiters:[{left:'$$',right:'$$',display:true},{left:'\\\\(',right:'\\\\)',display:false},{left:'$',right:'$',display:false}]});"></script>
<style>
:root{--bg:#ffffff;--fg:#1a1a1a;--muted:#666;--rule:#ddd;--accent:#1971c2;--card:#f7f7f9}
@media(prefers-color-scheme:dark){:root:where(:not([data-theme=light])){--bg:#14161a;--fg:#e6e6e6;--muted:#9aa0a6;--rule:#333;--accent:#74a9e8;--card:#1d2026}}
:root[data-theme=dark]{--bg:#14161a;--fg:#e6e6e6;--muted:#9aa0a6;--rule:#333;--accent:#74a9e8;--card:#1d2026}
body{margin:0;background:var(--bg);color:var(--fg);font:17px/1.6 Georgia,'Times New Roman',serif}
.wrap{max-width:900px;margin:0 auto;padding:24px 20px 80px}
h1,h2,h3,h4{font-family:system-ui,sans-serif;line-height:1.25}
h1{font-size:1.9em}h2{margin-top:1.8em}h3{margin-top:1.4em}h4{margin:.2em 0 .5em;font-size:.95em}
img{display:block;max-width:100%;margin:1em auto;border-radius:4px}
figure{margin:1.4em 0}
code{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:.85em;background:color-mix(in srgb,var(--fg) 7%,transparent);padding:.08em .3em;border-radius:3px}
code.gen{background:color-mix(in srgb,var(--fg) 5%,transparent)}
code.gen.block{display:block;white-space:pre-wrap;padding:.5em .7em;margin:.4em 0;border-radius:6px;font-size:.78em;line-height:1.45}
a{color:var(--accent)}hr{border:0;border-top:1px solid var(--rule);margin:2.2em 0}
p>em:first-child{color:var(--muted)}
.card{background:var(--card);border:1px solid var(--rule);border-radius:10px;padding:14px 16px;margin:1.2em 0;font-family:system-ui,sans-serif;font-size:.85em;line-height:1.45}
.pills{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:8px}
.pill{border:1px solid var(--rule);border-radius:999px;padding:1px 9px;font-size:.82em;color:var(--muted);background:var(--bg)}
.small{color:var(--muted);margin:.3em 0}
.cap{margin:.7em 0 .2em}.judge{font-style:italic;color:var(--muted);margin:.5em 0 .1em}
.cols2{display:grid;grid-template-columns:1fr 1fr;gap:18px}
@media(max-width:640px){.cols2{grid-template-columns:1fr}}
.brow{display:grid;grid-template-columns:11em 1fr 1fr;gap:8px;align-items:center;margin:2px 0}
.card .cols2 .brow{grid-template-columns:2.2em 1fr 1fr}
.card .cols2 .brow.wide{grid-template-columns:10.5em 1fr}
.brow.b4c{grid-template-columns:2.2em 1fr 1fr 16px 1fr 1fr}
.ghead .gh{font-weight:600;text-align:center}
.ghead .gh.s2{grid-column:span 2}
.ghead{margin:6px 0 0}
.anntok{display:inline-flex;flex-direction:column;align-items:center;vertical-align:top}
.annlab{font-size:.8em;color:var(--muted);line-height:1.1}
.kn{color:var(--muted);font-size:.9em;font-family:system-ui,sans-serif}
.readtok{color:#fff;border-radius:3px;padding:0 .3em}
.card u{text-decoration-color:var(--accent);text-underline-offset:2px}
.lrow{display:flex;align-items:center;gap:5px;margin:3px 0;flex-wrap:wrap}
.llab{font:600 .85em system-ui;color:var(--muted);width:2.4em}
code.gen.chip{padding:.15em .45em;border-radius:4px;white-space:pre}
.brow .btok{font-family:ui-monospace,monospace;text-align:right;font-weight:600}
.brow .btok.sm{font-weight:400;font-size:.85em;text-align:right}
.btrack{position:relative;height:14px;background:color-mix(in srgb,var(--fg) 6%,transparent);border-radius:3px;overflow:visible}
.btrack i{position:absolute;left:0;top:0;bottom:0;border-radius:3px}
.laxis{position:relative;height:13px}
.laxis em{position:absolute;top:0;height:4px;width:1px;background:var(--muted);opacity:.7}
.laxis u{position:absolute;top:4px;font:.62em/1.4 ui-monospace,monospace;color:var(--muted);text-decoration:none}
.btrack b{position:absolute;left:4px;top:-1px;font:10px/16px ui-monospace,monospace;font-weight:400;color:var(--fg);opacity:.85;white-space:nowrap}
.bhead .bh{font-size:.8em;color:var(--muted)}
.steergen{padding:2px 10px;margin:2px 0}
.leg{font-size:.8em;color:var(--muted)}.leg i{display:inline-block;width:10px;height:10px;border-radius:2px;margin:0 3px 0 10px;vertical-align:-1px}
#thm{position:fixed;top:12px;right:14px;font:13px system-ui;cursor:pointer;background:none;border:1px solid var(--rule);border-radius:6px;color:var(--fg);padding:3px 8px}
</style></head><body>
<button id="thm" onclick="const r=document.documentElement,c=r.dataset.theme||'';r.dataset.theme=c==='dark'?'light':c==='light'?'':'dark';localStorage.setItem('dgblogTheme',r.dataset.theme)">&#9681;</button>
<script>document.documentElement.dataset.theme=localStorage.getItem('dgblogTheme')||''</script>
<div class="wrap">
"""
TAIL = "\n</div></body></html>"

(ROOT / "post.html").write_text(HEAD + body + TAIL)
print(ROOT / "post.html")

if "card" in __import__("sys").argv:
    allcards = (ill_letters_example() + ill_probe_pair() + ill_steer_pair() + ill_jlens_pair()
                + ill_jlens_future() + ill_posthoc_case() + ill_selfcorr_steps())
    (ROOT / "card_preview.html").write_text(HEAD + allcards + TAIL)
    print(ROOT / "card_preview.html")
