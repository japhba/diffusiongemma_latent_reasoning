"""Render post.md -> post.html (minimal md subset) and inject native-HTML illustration blocks
at <!--ILLUST:name--> markers (data from data/*.json).

House conventions: system light/dark + toggle, no-cache meta, ~900px column, generations in
code font, judge digests italic, provenance pills, sqrt-scaled probability bars.
"""
import html
import json
import math
import re
from pathlib import Path

ROOT = Path("/workspace-vast/jbauer/dg_blog")
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


def bar_rows(entries, cls_of, pmax=1.0):
    rows = []
    for tok, p1, p2 in entries:
        col = CL[cls_of(tok)]
        w1, w2 = (100 * math.sqrt(p / pmax) for p in (p1, p2))
        rows.append(
            f'<div class="brow"><span class="btok" style="color:{col}">{E(tok)}</span>'
            f'<span class="btrack"><i style="width:{w1:.1f}%;background:{col};opacity:.35"></i>'
            f'<b class="bv1">{p1:.3g}</b></span>'
            f'<span class="btrack"><i style="width:{w2:.1f}%;background:{col}"></i>'
            f'<b class="bv2">{p2:.3g}</b></span></div>')
    return ('<div class="brow bhead"><span class="btok"></span><span class="bh">before</span>'
            '<span class="bh">after</span></div>' + "".join(rows))


def pills(*ps):
    return '<div class="pills">' + "".join(f'<span class="pill">{E(p)}</span>' for p in ps) + "</div>"


def ill_letters_example():
    d = json.load(open(DATA / "letters_example.json"))
    cls = lambda tok: ("inj" if tok == d["x"] else "tgt" if tok == d["img"] else
                       "att" if tok in (d["ja"], d["nat"]) else "oth")
    return f"""<div class="card">
{pills(f"letter arithmetic +{d['k']} (UPPER→UPPER)", f"sheet seed s=0, step t={d['t']}→{d['t']+1}",
       f"inject ε={d['eps']:g} on '{d['x']}' (stays sub-leading)", f"{d['draws']} paired renoise draws")}
<p class="small">prompt: {E(d['prompt'])}</p>
<p class="small">natural generation: <code class="gen">{E(d['final'])}</code>
&nbsp;(committed operand '{E(d['nat'])}', natural answer '{E(d['ja'])}')</p>
<div class="cols2">
<div><h4>operand slot A — sheet S<sup>t</sup>, base → +ε on '{E(d['x'])}'</h4>{bar_rows(d['A'], cls)}</div>
<div><h4>answer slot B — one step later, base → perturbed</h4>{bar_rows(d['B'], cls)}</div>
</div>
<p class="cap"><em>The leader '{E(d['nat'])}' keeps rank 1 and the canvas never changes; yet one step later the
answer slot has swung from '{E(d['ja'])}' (0.996 → 0.05) to '{E(d['img'])}' = '{E(d['x'])}'+{d['k']}
(0.0002 → 0.91): the answer position tracks the <b>belief</b> at the operand position, not the committed text.</em></p>
<span class="leg"><i style="background:{CL['inj']}"></i> injected source
<i style="background:{CL['tgt']}"></i> its image
<i style="background:{CL['att']}"></i> committed operand / natural answer
<i style="background:{CL['oth']}"></i> other &nbsp;·&nbsp; bars √-scaled</span>
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
{pills("concept: world news (161_agnews_0)", f"same gemma-trained probe, layer {d['layer']}", "held-out test texts")}
<div class="cols2">{side(d['pos'], "positive (world news)")}{side(d['neg'], "negative (balanced other)")}</div>
</div>"""


def ill_steer_pair():
    d = json.load(open(DATA / "steer_pair.json"))
    strip = lambda t: "".join(ch for ch in t if ord(ch) < 0x2500 or ch in "💖✨")
    def gen(txt, lab, col):
        return (f'<div class="steergen" style="border-left:3px solid {col}"><b style="color:{col}">{lab}</b>'
                f'<code class="gen block">{E(strip(txt))} …</code></div>')
    return f"""<div class="card">
{pills(f"task: {d['task']}", f"direction: {d['cell']}", "same carrier prompt, ±steer",
       f"judge confidence {d['confidence']:g}")}
<p class="small">carrier: <code class="gen">{E(d['carrier'])}</code></p>
<div class="cols2">{gen(d['pos'], "+steer (toward happiness)", "#2f9e44")}{gen(d['neg'], "−steer (away)", "#c2255c")}</div>
<p class="judge">judge: {E(d['judge'])}</p>
</div>"""


def ill_jlens_pair():
    d = json.load(open(DATA / "jlens_pair.json"))
    def side(rec, col):
        insp = f'<p class="judge">inspection: {E(rec["inspection"])}</p>' if rec.get("inspection") else ""
        return (f'<div><h4 style="color:{col}">{E(rec["label"])}</h4><p class="small">{E(rec["desc"])}</p>'
                f'<code class="gen block">A = {rec["A"]:.2f} &nbsp; top-20 matches: {E(rec["matches"])}</code>{insp}</div>')
    return f"""<div class="card">
{pills("J-Lens percepts", "score A = 1 − e^−n", "judge: gemini-3-flash (inspection rungs)")}
<div class="cols2">{side(d['hit'], "#2f9e44")}{side(d['miss'], "#c2255c")}</div>
</div>"""


GEN = {"letters_example": ill_letters_example, "probe_pair": ill_probe_pair,
       "steer_pair": ill_steer_pair, "jlens_pair": ill_jlens_pair}
for j, name in enumerate(ILL):
    blk = GEN[name]()
    body = re.sub(rf"(<p>)?\x02I{j}\x02(</p>)?", lambda m: blk, body)

HTML = """<!DOCTYPE html>
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
.brow .btok{font-family:ui-monospace,monospace;text-align:right;font-weight:600}
.brow .btok.sm{font-weight:400;font-size:.85em;text-align:right}
.btrack{position:relative;height:14px;background:color-mix(in srgb,var(--fg) 6%,transparent);border-radius:3px;overflow:visible}
.btrack i{position:absolute;left:0;top:0;bottom:0;border-radius:3px}
.btrack b{position:absolute;left:4px;top:-1px;font:10px/16px ui-monospace,monospace;font-weight:400;color:var(--fg);opacity:.85}
.bhead .bh{font-size:.8em;color:var(--muted)}
.steergen{padding:2px 10px;margin:2px 0}
.leg{font-size:.8em;color:var(--muted)}.leg i{display:inline-block;width:10px;height:10px;border-radius:2px;margin:0 3px 0 10px;vertical-align:-1px}
#thm{position:fixed;top:12px;right:14px;font:13px system-ui;cursor:pointer;background:none;border:1px solid var(--rule);border-radius:6px;color:var(--fg);padding:3px 8px}
</style></head><body>
<button id="thm" onclick="const r=document.documentElement,c=r.dataset.theme||'';r.dataset.theme=c==='dark'?'light':c==='light'?'':'dark';localStorage.setItem('dgblogTheme',r.dataset.theme)">&#9681;</button>
<script>document.documentElement.dataset.theme=localStorage.getItem('dgblogTheme')||''</script>
<div class="wrap">
""" + body + """
</div></body></html>"""

(ROOT / "post.html").write_text(HTML)
print(ROOT / "post.html", len(HTML))
