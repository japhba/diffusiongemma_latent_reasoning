"""Render post.md -> lesswrong.md, ready to paste into the LessWrong Markdown editor.

Transforms: drop the H1 (LW takes the title separately); rewrite relative figs/ image
references to the public GitHub raw URLs (repo must be public for them to resolve).
Math ($...$ inline, $$...$$ display) and inline code spans paste through unchanged —
both are supported by the LW Markdown editor.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = "https://raw.githubusercontent.com/japhba/diffusiongemma_latent_reasoning/main/"

lines = (ROOT / "post.md").read_text().splitlines()
assert lines[0].startswith("# "), "post.md must start with the H1 title"
title = lines[0][2:].strip()
body = "\n".join(lines[1:]).lstrip("\n")
body, n = re.subn(r"\]\(figs/", f"]({RAW}figs/", body)

out = ROOT / "lesswrong.md"
out.write_text(body + "\n")
print(f"{out}  (title: {title!r}, {n} figure URLs rewritten)")
for m in re.finditer(r"!\[[^\]]*\]\((\S+?)\)", body):
    print(" ", m.group(1))
