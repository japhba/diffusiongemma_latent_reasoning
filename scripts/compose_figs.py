"""Compose appendix figures: matrix panel (top) + rendered HTML example card (bottom) -> one PNG.
Also promotes the letters-example card to figs/fig2a_example_intervention.png.

Cards are 2x-device-scale screenshots of the post.html illustration cards
(figs/parts/card_*.png, captured on aws-static via playwright); recapture them after
changing the card HTML, then re-run this.
"""
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "figs"
P = FIGS / "parts"
PAD, BG = 36, (255, 255, 255)


def compose_v(top, bottom, out):
    a, b = Image.open(top).convert("RGB"), Image.open(bottom).convert("RGB")
    W = max(a.width, b.width)
    canvas = Image.new("RGB", (W, a.height + b.height + PAD), BG)
    canvas.paste(a, ((W - a.width) // 2, 0))
    canvas.paste(b, ((W - b.width) // 2, a.height + PAD))
    canvas.save(out)
    print(out, canvas.size)


compose_v(P / "figA2_matrix.png", P / "card_probe_pair.png", FIGS / "figA2_probe_retention.png")
compose_v(P / "figA3_matrix.png", P / "card_steer_pair.png", FIGS / "figA3_steer_retention.png")
compose_v(P / "figA4_matrix.png", P / "card_jlens_pair.png", FIGS / "figA4_jlens_retention.png")
Image.open(P / "card_letters_example.png").convert("RGB").save(FIGS / "fig2a_example_intervention.png")
print(FIGS / "fig2a_example_intervention.png")
Image.open(P / "card_jlens_future.png").convert("RGB").save(FIGS / "figA6_jlens_future.png")
print(FIGS / "figA6_jlens_future.png")
