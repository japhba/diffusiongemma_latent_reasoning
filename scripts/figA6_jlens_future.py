"""A6: J-Lens future-operation card data — order-ops minimal pair (word-div-sub):
the DG-bidirectional-fit lens read at the earlier token ' by' surfaces the upcoming
operation; swapping the single future token minus->plus switches the read.

Data: src_data/jlens_future_rows.json (extracted verbatim from
reports/concept_probes/jlens_future.html `const rows=`, builder
concept_probes/analyze_jlens_future.py; original = subtraction, counterfactual = addition).
Emits data/jlens_future.json consumed by build_html.py (card jlens_future).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LAYERS = [23, 21, 18, 12]

rows = json.load(open(ROOT / "src_data" / "jlens_future_rows.json"))
r = next(x for x in rows if x["name"] == "word-div-sub")

def variant(arm, prompt_key, op, rank_key, label):
    per = {e["layer"]: e for e in r[arm]["jlens"]}
    return {"prompt": r[prompt_key].strip(), "op": op, "label": label,
            "tops": {str(L): per[L]["top_tokens"][:5] for L in LAYERS},
            "ranks": {str(L): per[L][rank_key] for L in LAYERS}}

out = {"name": r["name"], "layers": LAYERS,
       "read_token": r["source_token"], "source_position": r["source_position"],
       "future_position": r["future_position"],
       "config_label": "DG-bidirectional-fit Jacobian lens",
       "variants": {
           "sub": variant("original", "prompt", r["target_token"], "target_rank",
                          "subtraction (original)"),
           "add": variant("counterfactual", "counterfactual_prompt", r["foil_token"], "foil_rank",
                          "addition (one-token counterfactual)")}}
DATA = ROOT / "data"; DATA.mkdir(exist_ok=True)
json.dump(out, open(DATA / "jlens_future.json", "w"), indent=1)
print(DATA / "jlens_future.json")
for k, v in out["variants"].items():
    print(k, v["prompt"], "| op", repr(v["op"]), "| ranks", v["ranks"])
