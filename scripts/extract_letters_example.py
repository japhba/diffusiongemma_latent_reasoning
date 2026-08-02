"""Extract the letter-arithmetic example intervention -> data/letters_example.json.

Cell: tmap.let UU3|hi|s0, inject x='H' at eps=0.45 (strictly subleading) at operand slot A.
A-panel sheet from the battery state UU3|src0|s0 (same capture s0, t=2), 'after' per the report's
sheet_vec promo rule: p*(1-eps) then +eps on x.  B-panel: tmap base vs arm (8 paired draws).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from payload import load_payload

ROOT = Path(__file__).resolve().parent.parent
UPP = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
D = load_payload()
tm = D["tmap"]["let"]
EPS = tm["eps"]
X = "H"
st = tm["states"]["UU3|hi|s0"]
c = next(c for c in tm["cells"] if c["st"] == "UU3|hi|s0" and c["x"] == X)
bst = D["doms"]["uu"]["states"]["UU3|src0|s0"]
assert bst["nat"] == UPP[[i for i in range(26) if UPP[i] == st["nat"]][0]] == "G"
k = st["k"]
img = UPP[UPP.index(X) + k]
ja = tm["field"][st["ja"]].strip() if isinstance(st["ja"], int) and st["ja"] < len(tm["field"]) else None

field = D["doms"]["uu"]["field"]  # 52 tokens, lower+UPPER
stA = dict(zip(field, bst["stA"]))
stAp = {w: p * (1 - EPS) for w, p in stA.items()}
stAp[X] = stAp.get(X, 0.0) + EPS

base = {field[q] if isinstance(field[q], str) else q: (st["base"][q] or 0.0) for q in range(52)}
base = {field[q]: (st["base"][q] or 0.0) for q in range(52)}
arm = {field[q]: (c["arm"][q] or 0.0) for q in range(52)}

# unified token rows: [tok, A_base, A_pert, B_base, B_pert], shared across all four bar tracks
vecs = (stA, stAp, base, arm)
mx = lambda w: max(v.get(w, 0) or 0 for v in vecs)
sel = [w for w in sorted(field, key=lambda w: -mx(w)) if mx(w) > 0][:9]
for w in (X, img, bst["nat"], bst["ja"]):
    if w not in sel:
        sel.append(w)
sel.sort(key=lambda w: -mx(w))
rows = [[w] + [round(v.get(w, 0) or 0, 5) for v in vecs] for w in sel]

out = dict(prompt=tm["prompts"]["UU3"], final=bst["final"], nat=bst["nat"], ja=bst["ja"],
           k=k, eps=EPS, x=X, img=img, draws=tm["draws"], t=2, rows=rows)
(ROOT / "data").mkdir(exist_ok=True)
json.dump(out, open(ROOT / "data" / "letters_example.json", "w"), indent=1)
print(ROOT / "data" / "letters_example.json")
print("img", img, "ja", bst["ja"], "P_base(img)", base[img], "P_pert(img)", arm[img],
      "P_base(ja)", base[bst["ja"]], "P_pert(ja)", arm[bst["ja"]])
