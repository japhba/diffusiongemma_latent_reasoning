"""Load the symbol-arithmetic study payload (src_data/symbol_arithmetic_payload.json).

Provenance: window.__DATA__ of activation_oracles_dev/reports/dg-planning/symbol_arithmetic.html
(builder: diffusiongemma/planning/build_superpos.py), extracted verbatim 2026-08-02; refreshed
2026-08-10 from the 2026-08-04 report build (adds doms mu/rf/kb/mn3/cp + tmap mu/rf/kb/mn3/cp;
the num/let/ll/uu subtrees are byte-identical to the 2026-08-02 extraction).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_payload():
    return json.load(open(ROOT / "src_data" / "symbol_arithmetic_payload.json"))


if __name__ == "__main__":
    d = load_payload()
    print(list(d.keys()))
    for dom, t in d["doms"].items():
        print(dom, t["label"], "eps0", t["eps0"], "states", len(t["states"]), "cells", len(t["cells"]),
              "ks", t["ks"], "nlvl", t["nlvl"])
