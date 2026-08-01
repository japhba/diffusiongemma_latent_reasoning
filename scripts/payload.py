"""Extract window.__DATA__ payload from symbol_arithmetic.html (cached to scratchpad)."""
import json
from pathlib import Path

HTML = Path("/workspace-vast/jbauer/activation_oracles_dev/reports/dg-planning/symbol_arithmetic.html")
CACHE = Path("/tmp/claude-2107/-workspace-vast-jbauer-activation-oracles-dev/29d76a9f-ac8a-4212-8cc9-989545728103/scratchpad/symbol_arithmetic_payload.json")


def load_payload():
    if CACHE.exists():
        return json.load(open(CACHE))
    txt = HTML.read_text()
    i = txt.index("window.__DATA__ = ") + len("window.__DATA__ = ")
    data, _ = json.JSONDecoder().raw_decode(txt[i:])
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    json.dump(data, open(CACHE, "w"))
    return data


if __name__ == "__main__":
    d = load_payload()
    print(list(d.keys()))
    for dom, t in d["doms"].items():
        print(dom, t["label"], "eps0", t["eps0"], "states", len(t["states"]), "cells", len(t["cells"]),
              "ks", t["ks"], "nlvl", t["nlvl"])
