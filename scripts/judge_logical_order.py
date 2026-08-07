"""LLM-judged logical ordering of content atoms, for the causality figure overlay.

For each captured GPQA / poem rollout (full texts in src_data/planning/full_texts.json):
the judge splits the text into atoms (in surface order) and lists, per atom, which other
atoms it logically DEPENDS on (whose content must be fixed before it is determined).
We compute each atom's derivation depth (longest dependency path) and
rho_logic = tie-aware Spearman(surface index, depth): +1 = the content itself is
left-to-right inductive, ~0 = atoms independent, -1 = right-to-left.
Zero-variance depths (all atoms independent) => rho_logic = 0.

Judge: Node V (self-hosted, OpenAI-compatible; NODEV_URL/NODEV_KEY from the repo .env),
thinking disabled, temperature 0. Sequential calls. Output (with per-text justification):
src_data/planning/judged_logical_order.json.
"""
import json
import os
import re
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ENV = Path("/workspace-vast/jbauer/activation_oracles_dev/.env")
for line in ENV.read_text().splitlines():
    m = re.match(r"^(NODEV_URL|NODEV_KEY|NODEV_MODEL)=(.*)$", line.strip())
    if m:
        os.environ.setdefault(m.group(1), m.group(2).strip('"'))
URL, KEY = os.environ["NODEV_URL"], os.environ["NODEV_KEY"]
MODEL = os.environ.get("NODEV_MODEL", "")

def chat(messages):
    body = dict(model=MODEL, messages=messages, temperature=0.0, max_tokens=2000,
                chat_template_kwargs={"enable_thinking": False})
    req = urllib.request.Request(f"{URL}/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json",
                                  "Authorization": f"Bearer {KEY}",
                                  "User-Agent": "curl/8.5.0"})  # RunPod proxy 403s Python-urllib
    return json.loads(urllib.request.urlopen(req, timeout=300).read())["choices"][0]["message"]["content"]

if not MODEL:
    req = urllib.request.Request(f"{URL}/models", headers={"Authorization": f"Bearer {KEY}", "User-Agent": "curl/8.5.0"})
    MODEL = json.loads(urllib.request.urlopen(req, timeout=30).read())["data"][0]["id"]
print("judge model:", MODEL)

PROMPT = """A language model was given this task:
---
{task}
---
and produced this text:
---
{text}
---
1. Split the text into its atomic content units ("atoms") — reasoning steps, claims, lines of verse, or items — in the order they appear (at most 12; merge trivial fragments).
2. For each atom, list which OTHER atoms it logically depends on: atoms whose content must already be fixed or derived for this atom's content to be determined. A computation step depends on the values it uses; a conclusion depends on the premises it draws on; an item that could be swapped or written independently depends on nothing. Judge the content's own logic, not the reading order.
Respond with ONLY JSON: {{"atoms": [{{"i": 0, "text": "...", "depends_on": []}}, ...], "justification": "one sentence"}}"""

def rankdata(v):
    v = np.asarray(v, float); order = np.argsort(v); ranks = np.empty(len(v)); sv = v[order]; i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks

def depth_of(atoms):
    memo = {}
    def d(i):
        if i in memo:
            return memo[i]
        memo[i] = 0
        deps = [x for x in atoms[i].get("depends_on", []) if 0 <= x < len(atoms) and x != i]
        memo[i] = 1 + max((d(x) for x in deps), default=-1)
        return memo[i]
    return [d(i) for i in range(len(atoms))]

texts = json.load(open(ROOT / "src_data" / "planning" / "full_texts.json"))
out = {}
for key, rec in texts.items():
    raw = chat([{"role": "user", "content": PROMPT.format(task=rec["prompt"], text=rec["text"])}])
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        j = json.loads(m.group(0))
        atoms = j["atoms"]
        depth = depth_of(atoms)
        if len(set(depth)) < 2:
            rho = 0.0
        else:
            rho = float(np.corrcoef(rankdata(range(len(atoms))), rankdata(depth))[0, 1])
        out[key] = dict(bench=rec["bench"], pid=rec["pid"], n_atoms=len(atoms), depth=depth,
                        rho_logic=round(rho, 3), justification=j.get("justification", ""),
                        atoms=[a.get("text", "")[:80] for a in atoms],
                        depends_on=[a.get("depends_on", []) for a in atoms])
        print(f"{key}: atoms {len(atoms)} rho_logic {rho:+.2f}")
    except Exception as e:
        print(f"{key}: SKIP ({e})")
json.dump(out, open(ROOT / "src_data" / "planning" / "judged_logical_order.json", "w"), indent=1)
skips = len(texts) - len(out)
assert skips <= max(1, len(texts) // 10), f"too many judge failures: {skips}"
for b in ("gpqa", "poem"):
    rhos = [v["rho_logic"] for v in out.values() if v["bench"] == b]
    print(f"{b}: n={len(rhos)} median rho_logic {np.median(rhos):+.2f}  [{min(rhos):+.2f},{max(rhos):+.2f}]")
