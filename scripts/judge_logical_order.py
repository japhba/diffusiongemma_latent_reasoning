"""LLM-judged logical ordering of content atoms, for the causality figure overlay.

For each captured GPQA / poem rollout (full texts in src_data/planning/full_texts.json):
the judge splits the text into atoms (in surface order) and then states the LOGICAL
DERIVATION ORDER of those atoms as an ordered list of groups (ties allowed: atoms in one
group are mutually order-independent). rho_logic = tie-aware Spearman(surface position,
group rank): +1 = the content is left-to-right inductive, ~0 = order-indifferent,
-1 = right-to-left. All atoms in a single group (fully independent) => rho_logic = 0.

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
ENV = Path(os.environ.get("DGLR_ENV_FILE", Path(__file__).resolve().parent.parent / ".env"))
if ENV.exists():  # NODEV_URL/NODEV_KEY may also come straight from the environment
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
2. Then give the LOGICAL DERIVATION ORDER of these atoms: the order in which their content must be fixed or derived, where an atom that builds on another comes in a later group. Express it as an ordered list of groups of atom indices; atoms within one group are mutually order-independent (could be derived in any order). If all atoms are independent, use a single group. Judge the content's own logic, not the reading order.
Respond with ONLY JSON: {{"atoms": [{{"i": 0, "text": "..."}}, ...], "order": [[indices of group 1], [indices of group 2], ...], "justification": "one sentence"}}"""

def rankdata(v):
    v = np.asarray(v, float); order = np.argsort(v); ranks = np.empty(len(v)); sv = v[order]; i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks

def group_ranks(order, n):
    rank = [None] * n
    for g, group in enumerate(order):
        for i in group:
            if isinstance(i, int) and 0 <= i < n and rank[i] is None:
                rank[i] = g
    assert all(r is not None for r in rank), "order does not cover all atoms"
    return rank

def parse_json(raw):
    m = re.search(r"\{.*\}", raw, re.S)
    s = m.group(0)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # GPQA texts carry LaTeX; the judge often quotes them with unescaped backslashes
        return json.loads(re.sub(r'\\(?![\\/"bfnrtu])', r"\\\\", s))

texts = json.load(open(ROOT / "src_data" / "planning" / "full_texts.json"))
out = {}
for key, rec in texts.items():
    msgs = [{"role": "user", "content": PROMPT.format(task=rec["prompt"], text=rec["text"])}]
    raw = chat(msgs)
    try:
        try:
            j = parse_json(raw)
        except Exception:
            raw = chat(msgs + [{"role": "assistant", "content": raw},
                               {"role": "user", "content": "Your JSON was invalid. Respond again with STRICTLY valid JSON only — escape every backslash as \\\\."}])
            j = parse_json(raw)
        atoms = j["atoms"]
        rank = group_ranks(j["order"], len(atoms))
        if len(set(rank)) < 2:
            rho = 0.0
        else:
            rho = float(np.corrcoef(rankdata(range(len(atoms))), rankdata(rank))[0, 1])
        out[key] = dict(bench=rec["bench"], pid=rec["pid"], n_atoms=len(atoms), rank=rank,
                        rho_logic=round(rho, 3), justification=j.get("justification", ""),
                        atoms=[a.get("text", "")[:80] for a in atoms], order=j["order"])
        print(f"{key}: atoms {len(atoms)} rho_logic {rho:+.2f}")
    except Exception as e:
        print(f"{key}: SKIP ({e})")
json.dump(out, open(ROOT / "src_data" / "planning" / "judged_logical_order.json", "w"), indent=1)
skips = len(texts) - len(out)
assert skips <= max(1, len(texts) // 10), f"too many judge failures: {skips}"
for b in ("gpqa", "poem"):
    rhos = [v["rho_logic"] for v in out.values() if v["bench"] == b]
    print(f"{b}: n={len(rhos)} median rho_logic {np.median(rhos):+.2f}  [{min(rhos):+.2f},{max(rhos):+.2f}]")
