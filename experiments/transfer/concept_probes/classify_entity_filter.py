"""Classify each SAE-Probes concept: does its positive class probe for a SINGLE ENTITY
(one specific named person/place/organization/lexical item/category value) or a general
property/topic/relation? Used to filter the probe-section headline to non-single-entity
concepts. LLM = gemini-3-flash-preview via OpenRouter (high concurrency OK), plus a
deterministic rule list for review. Writes out/saeprobes/concept_entity_filter.json."""

import json, os, re, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from dotenv import load_dotenv
from tqdm.auto import tqdm

from saeprobes_data import load_datasets

load_dotenv(Path(__file__).parent.parent / ".env")
OUT = Path(__file__).parent / "out/saeprobes"
KEY = os.environ["OPENROUTER_API_KEY"]
MODEL = "google/gemini-3-flash-preview"

RUBRIC = """You classify binary text-classification probe datasets.

Question: is the dataset's POSITIVE class defined by ONE SINGLE SPECIFIC ENTITY — i.e. one
named person, one place (state/city/borough/country/timezone city), one organization or party,
one specific lexical item/bigram, one specific disease (not a disease family), one specific
sport, one specific programming language, one specific occupation, or one specific
multiple-choice letter?

single_entity = true examples: "is this headline about Trump", "is this text about Florida",
"does this text contain the bigram 'credit card'", "is this abstract about Thyroid Cancer",
"does this athlete play football", "is this code Python", "is this person a journalist",
"is the answer choice A".

single_entity = false examples: general properties, qualities, relations or broad topics —
truthfulness, sentiment, toxicity, spam, grammaticality, entailment/paraphrase, being male,
being alive, broad topical classes (Politics, Technology, World news), emotion (sadness),
reasoning-category datasets, disease FAMILIES (cardiovascular diseases), AI-generated-text
detection, clickbait, media format (book vs song vs movie), support-ticket category.

Respond with EXACT JSON only: {"single_entity": true/false, "entity": "<the entity or null>",
"why": "<one sentence>"}"""


def ask(d):
    ex1 = [t[:300] for t, y in zip(d["texts_train"], d["y_train"]) if y == 1][:2]
    ex0 = [t[:300] for t, y in zip(d["texts_train"], d["y_train"]) if y == 0][:2]
    user = (f"Dataset tag: {d['tag']}\nLabel for y=1: {d['label_y1']}\nLabel for y=0: {d['label_y0']}\n"
            f"Examples labelled {d['label_y1']}:\n- " + "\n- ".join(ex1) +
            f"\nExamples labelled {d['label_y0']}:\n- " + "\n- ".join(ex0))
    for attempt in range(5):
        try:
            r = httpx.post("https://openrouter.ai/api/v1/chat/completions",
                           headers={"Authorization": f"Bearer {KEY}"},
                           json={"model": MODEL, "max_tokens": 1500,
                                 "reasoning": {"effort": "low"},
                                 "messages": [{"role": "system", "content": RUBRIC},
                                              {"role": "user", "content": user}]},
                           timeout=120)
            r.raise_for_status()
            txt = r.json()["choices"][0]["message"]["content"]
            m = re.search(r"\{.*\}", txt, re.S)
            return d["tag"], json.loads(m.group(0))
        except Exception as e:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


def main():
    ds = load_datasets()
    with ThreadPoolExecutor(16) as ex:
        res = dict(tqdm(ex.map(ask, ds), total=len(ds)))
    n1 = sum(v["single_entity"] for v in res.values())
    print(f"[entity_filter] {n1}/{len(res)} single-entity, {len(res) - n1} survive")
    for tag in sorted(res, key=lambda t: int(t.split('_')[0])):
        v = res[tag]
        print(f"  {'ENTITY ' if v['single_entity'] else 'keep   '} {tag:45s} {v.get('entity')}: {v['why']}")
    (OUT / "concept_entity_filter.json").write_text(json.dumps(res, indent=1))
    print("wrote concept_entity_filter.json")


if __name__ == "__main__":
    main()
