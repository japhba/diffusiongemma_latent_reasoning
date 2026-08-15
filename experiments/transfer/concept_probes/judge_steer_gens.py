"""Judge the steered generations (out/saeprobes/generations.json) for concept expression.

Needs an OpenAI-compatible judge endpoint: set NODEV_URL to the full
/v1/chat/completions URL and NODEV_KEY to its bearer key (originally
Qwen/Qwen3.6-35B-A3B-FP8 served via vLLM).

Per generation we ask for Trueness and Precision (0.0-1.0, 0.1 granularity):
  Trueness  — does the text actually exhibit the concept?
  Precision — is the text still coherent, on-prompt language (not degenerate)?
plus a one-line digest for the report. Steering success per (concept, arm) =
mean Trueness(+steer) − mean Trueness(−steer).

  python concept_probes/judge_steer_gens.py
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path

import pandas as pd
import requests

REPO = Path(os.environ.get("DGLR_ROOT", Path(__file__).resolve().parents[1]))
OUT = REPO / "concept_probes/out/saeprobes"
JUDGE_URL = os.environ.get("NODEV_URL")  # full OpenAI-compatible /v1/chat/completions URL
JUDGE_MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"
HEADERS = {"Authorization": "Bearer " + os.environ.get("NODEV_KEY", "")}


# The master sheet's Name/Probe-to columns are too thin for several families (agnews
# rows literally say "SAE BENCH"; world_country/us_state omit WHICH entity) — the
# judge needs a real description of the semantic positive class.
DESC_OVERRIDES = {
    "36_sciq_tf": "given science answer is factually correct (POS) vs incorrect (NEG)",
    "41_truthqa_tf": "given answer is truthful/correct (POS) vs false or misleading (NEG)",
    "42_temp_sense": "question–answer pair is temporally plausible (POS) vs implausible (NEG)",
    "44_phys_tf": "candidate physical-reasoning answer is correct (POS) vs incorrect (NEG)",
    "47_reasoning_tf": "candidate reasoning answer is correct (POS) vs incorrect (NEG)",
    "48_cm_correct": "behavior is morally wrong/unacceptable (POS) vs acceptable (NEG); Common Sense is the source family",
    "49_cm_isshort": "text is short (POS) vs a long narrative (NEG); Common Sense is the source family, not the target",
    "50_deon_isvalid": "duty or excuse pairing is valid (POS) vs invalid (NEG); Deontology is the source family",
    "51_just_is": "stated justification is sensible/valid (POS) vs unjustified or illogical (NEG); Justice is the source family",
    "52_virtue_is": "trait label fits the described behavior (POS) vs mismatches it (NEG); Virtue is the source family",
    "54_cs_tf": "commonsense statement is true (POS) vs false (NEG)",
    "90_glue_qnli": "the passage entails/answers the question (POS) vs does not entail or answer it (NEG)",
    "56_wikidatasex_or_gender": "described person is male (POS) vs female (NEG)",
    "57_wikidatais_alive": "described person is alive (POS) vs deceased (NEG)",
    "58_wikidatapolitical_party": "described politician is Republican (POS) vs Democratic (NEG)",
    "129_arith_mc_A": "the correct arithmetic answer is option A (POS) vs another option (NEG)",
    "157_amazon_5star": "Amazon review has a 5-star rating (POS) vs a 1-star rating (NEG); star rating is hidden metadata inferred from the review text",
    "161_agnews_0": "general/world news text (POS) vs source-code or technical-documentation text (NEG)",
    "162_agnews_1": "sports news text (POS) vs source-code or technical-documentation text (NEG)",
    "163_agnews_2": "business/financial news text (POS) vs source-code or technical-documentation text (NEG)",
    "124_world_country_United_States": "content about the United States (US places, cities, institutions)",
    "125_world_country_Italy": "content about Italy (Italian places, cities, culture)",
    "123_world_country_United_Kingdom": "content about the United Kingdom (British places, cities)",
    "119_us_state_TX": "content about Texas (Texan places, cities)",
    "117_us_state_FL": "content about Florida (Floridian places, cities)",
    "118_us_state_CA": "content about California (Californian places, cities)",
    "155_athlete_sport_basketball": "basketball (players, teams, the sport)",
    "154_athlete_sport_football": "football (players, teams, the sport)",
    "156_athlete_sport_baseball": "baseball (players, teams, the sport)",
    "105_click_bait": "clickbait style (sensationalist teaser headlines/content)",
    "96_spam_is": "spam content (unsolicited promotional/scam messages)",
    "110_aimade_humangpt3": "AI-generated text (as opposed to human-written)",
    "77_second-derivative": "the mathematical concept of the second derivative",
    "5_hist_fig_ismale": "a male historical figure",
    "6_hist_fig_isamerican": "an American historical figure",
    "7_hist_fig_ispolitician": "a politician",
    "159_code_Python": "Python source code",
    "139_news_class_Politics": "political news content (government, elections, policy)",
    "140_news_class_Technology": "business/financial news content (POS) vs other news categories (NEG); the upstream category name is stale",
    "141_news_class_Entertainment": "technology/science news content (POS) vs other news categories (NEG); the upstream category name is stale",
    "100_news_fake": "fake/alternative-media news (POS) vs real/mainstream Reuters news (NEG)",
    "113_movie_sent": "positive movie review sentiment",
    "121_us_timezone_New_York": "a US place in the Eastern time-zone region (POS) vs another US time-zone region (NEG)",
    "120_us_timezone_Chicago": "a US place in the Central time-zone region (POS) vs another US time-zone region (NEG)",
    "122_us_timezone_Los_Angeles": "a US place in the Pacific time-zone region (POS) vs another US time-zone region (NEG)",
    "131_temp_cat_Typical Time": "a question asking when/what time an event occurs (POS) vs one asking duration, frequency, or event ordering (NEG)",
    "143_cancer_cat_Lung_Cancer": "lung cancer (medical content)",
    "23_headline_ischina": "content about China",
    "149_twt_emotion_happiness": "happy/joyful emotional content",
    "95_toxic_is": "toxic/offensive content",
    "107_hate_offensive": "offensive/profane speech (POS) vs non-offensive speech (NEG)",
    "62_wikidata_occupation_ispolitician": "a politician (person description)",
    "94_ai_gen": "AI-generated text (as opposed to human-written)",
}


def _family_desc(tag: str) -> str | None:
    """Programmatic description for tag families the master sheet describes poorly."""
    parts = tag.split("_")
    try:
        num = int(parts[0])
    except ValueError:
        return None
    rest = "_".join(parts[1:])
    if 65 <= num <= 85:  # Gurnee context-bigram sets, tag = the phrase
        return f"content involving the phrase/topic '{rest.replace('-', ' ')}'"
    fam_rules = [
        ("nyc_borough_", lambda x: f"content about {x} (New York City borough)"),
        ("us_state_", lambda x: f"content about the US state {x}"),
        ("us_timezone_", lambda x: f"content about the {x.replace('_', ' ')} region of the US"),
        ("world_country_", lambda x: f"content about {x.replace('_', ' ')}"),
        ("art_type_", lambda x: f"a {x} (as a type of artwork/media)"),
        ("temp_cat_", lambda x: f"temporal reasoning about {x.replace('_', ' ').lower()}"),
        ("context_type_", lambda x: f"{x.replace('_', ' ').lower()} reasoning content"),
        ("glue_mnli_", lambda x: f"a premise–hypothesis pair in an {x} relation"),
        ("news_class_", lambda x: f"{x.lower()} news content"),
        ("cancer_cat_", lambda x: f"medical content about {x.replace('_', ' ').lower()}"),
        ("disease_class_", lambda x: f"medical content about {x.replace('_', ' ').lower()}"),
        ("twt_emotion_", lambda x: f"{x} emotional content"),
        ("it_tick_", lambda x: f"an IT support request about {x.replace('_', ' ').lower()}"),
        ("athlete_sport_", lambda x: f"{x} (players, teams, the sport)"),
        ("code_", lambda x: f"{x} source code"),
        ("headline_is", lambda x: f"a news headline about {x}"),
        ("wikidata_occupation_is", lambda x: f"a person who is a {x}"),
    ]
    for prefix, fn in fam_rules:
        if rest.startswith(prefix):
            return fn(rest[len(prefix):])
    return None


def concept_descriptions() -> dict[str, str]:
    master = pd.read_csv(REPO / "third_party/SAE-Probes/data/probing_datasets_MASTER.csv")
    master = master[master["Data type"] == "Binary Classification"]
    desc = {}
    name = None
    for _, r in master.iterrows():
        name = r["Dataset name"] if isinstance(r["Dataset name"], str) else name
        tag = Path(str(r["Dataset save name"])).stem
        probe_to = r["Probe to"] if isinstance(r["Probe to"], str) else ""
        desc[tag] = f"{name} — {probe_to}".strip(" —")
        fam = _family_desc(tag)
        if fam:
            desc[tag] = fam
    desc.update(DESC_OVERRIDES)
    # the original designed battery (stimuli.json) carries its own descriptions;
    # its keys (e.g. "mentions_food_dish") don't collide with the numbered SAE tags
    stim_path = REPO / "concept_probes/stimuli.json"
    if stim_path.exists():
        for c in json.loads(stim_path.read_text()):
            desc.setdefault(c["key"], c["description"])
    # extra batteries beyond SAE-Probes carry their own {tag: description} maps
    for extra in ["out/saeprobes/ae_desc.json", "out/saeprobes/repeown_desc.json",
                  "out/saeprobes/div_desc.json", "out/saeprobes/dom_desc.json",
                  "out/saeprobes/repe_ctrl_desc.json"]:
        p = REPO / "concept_probes" / extra
        if p.exists():
            desc.update(json.loads(p.read_text()))
    return desc


def judge_prompt(concept: str, text: str) -> str:
    """The canonical judge prompt — shared verbatim by every judge backend (Node V, OpenRouter)."""
    return f"""You are scoring a text generated by a language model that was steered toward (or away from) an internal concept direction.

Concept: "{concept}"
Generated text:
---
{text}
---

Score two things on a 0.0-1.0 scale with 0.1 granularity:
- trueness: to what degree does the text exhibit/express the concept? (1.0 = clearly and strongly present, 0.0 = absent)
- precision: is the text coherent, fluent language (regardless of the concept)? (1.0 = fully coherent, 0.0 = degenerate/garbled)

Also give a one-line digest (<=15 words) of what the text says, and a one-line justification
(<=25 words) explaining WHY you chose these trueness/precision scores (cite concrete evidence
from the text).
Answer with EXACTLY this JSON: {{"trueness": <float>, "precision": <float>, "digest": "<one line>", "justification": "<one line>"}}"""


def judge(concept: str, text: str) -> dict:
    assert JUDGE_URL and os.environ.get("NODEV_KEY"), "set NODEV_URL (full /v1/chat/completions URL) and NODEV_KEY"
    prompt = judge_prompt(concept, text)
    r = requests.post(JUDGE_URL, headers=HEADERS, timeout=180, json={
        "model": JUDGE_MODEL, "temperature": 0.0, "max_tokens": 200,
        "messages": [{"role": "user", "content": prompt}]})
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"]
    m = re.search(r"\{.*\}", content, re.DOTALL)
    return json.loads(m.group(0))


def main():
    gens = json.loads((OUT / "generations.json").read_text())
    desc = concept_descriptions()
    jobs = []  # (tag, run_idx, field, concept_desc, text)
    for tag, rec in gens.items():
        cdesc = desc.get(tag, tag)
        for i, run in enumerate(rec["runs"]):
            for field, text in run.items():
                if field == "prompt" or not isinstance(text, str) or not text.strip():
                    continue
                jobs.append((tag, i, field, cdesc, text))
    print(f"[judge] {len(jobs)} generations to score")

    def work(job):
        tag, i, field, cdesc, text = job
        try:
            return (tag, i, field, judge(cdesc, text))
        except Exception as e:
            return (tag, i, field, {"error": str(e)})

    with ThreadPoolExecutor(max_workers=16) as ex:
        scored = list(ex.map(work, jobs))
    n_err = sum(1 for *_, s in scored if "error" in s)
    assert n_err / max(len(scored), 1) < 0.05, f"{n_err}/{len(scored)} judge calls failed"

    out = {}
    for tag, i, field, s in scored:
        out.setdefault(tag, {}).setdefault(str(i), {})[field] = s

    # steering success per (concept, arm): mean trueness(+1) - mean trueness(-1)
    summary = {}
    for tag, runs in out.items():
        arms = {}
        for arm in ["gemma_native", "dg_native", "dg_transfer"]:
            tp = [r[f"{arm}_pos"]["trueness"] for r in runs.values()
                  if f"{arm}_pos" in r and "trueness" in r[f"{arm}_pos"]]
            tn = [r[f"{arm}_neg"]["trueness"] for r in runs.values()
                  if f"{arm}_neg" in r and "trueness" in r[f"{arm}_neg"]]
            tb = [r[f"{arm}_base"]["trueness"] for r in runs.values()
                  if f"{arm}_base" in r and "trueness" in r[f"{arm}_base"]]
            pp = [r[f"{arm}_pos"]["precision"] for r in runs.values()
                  if f"{arm}_pos" in r and "precision" in r[f"{arm}_pos"]]
            if tp and tn:
                arms[arm] = {"delta_trueness": round(sum(tp) / len(tp) - sum(tn) / len(tn), 3),
                             "base_trueness": round(sum(tb) / len(tb), 3) if tb else None,
                             "pos_precision": round(sum(pp) / len(pp), 3)}
        summary[tag] = arms
        print(f"[judge] {tag:<40} " + "  ".join(
            f"{a}:dT={v['delta_trueness']:+.2f}(prec {v['pos_precision']:.2f})" for a, v in arms.items()))

    (OUT / "judged_generations.json").write_text(json.dumps(
        {"scores": out, "summary": summary, "descriptions": desc}, indent=1))
    print(f"[judge] wrote {OUT / 'judged_generations.json'}")


if __name__ == "__main__":
    main()
