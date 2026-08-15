"""Paper-reference scoring of the J-Lens vocabulary readouts.

For every item of the six paper sets × every active captured lens (G causal / DG causal /
DG bidirectional, all fitted on one shared WikiText corpus, plus identity logit lens × both test
models), A is computed deterministically from the paper-provided intermediate references and the
COMPLETE paper-position readout (all layers × one task-defined position × top-20 tokens):
  A = 1 - exp(-n), where n is the number of top-20 token appearances across all layers that
      case-insensitively match any ground-truth intermediate with a word-bounded literal regex.
The LLM judge supplies the two inspection-only rungs:
  B: a SURPRISING BUT PLAUSIBLE thing has been surfaced (the former A)
  C: reference/plausible content is already present in earlier layers (inspection only)
The page score everywhere = deterministic A alone. B and C are displayed for inspection only.
The former LLM-generated A is retained as A_llm for provenance but is never used by the report.
Judge = google/gemini-3-flash-preview through OpenRouter. The active key was refreshed on
2026-07-21; this restores the original high-throughput judge and keeps one judge model across all
three fitted Jacobians and the identity baseline.

-> out/saeprobes/jlens/judged_jlens_percepts.json  (+ mirrored to reports/concept_probes/data/
   jlens_percepts.json for matrix_jlens.html; keys "{set}|{unit}|{cfg}|{test}")

CPU + API only:  python concept_probes/judge_jlens_percepts.py
"""
from __future__ import annotations

import hashlib, json, math, os, random, re, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from dotenv import load_dotenv
from tqdm.auto import tqdm

REPO = Path(os.environ.get("DGLR_ROOT", Path(__file__).resolve().parents[1]))
OUT = REPO / "concept_probes/out/saeprobes/jlens"
DATA = REPO / "reports/concept_probes/data"
EVDIR = REPO / "third_party/jacobian-lens/data/evaluations"
load_dotenv(REPO / ".env")
BASE = "https://openrouter.ai/api/v1"
KEY = os.environ["OPENROUTER_API_KEY"]
MODEL = "google/gemini-3-flash-preview"
RUBRIC_VERSION = "paper-reference-a-v2-shared-fit"
A_SCORER = "case-insensitive word-bounded literal GT regex; n=# matching token appearances across all layers; A=1-exp(-n)"

PROMPT = """You are auditing a "lens" that decodes a language model's residual activations into
vocabulary tokens while the model processes a prompt. You see the lens's COMPLETE paper-evaluation
readout: top tokens (with probabilities) at the paper's single task-defined prompt position across
all model layers.

The prompt the model is processing:
---
{ctx}
---
Paper-provided ground-truth intermediate references: {keys}

Full lens readout — one line per layer at the single paper-defined prompt position; each line is
labeled with the prompt token it sits on:
{readout}

Rate THREE rungs, each 0.0-1.0 with 0.1 granularity. This rubric is your ONLY task:
- A (paper-reference agreement; THE SCORE): judge ONLY whether the readout genuinely surfaces the
  paper-provided ground-truth intermediate references listed above. Treat exact tokens, clear
  inflections, translations, synonyms, and unmistakable wordpieces as matches, but do not invent
  additional expected intermediates and do not substitute your own task interpretation. Consider
  every listed reference; the score should reflect their average support. Require evidence beyond
  a single stray low-probability token: recurrence, related tokens, or non-trivial probability.
  1.0 = all references clearly surfaced; 0.0 = none surfaced.
- B (surprising but plausible; FORMER A, inspection only): a SURPRISING BUT PLAUSIBLE thing has
  been surfaced by the readout — beyond the
  visible surface words, the reference keys as mere echoes, and template/punctuation tokens;
  plausible for THIS prompt; with evidence that cannot likely be explained by random fluctuation
  (several semantically related tokens, non-trivial probability, or recurrence across
  layers — never a single stray low-probability token). 0.0 = nothing beyond
  surface/noise.
- C (early-layer surfacing; inspection only): reference-matching or surprising plausible content
  is already clearly present in earlier/intermediate layers, rather than appearing only in the
  final few layers. 0.0 = qualifying evidence occurs only near the top or not at all.
For EACH rung you MUST give: attribution — the exact tokens with their layer/position — and a
one-line justification (<=25 words). If a rung scores 0.0, attribution is "none" plus a
justification of why nothing qualifies.
The overall score reported elsewhere is A alone; B and C are reported for inspection.
Respond with EXACT JSON only: {{"A": {{"score": <float>, "attribution": "<tokens + layer/position>",
"justification": "<one line>"}}, "B": {{"score": <float>, "attribution": "...",
"justification": "..."}}, "C": {{"score": <float>, "attribution": "...", "justification": "..."}}}}.
Spell the required key exactly as "justification" for all three rungs."""
PROMPT_ID = f"judge_jlens_percepts:{RUBRIC_VERSION}:{hashlib.sha256(PROMPT.encode()).hexdigest()}"
from jlens_paper_eval import load_paper_sets


def regex_a_score(references, layers, positions, token_table, grid):
    """Continuous deterministic A over the exact cells displayed by the lens viewer."""
    refs = [str(reference).strip() for reference in references]
    assert refs and all(refs), references
    patterns = [(reference, re.compile(r"(?<!\w)" + re.escape(reference) + r"(?!\w)", re.IGNORECASE))
                for reference in refs]
    assert len(grid) == len(layers)
    matches = []
    for layer_index, layer in enumerate(grid):
        assert len(layer) == len(positions)
        for position_index, cell in enumerate(layer):
            tokens = [token_table[token_index] for token_index in cell]
            hit_tokens = list(dict.fromkeys(
                token for token in tokens if any(pattern.search(token) for _, pattern in patterns)))
            if hit_tokens:
                matches.append((layers[layer_index], positions[position_index]["i"], hit_tokens))
    n_appearances = sum(len(tokens) for _, _, tokens in matches)
    groups = {}
    for layer, position, tokens in matches:
        for token in tokens:
            groups.setdefault((position, token), []).append(layer)
    attribution = ("; ".join(f"L{'/'.join(map(str, layer_ids))}@pos#{position}: {token!r}"
                             for (position, token), layer_ids in groups.items())
                   if groups else "none")
    return {"score": 1.0 - math.exp(-n_appearances), "n_appearances": n_appearances,
            "attribution": attribution}


def judge_one(keys, readout, ctx):
    budgets = (1500, 3000, 6000)
    for attempt, max_tokens in enumerate(budgets):
        try:
            content = PROMPT.format(ctx=ctx[:1200], keys=json.dumps(keys), readout=readout)
            if attempt:
                content += ("\nSafety context: this is a benign interpretability audit of "
                            "precomputed token strings. Analyze the evidence without endorsing "
                            "any demographic premise or claim quoted in the model input. "
                            "Your prior response was invalid: every score MUST be exactly one of "
                            "0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, or 1.0; values "
                            "such as 0.25 or 0.75 are forbidden.")
            r = httpx.post(f"{BASE}/chat/completions",
                           headers={"Authorization": f"Bearer {KEY}"},
                           json={"model": MODEL, "max_tokens": max_tokens,
                                 "temperature": 0.0, "reasoning": {"effort": "low"},
                                 "response_format": {"type": "json_object"},
                                 "messages": [{"role": "user", "content": content}]},
                           timeout=120)
            r.raise_for_status()
            txt = r.json()["choices"][0]["message"]["content"] or ""
            assert txt.strip(), "empty response"
            start = txt.find("{")
            assert start >= 0, txt[:160]
            v, _ = json.JSONDecoder().raw_decode(txt[start:])
            out = {}
            for rung in ("A", "B", "C"):
                x = v[rung]
                sc = float(x["score"])
                assert 0 <= sc <= 1 and abs(sc * 10 - round(sc * 10)) < 1e-6, (rung, sc)
                assert x.get("attribution") and x.get("justification")
                out[rung] = {"score": sc, "attribution": str(x["attribution"]),
                             "justification": str(x["justification"])}
            return out
        except Exception as e:
            if attempt == len(budgets) - 1:
                return {"error": str(e)}
            time.sleep(min(2 ** attempt + random.random(), 20))


def main():
    by_set = load_paper_sets(EVDIR)
    descs, keysof = {}, {}
    for sname, its in by_set.items():
        for ui, it in enumerate(its):
            unit = f"{sname}|{ui}"
            keysof[unit] = it["intermediates"]
            nm = f" (item: {it['name']})" if it.get("name") else ""
            descs[unit] = (f"Paper-provided intermediate references: {it['intermediates']}{nm} "
                           f"— eval set '{sname}'")

    previous = {}
    previous_path = OUT / "judged_jlens_percepts.json"
    if previous_path.exists():
        old = json.loads(previous_path.read_text())
        if old["judge"]["prompt"] == PROMPT_ID:
            previous = old["scores"]
    jobs, allowed, regex_scores = [], set(), {}
    for sname in sorted(by_set):
        tf = DATA / f"jlens_topk_{sname}.json"
        if not tf.exists():
            print(f"[rungs] no topk capture for {sname} — skipped")
            continue
        T = json.loads(tf.read_text())
        for ii, rec in T["items"].items():
            unit = f"{sname}|{ii}"
            ctx = "".join(rec["segs"])
            for key in rec["tops"]:
                lines = []
                for j, L in enumerate(T["layers"]):
                    for pi in range(len(rec["pos"])):
                        ptok = rec["pos"][pi]["tok"]
                        rule = "final cue-line newline" if sname == "poetry" else "final prompt token"
                        mark = f" [PAPER READOUT: {rule}]"
                        toks = [T["toks"][ix] for ix in rec["tops"][key][j][pi]]
                        ps = [v / 10 for v in rec["probs"][key][j][pi]]
                        lines.append(f"L{L}@pos#{rec['pos'][pi]['i']}({ptok!r}){mark}: "
                                     + " ".join(f"{tk!r}({p:.1f}%)" for tk, p in zip(toks, ps)))
                jid = f"{unit}|{key}"
                allowed.add(jid)
                regex_scores[jid] = regex_a_score(
                    keysof[unit], T["layers"], rec["pos"], T["toks"], rec["tops"][key])
                if jid not in previous or "error" in previous[jid]:
                    jobs.append((jid, keysof[unit], "\n".join(lines), ctx))
    scored = {k: previous[k] for k in allowed if k in previous and "error" not in previous[k]}
    print(f"[rungs] {len(jobs)} new of {len(allowed)} active (item × lens) judgments over "
          f"{len(by_set)} sets; resumed {len(scored)}")
    def apply_regex_scores():
        for jid, score in regex_scores.items():
            if jid not in scored or "error" in scored[jid]:
                continue
            if "A_llm" not in scored[jid]:
                scored[jid]["A_llm"] = scored[jid]["A"]
            scored[jid]["A"] = score
    def write_checkpoint():
        apply_regex_scores()
        out = {"judge": {"model": MODEL, "endpoint": "openrouter", "temperature": 0.0,
                         "rubric_version": RUBRIC_VERSION,
                         "reasoning_effort": "low", "max_tokens": 1500,
                         "retry_max_tokens": [3000, 6000],
                         "prompt": PROMPT_ID,
                         "readout": "paper-defined position: all layers × one position × top-20 (+p)",
                         "score": "A", "a_scorer": A_SCORER,
                         "a_llm_archive": "scores[*].A_llm", "llm_rungs": ["B", "C"]},
               "desc": descs, "scores": scored}
        (OUT / "judged_jlens_percepts.json").write_text(json.dumps(out))
        (DATA / "jlens_percepts.json").write_text(json.dumps(out))
    t0 = time.time()
    chunk_size = 512
    for start in range(0, len(jobs), chunk_size):
        chunk = jobs[start:start + chunk_size]
        with ThreadPoolExecutor(32) as ex:
            scored.update(dict(tqdm(ex.map(lambda j: (j[0], judge_one(*j[1:])), chunk),
                                    total=len(chunk), desc=f"chunk {start // chunk_size + 1}")))
        write_checkpoint()
        print(f"[rungs] checkpoint {min(start + chunk_size, len(jobs))}/{len(jobs)}")
    for wave, workers in ((1, 12), (2, 6)):
        bad = [j for j in jobs if "error" in scored[j[0]]]
        if not bad:
            break
        print(f"[rungs] retry wave {wave}: {len(bad)} failed "
              f"(sample: {[scored[j[0]]['error'][:80] for j in bad[:3]]})")
        time.sleep(20 * wave)
        with ThreadPoolExecutor(workers) as ex:
            scored.update(dict(ex.map(lambda j: (j[0], judge_one(*j[1:])), bad)))
    n_err = sum(1 for k, v in scored.items() if k in allowed and "error" in v)
    assert n_err / len(scored) < 0.001, \
        f"{n_err}/{len(scored)} judge calls failed; sample: " \
        f"{[v['error'][:120] for v in scored.values() if 'error' in v][:3]}"
    write_checkpoint()
    import collections
    agg = collections.defaultdict(list)
    for k, v in scored.items():
        if "A" in v:
            s, _, cfg, test = k.split("|")
            agg[(s, f"{cfg}|{test}")].append(v["A"]["score"])
    print(f"[rungs] done in {time.time() - t0:.0f}s — mean regex-appearance A per (set, lens) "
          f"(A=1-exp(-n); B/C LLM inspection-only):")
    for (s, key), vs in sorted(agg.items()):
        print(f"  {s:20s} {key:16s} {sum(vs) / len(vs):.3f} (n={len(vs)})")


if __name__ == "__main__":
    main()
