"""Steering 2×2 redo — RepE tasks, plain DIFFERENCE OF MEANS, position-masked writes.

Tasks (RepE data): the 6 primary emotions (contrast: this emotion's scenarios vs the other
five's) + honesty (facts_true_false statements under an honest-vs-untruthful persona template,
paired). Fairness/harmlessness ship only as notebooks without local stimuli — noted, not run.

Vector = unit(mean(pos acts) − mean(neg acts)) per layer 2..28 — no PCA/LAT protocol.
Readers (donor axis) — THE HYBRID CONVENTION (2026-07-19): ONE wrapper (each stimulus in the
chat template, read at the last token = the assistant-turn start) × THREE read streams:
  g : gemma-4 causal                       (cells gg, gd)
  e : DG ENCODER stack (probe-aligned)     (cells eg, ed — the headline DG row)
  c : DG DECODER stream, stimulus as canvas over BOS (generation-matched; cells cg, cd)
Earlier eras retired to *.bak files: decoder_read (templated gemma + RAW decoder DG, gemini
judge) and rawenc_read (raw text, no template, enc-only — steered ~2/3 weaker: the wrapper is
required for actuation).

Write protocols (2026-07-19 final set; earlier pr / last80 retired):
  pr80   : THE HEADLINE, both targets — steer the LAST 80% OF THE PROMPT REGION only
           (positions ≥0.2·T_prompt; gemma: prefill only, decode untouched; DG: encoder
           positions). Band L9-19 odd, dose 0.35×resid.
  gen    : gemma targets — steer ALL GENERATION positions (every decode step; prefill
           untouched). Band L5-25 odd, dose 0.12×resid.
  canvas : DG targets — steer ALL canvas positions at every denoising step (the DG analog
           of gen). Band L5-25 odd, dose 0.12×resid.
Canonical read cap: max 512 tokens on every derivation read (a no-op — stimuli ≤150 tok).

Cells (donor,target): gg, gd, dg, dd × 2 protocols × ±sign × the RepE paper's own carriers
(repe_carriers.py — the single source of truth: 6 lie-inviting scenarios for honesty, 6
emotional provocations for the emotions; NOT the deprecated 16 generic neutral prompts).

--derive              -> out/saeprobes/dom_directions.pt + dom_meta.json + dom_desc.json
SAEP_SHARD=i/n (gen)  -> out/saeprobes/dom_gens_shard{i}.json

srun ... bash concept_probes/slurm/ensure_and_run.sh concept_probes/steer_dom_repe.py
"""
from __future__ import annotations

import functools
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

print = functools.partial(print, flush=True)
REPO = Path(os.environ.get("DGLR_ROOT", Path(__file__).resolve().parents[1]))
OUT = Path(os.environ.get("SAEP_OUT", REPO / "concept_probes/out/saeprobes"))
REPE_DATA = REPO / "third_party/representation-engineering/data"
# task set + artifact prefix. "repe" (default) = the 11 RepE control tasks -> dom_*; "aemo" =
# the Anthropic emotion-concepts battery (16 curated distinct emotions) re-derived under the RepE
# CONTRASTIVE paradigm -> aemo_*; "aemoall" = ALL 171 emotions (the firm transfer verdict) ->
# aemoall_*. Optional SAEP_CARR_CAP caps carriers/task (carriers don't drive the between-task CI,
# so the 171-verdict run uses 2 to stay single-GPU-feasible).
STEER_SET = os.environ.get("SAEP_STEER_SET", "repe")
PREFIX = {"aemo": "aemo", "aemoall": "aemoall"}.get(STEER_SET, "dom")
CARR_CAP = int(os.environ["SAEP_CARR_CAP"]) if os.environ.get("SAEP_CARR_CAP") else None

sys.path.insert(0, str(REPO / "concept_probes"))
import importlib.util
def _load(n, f):
    s = importlib.util.spec_from_file_location(n, REPO / f"concept_probes/{f}")
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
rcp = _load("rcp", "run_concept_probes.py")
rsg = _load("rsg", "run_saeprobes_gpu.py")
repe = _load("repe", "repe_steer.py")
cs2 = _load("cs2", "calibrate_steer2.py")
rdr = _load("rdr", "repe_dg_read.py")
rca = _load("rca", "repe_carriers.py")               # single source of truth for carriers

RL = list(range(2, 29))
WIDE = list(range(5, 26, 2)); DOSE_WIDE = 0.12       # last80: in-stream band
BAND = list(range(9, 20, 2)); DOSE_PR = 0.35         # pr: prompt-region band
EMOTIONS = ["happiness", "sadness", "anger", "fear", "disgust", "surprise"]
EMO_TPL = "Consider the {emotion} of the following scenario:\nScenario: {scenario}\nAnswer:"
HON_TPL = "Pretend you're an {persona} person making statements about the world.\nStatement: {s}\nAnswer:"

# Anthropic emotion-concepts battery under the RepE paradigm: 16 conceptually-distinct emotions
# spanning the affect space (the 6 basics + calm/love/pride/guilt/boredom/gratitude/desperation/
# jealousy/contempt/awe — includes the paper's causal highlights desperate & loving). Derived
# CONTRASTIVELY (this emotion's stories vs the OTHER 15's, sampled) — NOT the story-vs-neutral
# diff-of-means of emotion_stories_steer.py, which collapses all 171 onto one affect axis
# (participation ratio ~1.4); the contrastive contrast cancels the shared axis (cf. RepE's 6
# emotions, PR ~4.3/6). Same 200-cap and same adjective in the pos/neg wrapper as RepE.
AEMO_SEL = ["happy", "sad", "angry", "afraid", "disgusted", "surprised", "calm", "loving",
            "proud", "guilty", "bored", "grateful", "desperate", "jealous", "contemptuous", "awestruck"]
AEMO_TPL = "Consider the {emotion} feeling in the following scenario:\nScenario: {scenario}\nAnswer:"


def anthropic_emotion_tasks(cap=200):
    """[(tag, desc, pos_texts, neg_texts)] — RepE contrastive paradigm on the Anthropic emotion
    stories (ryancodrai/emotion-probes). pos = this emotion's stories, neg = pooled OTHER
    emotions' stories; both wrapped in AEMO_TPL with THIS emotion's adjective (constant context,
    so the contrast is purely the story content). aemo = the 16 curated distinct emotions;
    aemoall = all 171 (contrastive vs the other 170) — the firm transfer verdict."""
    ess = _load("ess", "emotion_stories_steer.py")
    by, _neutral = ess.load_data()
    sel = sorted(by) if STEER_SET == "aemoall" else AEMO_SEL
    w = 3 if len(sel) > 99 else 2                          # zero-pad tag index
    tasks = []
    for i, e in enumerate(sel):
        pos_s = by[e][:cap]
        others = [s for e2 in sel if e2 != e for s in by[e2][:cap]]
        np.random.default_rng(i).shuffle(others)
        neg_s = others[:len(pos_s)]
        tasks.append((f"aemo_{i:0{w}d}_{ess._slug(e)}", f"the emotion of {e}",
                      [AEMO_TPL.format(emotion=e, scenario=s) for s in pos_s],
                      [AEMO_TPL.format(emotion=e, scenario=s) for s in neg_s]))
    return tasks


def get_tasks():
    return anthropic_emotion_tasks() if STEER_SET in ("aemo", "aemoall") else task_stimuli()


def task_stimuli():
    """[(tag, desc, pos_texts, neg_texts)] — raw stimulus texts (templates applied)."""
    raw = {e: sorted(set(json.loads((REPE_DATA / f"emotions/{e}.json").read_text())))[:200]
           for e in EMOTIONS}
    tasks = []
    for i, e in enumerate(EMOTIONS):
        pos = [EMO_TPL.format(emotion=e, scenario=s) for s in raw[e]]
        neg_s = [s for e2 in EMOTIONS if e2 != e for s in raw[e2]]
        np.random.default_rng(i).shuffle(neg_s)
        neg = [EMO_TPL.format(emotion=e, scenario=s) for s in neg_s[:len(pos)]]
        tasks.append((f"re_{i}_{e}", f"the emotion of {e}", pos, neg))
    import csv
    stmts = [r["statement"] for r in csv.DictReader((REPE_DATA / "facts/facts_true_false.csv").open())]
    tasks.append(("rh_honesty", "honesty — truthful, sincere statements (vs deception and lying)",
                  [HON_TPL.format(persona="honest", s=s) for s in stmts],
                  [HON_TPL.format(persona="untruthful", s=s) for s in stmts]))
    tasks += _repe_control_tasks()
    return tasks


def _repe_control_tasks(cap=200):
    """The remaining RepE control tasks (harmful / fairness / language / memorization) — paired
    pos/neg stimuli lifted verbatim from the paper's own datasets, so the diff-of-means direction
    matches the paper's contrast. Carriers for each live in repe_carriers.py."""
    from datasets import load_dataset
    out = []

    # harmful: +direction = harmless. pos = harmless instructions, neg = harmful ones.
    # The judged concept MUST describe the POS pole (sign-convention fix 2026-07-19: the old
    # desc "refusal" described the NEG pole, so a perfectly-working direction scored Δ=−1).
    d = load_dataset("justinphan3110/harmful_harmless_instructions")["train"]
    sents = [s for row in d["sentence"] for s in row]
    labs = [l for row in d["label"] for l in row]
    harmless = [s for s, l in zip(sents, labs) if l == 1][:cap]
    harmful = [s for s, l in zip(sents, labs) if l == 0][:cap]
    n = min(len(harmless), len(harmful))
    out.append(("rc_harmful", "perceiving the request as harmless and safe — direct helpful "
                "engagement with it, rather than refusing or flagging it as harmful",
                harmless[:n], harmful[:n]))

    # fairness: +direction = anti-stereotype. StereoSet-race anti vs stereotype sentences.
    ss = load_dataset("McGill-NLP/stereoset", "intrasentence")["validation"]
    pos, neg = [], []
    for r in ss:
        if r["bias_type"] != "race":
            continue
        labs_, sents_ = r["sentences"]["gold_label"], r["sentences"]["sentence"]
        anti = [s for s, l in zip(sents_, labs_) if l == 0]
        ster = [s for s, l in zip(sents_, labs_) if l == 1]
        if anti and ster:
            pos.append(anti[0]); neg.append(ster[0])
        if len(pos) >= 150:
            break
    out.append(("rc_fairness", "fair, unbiased, non-stereotyped treatment of race and gender", pos, neg))

    # language: +direction = Vietnamese. Same request, VN- vs EN-instructed (the paper's contrast).
    ins = load_dataset("justinphan3110/sharegpt_instructions_small", split="train")["instructions"][:64]
    out.append(("rc_language", "a response written in the Vietnamese language",
                [f"{s}\nTrả lời câu hỏi trên bằng Tiếng Việt." for s in ins],
                [f"{s}\nAnswer the request above in English." for s in ins]))

    # memorization: +direction = verbatim recall. Real/popular quotes vs unseen/fabricated ones.
    real = json.loads((REPE_DATA / "memorization/quotes/popular_quotes.json").read_text())[:cap]
    fake = json.loads((REPE_DATA / "memorization/quotes/unseen_quotes.json").read_text())[:cap]
    m = min(len(real), len(fake))
    out.append(("rc_memorization", "verbatim reproduction of a well-known, memorized quotation",
                real[:m], fake[:m]))
    return out


def dom(P, N):
    """Unit diff-of-means per read layer: P,N = [n, len(RL), d]."""
    d = P.mean(0) - N.mean(0)                          # [len(RL), d]
    return {L: torch.tensor(d[j] / (np.linalg.norm(d[j]) + 1e-9), dtype=torch.float32)
            for j, L in enumerate(RL)}


def derive():
    tasks = get_tasks()
    (OUT / f"{PREFIX}_desc.json").write_text(json.dumps({t: desc for t, desc, _, _ in tasks}, indent=1))
    model_g, tok = rcp.load_model("gemma4", device_map={"": 0})
    model_d, _ = rcp.load_model("diffusiongemma", device_map={"": 0})
    bb, g_layers = rcp.locate(model_g)
    enc_lm = model_d.model.encoder.language_model
    enc_layers = enc_lm.layers
    dec_layers = model_d.model.decoder.layers
    def tpl(texts):                                # decoder canvas gets the rendered template
        return [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                        add_generation_prompt=True) for p in texts]
    dirs = {"g": {}, "e": {}, "c": {}}
    for tag, _desc, pos, neg in tasks:
        # g / e: chat-templated read via repe.last_token_acts_all (template applied internally)
        dirs["g"][tag] = dom(repe.last_token_acts_all(model_g, bb, g_layers, tok, pos, model_g.device),
                             repe.last_token_acts_all(model_g, bb, g_layers, tok, neg, model_g.device))
        dirs["e"][tag] = dom(repe.last_token_acts_all(model_d, enc_lm, enc_layers, tok, pos, model_d.device),
                             repe.last_token_acts_all(model_d, enc_lm, enc_layers, tok, neg, model_d.device))
        dirs["c"][tag] = dom(rdr.dg_decoder_read(model_d, dec_layers, tok, tpl(pos), model_d.device, max_len=512),
                             rdr.dg_decoder_read(model_d, dec_layers, tok, tpl(neg), model_d.device, max_len=512))
        print(f"[dom] {tag}: {len(pos)}+{len(neg)} templated stimuli read "
              f"(gemma-causal + DG-encoder + DG-decoder)")
    torch.save(dirs, OUT / f"{PREFIX}_directions.pt")
    print(f"[dom/derive] wrote {PREFIX}_directions.pt ({len(tasks)} tasks × 3 readers)")


def masked_add_hook(vec, proto):
    """pr80: prefill-only, positions ≥0.2·T (= last 80% of the prompt; encoder registration
    covers DG; AR decode steps untouched). gen: ONLY decode steps (T==1), prefill untouched.
    canvas: ALL positions at every forward (DG decoder registration → whole canvas, every
    denoising step)."""
    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        T = h.shape[1]
        if T == 1:
            if proto == "pr80":
                return out
            lo = 0                                 # gen / canvas: steer the decode step
        elif proto == "gen":
            return out                             # gen: prefill untouched
        elif proto == "pr80":
            lo = int(0.2 * T)
        else:
            lo = 0
        h = h.clone()
        h[:, lo:, :] = h[:, lo:, :] + vec.to(h.dtype)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
    return hook


NORM_PROMPT = "Describe an ordinary afternoon."      # carrier-independent residual-norm probe


def gen():
    shard_i, shard_n = (int(x) for x in os.environ.get("SAEP_SHARD", "0/1").split("/"))
    gens_path = OUT / (f"{PREFIX}_gens_shard{shard_i}.json" if shard_n > 1 else f"{PREFIX}_gens.json")
    done = json.loads(gens_path.read_text()) if gens_path.exists() else {}
    dirs = torch.load(OUT / f"{PREFIX}_directions.pt")
    # {tag: {carriers, prov ('repe'|'added'), n_orig}} — originals first, additions appended
    (OUT / f"{PREFIX}_carriers.json").write_text(json.dumps(
        {t: {"carriers": rca.for_task(t), "prov": rca.provenance(t), "n_orig": rca.n_orig(t)}
         for t in dirs["g"]}, indent=1))
    tags = list(dirs["g"])[shard_i::shard_n]
    donors = tuple(dirs)                           # ("g", "e", "c") — see module docstring

    model_g, tok = rcp.load_model("gemma4", device_map={"": 0})
    model_d, _ = rcp.load_model("diffusiongemma", device_map={"": 0})
    _, g_layers = rcp.locate(model_g)
    dec_layers = model_d.model.decoder.layers
    enc_layers = model_d.model.encoder.language_model.layers
    device = model_g.device

    # live per-layer residual norms on each WRITE stream (one unsteered gen per model)
    def capture_norms(layers, run):
        cap = {}
        def mk(L):
            def h(_m, _i, out): cap[L] = (out[0] if isinstance(out, tuple) else out)
            return h
        hs = [layers[L].register_forward_hook(mk(L)) for L in RL]
        try:
            run()
        finally:
            for h in hs:
                h.remove()
        return {L: float(cap[L].float().norm(dim=-1).mean()) for L in RL}
    g_rn = capture_norms(g_layers, lambda: rsg.gen_ar(model_g, tok, NORM_PROMPT))
    dec_rn = capture_norms(dec_layers, lambda: cs2.gen_dg_eager(model_d, tok, NORM_PROMPT, seed=0))
    enc_rn = capture_norms(enc_layers, lambda: cs2.gen_dg_eager(model_d, tok, NORM_PROMPT, seed=0))
    print(f"[dom/gen] shard {shard_i}/{shard_n}: {len(tags)} tasks; norms g13={g_rn[13]:.0f} "
          f"dec13={dec_rn[13]:.0f} enc13={enc_rn[13]:.0f}")

    # (target, proto) -> (module list, band, dose, norms)
    WRITE = {("g", "pr80"): (g_layers, BAND, DOSE_PR, g_rn),
             ("g", "gen"): (g_layers, WIDE, DOSE_WIDE, g_rn),
             ("d", "pr80"): (enc_layers, BAND, DOSE_PR, enc_rn),
             ("d", "canvas"): (dec_layers, WIDE, DOSE_WIDE, dec_rn)}

    for ci, tag in enumerate(tags):
        for pi, prompt in enumerate(rca.for_task(tag)[:CARR_CAP]):
            for donor in donors:
                dvec = dirs[donor][tag]
                for target in ("g", "d"):
                    for proto in (("pr80", "gen") if target == "g" else ("pr80", "canvas")):
                        layers, band, dose, rn = WRITE[(target, proto)]
                        for sign, sname in [(+1, "pos"), (-1, "neg")]:
                            key = f"{tag}|{pi}|{donor}{target}|{proto}|{sname}"
                            if key in done:
                                continue
                            hks = []
                            try:
                                for L in band:
                                    v = dvec[L].to(device) * (sign * dose * rn[L])
                                    hks.append(layers[L].register_forward_hook(
                                        masked_add_hook(v, proto)))
                                if target == "g":
                                    done[key] = rsg.gen_ar(model_g, tok, prompt)
                                else:
                                    done[key] = cs2.gen_dg_eager(model_d, tok, prompt, seed=ci * 37 + pi)
                            finally:
                                for h in hks:
                                    h.remove()
            gens_path.write_text(json.dumps(done))
        print(f"[dom/gen] {ci + 1}/{len(tags)} {tag} ({len(done)} gens)")
    gens_path.write_text(json.dumps(done))
    print(f"[dom/gen] wrote {gens_path} ({len(done)} entries)")


if __name__ == "__main__":
    if "--all" in sys.argv:                 # re-derive all directions, then resume/extend gens
        derive(); gen()
    elif "--derive" in sys.argv:
        derive()
    else:
        gen()
