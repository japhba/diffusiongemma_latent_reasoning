"""Full paper-protocol sweep with films: for every Fig-2 task we can run (GPQA, AMC/AIME,
LCB, IMO — N2C is Google-private), generate the paper's top-k ladder (and GPQA's f_p ladder)
at the model-default schedule, grading each rollout AND saving the full denoising film
(per-step argmax canvas + pre-truncation entropy), same shard format as ds_rb_capture_all.
IMO additionally gets the closure pair (soft/k1 at slow3) since it is a brand-new task.
Existing shards are skipped, so the closure arms captured earlier are not redone.

  films     -> exp/dg_lockin/rb_steps/<task>/<arm>/<pid>.json   {frames: [[ci,pieces,ents],..]}
  manifests -> commit_ds/acts_psweep/<task>/<arm>/manifest_w<WORKER>.jsonl (one line per rollout)

Env: WORKER/NWORKERS (idx-mod sharding), TASKS, ARMS_OVERRIDE, LIMIT_PER_ARM (smoke).
Seeds match the existing manifests exactly (gpqa 20000+i, math (aime?1000:0)+pid, lcb 50000+i),
imo is new at 60000+i — same-GPU-type replays are step-exact (validated for rb_steps).
"""
import json
import os
import re
import sys
import time
from pathlib import Path

sys.set_int_max_str_digits(1_000_000)  # degenerate loops can emit 1000s-digit "answers"

import numpy as np
import torch
from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion
from transformers.models.diffusion_gemma.generation_diffusion_gemma import (
    DiffusionGemmaGenerationConfig, EntropyBoundSamplerConfig,
)

from ds_ablate import MODEL_ID, f_k
from ds_ablate_bench import GPQA, dup8, extract_block, extract_letter, f_p
from ds_ablate_lcb import LCB, SUFFIX as LCB_SUFFIX, run_program
from ds_ablate_math import CH, PROBLEMS, SUFFIX as MATH_SUFFIX, extract_int
from ds_battery import CANVAS, build as battery_build

CD = Path(os.environ.get("DG_LOCKIN_DIR", str(Path(__file__).resolve().parent / "data")))
RB = Path(os.environ.get("DG_RB_DIR", str(CD / "rb_steps")))
MF = CD / "acts_psweep"
IMO = json.loads((CD / "imo_problems.json").read_text())
IMO_FULL = json.loads((CD / "imo_full_problems.json").read_text())
WORKER, NWORKERS = int(os.environ.get("WORKER", 0)), int(os.environ.get("NWORKERS", 1))
LIMIT = int(os.environ.get("LIMIT_PER_ARM", 0))

# arm -> (kind, val, slow, maxnew, T);  kind in {k, p}
KD = {f"k{k}": ("k", k, False, 8192, 48) for k in (1, 2, 4, 8, 16, 32, 64)}
PD = {m: ("p", p, False, 8192, 48) for m, p in
      [("p10", 0.1), ("p03", 0.03), ("p01", 0.01), ("p003", 0.003), ("p001", 0.001)]}
SOFT = {"soft": ("k", None, False, 8192, 48)}
SLOW = {f"{m}_slow3": ("k", k, True, 12288, 96)
        for m, k in [("soft", None), ("k1", 1), ("k2", 2), ("k4", 4), ("k8", 8)]}
B32 = {"soft_b32": ("k", None, False, 32768, 48), "k8_b32": ("k", 8, False, 32768, 48)}
# step-matched gentle: the gentle sampler's PACING knobs (eb 0.02, t 0.5-1.0) run inside the
# paper's own budgets (T=48 steps/canvas, 8192 tokens) — controls for "gentle only wins because it
# spends more denoising steps" (gpqa means: soft 321->734, k1 1276->1924 under slow3).
SLOWM = {f"{m}_slow3m": ("k", k, True, 8192, 48)
         for m, k in [("soft", None), ("k1", 1), ("k2", 2), ("k4", 4), ("k8", 8)]}
MATCHED = ["soft_slow3m", "k8_slow3m", "k4_slow3m", "k2_slow3m", "k1_slow3m"]
LADDER = [f"k{k}" for k in (2, 4, 8, 16, 32, 64)]
GENTLE_MIDK = ["k2_slow3", "k4_slow3", "k8_slow3"]
WORD_DS = ("univocalic", "lipogram", "piem", "self_count_words")
# word pool: battery protocol (single canvas, FIXED steps, no thinking); std T48/eb0.1, gentle T96/eb0.02
WORD_ARMS = ["soft", "k1", "k2", "k4", "k8",
             "soft_slow3", "k1_slow3", "k2_slow3", "k4_slow3", "k8_slow3"]
TASK_ARMS = {
    "gpqa": LADDER + list(PD) + GENTLE_MIDK,                  # soft,k1(+slow3) films exist
    "amc_aime": LADDER + GENTLE_MIDK,
    "lcb": LADDER + GENTLE_MIDK,
    "imo": ["soft", "k1"] + LADDER + ["soft_slow3", "k1_slow3"] + GENTLE_MIDK,
    "word": WORD_ARMS,
    "imo_full": ["soft_b32", "k8_b32"],
}
ALL_ARMS = {**KD, **PD, **SOFT, **SLOW, **B32, **SLOWM}
TASKS = os.environ.get("TASKS", "gpqa,amc_aime,lcb,imo,word").split(",")
if os.environ.get("ARMS_OVERRIDE"):
    TASK_ARMS = {t: os.environ["ARMS_OVERRIDE"].split(",") for t in TASKS}


class SweepCapture(DiffusionGemmaForBlockDiffusion):
    """f_k / f_p on the self-conditioning logits (paper's intervention point) + film capture:
    per denoising step the argmax canvas and the PRE-truncation per-position entropy."""
    _k = _p = None

    def _begin_capture(self):
        self._frames, self._ci = [], -1

    def _prepare_denoiser_inputs(self, *args, **kwargs):
        self._ci += 1
        return super()._prepare_denoiser_inputs(*args, **kwargs)

    def _denoising_step(self, *args, **kwargs):
        canvas, argmax_canvas, scl, finished = super()._denoising_step(*args, **kwargs)
        ent = torch.distributions.Categorical(logits=scl[0].float()).entropy()
        if self._k is not None:
            scl = f_k(scl, self._k)
        elif self._p is not None:
            scl = f_p(scl, self._p)
        self._frames.append((self._ci, argmax_canvas[0].to("cpu", torch.int32).tolist(),
                             [round(float(x), 1) for x in ent.cpu()]))
        return canvas, argmax_canvas, scl, finished


def items_for(task):
    if task == "gpqa":
        return [(p["pid"], p["problem"], 20000 + i, p) for i, p in enumerate(GPQA)]
    if task == "amc_aime":
        probs = [p for p in PROBLEMS if p["src"] == "amc"][:32] + \
                [p for p in PROBLEMS if p["src"] == "aime"][:16]
        return [(p["pid"], p["problem"] + MATH_SUFFIX,
                 (1000 if p["src"] == "aime" else 0) + int(p["pid"].split("_")[1]), p) for p in probs]
    if task == "lcb":
        return [(p["pid"], p["problem"] + LCB_SUFFIX, 50000 + i, p) for i, p in enumerate(LCB)]
    if task == "imo":
        return [(p["pid"], p["problem"] + MATH_SUFFIX, 60000 + i, p) for i, p in enumerate(IMO)]
    if task == "imo_full":
        return [(p["pid"], p["problem"] + MATH_SUFFIX, 70000 + i, p) for i, p in enumerate(IMO_FULL)]
    if task == "word":
        rows = [r for r in battery_build() if r["ds"] in WORD_DS]
        return [(r["rid"], r["prompt"], 7000 + r["i"], r) for r in rows]
    raise ValueError(task)


def grade(task, text, prob):
    if task == "gpqa":
        v, how = extract_letter(text)
        return dict(extracted=v, how=how, ok=bool(v == prob["answer"]),
                    ok_strict=bool(how == "boxed" and v == prob["answer"]))
    if task in ("amc_aime", "imo", "imo_full"):
        v = extract_int(text)
        m = re.findall(r"\\boxed\{([^{}]+)\}", text or "")
        sv = extract_int("\\boxed{" + m[-1] + "}") if m else None
        return dict(extracted=v, ok=bool(v == int(prob["answer"])),
                    ok_strict=bool(sv == int(prob["answer"])))
    if task == "lcb":
        block = extract_block(text)
        frac, syn = run_program(block, prob["tests"])
        return dict(pass_frac=frac, syntax_ok=bool(syn), has_block=block is not None,
                    ok=bool(frac == 1.0), ok_strict=bool(frac == 1.0))
    if task == "word":
        ok = bool(prob["check"](text))
        return dict(ok=ok, ok_strict=ok, ds=prob["ds"], i=prob["i"], answer=str(prob["answer"]),
                    wc=len(re.findall(r"[a-zA-Z']+", text)))
    raise ValueError(task)


def main():
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    tok = processor.tokenizer
    model = SweepCapture.from_pretrained(MODEL_ID, dtype="auto", device_map={"": "cuda:0"})
    model.eval()
    pad_token_id = getattr(model.config, "pad_token_id", 0) or 0
    eos_ids = model.config.eos_token_id
    eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
    skip = set(eos_ids) | {pad_token_id}

    default_canvas = model.config.canvas_length

    def gcfg(slow, maxnew, T, fixed=False):
        return DiffusionGemmaGenerationConfig(
            max_new_tokens=maxnew, max_denoising_steps=T,
            sampler_config=EntropyBoundSamplerConfig(entropy_bound=0.02 if slow else 0.1),
            t_min=0.5 if slow else 0.4, t_max=1.0 if slow else 0.8,
            stability_threshold=(T + 1) if fixed else 1,
            confidence_threshold=1e-9 if fixed else 0.005,
            pad_token_id=pad_token_id, eos_token_id=eos_ids,
        )

    idx = -1
    for task in TASKS:
        items = items_for(task)
        for arm in TASK_ARMS[task]:
            kind, val, slow, maxnew, T = ALL_ARMS[arm]
            out_rb = RB / task / arm
            out_rb.mkdir(parents=True, exist_ok=True)
            out_mf = MF / task / arm
            out_mf.mkdir(parents=True, exist_ok=True)
            mfp = out_mf / f"manifest_w{WORKER}.jsonl"
            word = task == "word"
            if not word:
                gen_config = gcfg(slow, maxnew, T)
            done_in_arm = 0
            for pid, content, seed, prob in items:
                idx += 1
                if idx % NWORKERS != WORKER:
                    continue
                if LIMIT and done_in_arm >= LIMIT:
                    continue
                f = out_rb / f"{pid}.json"
                if f.exists():
                    continue
                done_in_arm += 1
                SweepCapture._k = val if kind == "k" else None
                SweepCapture._p = val if kind == "p" else None
                if word:
                    c = CANVAS.get(prob["ds"], 64)
                    model.config.canvas_length = c
                    gen_config = gcfg(slow, c, T, fixed=True)
                    maxnew = c
                else:
                    model.config.canvas_length = default_canvas
                enc = processor.apply_chat_template(
                    [{"role": "user", "content": content}], tokenize=True,
                    add_generation_prompt=True, return_dict=True, return_tensors="pt",
                    enable_thinking=not word,
                ).to(model.device)
                torch.manual_seed(seed)
                model._begin_capture()
                t0 = time.time()
                o = model.generate(**enc, generation_config=gen_config)
                frames = model._frames
                model._frames = None
                final = o.sequences[0, enc["input_ids"].shape[1]:].to("cpu", torch.int32).numpy()
                kept = [int(x) for x in final if int(x) not in skip]
                text = CH.sub(" ", tok.decode(kept)).strip()
                ntok = int((~np.isin(final, list(skip))).sum())
                g = grade(task, text, prob)
                rec = dict(task=task, arm=arm, pid=pid, **g,
                           finished=bool(ntok < maxnew - 8), n_tokens=ntok, n_steps=len(frames),
                           dup8=dup8(kept), wall=round(time.time() - t0, 1),
                           final_text=text[-800:])
                with open(mfp, "a") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                shard = dict(frames=[[ci, tok.convert_ids_to_tokens(ids), ents]
                                     for ci, ids, ents in frames])
                tmp = f.with_suffix(f".tmp{WORKER}")
                tmp.write_text(json.dumps(shard, ensure_ascii=False))
                tmp.rename(f)
                print(f"[{task}/{arm}] {pid}: ok={rec['ok']} fin={rec['finished']} "
                      f"steps={rec['n_steps']} tok={ntok} dup8={rec['dup8']} "
                      f"{f.stat().st_size//1024}KB {rec['wall']:.0f}s", flush=True)
            n = len(list(out_rb.glob("*.json")))
            print(f"[{task}/{arm}] ARM PASSED ({n} shards on disk)", flush=True)
    print("DS_PAPER_SWEEP_DONE", flush=True)


if __name__ == "__main__":
    main()
