"""Fig-2 ladder on the paper's REMAINING benchmark families: GPQA (MCQ science) and code
(HumanEval as the public stand-in for N2C/LCB), modes soft/k1/k2/k4/k8, paper regime
(dynamic thinking, adaptive stopping). Plus a GPQA NO-THINKING ladder as a diagnostic
control: it removes the long-generation/stopping channel entirely, so any residual k1 drop
there is per-step conditioning damage, not failure-to-finish.

Per-record diagnostics for decomposing the drop (the math lesson: drop ~= finish drop):
  finished, n_tokens, n_steps, dup8 (repeated-8gram fraction = loop signature),
  GPQA: extraction method (boxed/phrase/paren); code: block present, syntax_ok, tests pass.

-> acts_bench/<task>/<mode>/manifest.json + ablate_bench.json   (resumable)
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor
from transformers.models.diffusion_gemma.generation_diffusion_gemma import (
    DiffusionGemmaGenerationConfig, EntropyBoundSamplerConfig,
)

from ds_ablate import MODEL_ID, Ablate, f_k  # noqa
from ds_ablate_math import CH, MathAblate

CD = Path(os.environ.get("DG_LOCKIN_DIR", str(Path(__file__).resolve().parent / "data")))
ROOT = Path(os.environ.get("ROOT", str(CD / "acts_bench")))
# mode -> ("k", k) for the paper's f_k, ("p", p) for the paper's f_p
_ALL = {"soft": ("k", None), "k1": ("k", 1), "k2": ("k", 2), "k4": ("k", 4), "k8": ("k", 8),
        "p10": ("p", 0.1), "p03": ("p", 0.03), "p01": ("p", 0.01),
        "p003": ("p", 0.003), "p001": ("p", 0.001)}
MODES = {m: _ALL[m] for m in os.environ.get("MODES", "soft,k1,k2,k4,k8").split(",")}
TASKS = os.environ.get("TASKS", "gpqa_nt,gpqa,humaneval").split(",")


def f_p(logits, p):
    """Paper's f_p: tokens with prob > p keep their probability; leftover mass is spread
    uniformly over the rest (same construction as f_k). Top-1 always retained."""
    pr = torch.softmax(logits.float(), dim=-1)
    keep = pr > p
    keep.scatter_(-1, pr.argmax(-1, keepdim=True), True)
    kept_mass = (pr * keep).sum(-1, keepdim=True)
    n_rest = (~keep).sum(-1, keepdim=True).clamp_min(1)
    rest = (1.0 - kept_mass).clamp_min(0) / n_rest
    q = torch.where(keep, pr, rest.expand_as(pr))
    return q.clamp_min(1e-20).log().to(logits.dtype)


class BenchAblate(MathAblate):
    """_k set -> f_k (via MathAblate); _p set -> f_p; both None -> soft."""
    _p = None

    def _denoising_step(self, *args, **kwargs):
        if self._p is None:
            return super()._denoising_step(*args, **kwargs)
        canvas, argmax_canvas, sc_logits, finished = super(Ablate, self)._denoising_step(*args, **kwargs)
        sc_logits = f_p(sc_logits, self._p)
        self._cap.append(None)
        return canvas, argmax_canvas, sc_logits, finished

GPQA = json.loads((CD / "gpqa_problems.json").read_text())
HE = json.loads((CD / "humaneval_problems.json").read_text())
HE_SUFFIX = ("\n\nReturn the complete function implementation (including the signature) "
             "in a single ```python code block.")


def extract_letter(text):
    m = re.findall(r"\\boxed\{\s*\\?(?:text|mathrm)?\{?\s*\(?([A-Da-d])\)?\s*\}?\s*\}", text or "")
    if m:
        return m[-1].upper(), "boxed"
    m = re.findall(r"(?:answer|option|choice)\s*(?:is|:)?\s*\**\s*\(?([A-D])\)?", text or "", re.I)
    if m:
        return m[-1].upper(), "phrase"
    m = re.findall(r"\(([A-D])\)", text or "")
    if m:
        return m[-1], "paren"
    return None, None


def extract_block(text):
    m = re.findall(r"```(?:python)?\s*\n(.*?)```", text or "", re.S)
    return m[-1] if m else None


def run_tests(block, prob):
    """Execute candidate programs against the HumanEval check. Returns (passed, syntax_ok)."""
    if block is None:
        return False, False
    try:
        compile(block, "<gen>", "exec")
        syntax_ok = True
    except SyntaxError:
        syntax_ok = False
    tail = "\n\n" + prob["test"] + f"\ncheck({prob['entry_point']})\n"
    for cand in (block, prob["prompt"] + "\n" + block):
        try:
            r = subprocess.run([sys.executable, "-c", cand + tail], timeout=20,
                               capture_output=True, text=True)
            if r.returncode == 0:
                return True, syntax_ok
        except subprocess.TimeoutExpired:
            pass
    return False, syntax_ok


def dup8(ids):
    if len(ids) < 16:
        return 0.0
    grams = [tuple(ids[i:i + 8]) for i in range(len(ids) - 7)]
    return round(1.0 - len(set(grams)) / len(grams), 4)


TASK_CFG = {
    # task: (problems, thinking, maxnew, seed_base)
    "gpqa":      (GPQA, True, 8192, 20000),
    "gpqa_nt":   (GPQA, False, 1024, 30000),
    "gpqa_nt2":  (GPQA, False, 2048, 30000),  # budget-recovery check on the no-think arm
    "humaneval": (HE, True, 8192, 40000),
}


def main():
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    tok = processor.tokenizer
    model = BenchAblate.from_pretrained(MODEL_ID, dtype="auto", device_map={"": "cuda:0"})
    model.eval()
    pad_token_id = getattr(model.config, "pad_token_id", 0) or 0
    eos_ids = model.config.eos_token_id
    eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
    skip = set(eos_ids) | {pad_token_id}

    def gcfg(maxnew):
        return DiffusionGemmaGenerationConfig(
            max_new_tokens=maxnew, max_denoising_steps=48,
            sampler_config=EntropyBoundSamplerConfig(entropy_bound=0.1),
            t_min=0.4, t_max=0.8, stability_threshold=1, confidence_threshold=0.005,
            pad_token_id=pad_token_id, eos_token_id=eos_ids,
        )

    for task in TASKS:
        probs, thinking, maxnew, seed0 = TASK_CFG[task]
        gen_config = gcfg(maxnew)
        for mode, (kind, val) in MODES.items():
            out = ROOT / task / mode
            out.mkdir(parents=True, exist_ok=True)
            mf = out / "manifest.json"
            manifest = json.loads(mf.read_text()) if mf.exists() else {}
            BenchAblate._k = val if kind == "k" else None
            BenchAblate._p = val if kind == "p" else None
            for pi, prob in enumerate(probs):
                pid = prob["pid"]
                if pid in manifest:
                    continue
                content = prob["problem"] if task.startswith("gpqa") else prob["prompt"] + HE_SUFFIX
                enc = processor.apply_chat_template(
                    [{"role": "user", "content": content}],
                    tokenize=True, add_generation_prompt=True, return_dict=True,
                    return_tensors="pt", enable_thinking=thinking,
                ).to(model.device)
                torch.manual_seed(seed0 + pi)
                model._begin_capture()
                t0 = time.time()
                o = model.generate(**enc, generation_config=gen_config)
                nsteps = len(model._cap)
                model._cap = None
                final = o.sequences[0, enc["input_ids"].shape[1]:].to("cpu", torch.int32).numpy()
                kept = [int(x) for x in final if int(x) not in skip]
                text = CH.sub(" ", tok.decode(kept)).strip()
                ntok = int((~np.isin(final, list(skip))).sum())
                finished = ntok < maxnew - 8
                rec = dict(finished=bool(finished), n_tokens=ntok, n_steps=nsteps, dup8=dup8(kept))
                if task.startswith("gpqa"):
                    v, how = extract_letter(text)
                    rec.update(answer=prob["answer"], domain=prob["domain"], extracted=v, how=how,
                               correct=bool(v == prob["answer"]),
                               correct_strict=bool(how == "boxed" and v == prob["answer"]),
                               final_text=text[-800:])
                else:
                    block = extract_block(text)
                    passed, syn = run_tests(block, prob)
                    rec.update(task_id=prob["task_id"], has_block=block is not None,
                               syntax_ok=bool(syn), correct=bool(passed),
                               block=(block or "")[:4000], final_text=text[-800:])
                manifest[pid] = rec
                mf.write_text(json.dumps(manifest, ensure_ascii=False))
                print(f"[{task}/{mode}] {pid}: ok={rec['correct']} fin={finished} steps={nsteps} "
                      f"tok={ntok} dup8={rec['dup8']} {time.time()-t0:.0f}s", flush=True)
            sel = list(manifest.values())
            fin = [m for m in sel if m["finished"]]
            print(f"[{task}/{mode}] SUMMARY acc={np.mean([m['correct'] for m in sel]):.3f} "
                  f"finish={np.mean([m['finished'] for m in sel]):.3f} "
                  f"acc|fin={np.mean([m['correct'] for m in fin]) if fin else float('nan'):.3f} "
                  f"dup8={np.mean([m['dup8'] for m in sel]):.3f} (n={len(sel)})", flush=True)

    res = {}
    for task in TASKS:
        for mode in MODES:
            p = ROOT / task / mode / "manifest.json"
            if not p.exists():
                continue
            man = json.loads(p.read_text())
            sel = list(man.values())
            fin = [m for m in sel if m["finished"]]
            d = dict(acc=float(np.mean([m["correct"] for m in sel])),
                     finish=float(np.mean([m["finished"] for m in sel])),
                     acc_given_finished=(float(np.mean([m["correct"] for m in fin])) if fin else None),
                     steps=float(np.mean([m["n_steps"] for m in sel])),
                     tokens=float(np.mean([m["n_tokens"] for m in sel])),
                     dup8=float(np.mean([m["dup8"] for m in sel])), n=len(sel))
            if task.startswith("gpqa"):
                d["acc_strict"] = float(np.mean([m["correct_strict"] for m in sel]))
                d["by_domain"] = {dom: float(np.mean([m["correct"] for m in sel if m["domain"] == dom]))
                                  for dom in sorted({m["domain"] for m in sel})}
            else:
                d["has_block"] = float(np.mean([m["has_block"] for m in sel]))
                d["syntax_ok"] = float(np.mean([m["syntax_ok"] for m in sel]))
            res.setdefault(task, {})[mode] = d
    json.dump(dict(modes=list(MODES), tasks=TASKS, per_task=res),
              open(CD / "ablate_bench.json", "w"), ensure_ascii=False, indent=1)
    print("DS_ABLATE_BENCH_DONE", flush=True)


if __name__ == "__main__":
    main()
