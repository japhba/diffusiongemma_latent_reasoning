"""Pacing-closure generalization, task 2: LiveCodeBench (paper Fig-2 benchmark; 32 public
stdin problems from code_generation_lite v1, execution-graded on up to 8 public+private
tests). Four points: soft / k1 at the default schedule (8192) + the matched slow3 pair
(entropy bound 0.02, 96 steps/canvas, 12288). -> acts_lcb/<mode>/manifest.json + ablate_lcb.json
"""
import json
import os
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

from ds_ablate import MODEL_ID  # noqa
from ds_ablate_math import CH, MathAblate
from ds_ablate_bench import dup8, extract_block

CD = Path(os.environ.get("DG_LOCKIN_DIR", str(Path(__file__).resolve().parent / "data")))
ROOT = Path(os.environ.get("ROOT", str(CD / "acts_lcb")))
LCB = json.loads((CD / "lcb_problems.json").read_text())
# mode: (k, slow, maxnew)
MODES = {"soft": (None, False, 8192), "k1": (1, False, 8192),
         "soft_slow3": (None, True, 12288), "k1_slow3": (1, True, 12288)}
SUFFIX = ("\n\nRead the input from stdin and print the answer(s) to stdout. "
          "Provide a complete Python program in a single ```python code block.")


def run_program(code, tests):
    """Fraction of tests passed (stdout compared line-wise, rstripped)."""
    if code is None:
        return 0.0, False
    try:
        compile(code, "<gen>", "exec")
        syn = True
    except SyntaxError:
        return 0.0, False
    npass = 0
    for t in tests:
        try:
            r = subprocess.run([sys.executable, "-c", code], input=t["input"], timeout=15,
                               capture_output=True, text=True)
            got = [l.rstrip() for l in r.stdout.strip().splitlines()]
            want = [l.rstrip() for l in t["output"].strip().splitlines()]
            npass += (r.returncode == 0 and got == want)
        except subprocess.TimeoutExpired:
            pass
    return npass / len(tests), syn


def main():
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    tok = processor.tokenizer
    model = MathAblate.from_pretrained(MODEL_ID, dtype="auto", device_map={"": "cuda:0"})
    model.eval()
    pad_token_id = getattr(model.config, "pad_token_id", 0) or 0
    eos_ids = model.config.eos_token_id
    eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
    skip = set(eos_ids) | {pad_token_id}

    def gcfg(slow, maxnew):
        return DiffusionGemmaGenerationConfig(
            max_new_tokens=maxnew, max_denoising_steps=96 if slow else 48,
            sampler_config=EntropyBoundSamplerConfig(entropy_bound=0.02 if slow else 0.1),
            t_min=0.5 if slow else 0.4, t_max=1.0 if slow else 0.8,
            stability_threshold=1, confidence_threshold=0.005,
            pad_token_id=pad_token_id, eos_token_id=eos_ids,
        )

    for mode, (k, slow, maxnew) in MODES.items():
        gen_config = gcfg(slow, maxnew)
        out = ROOT / mode
        out.mkdir(parents=True, exist_ok=True)
        mf = out / "manifest.json"
        manifest = json.loads(mf.read_text()) if mf.exists() else {}
        MathAblate._k = k
        for pi, prob in enumerate(LCB):
            pid = prob["pid"]
            if pid in manifest:
                continue
            enc = processor.apply_chat_template(
                [{"role": "user", "content": prob["problem"] + SUFFIX}],
                tokenize=True, add_generation_prompt=True, return_dict=True,
                return_tensors="pt", enable_thinking=True,
            ).to(model.device)
            torch.manual_seed(50000 + pi)
            model._begin_capture()
            t0 = time.time()
            o = model.generate(**enc, generation_config=gen_config)
            nsteps = len(model._cap)
            model._cap = None
            final = o.sequences[0, enc["input_ids"].shape[1]:].to("cpu", torch.int32).numpy()
            kept = [int(x) for x in final if int(x) not in skip]
            text = CH.sub(" ", tok.decode(kept)).strip()
            ntok = int((~np.isin(final, list(skip))).sum())
            block = extract_block(text)
            frac, syn = run_program(block, prob["tests"])
            manifest[pid] = dict(difficulty=prob["difficulty"], correct=bool(frac == 1.0),
                                 pass_frac=frac, has_block=block is not None, syntax_ok=bool(syn),
                                 finished=bool(ntok < maxnew - 8), n_tokens=ntok, n_steps=nsteps,
                                 dup8=dup8(kept), block=(block or "")[:4000], final_text=text[-800:])
            mf.write_text(json.dumps(manifest, ensure_ascii=False))
            m = manifest[pid]
            print(f"[lcb/{mode}] {pid}: ok={m['correct']} pass={frac:.2f} fin={m['finished']} "
                  f"steps={nsteps} tok={ntok} dup8={m['dup8']} {time.time()-t0:.0f}s", flush=True)
        sel = list(manifest.values())
        fin = [m for m in sel if m["finished"]]
        print(f"[lcb/{mode}] SUMMARY acc={np.mean([m['correct'] for m in sel]):.3f} "
              f"finish={np.mean([m['finished'] for m in sel]):.3f} "
              f"acc|fin={np.mean([m['correct'] for m in fin]) if fin else float('nan'):.3f} "
              f"dup8={np.mean([m['dup8'] for m in sel]):.3f} (n={len(sel)})", flush=True)

    res = {}
    for mode in MODES:
        p = ROOT / mode / "manifest.json"
        if not p.exists():
            continue
        sel = list(json.loads(p.read_text()).values())
        fin = [m for m in sel if m["finished"]]
        res[mode] = dict(acc=float(np.mean([m["correct"] for m in sel])),
                         pass_frac=float(np.mean([m["pass_frac"] for m in sel])),
                         finish=float(np.mean([m["finished"] for m in sel])),
                         acc_given_finished=(float(np.mean([m["correct"] for m in fin])) if fin else None),
                         has_block=float(np.mean([m["has_block"] for m in sel])),
                         syntax_ok=float(np.mean([m["syntax_ok"] for m in sel])),
                         steps=float(np.mean([m["n_steps"] for m in sel])),
                         tokens=float(np.mean([m["n_tokens"] for m in sel])),
                         dup8=float(np.mean([m["dup8"] for m in sel])), n=len(sel))
    json.dump(dict(modes=list(MODES), per_mode=res), open(CD / "ablate_lcb.json", "w"),
              ensure_ascii=False, indent=1)
    print("DS_ABLATE_LCB_DONE", flush=True)


if __name__ == "__main__":
    main()
