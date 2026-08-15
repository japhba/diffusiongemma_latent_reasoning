"""S^t ablation, PAPER-REGIME replication: AMC/AIME problems (integer answers), DYNAMIC THINKING
enabled, MODEL-DEFAULT sampler with ADAPTIVE STOPPING (T=48, tau 0.8->0.4, gamma=0.1,
stability_threshold=1, confidence_threshold=0.005), long multi-canvas generation. This is the
setting of the paper's fig:ablation_experiment, where k=1 collapses AMC/AIME 0.72->0.24.
Modes soft/k4/k1 (f_k identical to ds_ablate; the paper's ladder shows k=4 still below baseline).
-> exp/dg_lockin/pipe/commit_ds/acts_math/<mode>/manifest.json + ablate_math.json
"""
import os
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor
from transformers.models.diffusion_gemma.generation_diffusion_gemma import (
    DiffusionGemmaGenerationConfig,
    EntropyBoundSamplerConfig,
)

from ds_ablate import MODEL_ID, Ablate

MAXNEW = 2048  # up to 8 canvases; adaptive stopping usually ends far earlier
MODES = {"soft": None, "k4": 4, "k1": 1, "onehot": 0}
CD = Path(os.environ.get("DG_LOCKIN_DIR", str(Path(__file__).resolve().parent / "data")))
ROOT = CD / "acts_math"
PROBLEMS = json.loads((CD / "math_problems.json").read_text())
CH = re.compile(r"<\|channel>thought\s*<channel\|>|<\|?channel\|?>")
SUFFIX = "\n\nPut your final answer within \\boxed{}."


class MathAblate(Ablate):
    """k=None -> soft baseline (no truncation, no capture kept beyond len bookkeeping)."""

    def _denoising_step(self, *args, **kwargs):
        if self._k is None:
            out = super(Ablate, self)._denoising_step(*args, **kwargs)
            self._cap.append(None)
            return out
        canvas, argmax_canvas, sc_logits, finished = super(Ablate, self)._denoising_step(*args, **kwargs)
        from ds_ablate import f_k
        sc_logits = f_k(sc_logits, self._k)
        self._cap.append(None)
        return canvas, argmax_canvas, sc_logits, finished


def extract_int(text):
    m = re.findall(r"\\boxed\{([^{}]+)\}", text or "")
    cand = m[-1] if m else None
    if cand is None:
        m2 = re.findall(r"-?\d[\d,]*", (text or "").replace(" ", ""))
        cand = m2[-1] if m2 else None
    if cand is None:
        return None
    m3 = re.search(r"-?\d[\d,]*", cand.replace(" ", ""))
    return int(m3.group(0).replace(",", "")) if m3 else None


def main():
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    tok = processor.tokenizer
    model = MathAblate.from_pretrained(MODEL_ID, dtype="auto", device_map={"": "cuda:0"})
    model.eval()
    pad_token_id = getattr(model.config, "pad_token_id", 0) or 0
    eos_ids = model.config.eos_token_id
    eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
    skip = set(eos_ids) | {pad_token_id}
    # model-default sampler WITH adaptive stopping (generation_config.json values)
    gen_config = DiffusionGemmaGenerationConfig(
        max_new_tokens=MAXNEW, max_denoising_steps=48,
        sampler_config=EntropyBoundSamplerConfig(entropy_bound=0.1),
        t_min=0.4, t_max=0.8,
        stability_threshold=1, confidence_threshold=0.005,
        pad_token_id=pad_token_id, eos_token_id=eos_ids,
    )
    for mode, k in MODES.items():
        out = ROOT / mode
        out.mkdir(parents=True, exist_ok=True)
        mf = out / "manifest.json"
        manifest = json.loads(mf.read_text()) if mf.exists() else {}
        MathAblate._k = k
        for pi, prob in enumerate(PROBLEMS):
            pid = prob["pid"]
            if pid in manifest:
                continue
            enc = processor.apply_chat_template(
                [{"role": "user", "content": prob["problem"] + SUFFIX}],
                tokenize=True, add_generation_prompt=True, return_dict=True,
                return_tensors="pt", enable_thinking=True,
            ).to(model.device)
            torch.manual_seed(pi)
            model._begin_capture()
            t0 = time.time()
            o = model.generate(**enc, generation_config=gen_config)
            nsteps = len(model._cap); model._cap = None
            final = o.sequences[0, enc["input_ids"].shape[1]:].to("cpu", torch.int32).numpy()
            text = CH.sub(" ", tok.decode([int(x) for x in final if int(x) not in skip])).strip()
            v = extract_int(text)
            ok = v is not None and v == prob["answer"]
            manifest[pid] = dict(src=prob["src"], answer=prob["answer"], extracted=v, correct=bool(ok),
                                 n_tokens=int((~np.isin(final, list(skip))).sum()),
                                 n_steps=nsteps, final_text=text[-800:])
            mf.write_text(json.dumps(manifest, ensure_ascii=False))
            print(f"[{mode}] {pid}: correct={ok} ({v} vs {prob['answer']}) steps={nsteps} "
                  f"tok={manifest[pid]['n_tokens']} {time.time()-t0:.0f}s", flush=True)
        for src in ("amc", "aime"):
            accs = [m["correct"] for m in manifest.values() if m["src"] == src]
            print(f"[{mode}] {src}: {np.mean(accs):.3f} (n={len(accs)})", flush=True)

    res = {}
    for mode in MODES:
        man = json.loads((ROOT / mode / "manifest.json").read_text())
        for src in ("amc", "aime"):
            res.setdefault(src, {})[mode] = float(np.mean([m["correct"] for m in man.values() if m["src"] == src]))
        res.setdefault("all", {})[mode] = float(np.mean([m["correct"] for m in man.values()]))
    json.dump(dict(modes=list(MODES), per_ds=res),
              open(ROOT.parent / "ablate_math.json", "w"), ensure_ascii=False)
    print("DS_ABLATE_MATH_DONE", flush=True)


if __name__ == "__main__":
    main()
