"""Part-3 S^t bottleneck ablation: does the DISTRIBUTIONAL information in the self-conditioning
matrix causally matter, and where? Reproduces the transparency paper's f_k intervention
(sec 4.1 / fig:ablation_experiment): restrict S^t to the top-k tokens per position, with the
retained tokens' post-softmax probabilities unchanged and the remaining mass spread uniformly
(their Alg. f_p/f_k construction), for ALL steps and positions. Sampling within a step is
untouched — only the information passed BETWEEN steps is truncated (canvas tokens pass as usual).

Modes: k1 (top-1 only = fully interpretable bottleneck), k2, k8 (paper: ≈ baseline).
Baseline soft-S^t = the existing Part-3 captures (same battery, same seeds).
-> exp/dg_lockin/pipe/commit_ds/acts_ablate/<mode>/<rid>.npz + manifest.json   (resumable)
"""
import os
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion
from transformers.models.diffusion_gemma.generation_diffusion_gemma import (
    DiffusionGemmaGenerationConfig,
    EntropyBoundSamplerConfig,
)

from ds_battery import CANVAS, build

MODEL_ID = "google/diffusiongemma-26B-A4B-it"
T, C, TOPK = 48, 64, 8
MODES = {"k1": 1, "k2": 2, "k8": 8, "onehot": 0}
CD = Path(os.environ.get("DG_LOCKIN_DIR", str(Path(__file__).resolve().parent / "data")))
ROOT = CD / "acts_ablate"
CH = re.compile(r"<\|channel>thought\s*<channel\|>|<\|?channel\|?>")


def f_k(logits, k):
    """Paper's f_k: probs of the kept top-k tokens unchanged; leftover mass uniform on the rest.
    Implemented in prob space (log p' are valid logits). k=0 = ONE-HOT argmax (the paper's
    mentioned "-inf alternative"): S^t becomes exactly the top token's embedding — zero uniform
    haze, in-distribution-shaped (a maximally confident state), alternatives fully cut."""
    if k == 0:
        idx = logits.argmax(-1, keepdim=True)
        out = torch.full_like(logits, -60.0)
        out.scatter_(-1, idx, torch.zeros_like(idx, dtype=logits.dtype))
        return out
    p = torch.softmax(logits.float(), dim=-1)
    top, idx = p.topk(k, dim=-1)
    rest = (1.0 - top.sum(-1, keepdim=True)).clamp_min(0) / (p.shape[-1] - k)
    q = rest.expand_as(p).clone()
    q.scatter_(-1, idx, top)
    return q.clamp_min(1e-20).log().to(logits.dtype)


class Ablate(DiffusionGemmaForBlockDiffusion):
    _k = None

    def _begin_capture(self):
        self._cap = []

    def _denoising_step(self, *args, **kwargs):
        canvas, argmax_canvas, sc_logits, finished = super()._denoising_step(*args, **kwargs)
        sc_logits = f_k(sc_logits, self._k)
        probs = torch.softmax(sc_logits.float(), dim=-1)
        p, ids = probs.topk(TOPK, dim=-1)
        self._cap.append((ids[0].to("cpu", torch.int32), p[0].to("cpu", torch.float16)))
        return canvas, argmax_canvas, sc_logits, finished


def main():
    rows = build()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    tok = processor.tokenizer
    model = Ablate.from_pretrained(MODEL_ID, dtype="auto", device_map={"": "cuda:0"})
    model.eval()
    pad_token_id = getattr(model.config, "pad_token_id", 0) or 0
    eos_ids = model.config.eos_token_id
    eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
    skip = set(eos_ids) | {pad_token_id}

    def gen_config_for(c):
        return DiffusionGemmaGenerationConfig(
            max_new_tokens=c, max_denoising_steps=T,
            sampler_config=EntropyBoundSamplerConfig(entropy_bound=0.1),
            t_min=0.4, t_max=0.8,
            stability_threshold=T + 1, confidence_threshold=1e-9,
            pad_token_id=pad_token_id, eos_token_id=eos_ids,
        )

    for mode, k in MODES.items():
        out = ROOT / mode
        out.mkdir(parents=True, exist_ok=True)
        mf = out / "manifest.json"
        manifest = json.loads(mf.read_text()) if mf.exists() else {}
        Ablate._k = k
        for r in rows:
            f = out / f"{r['rid']}.npz"
            if f.exists() and r["rid"] in manifest:
                continue
            enc = processor.apply_chat_template(
                [{"role": "user", "content": r["prompt"]}],
                tokenize=True, add_generation_prompt=True, return_dict=True,
                return_tensors="pt", enable_thinking=False,
            ).to(model.device)
            c_row = CANVAS.get(r["ds"], C)
            model.config.canvas_length = c_row
            torch.manual_seed(7000 + r["i"])
            model._begin_capture()
            t0 = time.time()
            o = model.generate(**enc, generation_config=gen_config_for(c_row))
            steps = model._cap; model._cap = None
            assert len(steps) == T
            final = o.sequences[0, enc["input_ids"].shape[1]:].to("cpu", torch.int32).numpy()
            ids = torch.stack([s[0] for s in steps]).numpy()
            ps = torch.stack([s[1] for s in steps]).numpy()
            np.savez_compressed(f, ids=ids, probs=ps, final=final)
            text = tok.decode([int(x) for x in final if int(x) not in skip]).strip()
            clean = CH.sub(" ", text).strip()
            ok = bool(r["check"](clean))
            manifest[r["rid"]] = dict(ds=r["ds"], i=r["i"], answer=r["answer"], final_text=text,
                                      final_text_clean=clean, correct=ok, C=c_row)
            mf.write_text(json.dumps(manifest, ensure_ascii=False))
            print(f"[{mode}] {r['rid']}: correct={ok} {time.time()-t0:.1f}s", flush=True)
        accs = {}
        for m in manifest.values():
            accs.setdefault(m["ds"], []).append(m["correct"])
        print(f"[{mode}] done: " + " ".join(f"{d}={np.mean(v):.2f}" for d, v in sorted(accs.items())), flush=True)
    print("DS_ABLATE_DONE", flush=True)


if __name__ == "__main__":
    main()
