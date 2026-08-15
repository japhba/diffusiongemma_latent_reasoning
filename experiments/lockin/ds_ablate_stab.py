"""Decoding-STABILIZER suite (the constructive test of the fig2_trunc conclusion): can the
k1 collapse be recovered by stabilizing generation, WITHOUT giving back any sub-leading tail
information? GPQA think, 8192 budget, same seeds as ds_ablate_bench.

Modes:
  k1_rep   : k1 + windowed presence penalty r=1.3 on decoder logits (tokens seen in the last
             256 committed ids or the current argmax canvas; pad/eos exempt). Visible-text-only.
  k1_slow  : k1 + slow-commit sampler (t 0.5->1.0, entropy_bound 0.05). Pacing-only.
  k1_ema   : k1 + EMA self-conditioning (beta=0.5 mixture of successive TRUNCATED states,
             reset per canvas). Damps inter-step oscillation; only top-1 identities involved.
  soft_rep : the r=1.3 penalty on soft (benignity control for the stabilizer itself).

-> acts_stab/gpqa/<mode>/manifest.json + ablate_stab.json   (resumable)
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor
from transformers.models.diffusion_gemma.generation_diffusion_gemma import (
    DiffusionGemmaGenerationConfig, EntropyBoundSamplerConfig,
)

from ds_ablate import MODEL_ID, Ablate, f_k
from ds_ablate_math import CH, MathAblate
from ds_ablate_bench import GPQA, dup8, extract_letter

CD = Path(os.environ.get("DG_LOCKIN_DIR", str(Path(__file__).resolve().parent / "data")))
ROOT = Path(os.environ.get("ROOT", str(CD / "acts_stab")))
MODES = os.environ.get("STAB_MODES", "k1_rep,k1_slow,k1_ema,soft_rep").split(",")
CNT_MIN = 8  # count-threshold penalty: only tokens with >= this many window occurrences
MAXNEW, T, WIN = 8192, 48, 256


class StabAblate(MathAblate):
    """_k via MathAblate; _rep = penalty factor (presence if _cntmin is None, else
    count-threshold); _ema = beta for sc smoothing."""
    _rep = None
    _ema = None
    _cntmin = None

    def _begin_capture(self):
        super()._begin_capture()
        self._pma = None

    def _prepare_denoiser_inputs(self, *args, **kwargs):
        self._pma = None  # EMA resets per canvas
        return super()._prepare_denoiser_inputs(*args, **kwargs)

    def _denoising_step(self, *args, **kwargs):
        if self._rep is not None:
            base_lp = kwargs["logits_processor"]
            ctx = torch.cat([kwargs["input_ids"][0, -WIN:], kwargs["argmax_canvas"][0]])
            ctx = ctx[~torch.isin(ctx, self._exempt.to(ctx.device))]
            if self._cntmin is None:
                ctx = torch.unique(ctx)
            else:
                vals, cnts = torch.unique(ctx, return_counts=True)
                ctx = vals[cnts >= self._cntmin]
            r = self._rep

            class _LP:
                def __call__(self, input_ids, scores, cur_step=None):
                    s = base_lp(input_ids, scores, cur_step=cur_step)
                    sel = s[..., ctx]
                    s[..., ctx] = torch.where(sel > 0, sel / r, sel * r)
                    return s

            kwargs = dict(kwargs, logits_processor=_LP())
        canvas, argmax_canvas, sc_logits, finished = super()._denoising_step(*args, **kwargs)
        if self._ema is not None:
            p = torch.softmax(sc_logits.float(), dim=-1)
            self._pma = p if self._pma is None else self._ema * self._pma + (1 - self._ema) * p
            sc_logits = self._pma.clamp_min(1e-20).log().to(sc_logits.dtype)
        return canvas, argmax_canvas, sc_logits, finished


CFG = {
    # mode: dict(k, rep, cntmin, ema, slow)  — slow: 0=default sampler, 1=slow, 2=slower
    "k1_rep":     dict(k=1, rep=1.3, cntmin=None, ema=None, slow=0),
    "k1_slow":    dict(k=1, rep=None, cntmin=None, ema=None, slow=1),
    "k1_ema":     dict(k=1, rep=None, cntmin=None, ema=0.5, slow=0),
    "soft_rep":   dict(k=None, rep=1.3, cntmin=None, ema=None, slow=0),
    "k1_cnt":     dict(k=1, rep=1.5, cntmin=CNT_MIN, ema=None, slow=0),
    "soft_cnt":   dict(k=None, rep=1.5, cntmin=CNT_MIN, ema=None, slow=0),
    "k1_slow2":   dict(k=1, rep=None, cntmin=None, ema=None, slow=2),
    "k1_slowcnt": dict(k=1, rep=1.5, cntmin=CNT_MIN, ema=None, slow=1),
    # round 3: push the pacing curve + matched-schedule soft references + composition
    "k1_slow3":    dict(k=1, rep=None, cntmin=None, ema=None, slow=3, maxnew=12288),
    "soft_slow3":  dict(k=None, rep=None, cntmin=None, ema=None, slow=3, maxnew=12288),
    "soft_slow2":  dict(k=None, rep=None, cntmin=None, ema=None, slow=2, maxnew=12288),
    "k1_slow2ema": dict(k=1, rep=None, cntmin=None, ema=0.5, slow=2, maxnew=12288),
    # pace-only gentle (2026-08-14): slow3's eb 0.02 + 96 steps + 12288 budget, but the
    # STANDARD temperature range t 0.8->0.4 — does the loop-kill survive without the heat?
    "k1_pace3":    dict(k=1, rep=None, cntmin=None, ema=None, slow=3, maxnew=12288, stdtemp=True),
    "soft_pace3":  dict(k=None, rep=None, cntmin=None, ema=None, slow=3, maxnew=12288, stdtemp=True),
    "k2_pace3":    dict(k=2, rep=None, cntmin=None, ema=None, slow=3, maxnew=12288, stdtemp=True),
    "k4_pace3":    dict(k=4, rep=None, cntmin=None, ema=None, slow=3, maxnew=12288, stdtemp=True),
    "k8_pace3":    dict(k=8, rep=None, cntmin=None, ema=None, slow=3, maxnew=12288, stdtemp=True),
}


def main():
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    tok = processor.tokenizer
    model = StabAblate.from_pretrained(MODEL_ID, dtype="auto", device_map={"": "cuda:0"})
    model.eval()
    pad_token_id = getattr(model.config, "pad_token_id", 0) or 0
    eos_ids = model.config.eos_token_id
    eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
    skip = set(eos_ids) | {pad_token_id}
    StabAblate._exempt = torch.tensor(sorted(skip), dtype=torch.long)

    def gcfg(slow, maxnew, stdtemp=False):
        eb = {0: 0.1, 1: 0.05, 2: 0.03, 3: 0.02}[slow]
        steps = {0: T, 1: T, 2: 64, 3: 96}[slow]
        return DiffusionGemmaGenerationConfig(
            max_new_tokens=maxnew, max_denoising_steps=steps,
            sampler_config=EntropyBoundSamplerConfig(entropy_bound=eb),
            t_min=0.4 if stdtemp else (0.5 if slow else 0.4),
            t_max=0.8 if stdtemp else (1.0 if slow else 0.8),
            stability_threshold=1, confidence_threshold=0.005,
            pad_token_id=pad_token_id, eos_token_id=eos_ids,
        )

    for mode in MODES:
        c = CFG[mode]
        mn = c.get("maxnew", MAXNEW)
        gen_config = gcfg(c["slow"], mn, c.get("stdtemp", False))
        out = ROOT / "gpqa" / mode
        out.mkdir(parents=True, exist_ok=True)
        mf = out / "manifest.json"
        manifest = json.loads(mf.read_text()) if mf.exists() else {}
        StabAblate._k, StabAblate._rep = c["k"], c["rep"]
        StabAblate._ema, StabAblate._cntmin = c["ema"], c["cntmin"]
        for pi, prob in enumerate(GPQA):
            pid = prob["pid"]
            if pid in manifest:
                continue
            enc = processor.apply_chat_template(
                [{"role": "user", "content": prob["problem"]}],
                tokenize=True, add_generation_prompt=True, return_dict=True,
                return_tensors="pt", enable_thinking=True,
            ).to(model.device)
            torch.manual_seed(20000 + pi)
            model._begin_capture()
            t0 = time.time()
            o = model.generate(**enc, generation_config=gen_config)
            nsteps = len(model._cap)
            model._cap = None
            final = o.sequences[0, enc["input_ids"].shape[1]:].to("cpu", torch.int32).numpy()
            kept = [int(x) for x in final if int(x) not in skip]
            text = CH.sub(" ", tok.decode(kept)).strip()
            ntok = int((~np.isin(final, list(skip))).sum())
            v, how = extract_letter(text)
            manifest[pid] = dict(
                answer=prob["answer"], domain=prob["domain"], extracted=v, how=how,
                correct=bool(v == prob["answer"]),
                correct_strict=bool(how == "boxed" and v == prob["answer"]),
                finished=bool(ntok < mn - 8), n_tokens=ntok, n_steps=nsteps,
                dup8=dup8(kept), final_text=text[-800:])
            mf.write_text(json.dumps(manifest, ensure_ascii=False))
            m = manifest[pid]
            print(f"[{mode}] {pid}: ok={m['correct']} fin={m['finished']} steps={nsteps} "
                  f"tok={ntok} dup8={m['dup8']} {time.time()-t0:.0f}s", flush=True)
        sel = list(manifest.values())
        fin = [m for m in sel if m["finished"]]
        print(f"[{mode}] SUMMARY acc={np.mean([m['correct'] for m in sel]):.3f} "
              f"finish={np.mean([m['finished'] for m in sel]):.3f} "
              f"acc|fin={np.mean([m['correct'] for m in fin]) if fin else float('nan'):.3f} "
              f"dup8={np.mean([m['dup8'] for m in sel]):.3f} (n={len(sel)})", flush=True)

    res = {}
    for mode in MODES:
        p = ROOT / "gpqa" / mode / "manifest.json"
        if not p.exists():
            continue
        sel = list(json.loads(p.read_text()).values())
        fin = [m for m in sel if m["finished"]]
        res[mode] = dict(acc=float(np.mean([m["correct"] for m in sel])),
                         acc_strict=float(np.mean([m["correct_strict"] for m in sel])),
                         finish=float(np.mean([m["finished"] for m in sel])),
                         acc_given_finished=(float(np.mean([m["correct"] for m in fin])) if fin else None),
                         steps=float(np.mean([m["n_steps"] for m in sel])),
                         tokens=float(np.mean([m["n_tokens"] for m in sel])),
                         dup8=float(np.mean([m["dup8"] for m in sel])), n=len(sel))
    json.dump(dict(modes=MODES, per_mode=res), open(CD / "ablate_stab.json", "w"),
              ensure_ascii=False, indent=1)
    print("DS_ABLATE_STAB_DONE", flush=True)


if __name__ == "__main__":
    main()
