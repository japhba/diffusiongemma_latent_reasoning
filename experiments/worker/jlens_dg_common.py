"""Shared machinery for fitting/applying the Jacobian lens (third_party/jacobian-lens) on
DiffusionGemma. The lens transports a decoder residual h_l into the final-layer basis with the
corpus-average input-output Jacobian J_l = E[dh_L29/dh_l], then decodes it with the model's own
unembedding: lens_l(h) = softcap_tanh(lm_head(norm(J_l @ h))).

DG adaptation: the expectation is over DG's OPERATING distribution — denoising-step decoder
forwards (mid-trajectory canvas + self-conditioning logits + prompt KV cache), with cotangents
summed over all valid (non-pad-final) canvas positions (attention is bidirectional, so there is
no current-and-future restriction) and averaged over the same positions as sources.
"""
import math
import os
import sys
import time
from pathlib import Path

import torch

HERE = str(Path(__file__).resolve().parent)  # server.py lives next to this file
JLENS_REPO = os.environ.get("JLENS_REPO", str(Path(HERE) / "jacobian-lens"))  # clone of the jacobian-lens repo (provides `jlens`)
for p in (HERE, JLENS_REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

from server import MODEL_ID, TracingDiffusionGemma, _basin_ids  # noqa: E402
from transformers import AutoProcessor  # noqa: E402
from transformers.cache_utils import DynamicCache  # noqa: E402
from transformers.models.diffusion_gemma.generation_diffusion_gemma import (  # noqa: E402
    DiffusionGemmaGenerationConfig,
    EntropyBoundSamplerConfig,
)
from jlens.hooks import ActivationRecorder  # noqa: E402

# Matches capture_dg_acts.py COMMON — the regime of all bimodal/lock-in studies.
COMMON = dict(T=16, C=48, t_max=1.3, t_min=0.8, entropy_bound=0.3, enable_thinking=False)
N_LAYERS, D_MODEL, TARGET_LAYER = 30, 2816, 29
SOURCE_LAYERS = list(range(TARGET_LAYER))


class JLensDG(TracingDiffusionGemma):
    """Adds per-step decoder-INPUT capture (current canvas + self-conditioning logits) so a
    denoising step can later be replayed with grad enabled for the Jacobian fit."""

    def _begin_step_capture(self, steps, scl=True):
        self._cap_steps, self._cap, self._cap_i = set(steps), {}, 0
        self._cap_scl = scl

    def _end_step_capture(self):
        self._cap_steps = None
        return self._cap

    def _denoising_step(self, *args, **kwargs):
        if getattr(self, "_cap_steps", None) is not None:
            if self._cap_i in self._cap_steps:
                scl = kwargs["self_conditioning_logits"] if getattr(self, "_cap_scl", True) else None
                self._cap[self._cap_i] = {
                    "canvas": kwargs["current_canvas"].detach().cpu(),
                    "scl": None if scl is None else scl.detach().to("cpu", torch.bfloat16),
                }
            self._cap_i += 1
        return super()._denoising_step(*args, **kwargs)


def load_dg():
    print(f"[jlens_dg] loading {MODEL_ID} ...", flush=True)
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = JLensDG.from_pretrained(MODEL_ID, dtype="auto", device_map={"": "cuda:0"})
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"[jlens_dg] loaded in {time.time() - t0:.1f}s device={model.device}", flush=True)
    return model, processor


def pad_id_set(model):
    eos = model.config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    return [getattr(model.config, "pad_token_id", 0) or 0] + list(eos)


def rollout(model, processor, prompt, *, seed, basin_a_ids=None, basin_b_ids=None,
            light=True, lens=False, lens_layers=None, capture_steps=None, capture_scl=True, top_k=8,
            initial_canvas_ids=None, clamp_positions=None, burnin_positions=None,
            burnin_steps=0, burnin_reset_self_conditioning=False, **overrides):
    """One traced rollout in the standard bimodal regime. Returns trace steps, final canvas,
    and (optionally) captured step inputs / per-layer residual buffers."""
    p = {**COMMON, **overrides}
    tok = processor.tokenizer
    pad_ids = pad_id_set(model)
    torch.manual_seed(seed)
    model.config.canvas_length = p["C"]
    enc = processor.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt", enable_thinking=p["enable_thinking"]).to(model.device)
    gen_config = DiffusionGemmaGenerationConfig(
        max_new_tokens=p["C"], max_denoising_steps=p["T"],
        sampler_config=EntropyBoundSamplerConfig(entropy_bound=p["entropy_bound"]),
        t_min=p["t_min"], t_max=p["t_max"],
        stability_threshold=p["T"] + 1, confidence_threshold=1e-9,
        pad_token_id=pad_ids[0], eos_token_id=model.config.eos_token_id)
    initial_canvas = None
    if initial_canvas_ids is not None:
        assert len(initial_canvas_ids) == p["C"]
        initial_canvas = torch.tensor([initial_canvas_ids], dtype=torch.long, device=model.device)
    clamp_canvas = initial_canvas
    clamp_mask = None
    if clamp_positions is not None:
        assert initial_canvas is not None
        clamp_mask = torch.zeros(p["C"], dtype=torch.bool, device=model.device)
        clamp_mask[torch.tensor(clamp_positions, dtype=torch.long, device=model.device)] = True
    burnin_mask = None
    if burnin_positions is not None:
        burnin_mask = torch.zeros(p["C"], dtype=torch.bool, device=model.device)
        burnin_mask[torch.tensor(burnin_positions, dtype=torch.long, device=model.device)] = True
    model._begin_trace(top_k=top_k, pad_ids=pad_ids, basin_a_ids=basin_a_ids,
                       basin_b_ids=basin_b_ids, light=light, clamp_canvas=clamp_canvas,
                       clamp_mask=clamp_mask, clamp_from_step=0, burnin_mask=burnin_mask,
                       burnin_steps=burnin_steps,
                       burnin_reset_self_conditioning=burnin_reset_self_conditioning)
    if capture_steps is not None:
        model._begin_step_capture(capture_steps, scl=capture_scl)
    if lens:
        model._begin_lens(layers=lens_layers)
    generate_kwargs = dict(enc)
    if initial_canvas is not None:
        generate_kwargs["decoder_input_ids"] = initial_canvas
    out = model.generate(**generate_kwargs, generation_config=gen_config)
    steps = model._end_trace()
    cap = model._end_step_capture() if capture_steps is not None else None
    lens_buf = model._end_lens() if lens else None
    prompt_len = enc["input_ids"].shape[1]
    canvas_ids = out.sequences[0][prompt_len:prompt_len + p["C"]].tolist()
    final_text = tok.decode([t for t in canvas_ids if t not in set(pad_ids)], skip_special_tokens=False)
    return dict(steps=steps, cap=cap, lens_buf=lens_buf, canvas_ids=canvas_ids,
                final_text=final_text, enc=enc, pad_ids=pad_ids, T=len(steps), C=p["C"])


def prefill_replay(model, enc, B, C):
    """Re-prefill the encoder at batch size B (replicated prompt) and return everything a
    denoising-step decoder replay needs: (past_kv, decoder_attention_mask, decoder_position_ids)."""
    device = model.device
    input_ids = enc["input_ids"].to(device).repeat(B, 1)
    cur_len = input_ids.shape[1]
    past = DynamicCache(config=model.config.get_text_config(decoder=True))
    encoder_position_ids = torch.arange(0, cur_len, dtype=torch.int32, device=device).unsqueeze(0)
    attention_mask = torch.ones((B, cur_len), dtype=torch.bool, device=device)
    unproc, enc_mask = model._prepare_encoder_inputs(
        input_ids=input_ids, attention_mask=attention_mask,
        encoder_position_ids=encoder_position_ids, past_key_values=past,
        is_prefill=True, canvas_length=C, batch_size=B)
    with torch.no_grad():
        enc_out = model.model.encoder(input_ids=unproc, attention_mask=enc_mask,
                                      past_key_values=past, position_ids=encoder_position_ids)
    dec_attn = torch.nn.functional.pad(attention_mask, (0, C), value=True)
    dec_pos = torch.arange(cur_len, cur_len + C, dtype=torch.int32, device=device).unsqueeze(0)
    return enc_out.past_key_values, dec_attn, dec_pos


def replay_decoder(model, past, dec_attn, dec_pos, canvas, scl, B):
    """One denoising-step decoder forward on a captured step input, replicated B times along the
    batch axis. Caller manages grad mode / hooks. Returns the decoder output (normed hidden)."""
    canvas = canvas.to(model.device).expand(B, -1)
    scl_r = None if scl is None else scl.to(model.device, torch.bfloat16).expand(B, -1, -1)
    return model.model.decoder(
        decoder_input_ids=canvas, self_conditioning_logits=scl_r,
        decoder_attention_mask=dec_attn, past_key_values=past, decoder_position_ids=dec_pos)


def unembed(model, residual, softcap=True):
    """Residual [..., d] -> fp32 logits via the model's own final norm + lm_head (+ softcap).
    The matmul runs in the head's dtype (bf16), matching run_logitlens' precision."""
    w = model.lm_head.weight
    logits = model.lm_head(model.model.decoder.norm(residual.to(w.dtype))).float()
    if softcap:
        sc = float(model.final_logit_softcapping)
        logits = sc * torch.tanh(logits / sc)
    return logits


def jacobian_for_step(model, past, dec_attn, dec_pos, canvas, scl, valid_positions,
                      *, dim_batch, source_layers=SOURCE_LAYERS, target_layer=TARGET_LAYER,
                      fidelity_ref=None):
    """jlens.fitting.jacobian_for_prompt adapted to a replayed DG denoising step.

    Returns (jacobians {layer: [d,d] fp32 cpu}, n_valid, fidelity) where fidelity is the fraction
    of valid canvas positions whose replayed argmax matches the rollout trace (None if no ref)."""
    d = D_MODEL
    layers = model.model.decoder.layers
    jac = {l: torch.zeros(d, d, dtype=torch.float32) for l in source_layers}
    n_passes = math.ceil(d / dim_batch)
    with ActivationRecorder(layers, at=[*source_layers, target_layer],
                            start_graph_at=min(source_layers)) as rec, torch.enable_grad():
        replay_decoder(model, past, dec_attn, dec_pos, canvas, scl, dim_batch)
        tgt = rec.activations[target_layer]          # [B, C, d]
        srcs = [rec.activations[l] for l in source_layers]
        vp = valid_positions.to(tgt.device)
        fidelity = None
        if fidelity_ref is not None:
            with torch.no_grad():
                am = unembed(model, tgt[0], softcap=False).argmax(-1)
                ref = fidelity_ref.to(am.device)
                fidelity = float((am[vp] == ref[vp]).float().mean())
        bidx = torch.arange(dim_batch, device=tgt.device)
        cot = torch.zeros_like(tgt)
        for pi, d0 in enumerate(range(0, d, dim_batch)):
            n = min(dim_batch, d - d0)
            cot.zero_()
            cot[bidx[:n, None], vp[None, :], d0 + bidx[:n, None]] = 1.0
            grads = torch.autograd.grad(outputs=tgt, inputs=srcs, grad_outputs=cot,
                                        retain_graph=(pi < n_passes - 1))
            for l, g in zip(source_layers, grads, strict=True):
                jac[l][d0:d0 + n, :] = g[:n][:, vp.to(g.device), :].float().mean(dim=1).cpu()
            del grads
    return jac, int(len(vp)), fidelity
