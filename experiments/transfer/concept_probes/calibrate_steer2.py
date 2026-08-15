"""Calibration round 2 — steer DiffusionGemma where it can actually hear it.

Round-1 discovery: DG's canvas is denoised by a SEPARATE decoder stack
(model.decoder.layers, DiffusionGemmaDecoderTextLayer, no weight sharing with the
encoder), and generate() compiles the denoise forward (fullgraph) so module hooks
never fire there. Everything steered so far touched only the ENCODER (context)
representation. This script steers both stacks properly, eager via a
non-compileable DynamicCache (the compile gate is past_key_values.is_compileable):

  dg_enc_native    encoder-layer hook, DG-encoder-fit unit dir (context steering)
  dg_enc_transfer  encoder-layer hook, GEMMA-fit unit dir (the transfer arm)
  dg_dec_native    decoder-layer hook, DECODER-fit unit dir (true canvas steering)
  dg_dec_transfer  decoder-layer hook, GEMMA-fit unit dir (cross-stack transfer)

Decoder directions are fit on decoder activations of the same train texts: one
forward per batch with the text as a clean canvas (input_ids = BOS prompt,
decoder_input_ids = text), mean-pooled over non-pad canvas positions.
Also records cos(dec_dir, enc_dir) and cos(dec_dir, gemma_dir) per layer.

Relative dose everywhere: resid += coeff * ||resid_pos|| * unit.
-> out/saeprobes/calibration_gens2.json + decoder_dir_cos.json

srun ... bash concept_probes/slurm/ensure_and_run.sh concept_probes/calibrate_steer2.py
"""
from __future__ import annotations

import functools
import json
import os
from pathlib import Path

import numpy as np
import torch

print = functools.partial(print, flush=True)
REPO = Path(os.environ.get("DGLR_ROOT", Path(__file__).resolve().parents[1]))
OUT = Path(os.environ.get("SAEP_OUT", REPO / "concept_probes/out/saeprobes"))

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util
spec = importlib.util.spec_from_file_location("rcp", REPO / "concept_probes/run_concept_probes.py")
rcp = importlib.util.module_from_spec(spec); spec.loader.exec_module(rcp)
spec3 = importlib.util.spec_from_file_location("cst", REPO / "concept_probes/calibrate_steer.py")
cst = importlib.util.module_from_spec(spec3); spec3.loader.exec_module(cst)
import saeprobes_data as sd  # noqa: E402

LAYERS = cst.LAYERS          # [13, 16, 19, 22]
DOSES = cst.DOSES            # [0.1, 0.25, 0.5]
PROMPTS = cst.PROMPTS
CANVAS, MAXSTEPS = 96, 24


def rel_hook(unit: torch.Tensor, coeff: float):
    """Relative-dose steering at every position of this stack's forward."""
    def hook(_m, _i, out):
        is_tuple = isinstance(out, tuple)
        resid = out[0] if is_tuple else out
        resid = resid + coeff * resid.norm(dim=-1, keepdim=True) * unit.to(resid.dtype)
        return (resid, *out[1:]) if is_tuple else resid
    return hook


@torch.no_grad()
def gen_dg_eager(model, tok, prompt, seed=0):
    """gen_dg with a non-compileable cache so decoder-layer hooks fire per step."""
    from transformers import DynamicCache
    from transformers.models.diffusion_gemma.generation_diffusion_gemma import (
        DiffusionGemmaGenerationConfig, EntropyBoundSamplerConfig)
    torch.manual_seed(seed)
    model.config.canvas_length = CANVAS
    enc = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True,
                                  add_generation_prompt=True, return_dict=True,
                                  return_tensors="pt").to(model.device)
    eos = model.config.eos_token_id; eos = eos if isinstance(eos, list) else [eos]
    pad = getattr(model.config, "pad_token_id", 0) or 0
    gc = DiffusionGemmaGenerationConfig(
        max_new_tokens=CANVAS, max_denoising_steps=MAXSTEPS,
        sampler_config=EntropyBoundSamplerConfig(entropy_bound=0.1),
        t_min=0.4, t_max=0.8, stability_threshold=1, confidence_threshold=0.005,
        pad_token_id=pad, eos_token_id=eos)
    out = model.generate(**enc, generation_config=gc, past_key_values=DynamicCache())
    seq = out.sequences[0]; plen = enc["input_ids"].shape[1]
    canvas = seq[plen : plen + CANVAS].tolist()
    bad = set([pad] + list(eos))
    return tok.decode([t for t in canvas if t not in bad], skip_special_tokens=False).strip()


@torch.no_grad()
def decoder_acts(model, tok, texts, dec_layers, device, batch=8, max_len=96):
    """Decoder-stack representations of given texts: text fed as a CLEAN canvas
    (input_ids = BOS prompt), mean-pooled over non-pad canvas positions.
    Returns [N, n_layers, d]."""
    cap = {}
    def mk(li):
        def h(_m, _i, out): cap[li] = out[0] if isinstance(out, tuple) else out
        return h
    handles = [dec_layers[li].register_forward_hook(mk(li)) for li in LAYERS]
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.pad_token_id
    feats = []
    try:
        tok.padding_side = "right"; tok.truncation_side = "left"
        for s in range(0, len(texts), batch):
            chunk = texts[s : s + batch]
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                      max_length=max_len, add_special_tokens=False)
            ids = enc["input_ids"].to(device); attn = enc["attention_mask"].to(device)
            B, T = ids.shape
            prompt_ids = torch.full((B, 1), bos, dtype=torch.long, device=device)
            cap.clear()
            model(input_ids=prompt_ids,
                  attention_mask=torch.ones_like(prompt_ids, dtype=torch.bool),
                  decoder_input_ids=ids,
                  decoder_position_ids=torch.arange(1, 1 + T, device=device).unsqueeze(0).expand(B, -1))
            m = attn.unsqueeze(-1).float()
            denom = attn.sum(1, keepdim=True).clamp_min(1).float()
            feats.append(torch.stack(
                [(cap[li].float() * m).sum(1) / denom for li in LAYERS], dim=1).cpu())
    finally:
        for h in handles:
            h.remove()
    return torch.cat(feats).numpy()


def unit(v):
    return v / (np.linalg.norm(v) + 1e-8)


def main():
    out_path = OUT / "calibration_gens2.json"
    done = json.loads(out_path.read_text()) if out_path.exists() else {}
    steer = json.loads((OUT / "steer_logprob.json").read_text())
    dirs_meta = torch.load(OUT / "directions.pt")["concepts"]
    env_tags = os.environ.get("SAEP_TAGS")
    concepts = env_tags.split(",") if env_tags else cst.pick_concepts(dirs_meta, steer)
    print(f"[calib2] concepts: {concepts}")
    datasets = {d["tag"]: d for d in sd.load_datasets() if d["tag"] in set(concepts)}

    model, tok = rcp.load_model("diffusiongemma")
    _, enc_layers = rcp.locate(model)                 # encoder text stack
    dec_layers = model.model.decoder.layers           # decoder (canvas) stack
    device = model.device
    print(f"[calib2] encoder stack: {type(enc_layers[0]).__name__} x{len(enc_layers)}; "
          f"decoder stack: {type(dec_layers[0]).__name__} x{len(dec_layers)}")

    # --- hook-liveness check: decoder hooks MUST fire during eager generate ---
    seen = []
    h = dec_layers[LAYERS[0]].register_forward_hook(
        lambda _m, _i, out: seen.append(tuple((out[0] if isinstance(out, tuple) else out).shape)))
    try:
        _ = gen_dg_eager(model, tok, PROMPTS[0], seed=0)
    finally:
        h.remove()
    assert seen, "decoder hooks did NOT fire during eager generate — DynamicCache gate failed"
    print(f"[calib2] decoder hook liveness OK: {len(seen)} calls, shapes {set(seen)}")

    # --- per-concept unit directions in all three spaces ---
    dir_cos = {}
    all_dirs = {}
    for tag in concepts:
        z = np.load(OUT / "acts" / f"{tag}.npz")
        layer_ids = z["layer_ids"].tolist(); y = z["y_train"]
        d = datasets[tag]
        dec = decoder_acts(model, tok, d["texts_train"], dec_layers, device)
        y_arr = np.array(d["y_train"])
        per_layer = {}
        cosrow = {}
        for j, L in enumerate(LAYERS):
            je = layer_ids.index(L)
            g = unit(z["g_train"][y == 1, je, :].astype(np.float64).mean(0)
                     - z["g_train"][y == 0, je, :].astype(np.float64).mean(0))
            e = unit(z["d_train"][y == 1, je, :].astype(np.float64).mean(0)
                     - z["d_train"][y == 0, je, :].astype(np.float64).mean(0))
            dd = unit(dec[y_arr == 1, j, :].astype(np.float64).mean(0)
                      - dec[y_arr == 0, j, :].astype(np.float64).mean(0))
            per_layer[L] = {"gemma": torch.tensor(g, dtype=torch.float32),
                            "enc": torch.tensor(e, dtype=torch.float32),
                            "dec": torch.tensor(dd, dtype=torch.float32)}
            cosrow[L] = {"dec_enc": float(dd @ e), "dec_gemma": float(dd @ g),
                         "enc_gemma": float(e @ g)}
        all_dirs[tag] = per_layer
        dir_cos[tag] = cosrow
        print(f"[calib2] dirs {tag}: cos(dec,enc)@L16={cosrow[16]['dec_enc']:+.2f} "
              f"cos(dec,gemma)@L16={cosrow[16]['dec_gemma']:+.2f}")
    (OUT / "decoder_dir_cos.json").write_text(json.dumps(dir_cos, indent=1))

    # --- generation grid ---
    ARMS = {  # arm -> (layers ModuleList, direction space)
        "dg_enc_native": (enc_layers, "enc"),
        "dg_enc_transfer": (enc_layers, "gemma"),
        "dg_dec_native": (dec_layers, "dec"),
        "dg_dec_transfer": (dec_layers, "gemma"),
    }
    n_done = 0
    for tag in concepts:
        for pi, prompt in enumerate(PROMPTS):
            for arm, (layers, space) in ARMS.items():
                basekey = f"{tag}|{pi}|{arm}|base"
                if basekey not in done:
                    done[basekey] = gen_dg_eager(model, tok, prompt, seed=pi)
                for L in LAYERS:
                    u = all_dirs[tag][L][space].to(device)
                    for dose in DOSES:
                        for sign, sname in [(+1, "pos"), (-1, "neg")]:
                            key = f"{tag}|{pi}|{arm}|L{L}|{dose}|{sname}"
                            if key in done:
                                continue
                            h = layers[L].register_forward_hook(rel_hook(u, sign * dose))
                            try:
                                done[key] = gen_dg_eager(model, tok, prompt, seed=pi)
                            finally:
                                h.remove()
                            n_done += 1
                            if n_done % 50 == 0:
                                out_path.write_text(json.dumps(done))
                                print(f"[calib2] {n_done} gens — last: {key} -> {done[key][:80]!r}")
        out_path.write_text(json.dumps(done))
        print(f"[calib2] concept {tag} DONE ({len(done)} entries)")
    out_path.write_text(json.dumps(done))
    print(f"[calib2] wrote {out_path} ({len(done)} entries)")


if __name__ == "__main__":
    main()
