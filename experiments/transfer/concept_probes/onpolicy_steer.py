"""CANONICAL (CAA-style, on-policy) steering vectors — the closing experiment.

Everything read-time failed to steer generations (see report §4/§5b). The canonical
recipe derives vectors from states the model occupies WHILE GENERATING:

  1. SEED: prompt the model to continue concept-positive vs concept-negative texts
     (24 each, greedy, chat format) and keep the generated continuations.
  2. DERIVE: mean residual activation over the GENERATED tokens at mid-depth layers
     {13, 16} — gemma: one causal forward over prompt+continuation, read the
     continuation positions (identical to generation-time states); DG: decoder-mode
     forward with the continuation as a clean canvas over the prompt's KV cache
     (≈ final denoising step state).  v = mean(pos-seeded) − mean(neg-seeded), RAW
     (unnormalized, CAA convention).
  3. STEER: add coeff·v at ONE layer, GENERATED positions only (CAA: "all positions
     after the prompt"), coeff ∈ {±1, ±3, ±10} (plus effective relative magnitude
     logged), same carrier prompts + judge as the calibration rounds.

Arms: gemma_onpolicy (AR decode), dg_dec_onpolicy (eager denoising, decoder stack).
-> out/saeprobes/onpolicy_gens.json, onpolicy_vectors.pt, onpolicy_meta.json

srun ... bash concept_probes/slurm/ensure_and_run.sh concept_probes/onpolicy_steer.py
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
def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, REPO / f"concept_probes/{fname}")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod
rcp = _load("rcp", "run_concept_probes.py")
rsg = _load("rsg", "run_saeprobes_gpu.py")
cst = _load("cst", "calibrate_steer.py")
cs2 = _load("cs2", "calibrate_steer2.py")
import saeprobes_data as sd  # noqa: E402

LAYERS = [13, 16]
COEFFS = [1.0, 3.0, 10.0]
N_SEEDS = 24                # per class, greedy continuations
CONT_TOKENS = 64
PROMPTS = cst.PROMPTS       # same carriers as the calibration rounds
CANVAS = 96


def const_gen_hook(vec: torch.Tensor, plen: int):
    """CAA application: add the raw vector at GENERATED positions only."""
    def hook(_m, _i, out):
        is_tuple = isinstance(out, tuple)
        resid = out[0] if is_tuple else out
        T = resid.shape[1]
        if T == 1:
            sl = slice(0, 1)              # AR decode step
        elif T == plen:
            return out                    # AR prefill
        elif T == CANVAS:
            sl = slice(0, T)              # DG decoder canvas pass
        elif T == plen + CANVAS:
            sl = slice(plen, T)
        else:
            sl = slice(max(0, T - CANVAS), T)
        resid = resid.clone()
        resid[:, sl, :] = resid[:, sl, :] + vec.to(resid.dtype)
        return (resid, *out[1:]) if is_tuple else resid
    return hook


def cont_prompt(text: str) -> str:
    return f"Continue this text naturally:\n\n{text}"


@torch.no_grad()
def continuations(model, tok, texts, gen_fn, device):
    """Greedy continuations of seed texts (chat format). Returns list[str]."""
    outs = []
    for i, t in enumerate(texts):
        outs.append(gen_fn(model, tok, cont_prompt(t[:600]), seed=0))
    return outs


@torch.no_grad()
def gemma_gen_acts(model, backbone, layers, tok, seeds, conts, device, batch=8):
    """Mean residual over CONTINUATION positions at LAYERS, one causal forward
    over prompt+continuation (identical to generation-time states). [N, L, d]"""
    feats = []
    for s in range(0, len(seeds), batch):
        msgs, plens = [], []
        for seed, cont in zip(seeds[s:s + batch], conts[s:s + batch]):
            enc = tok.apply_chat_template([{"role": "user", "content": cont_prompt(seed[:600])}],
                                          tokenize=True, add_generation_prompt=True,
                                          return_dict=True)["input_ids"]
            enc = list(enc)
            plens.append(len(enc))
            msgs.append(enc + tok(cont, add_special_tokens=False)["input_ids"][:CONT_TOKENS])
        maxlen = max(len(m) for m in msgs)
        pad = tok.pad_token_id
        ids = torch.full((len(msgs), maxlen), pad, dtype=torch.long)
        attn = torch.zeros_like(ids)
        for i, m in enumerate(msgs):
            ids[i, :len(m)] = torch.tensor(m); attn[i, :len(m)] = 1
        cap = rsg.capture(backbone, layers, LAYERS, ids.to(device), attn.to(device))
        for i, m in enumerate(msgs):
            gen_slice = slice(plens[i], len(m))
            if gen_slice.stop <= gen_slice.start:
                continue
            feats.append(torch.stack([cap[L][i, gen_slice].float().mean(0).cpu() for L in LAYERS]))
    return torch.stack(feats).numpy()


@torch.no_grad()
def dg_gen_acts(model, tok, seeds, conts, dec_layers, device):
    """DG decoder-mode representation of its own continuations: continuation as a
    clean canvas over the prompt's encoder KV cache. [N, L, d]"""
    from transformers import DynamicCache
    feats = []
    cap = {}
    def mk(li):
        def h(_m, _i, out): cap[li] = out[0] if isinstance(out, tuple) else out
        return h
    handles = [dec_layers[li].register_forward_hook(mk(li)) for li in LAYERS]
    try:
        for seed, cont in zip(seeds, conts):
            ids = tok(cont, add_special_tokens=False, return_tensors="pt",
                      truncation=True, max_length=CANVAS)["input_ids"].to(device)
            if ids.shape[1] < 4:
                continue
            enc = tok.apply_chat_template([{"role": "user", "content": cont_prompt(seed[:600])}],
                                          tokenize=True, add_generation_prompt=True,
                                          return_dict=True, return_tensors="pt").to(device)
            plen = enc["input_ids"].shape[1]
            cap.clear()
            model(input_ids=enc["input_ids"],
                  attention_mask=torch.ones_like(enc["input_ids"], dtype=torch.bool),
                  decoder_input_ids=ids,
                  decoder_position_ids=torch.arange(plen, plen + ids.shape[1], device=device).unsqueeze(0))
            feats.append(torch.stack([cap[L][0].float().mean(0).cpu() for L in LAYERS]))
    finally:
        for h in handles:
            h.remove()
    return torch.stack(feats).numpy()


def main():
    gens_path = OUT / "onpolicy_gens.json"
    done = json.loads(gens_path.read_text()) if gens_path.exists() else {}
    vec_path = OUT / "onpolicy_vectors.pt"
    vectors = torch.load(vec_path) if vec_path.exists() else {}
    steer = json.loads((OUT / "steer_logprob.json").read_text())
    dirs_meta = torch.load(OUT / "directions.pt")["concepts"]
    env_tags = os.environ.get("SAEP_TAGS")
    concepts = env_tags.split(",") if env_tags else cst.pick_concepts(dirs_meta, steer)
    datasets = {d["tag"]: d for d in sd.load_datasets() if d["tag"] in set(concepts)}

    model_g, tok = rcp.load_model("gemma4")
    bb_g, layers_g = rcp.locate(model_g)
    model_d, _ = rcp.load_model("diffusiongemma")
    _, enc_layers_d = rcp.locate(model_d)
    dec_layers = model_d.model.decoder.layers
    device = model_g.device

    meta = json.loads((OUT / "onpolicy_meta.json").read_text()) if (OUT / "onpolicy_meta.json").exists() else {}
    for tag in concepts:
        if tag in vectors:
            continue
        d = datasets[tag]
        # semantic polarity (label fix): seed with true concept-positive texts
        sem_pos = 0 if d["flipped"] else 1
        pos_seeds = [t for t, y in zip(d["texts_train"], d["y_train"]) if y == sem_pos][:N_SEEDS]
        neg_seeds = [t for t, y in zip(d["texts_train"], d["y_train"]) if y != sem_pos][:N_SEEDS]

        rec = {}
        for mk_, model, gen_fn in [("gemma", model_g, rsg.gen_ar), ("dg", model_d, cs2.gen_dg_eager)]:
            pos_cont = continuations(model, tok, pos_seeds, gen_fn, device)
            neg_cont = continuations(model, tok, neg_seeds, gen_fn, device)
            if mk_ == "gemma":
                A = gemma_gen_acts(model_g, bb_g, layers_g, tok, pos_seeds, pos_cont, device)
                B = gemma_gen_acts(model_g, bb_g, layers_g, tok, neg_seeds, neg_cont, device)
            else:
                A = dg_gen_acts(model_d, tok, pos_seeds, pos_cont, dec_layers, device)
                B = dg_gen_acts(model_d, tok, neg_seeds, neg_cont, dec_layers, device)
            for j, L in enumerate(LAYERS):
                v = A[:, j, :].mean(0) - B[:, j, :].mean(0)
                rec[(mk_, L)] = torch.tensor(v, dtype=torch.float32)
                resid_norm = float(np.linalg.norm(A[:, j, :], axis=1).mean())
                meta.setdefault(tag, {})[f"{mk_}_L{L}"] = {
                    "v_norm": float(np.linalg.norm(v)), "resid_norm": resid_norm,
                    "rel": float(np.linalg.norm(v) / resid_norm)}
            print(f"[onpolicy] {tag} {mk_}: |v|/|resid| = "
                  f"{meta[tag][f'{mk_}_L16']['rel']:.3f} @L16 "
                  f"(example cont: {pos_cont[0][:60]!r})")
        vectors[tag] = rec
        torch.save(vectors, vec_path)
        (OUT / "onpolicy_meta.json").write_text(json.dumps(meta, indent=1))

    # cos vs read-time directions (diagnostic)
    for tag in concepts:
        z = np.load(OUT / "acts" / f"{tag}.npz")
        lids = z["layer_ids"].tolist(); y = z["y_train"]
        j16 = lids.index(16)
        read_dir = z["g_train"][y == 1, j16, :].astype(np.float64).mean(0) - \
                   z["g_train"][y == 0, j16, :].astype(np.float64).mean(0)
        v = vectors[tag][("gemma", 16)].numpy().astype(np.float64)
        sign = -1.0 if datasets[tag]["flipped"] else 1.0
        cos = sign * float(v @ read_dir / (np.linalg.norm(v) * np.linalg.norm(read_dir) + 1e-8))
        meta[tag]["cos_onpolicy_readtime_L16"] = cos
    (OUT / "onpolicy_meta.json").write_text(json.dumps(meta, indent=1))

    # --- steered generations, CAA application ---
    n = 0
    for tag in concepts:
        arms = {"gemma_onpolicy": (model_g, layers_g, rsg.gen_ar, "gemma"),
                "dg_dec_onpolicy": (model_d, dec_layers, cs2.gen_dg_eager, "dg")}
        for pi, prompt in enumerate(PROMPTS):
            plen = cst.prompt_len(tok, prompt, device)
            for arm, (model, layers, gen_fn, mk_) in arms.items():
                basekey = f"{tag}|{pi}|{arm}|base"
                if basekey not in done:
                    done[basekey] = gen_fn(model, tok, prompt, seed=pi)
                for L in LAYERS:
                    v = vectors[tag][(mk_, L)].to(device)
                    for coeff in COEFFS:
                        for sign, sname in [(+1, "pos"), (-1, "neg")]:
                            key = f"{tag}|{pi}|{arm}|L{L}|{coeff}|{sname}"
                            if key in done:
                                continue
                            h = layers[L].register_forward_hook(const_gen_hook(sign * coeff * v, plen))
                            try:
                                done[key] = gen_fn(model, tok, prompt, seed=pi)
                            finally:
                                h.remove()
                            n += 1
                            if n % 50 == 0:
                                gens_path.write_text(json.dumps(done))
                                print(f"[onpolicy] {n} gens — last: {key} -> {done[key][:80]!r}")
        gens_path.write_text(json.dumps(done))
        print(f"[onpolicy] concept {tag} DONE ({len(done)} entries)")
    gens_path.write_text(json.dumps(done))
    print(f"[onpolicy] wrote {gens_path} ({len(done)} entries)")


if __name__ == "__main__":
    main()
