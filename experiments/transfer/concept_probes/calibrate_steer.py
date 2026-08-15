"""Steering calibration: find the (layer, dose) window where steering visibly moves
GENERATIONS while text stays coherent — the knob the probe-layer/constant-vector
design got wrong (see report §5).

Changes vs run_saeprobes_gpu.phase_generate:
  - steering layer is a swept MID-DEPTH band {13,16,19,22}, not the probe's layer;
  - dose is RELATIVE: add coeff * ||resid at position|| * unit(dir) (layer/context
    invariant), instead of a constant Venhoff-norm vector;
  - only GENERATED positions are steered (AR decode steps / DG canvas), never the prompt;
  - content-inviting carrier prompts (the chat-persona attractor masked weak steering).

Concepts: top-8 topical ones (geography/news/code/sentiment/medical) by gemma probe AUC.
Arms: gemma_native, dg_native, dg_transfer(gemma dir on DG). Unit diff-of-means
directions are recomputed per swept layer from the saved acts (all layers are in acts/).

-> out/saeprobes/calibration_gens.json  (judge on the workbench with judge_calibration.py)

Run:
  srun ... bash concept_probes/slurm/ensure_and_run.sh concept_probes/calibrate_steer.py
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
spec2 = importlib.util.spec_from_file_location("rsg", REPO / "concept_probes/run_saeprobes_gpu.py")
rsg = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(rsg)

LAYERS = [13, 16, 19, 22]
DOSES = [0.1, 0.25, 0.5]
N_CONCEPTS = 8
CANVAS = 96  # matches gen_ar max_new / gen_dg C
TOPICAL = {"geography", "news/topic", "code/format", "sentiment/tone", "medical"}
PROMPTS = ["Write a short paragraph about anything you like.",
           "Here is a short text:"]


def gen_steer_hook(unit: torch.Tensor, coeff: float, plen: int):
    """Position-gated relative-dose steering: add coeff * ||resid_pos|| * unit at
    GENERATED positions only. Shape cases: [B,1,d] = AR decode step (steer);
    [B,plen,d] = AR prefill (skip); [B,plen+C,d] = DG full-seq denoising pass
    (steer >= plen); [B,C,d] = DG canvas-only pass (steer all); anything else:
    assume the last CANVAS positions are the canvas."""
    def hook(_m, _i, out):
        is_tuple = isinstance(out, tuple)
        resid = out[0] if is_tuple else out
        T = resid.shape[1]
        if T == 1:
            sl = slice(0, 1)
        elif T == plen:
            return out
        elif T == plen + CANVAS:
            sl = slice(plen, T)
        elif T == CANVAS:
            sl = slice(0, T)
        else:
            sl = slice(max(0, T - CANVAS), T)
        resid = resid.clone()
        r = resid[:, sl, :]
        resid[:, sl, :] = r + coeff * r.norm(dim=-1, keepdim=True) * unit.to(r.dtype)
        return (resid, *out[1:]) if is_tuple else resid
    return hook


def pick_concepts(dirs, steer):
    scored = []
    for tag, D in dirs.items():
        if D["category"] not in TOPICAL:
            continue
        e = steer.get(tag, {}).get("arms", {}).get("gemma_native_clean", {}).get("effect_pm1", 0)
        scored.append((D["auc_gemma_test"] + 0.05 * e, tag))
    top = [t for _, t in sorted(scored, reverse=True)[:N_CONCEPTS]]
    print(f"[calib] concepts: {top}")
    return top


def unit_dirs_at_layers(tag):
    """{(model, L): unit diff-of-means} from the saved acts (train split, all sweep layers)."""
    z = np.load(OUT / "acts" / f"{tag}.npz")
    layer_ids = z["layer_ids"].tolist()
    y = z["y_train"]
    out = {}
    for mk, key in [("gemma", "g_train"), ("dg", "d_train")]:
        A = z[key].astype(np.float32)
        for L in LAYERS:
            j = layer_ids.index(L)
            v = A[y == 1, j, :].mean(0) - A[y == 0, j, :].mean(0)
            out[(mk, L)] = torch.tensor(v / (np.linalg.norm(v) + 1e-8))
    return out


def prompt_len(tok, prompt, device):
    enc = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True,
                                  add_generation_prompt=True, return_dict=True,
                                  return_tensors="pt").to(device)
    return enc["input_ids"].shape[1]


def main():
    out_path = OUT / "calibration_gens.json"
    done = json.loads(out_path.read_text()) if out_path.exists() else {}
    dirs = torch.load(OUT / "directions.pt")["concepts"]
    steer = json.loads((OUT / "steer_logprob.json").read_text())
    concepts = pick_concepts(dirs, steer)

    model_g, tok = rcp.load_model("gemma4")
    _, layers_g = rcp.locate(model_g)
    model_d, _ = rcp.load_model("diffusiongemma")
    _, layers_d = rcp.locate(model_d)
    device = model_g.device

    n_done = 0
    for tag in concepts:
        dirs_l = unit_dirs_at_layers(tag)
        arms = {"gemma_native": (model_g, layers_g, rsg.gen_ar, "gemma", "gemma"),
                "dg_native": (model_d, layers_d, rsg.gen_dg, "dg", "dg"),
                "dg_transfer": (model_d, layers_d, rsg.gen_dg, "dg", "gemma")}
        for pi, prompt in enumerate(PROMPTS):
            plen = prompt_len(tok, prompt, device)
            for arm, (model, layers, gen, _mk, dirsrc) in arms.items():
                basekey = f"{tag}|{pi}|{arm}|base"
                if basekey not in done:
                    done[basekey] = gen(model, tok, prompt, seed=pi)
                for L in LAYERS:
                    unit = dirs_l[(dirsrc, L)].to(device)
                    for dose in DOSES:
                        for sign, sname in [(+1, "pos"), (-1, "neg")]:
                            key = f"{tag}|{pi}|{arm}|L{L}|{dose}|{sname}"
                            if key in done:
                                continue
                            h = layers[L].register_forward_hook(
                                gen_steer_hook(unit, sign * dose, plen))
                            try:
                                done[key] = gen(model, tok, prompt, seed=pi)
                            finally:
                                h.remove()
                            n_done += 1
                            if n_done % 50 == 0:
                                out_path.write_text(json.dumps(done))
                                print(f"[calib] {n_done} gens (+{len(done)} total) — last: {key}"
                                      f" -> {done[key][:80]!r}")
        out_path.write_text(json.dumps(done))
        print(f"[calib] concept {tag} DONE ({len(done)} entries)")
    out_path.write_text(json.dumps(done))
    print(f"[calib] wrote {out_path} ({len(done)} entries)")


if __name__ == "__main__":
    main()
