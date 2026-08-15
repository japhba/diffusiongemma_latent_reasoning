"""Fix gemma steering: dose-normalized, scope-extended sweep with the existing
on-policy vectors (gemma only — no DG load).

Hypotheses from the 24-concept round (§5.3): failures cluster where the raw
coefficient never reached the working band. Winners sat at an EFFECTIVE dose of
0.7–1.3x the local residual norm; weak vectors (semantically-adjacent negatives,
|v|/|resid| 0.06–0.13) were under-dosed at coeff 3 and only sometimes rescued at 10.
Also: we never steered gemma's PROMPT representation (DG's best effects were
context-side), and never injected at two layers at once.

Grid per concept (24 on-policy concepts, vectors from onpolicy_vectors.pt):
  arm    gemma_gen   generated positions only, one layer   (the §5.3 baseline scope)
         gemma_all   ALL positions incl. prompt, one layer (context + generation)
         gemma_both  generated positions, L13 AND L16 together
  layer  L13, L16 (both for gemma_both)
  dose   RELATIVE {0.5, 0.9, 1.3} x local residual norm    (coeff = rel*resid/|v|)
  sign   +/-, carriers: the 2 standard ones. Greedy, judged as usual.

-> out/saeprobes/gemma_fix_gens.json

srun ... bash concept_probes/slurm/ensure_and_run.sh concept_probes/gemma_fix_steer.py
"""
from __future__ import annotations

import functools
import json
import os
from pathlib import Path

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
ops = _load("ops", "onpolicy_steer.py")

LAYERS = [13, 16]
REL_DOSES = [0.5, 0.9, 1.3]
PROMPTS = cst.PROMPTS


def gated_hook(vec: torch.Tensor, plen: int, gate: bool):
    """Add vec at generated positions (gate=True) or ALL positions (gate=False)."""
    def hook(_m, _i, out):
        is_tuple = isinstance(out, tuple)
        resid = out[0] if is_tuple else out
        T = resid.shape[1]
        if gate and T == plen:
            return out                       # prefill untouched in gated mode
        sl = slice(0, 1) if T == 1 else slice(0, T) if not gate else slice(plen, T)
        if gate and 1 < T <= plen:
            return out
        resid = resid.clone()
        resid[:, sl, :] = resid[:, sl, :] + vec.to(resid.dtype)
        return (resid, *out[1:]) if is_tuple else resid
    return hook


def main():
    out_path = OUT / "gemma_fix_gens.json"
    done = json.loads(out_path.read_text()) if out_path.exists() else {}
    vectors = torch.load(OUT / "onpolicy_vectors.pt")
    meta = json.loads((OUT / "onpolicy_meta.json").read_text())

    model, tok = rcp.load_model("gemma4")
    _, layers = rcp.locate(model)
    device = model.device

    tags = [t for t in vectors if t in meta]
    print(f"[fix] {len(tags)} concepts")
    n = 0
    for tag in tags:
        vecs = {}
        for L in LAYERS:
            v = vectors[tag][("gemma", L)].to(device)
            m = meta[tag][f"gemma_L{L}"]
            vecs[L] = {rel: v * (rel * m["resid_norm"] / max(m["v_norm"], 1e-6))
                       for rel in REL_DOSES}
        for pi, prompt in enumerate(PROMPTS):
            plen = cst.prompt_len(tok, prompt, device)
            basekey = f"{tag}|{pi}|gemma_gen|base"
            if basekey not in done:
                done[basekey] = rsg.gen_ar(model, tok, prompt, seed=pi)
                for arm in ["gemma_all", "gemma_both"]:
                    done[f"{tag}|{pi}|{arm}|base"] = done[basekey]
            cells = []
            for L in LAYERS:
                cells += [("gemma_gen", [(layers[L], True)], f"L{L}"),
                          ("gemma_all", [(layers[L], False)], f"L{L}")]
            cells.append(("gemma_both", [(layers[13], True), (layers[16], True)], "L1316"))
            for arm, hookspecs, Lname in cells:
                for rel in REL_DOSES:
                    for sign, sname in [(+1, "pos"), (-1, "neg")]:
                        key = f"{tag}|{pi}|{arm}|{Lname}|{rel}|{sname}"
                        if key in done:
                            continue
                        handles = []
                        try:
                            for mod, gate in hookspecs:
                                L_here = 13 if mod is layers[13] else 16
                                handles.append(mod.register_forward_hook(
                                    gated_hook(sign * vecs[L_here][rel], plen, gate)))
                            done[key] = rsg.gen_ar(model, tok, prompt, seed=pi)
                        finally:
                            for h in handles:
                                h.remove()
                        n += 1
                        if n % 50 == 0:
                            out_path.write_text(json.dumps(done))
                            print(f"[fix] {n} gens — last: {key} -> {done[key][:80]!r}")
        out_path.write_text(json.dumps(done))
        print(f"[fix] concept {tag} DONE ({len(done)} entries)")
    out_path.write_text(json.dumps(done))
    print(f"[fix] wrote {out_path} ({len(done)} entries)")


if __name__ == "__main__":
    main()
