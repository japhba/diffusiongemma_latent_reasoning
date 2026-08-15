"""Causally-verified concept/feature probes for a single model.

Pipeline (model-uniform; runs identically for an autoregressive transformer and
for DiffusionGemma treated as a SINGLE forward pass — no denoising loop, we just
run the text backbone once):

  1. load model + tokenizer; locate the decoder-layer ModuleList + lm head.
  2. EXTRACT: for each concept's positive/negative stimuli, run one forward pass
     over the text backbone and capture the mean-pooled residual stream at a set
     of candidate layers (forward hooks; works for any architecture).
  3. PROBE: per concept, per candidate layer, fit a difference-of-means direction
     on a train split and score held-out detection AUC. Pick the best layer.
  4. CAUSALLY VERIFY (the bar that makes a probe "causal", single forward pass):
     steer the residual at the chosen layer by +/- alpha * unit(direction) *
     ||resid|| on a battery of neutral carrier prompts and measure the change in
     next-token log-prob of concept-diagnostic tokens relative to foil tokens.
     A probe is VERIFIED iff detection AUC is high AND steering moves the model's
     output toward the concept with the right sign, consistently across carriers.
  5. save directions + metadata for the verified probes.

Usage (on a GPU node):
  python concept_probes/run_concept_probes.py --model gemma4 \
      --out concept_probes/out/gemma4
  # quick end-to-end self-test on a tiny subset:
  python concept_probes/run_concept_probes.py --model qwen3-8b --selftest
"""
from __future__ import annotations

import argparse
import functools
import json
import time
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

print = functools.partial(print, flush=True)

REPO = Path(os.environ.get("DGLR_ROOT", Path(__file__).resolve().parents[1]))

MODEL_IDS = {
    "qwen3-8b": "Qwen/Qwen3-8B",
    "gemma4": "google/gemma-4-26b-a4b-it",
    "diffusiongemma": "google/diffusiongemma-26B-A4B-it",
}

# Neutral carrier prompts for the causal steering metric. Concept-neutral stems
# the model naturally continues; we read the next-token distribution at the last
# position under +/- steering.
CARRIERS = [
    "The weather this morning was",
    "She opened the door and",
    "After thinking about it for a while, I",
    "The meeting started late because",
    "He picked up the phone and said",
    "Yesterday we went to the",
    "The report concluded that",
    "When the train finally arrived,",
    "My neighbor told me that",
    "The first thing you notice is",
    "It all began when",
    "Looking back on the year, the",
    "The committee decided to",
    "On the table there was a",
    "They walked along the river and",
    "The teacher explained that",
    "In the end, what mattered most was",
    "The package arrived and inside it",
    "Every morning she would",
    "The old house at the end of the street",
    "According to the new plan, the team will",
    "He stared at the screen and",
    "The children gathered around to",
    "From the window I could see",
]


# ---------------------------------------------------------------------------
# Model loading + structure location.
# ---------------------------------------------------------------------------
def load_model(key: str, device_map="auto"):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    mid = MODEL_IDS[key]
    print(f"[load] {key} -> {mid}  (device_map={device_map})")
    t0 = time.time()
    if key == "diffusiongemma":
        from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion
        proc = AutoProcessor.from_pretrained(mid)
        tok = proc.tokenizer
        model = DiffusionGemmaForBlockDiffusion.from_pretrained(mid, dtype="auto", device_map=device_map)
    else:
        tok = AutoTokenizer.from_pretrained(mid)
        try:
            model = AutoModelForCausalLM.from_pretrained(mid, dtype="auto", device_map=device_map)
        except Exception as e:
            print(f"[load] AutoModelForCausalLM failed ({e}); trying image-text-to-text")
            from transformers import AutoModelForImageTextToText
            model = AutoModelForImageTextToText.from_pretrained(mid, dtype="auto", device_map=device_map)
    model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"[load] done in {time.time()-t0:.1f}s  device={model.device}")
    return model, tok


def locate(model) -> tuple[nn.Module, nn.ModuleList]:
    """Find (text_backbone, decoder_layers). The backbone is the parent module
    that owns the `.layers` ModuleList and exposes a forward(input_ids=...)."""
    best = None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) > 0:
            child = mod[0]
            if hasattr(child, "self_attn") or "DecoderLayer" in type(child).__name__:
                if best is None or len(mod) > len(best[1]):
                    best = (name, mod)
    if best is None:
        raise RuntimeError("could not locate decoder layers")
    layers_name, layers = best
    parent_name = layers_name.rsplit(".layers", 1)[0] if layers_name.endswith(".layers") else layers_name.rsplit(".", 1)[0]
    backbone = model.get_submodule(parent_name) if parent_name else model
    print(f"[locate] layers='{layers_name}' (n={len(layers)})  backbone='{parent_name}' ({type(backbone).__name__})")
    return backbone, layers


# ---------------------------------------------------------------------------
# Extraction: mean-pooled residual at candidate layers, one forward pass.
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract(backbone, layers, tok, texts: list[str], cand_layers: list[int],
            device, batch_size: int = 16, max_len: int = 64) -> np.ndarray:
    """Returns array [n_texts, n_cand_layers, d_model] of mean-pooled residuals
    (mean over non-BOS, non-pad tokens)."""
    captured: dict[int, torch.Tensor] = {}

    def mk_hook(li):
        def hook(_m, _i, out):
            captured[li] = out[0] if isinstance(out, tuple) else out
        return hook

    handles = [layers[li].register_forward_hook(mk_hook(li)) for li in cand_layers]
    feats = []
    try:
        tok.padding_side = "right"
        for s in range(0, len(texts), batch_size):
            batch = texts[s : s + batch_size]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                      max_length=max_len).to(device)
            captured.clear()
            backbone(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            mask = enc["attention_mask"].clone()
            mask[:, 0] = 0  # drop BOS from the pool
            denom = mask.sum(1, keepdim=True).clamp_min(1).float()  # [B,1]
            per_layer = []
            for li in cand_layers:
                h = captured[li].float()  # [B,L,d]
                m = mask.unsqueeze(-1).float()
                pooled = (h * m).sum(1) / denom  # [B,d]
                per_layer.append(pooled.cpu())
            feats.append(torch.stack(per_layer, dim=1))  # [B, n_layers, d]
    finally:
        for h in handles:
            h.remove()
    return torch.cat(feats, 0).numpy()


# ---------------------------------------------------------------------------
# Probe fitting.
# ---------------------------------------------------------------------------
def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv), dtype=float)
    ranks[order] = np.arange(1, len(allv) + 1)
    rp = ranks[: len(pos)].sum()
    n1, n2 = len(pos), len(neg)
    return float((rp - n1 * (n1 + 1) / 2) / (n1 * n2))


def fit_concept(pos_feats: np.ndarray, neg_feats: np.ndarray, cand_layers: list[int],
                seed: int) -> dict:
    """pos_feats/neg_feats: [n, n_layers, d]. For each layer fit diff-of-means on
    a train split, score held-out AUC; pick best layer; return direction over ALL
    data at that layer."""
    rng = np.random.default_rng(seed)
    npos, nlay, d = pos_feats.shape
    nneg = neg_feats.shape[0]
    pi = rng.permutation(npos); ni = rng.permutation(nneg)
    ptr, pte = pi[: int(0.7 * npos)], pi[int(0.7 * npos):]
    ntr, nte = ni[: int(0.7 * nneg)], ni[int(0.7 * nneg):]

    per_layer = []
    for j, li in enumerate(cand_layers):
        P, N = pos_feats[:, j, :], neg_feats[:, j, :]
        mu = np.concatenate([P[ptr], N[ntr]], 0).mean(0)
        sd = np.concatenate([P[ptr], N[ntr]], 0).std(0) + 1e-6
        Ptr = (P[ptr] - mu) / sd; Ntr = (N[ntr] - mu) / sd
        w = Ptr.mean(0) - Ntr.mean(0)
        w = w / (np.linalg.norm(w) + 1e-8)
        sp = ((P[pte] - mu) / sd) @ w
        sn = ((N[nte] - mu) / sd) @ w
        per_layer.append((int(li), float(auc(sp, sn))))

    # Pick best AUC; break ties (common: AUC saturates to ~1.0 across many layers)
    # toward the most CENTRAL layer — mid-depth directions are more abstract and
    # steer behaviour more cleanly than the earliest lexical-detector layer.
    best_auc = max(a for _, a in per_layer)
    center = float(np.median(cand_layers))
    near = [(li, a) for li, a in per_layer if a >= best_auc - 0.01]
    li_best = min(near, key=lambda t: abs(t[0] - center))[0]
    best = {"layer_idx": li_best, "auc": next(a for li, a in per_layer if li == li_best),
            "auc_by_layer": per_layer}

    # Refit the steering direction on ALL data at the chosen layer (raw resid space,
    # unit norm) — what the hook adds.
    j = cand_layers.index(best["layer_idx"])
    P, N = pos_feats[:, j, :], neg_feats[:, j, :]
    raw_dir = P.mean(0) - N.mean(0)
    raw_dir = raw_dir / (np.linalg.norm(raw_dir) + 1e-8)
    best["direction"] = raw_dir.astype(np.float32)
    return best


# ---------------------------------------------------------------------------
# Causal verification: steer at the chosen layer, measure next-token logit shift.
# ---------------------------------------------------------------------------
def all_pos_steer_hook(unit_dir: torch.Tensor, coeff: float):
    """Add coeff * unit_dir * ||resid|| to EVERY position's residual."""
    def hook(_m, _i, out):
        is_tuple = isinstance(out, tuple)
        resid = out[0] if is_tuple else out
        norms = resid.norm(dim=-1, keepdim=True)
        resid = resid + (coeff * norms) * unit_dir.to(resid.dtype)
        return (resid, *out[1:]) if is_tuple else resid
    return hook


@torch.no_grad()
def logits_last(backbone, lm_head, softcap, tok, prompts, device):
    """Next-token logits at the last real token for each prompt (left-padded)."""
    tok.padding_side = "left"
    enc = tok(prompts, return_tensors="pt", padding=True).to(device)
    out = backbone(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    h = out.last_hidden_state[:, -1, :]  # left-pad => last token is real for all
    logits = lm_head(h).float()
    if softcap:
        logits = softcap * torch.tanh(logits / softcap)
    return torch.log_softmax(logits, dim=-1)  # [B, V]


@torch.no_grad()
def causal_verify(backbone, layers, lm_head, softcap, tok, concept: dict,
                  layer_idx: int, direction: np.ndarray, alpha: float, device) -> dict:
    """Steer +/- alpha at layer_idx; measure concept-token vs foil-token logp shift."""
    unit = torch.tensor(direction, device=device)
    unit = unit / (unit.norm() + 1e-8)

    def tok_ids(words):
        ids = []
        for w in words:
            t = tok.encode(" " + w, add_special_tokens=False)
            if t:
                ids.append(t[0])
        return sorted(set(ids))

    pos_ids = tok_ids(concept["pos_tokens"])
    neg_ids = tok_ids(concept["neg_tokens"])
    if len(pos_ids) < 2 or len(neg_ids) < 2:
        return {"causal_effect": 0.0, "sign_consistency": 0.0, "monotone": False,
                "n_pos_ids": len(pos_ids), "n_neg_ids": len(neg_ids)}

    pos_ids_t = torch.tensor(pos_ids, device=device)
    neg_ids_t = torch.tensor(neg_ids, device=device)

    def concept_score(coeff):
        if coeff == 0.0:
            lp = logits_last(backbone, lm_head, softcap, tok, CARRIERS, device)
        else:
            hook = all_pos_steer_hook(unit, coeff)
            h = layers[layer_idx].register_forward_hook(hook)
            try:
                lp = logits_last(backbone, lm_head, softcap, tok, CARRIERS, device)
            finally:
                h.remove()
        sp = lp.index_select(1, pos_ids_t).mean(1)  # [B]
        sn = lp.index_select(1, neg_ids_t).mean(1)
        return (sp - sn).cpu().numpy()  # per-carrier concept score

    s_minus = concept_score(-alpha)
    s_zero = concept_score(0.0)
    s_plus = concept_score(+alpha)
    per_carrier = s_plus - s_minus
    return {
        "causal_effect": float(per_carrier.mean()),
        "sign_consistency": float((per_carrier > 0).mean()),
        "monotone": bool(s_plus.mean() > s_zero.mean() > s_minus.mean()),
        "score_minus": float(s_minus.mean()),
        "score_zero": float(s_zero.mean()),
        "score_plus": float(s_plus.mean()),
        "n_pos_ids": len(pos_ids), "n_neg_ids": len(neg_ids),
    }


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_IDS))
    ap.add_argument("--stimuli", default=str(REPO / "concept_probes/stimuli.json"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--alpha", type=float, default=4.0, help="steering coefficient")
    ap.add_argument("--auc-thresh", type=float, default=0.85)
    ap.add_argument("--effect-thresh", type=float, default=0.5)
    ap.add_argument("--sign-thresh", type=float, default=0.7)
    ap.add_argument("--n-layers-sweep", type=int, default=9)
    ap.add_argument("--selftest", action="store_true", help="tiny subset, prints diagnostics")
    ap.add_argument("--max-concepts", type=int, default=0, help="0 = all")
    ap.add_argument("--only", default="", help="comma-separated concept keys (overrides max-concepts)")
    args = ap.parse_args()

    concepts = json.loads(Path(args.stimuli).read_text())
    if args.only:
        keys = set(args.only.split(","))
        concepts = [c for c in concepts if c["key"] in keys]
    elif args.selftest and args.max_concepts == 0:
        args.max_concepts = 8
        concepts = concepts[: args.max_concepts]
    elif args.max_concepts:
        concepts = concepts[: args.max_concepts]
    print(f"[main] {len(concepts)} concepts; model={args.model} alpha={args.alpha}")

    model, tok = load_model(args.model)
    backbone, layers = locate(model)
    device = model.device
    n_layers = len(layers)
    lm_head = model.get_output_embeddings()
    softcap = float(getattr(model.config, "final_logit_softcapping", 0) or
                    getattr(getattr(model.config, "text_config", model.config), "final_logit_softcapping", 0) or 0)

    cand = sorted(set(int(round(f * (n_layers - 1)))
                      for f in np.linspace(0.15, 0.95, args.n_layers_sweep)))
    print(f"[main] candidate layers: {cand}  (of {n_layers})  softcap={softcap}")

    if args.selftest:
        # sanity: unsteered top continuation tokens for a couple of carriers
        lp = logits_last(backbone, lm_head, softcap, tok, CARRIERS[:2], device)
        for i in range(2):
            top = lp[i].topk(8).indices.tolist()
            print(f"[selftest] carrier {i!r} top-next: {tok.convert_ids_to_tokens(top)}")

    results = []
    t0 = time.time()
    for ci, c in enumerate(concepts):
        pf = extract(backbone, layers, tok, c["positive"], cand, device)
        nf = extract(backbone, layers, tok, c["negative"], cand, device)
        probe = fit_concept(pf, nf, cand, seed=ci)
        cv = causal_verify(backbone, layers, lm_head, softcap, tok, c,
                           probe["layer_idx"], probe["direction"], args.alpha, device)
        verified = (probe["auc"] >= args.auc_thresh and
                    cv["causal_effect"] >= args.effect_thresh and
                    cv["sign_consistency"] >= args.sign_thresh)
        rec = {
            "category": c["category"], "key": c["key"], "description": c["description"],
            "layer": probe["layer_idx"], "n_layers": n_layers,
            "auc": round(probe["auc"], 4),
            "causal_effect": round(cv["causal_effect"], 4),
            "sign_consistency": round(cv["sign_consistency"], 4),
            "monotone": cv["monotone"],
            "score_minus": round(cv.get("score_minus", 0), 4),
            "score_zero": round(cv.get("score_zero", 0), 4),
            "score_plus": round(cv.get("score_plus", 0), 4),
            "n_pos_ids": cv["n_pos_ids"], "n_neg_ids": cv["n_neg_ids"],
            "alpha": args.alpha, "verified": bool(verified),
            "_direction": probe["direction"],
        }
        results.append(rec)
        flag = "VERIFIED" if verified else "        "
        print(f"[{ci+1:>3}/{len(concepts)}] {flag} {c['key']:<26} L{probe['layer_idx']:<2} "
              f"auc={probe['auc']:.3f} eff={cv['causal_effect']:+.3f} "
              f"sign={cv['sign_consistency']:.2f} mono={cv['monotone']}")

    n_ver = sum(r["verified"] for r in results)
    print(f"\n[main] verified {n_ver}/{len(results)} concepts in {time.time()-t0:.0f}s")

    if args.out:
        out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
        dirs = {r["key"]: torch.tensor(r.pop("_direction")) for r in results}
        torch.save({"model": args.model, "model_id": MODEL_IDS[args.model],
                    "directions": dirs, "n_layers": n_layers}, out / "directions.pt")
        (out / "probes.json").write_text(json.dumps(results, indent=1))
        print(f"[main] wrote {out}/directions.pt + probes.json")
    else:
        for r in results:
            r.pop("_direction", None)


if __name__ == "__main__":
    main()
