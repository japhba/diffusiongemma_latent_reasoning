"""Gemma-4 vs DiffusionGemma on the SAE-Probes concepts (arXiv 2502.16681) — GPU phases.

Revision of the synthetic-stimuli pipeline: all data is REAL text from the paper's
binary probing datasets (saeprobes_data.py). Both 26B models are loaded once
(shared tokenizer -> position-aligned). Steering follows Venhoff et al.
(arXiv 2506.18167): difference-of-means vectors (concept-positive mean minus
overall mean), norm-matched to the overall mean-activation norm at the chosen
layer, added at every position of ONE layer.

Phases (idempotent; `--phase all` runs everything, artifacts under out/saeprobes/):

  rsa      per-(layer, position) cosine between the two models' residual streams
           over the concept texts, clean AND 10%-noised (same corrupted ids into
           both models) -> repr_cosine_clean.npz / repr_cosine_noised.npz
  extract  per dataset: LAST-token residuals (SAE-Probes read convention) at a
           9-layer sweep, train+test x {clean, noised-context} x both models
           -> acts/<tag>.npz (float16) ; plus gemma-fit diff-of-means directions
           (best layer by held-out AUC) + DG-native directions -> directions.pt
  steer    logprob steering: Venhoff vectors at coeff {-2,-1,0,+1,+2}; read the
           next-token logp of the TRUE continuation at 4 prefix cut points per
           held-out text (last-position read = leak-free under DG's bidirectional
           single-forward convention). Arms: gemma+native, dg+native,
           dg+transfer(gemma dir), x {clean, noised} -> steer_logprob.json
  generate actual generations under steering for the top concepts (gemma via AR
           generate, DG via its denoising loop; hooks persist across steps).
           Arms: gemma+native, dg+native, dg+transfer -> generations.json
           (judged later on the workbench via judge_steer_gens.py)

Run (one GPU, H200):
  srun -p general,overflow --qos=high --gres=gpu:1 --cpus-per-task=16 --mem=200G \
    --time=12:00:00 --job-name=saep_cprobe \
    bash concept_probes/slurm/ensure_and_run.sh concept_probes/run_saeprobes_gpu.py --phase all
"""
from __future__ import annotations

import argparse
import functools
import json
import time
from pathlib import Path

import numpy as np
import torch

print = functools.partial(print, flush=True)
REPO = Path(os.environ.get("DGLR_ROOT", Path(__file__).resolve().parents[1]))
import os
OUT = Path(os.environ.get("SAEP_OUT", REPO / "concept_probes/out/saeprobes"))

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util
spec = importlib.util.spec_from_file_location("rcp", REPO / "concept_probes/run_concept_probes.py")
rcp = importlib.util.module_from_spec(spec); spec.loader.exec_module(rcp)
import saeprobes_data as sd  # noqa: E402

NOISE_FRAC = 0.10
P = 128                      # position axis of the cosine map
MAX_LEN = 512                # last-token read window (truncation_side=left, paper convention)
STEER_COEFFS = [-2.0, -1.0, 0.0, 1.0, 2.0]
N_STEER_TEXTS = 8            # held-out texts per class for the logprob steer metric
N_CUTS = 4                   # prefix cut points per text
GEN_TOP_K = 12               # concepts taken into the generation phase
# Venhoff-style application strength, per model: gemma (AR) degenerates at ±2 while
# DG's denoising loop stays fluent there but barely moves at ±1.
GEN_COEFF = {"gemma": 1.0, "dg": 2.0}
GEN_PROMPTS = ["Tell me about your day.", "Here is a short note:", "The following text continues:"]


def cand_layers(n_layers: int, n_sweep: int = 9) -> list[int]:
    return sorted(set(int(round(f * (n_layers - 1))) for f in np.linspace(0.15, 0.95, n_sweep)))


# ---------------------------------------------------------------------------
# Extraction: LAST-token residual (SAE-Probes convention), optionally noised ids.
# ---------------------------------------------------------------------------
@torch.no_grad()
def encode(tok, texts, device):
    tok.padding_side = "right"; tok.truncation_side = "left"
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN)
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


@torch.no_grad()
def capture(backbone, layers, layer_ids, ids, attn):
    cap = {}
    def mk(li):
        def h(_m, _i, out): cap[li] = out[0] if isinstance(out, tuple) else out
        return h
    handles = [layers[li].register_forward_hook(mk(li)) for li in layer_ids]
    try:
        backbone(input_ids=ids, attention_mask=attn)
    finally:
        for h in handles:
            h.remove()
    return cap


@torch.no_grad()
def last_token_acts(backbone, layers, layer_ids, ids, attn) -> torch.Tensor:
    """[B, n_layers, d] residual at the last real token."""
    cap = capture(backbone, layers, layer_ids, ids, attn)
    last = attn.sum(1) - 1  # [B]
    b = torch.arange(ids.shape[0], device=ids.device)
    return torch.stack([cap[li][b, last].float().cpu() for li in layer_ids], dim=1)


@torch.no_grad()
def extract_pair(bb_g, Lg, bb_d, Ld, tok, texts, layer_ids, device, noise_seed=None,
                 vocab_size=0, batch=16):
    """Both models on the same (optionally noised) ids. Returns (g, d) [N, n_layers, dim]."""
    gs, ds = [], []
    for s in range(0, len(texts), batch):
        ids, attn = encode(tok, texts[s : s + batch], device)
        if noise_seed is not None:
            nid = sd.noise_ids(ids.cpu().numpy(), attn.cpu().numpy(), NOISE_FRAC,
                               noise_seed + s, vocab_size, protect_last=True)
            ids = torch.tensor(nid, device=device)
        gs.append(last_token_acts(bb_g, Lg, layer_ids, ids, attn))
        ds.append(last_token_acts(bb_d, Ld, layer_ids, ids, attn))
    return torch.cat(gs).numpy(), torch.cat(ds).numpy()


# ---------------------------------------------------------------------------
# Phase RSA: per-(layer, position) cosine, clean + noised.
# ---------------------------------------------------------------------------
@torch.no_grad()
def phase_rsa(bb_g, Lg, bb_d, Ld, tok, datasets, device, vocab_size, batch=8, force=False):
    nL = len(Lg)
    docs = [(d["category"], t) for d in datasets for t in d["texts_test"]]
    regimes = sorted({r for r, _ in docs})
    ri = {r: i for i, r in enumerate(regimes)}
    for cond in ["clean", "noised"]:
        out_path = OUT / f"repr_cosine_{cond}.npz"
        if out_path.exists() and not force:
            print(f"[rsa] {out_path.name} exists, skip"); continue
        cos_sum = np.zeros((nL, P)); cnt = np.zeros((nL, P))
        cos_sum_r = np.zeros((len(regimes), nL, P)); cnt_r = np.zeros((len(regimes), nL, P))
        perdoc = np.full((len(docs), nL), np.nan)
        perdoc_reg = np.array([ri[r] for r, _ in docs])
        t0 = time.time()
        for s in range(0, len(docs), batch):
            chunk = docs[s : s + batch]
            tok.padding_side = "right"; tok.truncation_side = "left"
            enc = tok([t for _, t in chunk], return_tensors="pt", padding=True,
                      truncation=True, max_length=P)
            ids = enc["input_ids"].to(device); attn = enc["attention_mask"].to(device)
            if cond == "noised":
                nid = sd.noise_ids(ids.cpu().numpy(), attn.cpu().numpy(), NOISE_FRAC,
                                   1000 + s, vocab_size, protect_last=False)
                ids = torch.tensor(nid, device=device)
            cg = capture(bb_g, Lg, list(range(nL)), ids, attn)
            cd = capture(bb_d, Ld, list(range(nL)), ids, attn)
            m = attn.bool(); mm = m.cpu().numpy().astype(np.float64)
            ndok = np.maximum(mm.sum(1), 1)
            L = ids.shape[1]
            for li in range(nL):
                g = cg[li].float(); d = cd[li].float()
                c = ((g * d).sum(-1) / (g.norm(dim=-1) * d.norm(dim=-1) + 1e-8))
                c = c.masked_fill(~m, 0.0).cpu().numpy()
                cos_sum[li, :L] += c.sum(0); cnt[li, :L] += mm.sum(0)
                perdoc[s : s + len(chunk), li] = (c * mm).sum(1) / ndok
                for bi, (r, _) in enumerate(chunk):
                    cos_sum_r[ri[r], li, :L] += c[bi]; cnt_r[ri[r], li, :L] += mm[bi]
            if (s // batch) % 100 == 0:
                print(f"[rsa/{cond}] {s + len(chunk)}/{len(docs)} "
                      f"({(s + len(chunk)) / max(time.time() - t0, 1e-6):.0f} doc/s)")
        np.savez(out_path, cos=cos_sum / np.maximum(cnt, 1), cnt=cnt,
                 cos_r=cos_sum_r / np.maximum(cnt_r, 1), cnt_r=cnt_r,
                 regimes=np.array(regimes), n_layers=nL, P=P,
                 perdoc=perdoc, perdoc_reg=perdoc_reg)
        print(f"[rsa/{cond}] saved {out_path}  mean cos={(cos_sum / np.maximum(cnt, 1))[cnt > 0].mean():.3f}")


# ---------------------------------------------------------------------------
# Phase EXTRACT: per-dataset acts + gemma-fit directions.
# ---------------------------------------------------------------------------
def phase_extract(bb_g, Lg, bb_d, Ld, tok, datasets, device, vocab_size, force=False):
    acts_dir = OUT / "acts"; acts_dir.mkdir(parents=True, exist_ok=True)
    layer_ids = cand_layers(len(Lg))
    directions = {}
    dir_path = OUT / "directions.pt"
    if dir_path.exists() and not force:
        directions = torch.load(dir_path)["concepts"]
    t0 = time.time()
    for di, d in enumerate(datasets):
        f = acts_dir / f"{d['tag']}.npz"
        if f.exists() and d["tag"] in directions and not force:
            continue
        g_tr, d_tr = extract_pair(bb_g, Lg, bb_d, Ld, tok, d["texts_train"], layer_ids, device)
        g_te, d_te = extract_pair(bb_g, Lg, bb_d, Ld, tok, d["texts_test"], layer_ids, device)
        g_te_n, d_te_n = extract_pair(bb_g, Lg, bb_d, Ld, tok, d["texts_test"], layer_ids,
                                      device, noise_seed=7000 + di, vocab_size=vocab_size)
        np.savez(f, layer_ids=np.array(layer_ids),
                 y_train=np.array(d["y_train"]), y_test=np.array(d["y_test"]),
                 g_train=g_tr.astype(np.float16), d_train=d_tr.astype(np.float16),
                 g_test=g_te.astype(np.float16), d_test=d_te.astype(np.float16),
                 g_test_noised=g_te_n.astype(np.float16), d_test_noised=d_te_n.astype(np.float16))

        # Gemma-fit diff-of-means probe (train split), best layer by held-out AUC on
        # gemma test; then the Venhoff steering stats at that layer for BOTH models.
        y_tr = np.array(d["y_train"]); y_te = np.array(d["y_test"])
        probe = rcp.fit_concept(g_tr[y_tr == 1], g_tr[y_tr == 0], layer_ids, seed=di)
        j = layer_ids.index(probe["layer_idx"])
        te_score = g_te[:, j, :] @ probe["direction"]
        auc_test = rcp.auc(te_score[y_te == 1], te_score[y_te == 0])

        def venhoff(train_acts):  # (vector, overall_norm) at chosen layer j
            A = train_acts[:, j, :].astype(np.float64)
            overall = A.mean(0)
            vec = A[y_tr == 1].mean(0) - overall
            vec = vec * (np.linalg.norm(overall) / (np.linalg.norm(vec) + 1e-8))
            return vec.astype(np.float32), float(np.linalg.norm(overall))

        v_g, norm_g = venhoff(g_tr)
        v_d, norm_d = venhoff(d_tr)
        directions[d["tag"]] = {
            "category": d["category"], "layer": int(probe["layer_idx"]),
            "auc_gemma_val": float(probe["auc"]), "auc_gemma_test": float(auc_test),
            "auc_by_layer": probe["auc_by_layer"],
            "dir_gemma": torch.tensor(probe["direction"]),      # unit diff-of-means (probe)
            "steer_gemma": torch.tensor(v_g),                   # Venhoff-normalized steering vector
            "steer_dg_native": torch.tensor(v_d),
            "overall_norm_gemma": norm_g, "overall_norm_dg": norm_d,
        }
        torch.save({"layer_ids": layer_ids, "concepts": directions}, dir_path)
        print(f"[extract {di + 1:>3}/{len(datasets)}] {d['tag']:<40} L{probe['layer_idx']:<2} "
              f"auc={auc_test:.3f}  ({(time.time() - t0) / 60:.0f}min)")
    print(f"[extract] done: {len(directions)} concepts -> {dir_path}")


# ---------------------------------------------------------------------------
# Phase STEER: logprob steering on held-out real text.
# ---------------------------------------------------------------------------
def add_vec_hook(vec: torch.Tensor, coeff: float):
    """Venhoff application: add coeff * vec (already norm-matched) at every position."""
    def hook(_m, _i, out):
        is_tuple = isinstance(out, tuple)
        resid = out[0] if is_tuple else out
        resid = resid + coeff * vec.to(resid.dtype)
        return (resid, *out[1:]) if is_tuple else resid
    return hook


@torch.no_grad()
def true_next_logp(backbone, layers, lm_head, softcap, tok, rows, layer, vec, coeff, device, batch=32):
    """rows: list[(prefix_ids list, true_next_id)]. Returns np[len(rows)] of
    logp(true_next) at the final prefix position, under steering (coeff*vec at layer)."""
    hooks = []
    if coeff != 0.0:
        hooks.append(layers[layer].register_forward_hook(add_vec_hook(vec.to(device), coeff)))
    try:
        out = np.zeros(len(rows))
        for s in range(0, len(rows), batch):
            chunk = rows[s : s + batch]
            maxlen = max(len(p) for p, _ in chunk)
            pad = tok.pad_token_id
            ids = torch.full((len(chunk), maxlen), pad, dtype=torch.long)
            attn = torch.zeros((len(chunk), maxlen), dtype=torch.long)
            for i, (p, _) in enumerate(chunk):  # left-pad -> last position is real for all
                ids[i, maxlen - len(p):] = torch.tensor(p); attn[i, maxlen - len(p):] = 1
            o = backbone(input_ids=ids.to(device), attention_mask=attn.to(device))
            h = o.last_hidden_state[:, -1, :]
            logits = lm_head(h).float()
            if softcap:
                logits = softcap * torch.tanh(logits / softcap)
            lp = torch.log_softmax(logits, dim=-1)
            tgt = torch.tensor([t for _, t in chunk], device=lp.device)
            out[s : s + len(chunk)] = lp[torch.arange(len(chunk)), tgt].cpu().numpy()
        return out
    finally:
        for h in hooks:
            h.remove()


def make_rows(tok, texts, vocab_size, noise_seed=None):
    """Per text: N_CUTS (prefix, true_next) pairs at evenly spaced cut points.
    Under noising only the PREFIX is corrupted; the read target stays the clean token."""
    tok.truncation_side = "left"
    rows, owner = [], []
    for ti, t in enumerate(texts):
        clean = tok(t, truncation=True, max_length=MAX_LEN)["input_ids"]
        if len(clean) < 4:  # many concept texts are bare entity names (~5 tokens)
            continue
        ids = clean
        if noise_seed is not None:
            arr = np.array(clean)[None, :]
            attn = np.ones_like(arr)
            ids = sd.noise_ids(arr, attn, NOISE_FRAC, noise_seed + ti, vocab_size,
                               protect_last=False)[0].tolist()
        cuts = np.linspace(0.5, 0.95, N_CUTS)
        for c in cuts:
            k = max(1, min(int(len(clean) * c), len(clean) - 1))
            rows.append((ids[:k], clean[k])); owner.append(ti)
    return rows, np.array(owner)


def phase_steer(models, tok, datasets, device, vocab_size, force=False):
    out_path = OUT / "steer_logprob.json"
    done = json.loads(out_path.read_text()) if out_path.exists() and not force else {}
    dirs = torch.load(OUT / "directions.pt")["concepts"]
    by_tag = {d["tag"]: d for d in datasets}
    t0 = time.time()
    for ci, (tag, D) in enumerate(dirs.items()):
        if tag in done:
            continue
        d = by_tag[tag]
        y_te = np.array(d["y_test"])
        pos_texts = [t for t, y in zip(d["texts_test"], y_te) if y == 1][:N_STEER_TEXTS]
        neg_texts = [t for t, y in zip(d["texts_test"], y_te) if y == 0][:N_STEER_TEXTS]
        L = D["layer"]
        arms = {  # (model_key, vector) — probes are TRAINED ON GEMMA; dg+transfer is the transfer arm
            "gemma_native": ("gemma", D["steer_gemma"]),
            "dg_native": ("dg", D["steer_dg_native"]),
            "dg_transfer": ("dg", D["steer_gemma"]),
        }
        rec = {"category": D["category"], "layer": L, "auc_gemma_test": D["auc_gemma_test"], "arms": {}}
        for arm, (mk, vec) in arms.items():
            bb, layers, lm_head, softcap = models[mk]
            for cond in ["clean", "noised"]:
                seed = None if cond == "clean" else 9000 + ci
                rp, op = make_rows(tok, pos_texts, vocab_size, seed)
                rn, on = make_rows(tok, neg_texts, vocab_size, seed)
                if len(rp) < 2 or len(rn) < 2:
                    print(f"[steer] {tag} {arm}/{cond}: too few rows ({len(rp)}p/{len(rn)}n), skipping")
                    continue
                per_coeff = {}
                for coeff in STEER_COEFFS:
                    lp_p = true_next_logp(bb, layers, lm_head, softcap, tok, rp, L, vec, coeff, device)
                    lp_n = true_next_logp(bb, layers, lm_head, softcap, tok, rn, L, vec, coeff, device)
                    per_coeff[str(coeff)] = {"pos": lp_p.tolist(), "neg": lp_n.tolist()}
                # steer effect: [pos-text logp gain] - [neg-text logp gain], +coeff vs -coeff
                eff = ((np.mean(per_coeff["1.0"]["pos"]) - np.mean(per_coeff["-1.0"]["pos"]))
                       - (np.mean(per_coeff["1.0"]["neg"]) - np.mean(per_coeff["-1.0"]["neg"])))
                rec["arms"][f"{arm}_{cond}"] = {"effect_pm1": float(eff), "per_coeff": per_coeff,
                                                "owner_pos": op.tolist(), "owner_neg": on.tolist()}
        done[tag] = rec
        out_path.write_text(json.dumps(done))
        def _e(a):
            v = rec["arms"].get(a, {}).get("effect_pm1")
            return f"{v:+.3f}" if v is not None else "  n/a"
        print(f"[steer {ci + 1:>3}/{len(dirs)}] {tag:<40} "
              f"g={_e('gemma_native_clean')} dgN={_e('dg_native_clean')} "
              f"dgT={_e('dg_transfer_clean')} ({(time.time() - t0) / 60:.0f}min)")
    print(f"[steer] done -> {out_path}")


# ---------------------------------------------------------------------------
# Phase GENERATE: steered generations (AR generate / DG denoising loop).
# ---------------------------------------------------------------------------
@torch.no_grad()
def gen_ar(model, tok, prompt, max_new=96, seed=0):
    torch.manual_seed(seed)
    enc = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True,
                                  add_generation_prompt=True, return_dict=True,
                                  return_tensors="pt").to(model.device)
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.pad_token_id)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


@torch.no_grad()
def gen_dg(model, tok, prompt, C=96, T=24, seed=0):
    from transformers.models.diffusion_gemma.generation_diffusion_gemma import (
        DiffusionGemmaGenerationConfig, EntropyBoundSamplerConfig)
    torch.manual_seed(seed)
    model.config.canvas_length = C
    enc = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True,
                                  add_generation_prompt=True, return_dict=True,
                                  return_tensors="pt").to(model.device)
    eos = model.config.eos_token_id; eos = eos if isinstance(eos, list) else [eos]
    pad = getattr(model.config, "pad_token_id", 0) or 0
    gc = DiffusionGemmaGenerationConfig(
        max_new_tokens=C, max_denoising_steps=T,
        sampler_config=EntropyBoundSamplerConfig(entropy_bound=0.1),
        t_min=0.4, t_max=0.8, stability_threshold=1, confidence_threshold=0.005,
        pad_token_id=pad, eos_token_id=eos)
    out = model.generate(**enc, generation_config=gc)
    seq = out.sequences[0]; plen = enc["input_ids"].shape[1]
    canvas = seq[plen : plen + C].tolist()
    bad = set([pad] + list(eos))
    return tok.decode([t for t in canvas if t not in bad], skip_special_tokens=False).strip()


def phase_generate(models_full, tok, device, force=False):
    out_path = OUT / "generations.json"
    done = json.loads(out_path.read_text()) if out_path.exists() and not force else {}
    dirs = torch.load(OUT / "directions.pt")["concepts"]
    steer = json.loads((OUT / "steer_logprob.json").read_text())
    # top concepts: readable on gemma AND steerable on gemma (the trained-probe model)
    ranked = sorted(dirs, key=lambda t: -(dirs[t]["auc_gemma_test"]
                                          + 0.1 * steer.get(t, {}).get("arms", {})
                                          .get("gemma_native_clean", {}).get("effect_pm1", 0)))
    top = ranked[:GEN_TOP_K]
    (model_g, layers_g), (model_d, layers_d) = models_full["gemma"], models_full["dg"]
    for tag in top:
        if tag in done:
            continue
        D = dirs[tag]; L = D["layer"]
        arms = {"gemma_native": (model_g, layers_g, gen_ar, D["steer_gemma"], GEN_COEFF["gemma"]),
                "dg_native": (model_d, layers_d, gen_dg, D["steer_dg_native"], GEN_COEFF["dg"]),
                "dg_transfer": (model_d, layers_d, gen_dg, D["steer_gemma"], GEN_COEFF["dg"])}
        rec = {"category": D["category"], "layer": L, "runs": []}
        for pi, prompt in enumerate(GEN_PROMPTS):
            row = {"prompt": prompt}
            for arm, (model, layers, gen, vec, c0) in arms.items():
                for coeff, name in [(0.0, "base"), (+c0, "pos"), (-c0, "neg")]:
                    h = None
                    if coeff != 0.0:
                        h = layers[L].register_forward_hook(add_vec_hook(vec.to(device), coeff))
                    try:
                        row[f"{arm}_{name}"] = gen(model, tok, prompt, seed=pi)
                    finally:
                        if h is not None:
                            h.remove()
            rec["runs"].append(row)
            print(f"[gen] {tag} prompt {pi}: "
                  f"dgT+: {row['dg_transfer_pos'][:100]!r}")
        done[tag] = rec
        out_path.write_text(json.dumps(done, indent=1))
    print(f"[generate] done ({len(done)} concepts) -> {out_path}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all", choices=["all", "rsa", "extract", "steer", "generate"])
    ap.add_argument("--max-datasets", type=int, default=0, help="0 = all (smoke: e.g. 4)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    datasets = sd.load_datasets()
    if args.max_datasets:
        datasets = datasets[: args.max_datasets]

    model_g, tok = rcp.load_model("gemma4")
    bb_g, Lg = rcp.locate(model_g)
    model_d, tok_d = rcp.load_model("diffusiongemma")
    bb_d, Ld = rcp.locate(model_d)
    assert model_g.device == model_d.device, "cosine/steer phases need co-resident models"
    assert tok.encode("The cat sat on the mat.") == tok_d.encode("The cat sat on the mat."), \
        "tokenizer mismatch — position-aligned comparison invalid"
    device = model_g.device
    vocab = int(tok.vocab_size)

    def head(m):
        lm = m.get_output_embeddings()
        sc = float(getattr(m.config, "final_logit_softcapping", 0) or
                   getattr(getattr(m.config, "text_config", m.config), "final_logit_softcapping", 0) or 0)
        return lm, sc

    lm_g, sc_g = head(model_g); lm_d, sc_d = head(model_d)
    models = {"gemma": (bb_g, Lg, lm_g, sc_g), "dg": (bb_d, Ld, lm_d, sc_d)}
    models_full = {"gemma": (model_g, Lg), "dg": (model_d, Ld)}

    if args.phase in ("all", "rsa"):
        phase_rsa(bb_g, Lg, bb_d, Ld, tok, datasets, device, vocab, force=args.force)
    if args.phase in ("all", "extract"):
        phase_extract(bb_g, Lg, bb_d, Ld, tok, datasets, device, vocab, force=args.force)
    if args.phase in ("all", "steer"):
        phase_steer(models, tok, datasets, device, vocab, force=args.force)
    if args.phase in ("all", "generate"):
        phase_generate(models_full, tok, device, force=args.force)


if __name__ == "__main__":
    main()
