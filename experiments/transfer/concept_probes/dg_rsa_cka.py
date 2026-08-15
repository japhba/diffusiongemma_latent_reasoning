"""True RSA + CKA per-layer curves for the three stream pairs (g×enc, g×dec, enc×dec),
complementing §1.1's raw matched-text cosine and the 5-layer CKA/Procrustes table:

  raw cos      : mean over texts of cos(x_i, y_i)              — basis-dependent
  linear CKA   : ||Xc'Yc||_F^2 / (||Xc'Xc||_F ||Yc'Yc||_F)     — rotation-invariant, spectrum-weighted
  RSA          : Spearman corr of the two streams' RDMs        — rotation-invariant, 2nd-order
                 (RDM = 1 - pairwise cosine over texts, upper triangle)
  Procrustes   : mean matched cos after ONE optimal orthogonal map per layer;
                 split-half (fit on half the texts, score on the held-out half) + full-fit ref.

Texts: 64 per entity-filtered concept (48 train + 16 test, balanced), mean-pooled over real
tokens in the 96-token window per stream, ALL layers.

Extract (GPU): -> out/saeprobes/dg_rsa_cka_acts.npz
Fit (CPU):     --fit -> out/saeprobes/dg_rsa_cka_curves.json

srun ... bash concept_probes/slurm/ensure_and_run.sh concept_probes/dg_rsa_cka.py
"""
from __future__ import annotations

import functools
import json
import os
import sys
from pathlib import Path

import numpy as np

print = functools.partial(print, flush=True)
REPO = Path(os.environ.get("DGLR_ROOT", Path(__file__).resolve().parents[1]))
OUT = Path(os.environ.get("SAEP_OUT", REPO / "concept_probes/out/saeprobes"))
MAX_LEN = 96
PER_CONCEPT = 64
PAIRS = [("g", "enc"), ("g", "dec"), ("enc", "dec")]


def _load(n, f):
    import importlib.util
    s = importlib.util.spec_from_file_location(n, REPO / f"concept_probes/{f}")
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


def extract():
    import torch
    sys.path.insert(0, str(REPO / "concept_probes"))
    rcp = _load("rcp", "run_concept_probes.py")
    sd = _load("sd", "saeprobes_data.py")

    keep = {t for t, v in json.loads((OUT / "concept_entity_filter.json").read_text()).items()
            if not v["single_entity"]}
    rng = np.random.default_rng(7)
    texts, tags = [], []
    for d in sd.load_datasets():
        if d["tag"] not in keep:
            continue
        pool = [(t, "te") for t in d["texts_test"][:16]]
        tr = list(d["texts_train"]); rng.shuffle(tr)
        pool += [(t, "tr") for t in tr[:PER_CONCEPT - len(pool)]]
        texts += [t for t, _ in pool]; tags += [d["tag"]] * len(pool)
    print(f"[rsacka] {len(texts)} texts from {len(set(tags))} concepts")

    model_d, tok = rcp.load_model("diffusiongemma", device_map={"": 0})
    model_g, _ = rcp.load_model("gemma4", device_map={"": 0})
    _, g_layers = rcp.locate(model_g)
    enc_lm = model_d.model.encoder.language_model
    stream_layers = {"g": g_layers, "enc": enc_lm.layers, "dec": model_d.model.decoder.layers}
    nL = min(len(v) for v in stream_layers.values())
    device = model_d.device
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.pad_token_id
    print(f"[rsacka] layers: {[f'{k}={len(v)}' for k, v in stream_layers.items()]} -> {nL}")

    cap = {}
    def mk(L):
        def h(_m, _i, out): cap[L] = out[0] if isinstance(out, tuple) else out
        return h

    X = np.zeros((len(texts), 3, nL, model_g.config.get_text_config().hidden_size), dtype=np.float16)
    with torch.no_grad():
        for s in range(0, len(texts), 16):
            chunk = [t[:600] for t in texts[s:s + 16]]
            tok.padding_side = "right"; tok.truncation_side = "left"
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                      max_length=MAX_LEN, add_special_tokens=False)
            ids = enc["input_ids"].to(device); attn = enc["attention_mask"].to(device)
            B, T = ids.shape
            m = attn.unsqueeze(1).unsqueeze(-1).float()
            for si, name in enumerate(("g", "enc", "dec")):
                layers = stream_layers[name]
                hs = [layers[L].register_forward_hook(mk(L)) for L in range(nL)]
                try:
                    cap.clear()
                    if name == "dec":
                        p = torch.full((B, 1), bos, dtype=torch.long, device=device)
                        model_d(input_ids=p, attention_mask=torch.ones_like(p, dtype=torch.bool),
                                decoder_input_ids=ids,
                                decoder_position_ids=torch.arange(1, 1 + T, device=device).unsqueeze(0).expand(B, -1))
                    elif name == "enc":
                        enc_lm(input_ids=ids, attention_mask=attn)
                    else:
                        model_g(input_ids=ids, attention_mask=attn)
                    h = torch.stack([cap[L].float() for L in range(nL)], 1)     # [B, L, T, d]
                    mp = (h * m).sum(2) / m.sum(2).clamp_min(1)
                    X[s:s + B, si] = mp.cpu().numpy().astype(np.float16)
                finally:
                    for hk in hs:
                        hk.remove()
            if (s // 16) % 20 == 0:
                print(f"[rsacka] {s + B}/{len(texts)}")
    np.savez(OUT / "dg_rsa_cka_acts.npz", X=X, tags=np.array(tags), n_layers=nL)
    print(f"[rsacka/extract] wrote dg_rsa_cka_acts.npz {X.shape}")


def fit():
    from scipy.stats import spearmanr
    z = np.load(OUT / "dg_rsa_cka_acts.npz", allow_pickle=True)
    X = z["X"].astype(np.float64); nL = int(z["n_layers"])
    N = X.shape[0]
    rng = np.random.default_rng(0)
    half = rng.permutation(N); h1, h2 = half[:N // 2], half[N // 2:]
    iu = np.triu_indices(N, 1)
    SI = {"g": 0, "enc": 1, "dec": 2}

    def cka(A, B):
        A = A - A.mean(0); B = B - B.mean(0)
        return float(np.linalg.norm(A.T @ B) ** 2 /
                     (np.linalg.norm(A.T @ A) * np.linalg.norm(B.T @ B) + 1e-12))

    def unit(A):
        return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)

    def proc_map(A, B):
        Ac, Bc = A - A.mean(0), B - B.mean(0)
        U, _, Vt = np.linalg.svd(Ac.T @ Bc, full_matrices=False)
        return U @ Vt

    def matched_cos(A, B):
        return float((unit(A) * unit(B)).sum(1).mean())

    res = {f"{a}_{b}": {k: [] for k in ("cos", "cka", "rsa", "proc_holdout", "proc_full")}
           for a, b in PAIRS}
    for li in range(nL):
        rdm = {}
        for s, si in SI.items():
            A = unit(X[:, si, li])
            rdm[s] = (1.0 - A @ A.T)[iu]
        for a, b in PAIRS:
            A, B = X[:, SI[a], li], X[:, SI[b], li]
            r = res[f"{a}_{b}"]
            r["cos"].append(matched_cos(A, B))
            r["cka"].append(cka(A, B))
            r["rsa"].append(float(spearmanr(rdm[a], rdm[b]).statistic))
            R = proc_map(A[h1], B[h1])
            r["proc_holdout"].append(matched_cos(A[h2] @ R, B[h2]))
            r["proc_full"].append(matched_cos(A @ proc_map(A, B), B))
        print(f"[rsacka/fit] L{li}: " + " ".join(
            f"{a}_{b} cka={res[f'{a}_{b}']['cka'][-1]:.2f} rsa={res[f'{a}_{b}']['rsa'][-1]:.2f}"
            for a, b in PAIRS))
    out = {"n_layers": nL, "n_texts": N, "pairs": res,
           "measures": ["cos", "cka", "rsa", "proc_holdout", "proc_full"]}
    (OUT / "dg_rsa_cka_curves.json").write_text(json.dumps(out, indent=1))
    print("[rsacka/fit] wrote dg_rsa_cka_curves.json")


if __name__ == "__main__":
    (fit if "--fit" in sys.argv else extract)()
