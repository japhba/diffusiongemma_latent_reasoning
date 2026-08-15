"""§2.2 mode sub-rows, unified with the headline: the {causal, bidirectional·last,
bidirectional·mean} family re-measured on the SAME concepts (entity-filtered set), SAME budget
(topped-up 1024/class train + canonical 256 test), SAME 9-layer grid AND (since 2026-07-19)
the SAME canonical 512-token window as the headline 2×2 — so the sub-rows differ from the big
cell number ONLY by the attention mode (the causal·512 g row equals the headline by
construction). Window is env-tunable (SAEP_FAM_MAXLEN; the retired 96-window run lives in
family1024/ + dg_family_1024.json).

Reads per (text, stream in {g, enc, dec}): last-real-token + mean-pool at layers {4,7,..,28},
MAX_LEN-token window (left truncation). Decoder feed = text as its own exact-length canvas over
a BOS prompt (length-bucketed, NO padding — pads would be in-attention for the bidirectional
mode). NOTE: at 512 the decoder canvas is read well outside its C=96 generation regime.

Extract (GPU, shardable): SAEP_SHARD=i/n -> out/saeprobes/family1024_w{MAX_LEN}/{tag}.npz
Fit (CPU):               --fit           -> out/saeprobes/dg_family_1024_w{MAX_LEN}.json

srun ... bash concept_probes/slurm/ensure_and_run.sh concept_probes/dg_family_full.py
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
LAYERS = [4, 7, 10, 13, 16, 19, 22, 25, 28]      # the headline 2x2 layer grid
MAX_LEN = int(os.environ.get("SAEP_FAM_MAXLEN", "512"))   # canonical read window
FAM = OUT / f"family1024_w{MAX_LEN}"
TARGET = 1024


def _load(n, f):
    import importlib.util
    s = importlib.util.spec_from_file_location(n, REPO / f"concept_probes/{f}")
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


def kept_tags():
    e = json.loads((OUT / "concept_entity_filter.json").read_text())
    return sorted((t for t, v in e.items() if not v["single_entity"]),
                  key=lambda t: int(t.split("_")[0]))


def extract():
    import torch
    sys.path.insert(0, str(REPO / "concept_probes"))
    rcp = _load("rcp", "run_concept_probes.py")
    sd = _load("sd", "saeprobes_data.py")
    FAM.mkdir(exist_ok=True)

    shard_i, shard_n = (int(x) for x in os.environ.get("SAEP_SHARD", "0/1").split("/"))
    tags = kept_tags()[shard_i::shard_n]
    datasets = {d["tag"]: d for d in sd.load_datasets()}

    model, tok = rcp.load_model("diffusiongemma", device_map={"": 0})
    model_g, _ = rcp.load_model("gemma4", device_map={"": 0})
    _, g_layers = rcp.locate(model_g)
    device = model.device
    enc_lm = model.model.encoder.language_model
    stream_layers = {"g": g_layers, "enc": enc_lm.layers, "dec": model.model.decoder.layers}
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.pad_token_id

    def tok_ids(text):
        ids = tok(text, add_special_tokens=False)["input_ids"]
        return ids[-MAX_LEN:]                      # left truncation, matches prior protocols

    @torch.no_grad()
    def read_all(texts, batch=24):
        """[N, 3 streams, 2 reads(last, mean), L, d] — exact-length buckets, no padding."""
        toks = [tok_ids(t) for t in texts]
        order = sorted(range(len(toks)), key=lambda i: len(toks[i]))
        outN = [None] * len(toks)
        cap = {}
        def mk(L):
            def h(_m, _i, out): cap[L] = out[0] if isinstance(out, tuple) else out
            return h
        i = 0
        while i < len(order):
            grp = [order[i]]
            while (len(grp) < batch and i + len(grp) < len(order)
                   and len(toks[order[i + len(grp)]]) == len(toks[grp[0]])):
                grp.append(order[i + len(grp)])
            i += len(grp)
            ids = torch.tensor([toks[j] for j in grp], dtype=torch.long, device=device)
            B, T = ids.shape
            per_stream = []
            for s in ("g", "enc", "dec"):
                layers = stream_layers[s]
                hs = [layers[L].register_forward_hook(mk(L)) for L in LAYERS]
                try:
                    cap.clear()
                    if s == "dec":
                        p = torch.full((B, 1), bos, dtype=torch.long, device=device)
                        model(input_ids=p, attention_mask=torch.ones_like(p, dtype=torch.bool),
                              decoder_input_ids=ids,
                              decoder_position_ids=torch.arange(1, 1 + T, device=device).unsqueeze(0).expand(B, -1))
                    elif s == "enc":
                        enc_lm(input_ids=ids, attention_mask=torch.ones_like(ids))
                    else:
                        model_g(input_ids=ids, attention_mask=torch.ones_like(ids))
                    h = torch.stack([cap[L].float() for L in LAYERS], 1)      # [B, L, T, d]
                    per_stream.append(torch.stack([h[:, :, -1, :], h.mean(2)], 1))  # [B, 2, L, d]
                finally:
                    for hk in hs:
                        hk.remove()
            feats = torch.stack(per_stream, 1).cpu().numpy().astype(np.float16)  # [B, 3, 2, L, d]
            for k, j in enumerate(grp):
                outN[j] = feats[k]
        return np.stack(outN)

    for ti, tag in enumerate(tags):
        fp = FAM / f"{tag}.npz"
        if fp.exists():
            print(f"[fam] skip {tag} (exists)")
            continue
        d = datasets[tag]
        tr_t, tr_y = sd.topup_train(d, target=TARGET)
        te_t, te_y = d["texts_test"], d["y_test"]
        np.savez(fp, ytr=np.array(tr_y), yte=np.array(te_y),
                 tr=read_all([t[:MAX_LEN * 8] for t in tr_t]),
                 te=read_all([t[:MAX_LEN * 8] for t in te_t]),
                 layers=np.array(LAYERS))
        print(f"[fam] {ti + 1}/{len(tags)} {tag}: train {len(tr_t)}, test {len(te_t)}")
    print(f"[fam/extract] shard {shard_i}/{shard_n} done")


def _fit_one(tag, eval_mask):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    z = np.load(FAM / f"{tag}.npz")                # reopen inside worker (loky can't pickle handles)
    ytr, yte = z["ytr"], z["yte"][eval_mask]
    TR, TE = z["tr"].astype(np.float32), z["te"].astype(np.float32)
    TE = TE[eval_mask]
    SI = {"g": 0, "enc": 1, "dec": 2}; RI = {"last": 0, "mean": 1}
    def cv_fit(X, y):
        skf = StratifiedKFold(5, shuffle=True, random_state=1)
        score = np.mean([roc_auc_score(y[va], LogisticRegression(C=0.1, solver="liblinear", max_iter=1000)
                         .fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]) for tr, va in skf.split(X, y)])
        return LogisticRegression(C=0.1, solver="liblinear", max_iter=1000).fit(X, y), float(score)

    PROTO = [("g", "last"), ("enc", "last"), ("dec", "last"), ("dec", "mean")]
    nL = len(LAYERS)
    auc = lambda mdl, X: float(roc_auc_score(yte, mdl.predict_proba(X)[:, 1]))
    # Select one layer on source-stream training CV, then hold that layer and fitted probe fixed
    # across every target in the row. The evaluation subset is exact-string-disjoint from training.
    P = {(s, r): [cv_fit(TR[:, SI[s], RI[r], j, :], ytr) for j in range(nL)] for (s, r) in PROTO}
    best_j = {proto: int(np.argmax([v for _, v in fits])) for proto, fits in P.items()}
    jg = best_j[("g", "last")]
    res = {"gg": auc(P[("g", "last")][jg][0], TE[:, 0, 0, jg, :])}
    for (s, r) in PROTO[1:]:
        key = f"{s}_{r}"
        jd = best_j[(s, r)]
        res[f"gd|{key}"] = auc(P[("g", "last")][jg][0], TE[:, SI[s], RI[r], jg, :])
        res[f"dg|{key}"] = auc(P[(s, r)][jd][0], TE[:, 0, 0, jd, :])
        res[f"dd|{key}"] = auc(P[(s, r)][jd][0], TE[:, SI[s], RI[r], jd, :])
        res[f"layer|{key}"] = int(LAYERS[jd])
    res.update({"layer|g_last": int(LAYERS[jg]), "n_train": int(len(ytr)),
                "n_test_eval": int(eval_mask.sum()), "protocol_version": 3})
    (FAM / f"fit_{tag}.json").write_text(json.dumps(res))       # per-concept checkpoint (resume)
    print(f"[fam/fit] {tag} done", flush=True)
    return tag, res


def fit():
    from joblib import Parallel, delayed
    shard_i, shard_n = (int(x) for x in os.environ.get("SAEP_SHARD", "0/1").split("/"))
    tags = [t for t in kept_tags() if (FAM / f"{t}.npz").exists()]
    def current(t):
        p = FAM / f"fit_{t}.json"
        return p.exists() and json.loads(p.read_text()).get("protocol_version") == 3
    todo = [t for t in tags[shard_i::shard_n] if not current(t)]
    print(f"[fam/fit] shard {shard_i}/{shard_n}: {len(todo)} to fit "
          f"({len(tags)} total, rest resumed/other shards)")
    sd = _load("sd", "saeprobes_data.py")
    pp = _load("pp", "probe_protocol.py")
    datasets = {d["tag"]: d for d in sd.load_datasets()}
    masks = {t: pp.train_disjoint_test_mask(datasets[t]) for t in tags}
    Parallel(n_jobs=int(os.environ["SLURM_CPUS_PER_TASK"]), verbose=10)(
        delayed(_fit_one)(t, masks[t]) for t in todo)
    missing = [t for t in tags if not current(t)]
    if missing:
        print(f"[fam/fit] {len(missing)} concepts still pending (other shards) — no assemble yet")
        return
    per = {t: json.loads((FAM / f"fit_{t}.json").read_text()) for t in tags}
    cells = sorted({k for v in per.values() for k in v if ("|" in k and not k.startswith("layer|")) or k == "gg"})
    out = {"layers": LAYERS, "max_len": MAX_LEN, "target": TARGET, "tags": tags,
           "mean": {c: float(np.mean([per[t][c] for t in tags])) for c in cells},
           "per_concept": per,
           "protocol": {"layer_selection": "source-stream training-only 5-fold CV AUC at fixed C=0.1",
                        "transfer_rule": "source layer and probe fixed across row",
                        "evaluation": "train-disjoint unique test strings", "C": 0.1, "version": 3}}
    (OUT / f"dg_family_1024_w{MAX_LEN}.json").write_text(json.dumps(out, indent=1))
    print("[fam/fit] means:", {c: round(v, 3) for c, v in out["mean"].items()})
    print(f"[fam/fit] wrote dg_family_1024_w{MAX_LEN}.json")


if __name__ == "__main__":
    (fit if "--fit" in sys.argv else extract)()
