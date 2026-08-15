"""RepE (Zou et al. 2310.01405, third_party/representation-engineering) on the full
SAE-Probes concept battery — gemma-4.

READING (LAT, PCARepReader recipe): per concept, 24 concept-positive + 24
concept-negative paper texts wrapped in the task template
    "Consider the amount of <concept description> in the following text:\n<text>\n
     The amount of <concept description> is"
(chat-templated); hidden state at the LAST token, EVERY layer; direction = first PC
of the paired differences (pos_i − neg_i, half randomly sign-flipped before PCA,
RepE convention), sign calibrated so concept-positive projects positive.

CONTROL (RepControl, linear_comb): the reading vectors injected simultaneously
across a LAYER BAND at every position during generation (our gated_hook, gate=False),
per-layer magnitude = rel x that layer's residual norm (dose discipline kept from
§5.3-fix so results are comparable across concepts):
    band_wide    L5,7,...,25 (11 layers)  rel 0.12 each
    band_mid     L9,11,...,19 (6 layers)  rel 0.20 each
    single_L13   L13                      rel 0.9   (continuity reference)
2 carriers, ±, judged as usual.

-> out/saeprobes/repe_directions.pt, repe_meta.json, repe_gens.json

srun ... bash concept_probes/slurm/ensure_and_run.sh concept_probes/repe_steer.py
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
gfx = _load("gfx", "gemma_fix_steer.py")
jsg = _load("jsg", "judge_steer_gens.py")
import saeprobes_data as sd  # noqa: E402

READ_LAYERS = list(range(2, 29))       # directions at every layer
# FULL data per class: canonical train split + the entire remaining pool outside
# train∪test (loader's texts_extra_*, capped 2048/class) — reading is single
# forwards, so even thousands of pairs stay cheap
N_STIM = None  # None = all available
PROMPTS = cst.PROMPTS
BANDS = {
    "band_wide": (list(range(5, 26, 2)), 0.12),
    "band_mid": (list(range(9, 20, 2)), 0.20),
    "single_L13": ([13], 0.9),
    # dose escalation (n256 round showed precision 0.91-0.98 at the doses above —
    # clear headroom; these cells only generate on a rerun, reusing cached directions)
    "band_wide_hi": (list(range(5, 26, 2)), 0.25),
    "band_mid_hi": (list(range(9, 20, 2)), 0.35),
}


def template(desc: str, text: str) -> str:
    return (f"Consider the amount of {desc} in the following text:\n"
            f"{text}\nThe amount of {desc} is")


@torch.no_grad()
def last_token_acts_all(model, bb, layers, tok, prompts, device, batch=8):
    """[N, n_read_layers, d] last-token hidden states of chat-templated prompts."""
    feats = []
    for s in range(0, len(prompts), batch):
        msgs = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=True,
                                        add_generation_prompt=True, return_dict=True)["input_ids"]
                for p in prompts[s:s + batch]]
        msgs = [list(m) for m in msgs]
        maxlen = max(len(m) for m in msgs)
        ids = torch.full((len(msgs), maxlen), tok.pad_token_id, dtype=torch.long)
        attn = torch.zeros_like(ids)
        for i, m in enumerate(msgs):  # left-pad so the last position is the read token
            ids[i, maxlen - len(m):] = torch.tensor(m)
            attn[i, maxlen - len(m):] = 1
        cap = rsg.capture(bb, layers, READ_LAYERS, ids.to(device), attn.to(device))
        feats.append(torch.stack([cap[L][:, -1].float().cpu() for L in READ_LAYERS], dim=1))
    return torch.cat(feats).numpy()


def pca_direction(pos: np.ndarray, neg: np.ndarray, seed: int):
    """RepE PCARepReader: first PC of paired differences (half sign-flipped), sign
    calibrated so positives project positive. Returns (unit dir, train acc)."""
    rng = np.random.default_rng(seed)
    diffs = pos - neg
    flip = rng.random(len(diffs)) < 0.5
    diffs = diffs * np.where(flip, -1.0, 1.0)[:, None]
    diffs = diffs - diffs.mean(0)
    # first principal component via SVD
    _, _, vt = np.linalg.svd(diffs, full_matrices=False)
    d = vt[0]
    proj_p, proj_n = pos @ d, neg @ d
    if np.mean(proj_p > proj_n) < 0.5:
        d = -d
    acc = float(np.mean((pos @ d) > (neg @ d)))
    return d.astype(np.float32), acc


def main():
    # concept sharding for multi-GPU: SAEP_SHARD="i/N" -> this process handles
    # datasets[i::N] and writes to *_shard{i} files (merge with repe_merge.py)
    shard_env = os.environ.get("SAEP_SHARD", "0/1")
    shard_i, shard_n = (int(x) for x in shard_env.split("/"))
    suffix = f"_shard{shard_i}" if shard_n > 1 else ""

    gens_path = OUT / f"repe_gens{suffix}.json"
    done = json.loads(gens_path.read_text()) if gens_path.exists() else {}
    dir_path = OUT / f"repe_directions{suffix}.pt"
    directions = torch.load(dir_path) if dir_path.exists() else {}
    meta = json.loads((OUT / f"repe_meta{suffix}.json").read_text()) \
        if (OUT / f"repe_meta{suffix}.json").exists() else {}

    datasets = sd.load_datasets()[shard_i::shard_n]
    print(f"[repe] shard {shard_i}/{shard_n}: {len(datasets)} concepts")
    desc = jsg.concept_descriptions()
    model, tok = rcp.load_model("gemma4")
    bb, layers = rcp.locate(model)
    device = model.device

    # ---- READING: template-primed PCA directions at every layer ----
    for di, d in enumerate(datasets):
        tag = d["tag"]
        if tag in directions:
            continue
        cdesc = desc.get(tag, tag)
        sem_pos = 0 if d["flipped"] else 1
        pos_texts = [t for t, y in zip(d["texts_train"], d["y_train"]) if y == sem_pos]
        neg_texts = [t for t, y in zip(d["texts_train"], d["y_train"]) if y != sem_pos]
        pos_texts += d["texts_extra_y0" if d["flipped"] else "texts_extra_y1"]
        neg_texts += d["texts_extra_y1" if d["flipped"] else "texts_extra_y0"]
        if N_STIM:
            pos_texts, neg_texts = pos_texts[:N_STIM], neg_texts[:N_STIM]
        n = min(len(pos_texts), len(neg_texts))
        P = last_token_acts_all(model, bb, layers, tok,
                                [template(cdesc, t[:800]) for t in pos_texts[:n]], device)
        N = last_token_acts_all(model, bb, layers, tok,
                                [template(cdesc, t[:800]) for t in neg_texts[:n]], device)
        rec, accs = {}, {}
        for j, L in enumerate(READ_LAYERS):
            dvec, acc = pca_direction(P[:, j, :].astype(np.float64),
                                      N[:, j, :].astype(np.float64), seed=di * 100 + L)
            rec[L] = torch.tensor(dvec)
            accs[f"L{L}"] = {"acc": round(acc, 3),
                             "resid_norm": float(np.linalg.norm(P[:, j, :], axis=1).mean())}
        directions[tag] = rec
        meta[tag] = accs
        torch.save(directions, dir_path)
        (OUT / f"repe_meta{suffix}.json").write_text(json.dumps(meta, indent=1))
        best_acc = max(v["acc"] for v in accs.values())
        print(f"[repe] read {di + 1:>3}/113 {tag:<42} best LAT acc={best_acc:.2f}")

    # ---- CONTROL: band injection during generation ----
    n = 0
    for d in datasets:
        tag = d["tag"]
        for pi, prompt in enumerate(PROMPTS):
            plen = cst.prompt_len(tok, prompt, device)
            basekey = f"{tag}|{pi}|repe_band_wide|base"
            if basekey not in done:
                b = rsg.gen_ar(model, tok, prompt, seed=pi)
                for band in BANDS:
                    done[f"{tag}|{pi}|repe_{band}|base"] = b
            for band, (band_layers, rel) in BANDS.items():
                for sign, sname in [(+1, "pos"), (-1, "neg")]:
                    key = f"{tag}|{pi}|repe_{band}|L{band_layers[0]}-{band_layers[-1]}|{rel}|{sname}"
                    if key in done:
                        continue
                    handles = []
                    try:
                        for L in band_layers:
                            vec = directions[tag][L].to(device) * \
                                (sign * rel * meta[tag][f"L{L}"]["resid_norm"])
                            handles.append(layers[L].register_forward_hook(
                                gfx.gated_hook(vec, plen, False)))
                        done[key] = rsg.gen_ar(model, tok, prompt, seed=pi)
                    finally:
                        for h in handles:
                            h.remove()
                    n += 1
                    if n % 50 == 0:
                        gens_path.write_text(json.dumps(done))
                        print(f"[repe] {n} gens — last: {key} -> {done[key][:70]!r}")
        gens_path.write_text(json.dumps(done))
    gens_path.write_text(json.dumps(done))
    print(f"[repe] wrote {gens_path} ({len(done)} entries)")


if __name__ == "__main__":
    main()
