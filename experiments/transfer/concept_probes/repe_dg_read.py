"""DG-native RepE reading vectors (LAT) in DiffusionGemma's DECODER mode — the source
directions for the DG-native half of the steering 2x2.

Mirrors repe_steer.py's reading phase, but reads DiffusionGemma in decoder mode:
each template-primed stimulus is fed as a canvas over a BOS prompt, and we take the
LAST canvas-position residual at every decoder layer; PCA of paired differences
(RepE convention) -> DG-native direction per (concept, layer). Full data, sharded.

-> out/saeprobes/repe_dg_directions{_shardi}.pt, repe_dg_meta{_shardi}.json
   (merge shards with repe_dg_merge below via `python repe_dg_read.py --merge`)

srun ... bash concept_probes/slurm/ensure_and_run.sh concept_probes/repe_dg_read.py
"""
from __future__ import annotations

import functools
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

print = functools.partial(print, flush=True)
REPO = Path(os.environ.get("DGLR_ROOT", Path(__file__).resolve().parents[1]))
OUT = Path(os.environ.get("SAEP_OUT", REPO / "concept_probes/out/saeprobes"))

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util
def _load(n, f):
    s = importlib.util.spec_from_file_location(n, REPO / f"concept_probes/{f}")
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
rcp = _load("rcp", "run_concept_probes.py")
rsg = _load("rsg", "run_saeprobes_gpu.py")
repe = _load("repe", "repe_steer.py")
jsg = _load("jsg", "judge_steer_gens.py")
import saeprobes_data as sd  # noqa: E402

READ_LAYERS = list(range(2, 29))


def merge():
    import glob
    dirs, meta = {}, {}
    for dp in sorted(glob.glob(str(OUT / "repe_dg_directions_shard*.pt"))):
        i = Path(dp).stem.rsplit("shard", 1)[1]
        dirs.update(torch.load(dp))
        meta.update(json.loads((OUT / f"repe_dg_meta_shard{i}.json").read_text()))
        print(f"[dgread/merge] shard {i}: {len(dirs)} concepts")
    torch.save(dirs, OUT / "repe_dg_directions.pt")
    (OUT / "repe_dg_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[dgread/merge] wrote canonical: {len(dirs)} concepts")


@torch.no_grad()
def dg_decoder_read(model, dec_layers, tok, texts, device, batch=8, max_len=128):
    """[N, n_layers, d] last-canvas-token decoder-mode residual of chat-free texts."""
    cap = {}
    def mk(li):
        def h(_m, _i, out): cap[li] = out[0] if isinstance(out, tuple) else out
        return h
    handles = [dec_layers[L].register_forward_hook(mk(L)) for L in READ_LAYERS]
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.pad_token_id
    feats = []
    try:
        tok.padding_side = "right"; tok.truncation_side = "left"
        for s in range(0, len(texts), batch):
            enc = tok(texts[s:s + batch], return_tensors="pt", padding=True,
                      truncation=True, max_length=max_len, add_special_tokens=False)
            ids = enc["input_ids"].to(device); attn = enc["attention_mask"].to(device)
            B, T = ids.shape
            prompt = torch.full((B, 1), bos, dtype=torch.long, device=device)
            cap.clear()
            model(input_ids=prompt,
                  attention_mask=torch.ones_like(prompt, dtype=torch.bool),
                  decoder_input_ids=ids,
                  decoder_position_ids=torch.arange(1, 1 + T, device=device).unsqueeze(0).expand(B, -1))
            last = attn.sum(1) - 1
            b = torch.arange(B, device=device)
            feats.append(torch.stack([cap[L][b, last].float().cpu() for L in READ_LAYERS], dim=1))
    finally:
        for h in handles:
            h.remove()
    return torch.cat(feats).numpy()


def main():
    if "--merge" in sys.argv:
        merge(); return
    shard_i, shard_n = (int(x) for x in os.environ.get("SAEP_SHARD", "0/1").split("/"))
    suffix = f"_shard{shard_i}" if shard_n > 1 else ""
    dir_path = OUT / f"repe_dg_directions{suffix}.pt"
    directions = torch.load(dir_path) if dir_path.exists() else {}
    meta = json.loads((OUT / f"repe_dg_meta{suffix}.json").read_text()) \
        if (OUT / f"repe_dg_meta{suffix}.json").exists() else {}

    desc = jsg.concept_descriptions()
    datasets = sd.load_datasets()[shard_i::shard_n]
    print(f"[dgread] shard {shard_i}/{shard_n}: {len(datasets)} concepts (decoder-mode LAT)")

    model, tok = rcp.load_model("diffusiongemma")
    dec_layers = model.model.decoder.layers

    for di, d in enumerate(datasets):
        tag = d["tag"]
        if tag in directions:
            continue
        cdesc = desc.get(tag, tag)
        sem_pos = 0 if d["flipped"] else 1
        pos = [t for t, y in zip(d["texts_train"], d["y_train"]) if y == sem_pos]
        neg = [t for t, y in zip(d["texts_train"], d["y_train"]) if y != sem_pos]
        pos += d["texts_extra_y0" if d["flipped"] else "texts_extra_y1"]
        neg += d["texts_extra_y1" if d["flipped"] else "texts_extra_y0"]
        n = min(len(pos), len(neg))
        P = dg_decoder_read(model, dec_layers, tok,
                            [repe.template(cdesc, t[:800]) for t in pos[:n]], model.device)
        N = dg_decoder_read(model, dec_layers, tok,
                            [repe.template(cdesc, t[:800]) for t in neg[:n]], model.device)
        rec, accs = {}, {}
        for j, L in enumerate(READ_LAYERS):
            dvec, acc = repe.pca_direction(P[:, j, :].astype(np.float64),
                                           N[:, j, :].astype(np.float64), seed=di * 100 + L)
            rec[L] = torch.tensor(dvec)
            accs[f"L{L}"] = {"acc": round(acc, 3),
                             "resid_norm": float(np.linalg.norm(P[:, j, :], axis=1).mean())}
        directions[tag] = rec
        meta[tag] = accs
        torch.save(directions, dir_path)
        (OUT / f"repe_dg_meta{suffix}.json").write_text(json.dumps(meta, indent=1))
        print(f"[dgread {di + 1:>3}] {tag:<42} best decoder-LAT acc={max(v['acc'] for v in accs.values()):.2f}")
    print(f"[dgread] shard {shard_i} done ({len(directions)} concepts)")


if __name__ == "__main__":
    main()
