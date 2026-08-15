"""Top-20 lens tokens at the paper-defined position across every layer for each J-Lens eval item.

For each item of the 6 paper eval sets (AVBench dropped 2026-07-18) and each test model
(gemma-4 causal / DG bidirectional decoder), captures residuals at the paper's single task-defined
text-token position across all lens layers, applies the three active fitted lenses
(G causal, DG causal, and DG bidirectional, all fit on the same WikiText token data) PLUS the identity
logit-lens baseline (test-side unembed, no transport), and stores the top-20 vocab tokens AND
their softmax probabilities (quantized to 1/1000) per (cfg|test, layer, position). Readout
geometry, length limit, chat-template offset and unembeds are IDENTICAL to jlens_eval_2x2.py.

Output: reports/concept_probes/data/jlens_topk_{set}.json (one file per set, token string table +
[nL][1][20] index arrays) — lazily fetched by matrix_jlens.html.

srun -p general --qos=high --gres=gpu:1 ... bash concept_probes/slurm/ensure_and_run.sh \
    concept_probes/jlens_topk_capture.py
"""
from __future__ import annotations

import functools
import glob
import json
import os
import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm

print = functools.partial(print, flush=True)
REPO = Path(os.environ.get("DGLR_ROOT", Path(__file__).resolve().parents[1]))
OUT = Path(os.environ.get("SAEP_OUT", REPO / "concept_probes/out/saeprobes")) / "jlens"
EVDIR = REPO / "third_party/jacobian-lens/data/evaluations"
DATA = REPO / "reports/concept_probes/data"
sys.path.insert(0, str(REPO / "concept_probes"))
sys.path.insert(0, str(REPO / "third_party/jacobian-lens"))

import importlib.util
def _load(n, f):
    s = importlib.util.spec_from_file_location(n, REPO / f"concept_probes/{f}")
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

MAX_TOK = 380                                    # fail loudly above the shared context limit
TOPK = 20
from jlens_paper_eval import load_paper_sets, paper_readout


def merged_lens(pattern):
    from jlens.lens import JacobianLens
    paths = sorted(glob.glob(str(OUT / pattern)))
    if not paths:
        return None
    lenses = [JacobianLens.load(p) for p in paths]
    return lenses[0] if len(lenses) == 1 else JacobianLens.merge(lenses)


def main():
    active = ("g_shared", "dgc_shared", "dgb_shared")
    print(f"[topk] fit configs: {', '.join(active)}")

    rcp = _load("rcp", "run_concept_probes.py")
    model_g, tok = rcp.load_model("gemma4", device_map={"": 0})
    model_d, _ = rcp.load_model("diffusiongemma", device_map={"": 0})
    _, g_layers = rcp.locate(model_g)
    enc_lm = model_d.model.encoder.language_model
    dec_layers = model_d.model.decoder.layers
    device = model_g.device
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.pad_token_id

    LENS_PAT = {name: f"{name}_lens_shard*.pt" for name in active}
    lenses = {c: merged_lens(LENS_PAT[c]) for c in active}
    assert all(lenses.values()), f"missing lens checkpoints for {list(lenses)}"
    assert {lens.n_prompts for lens in lenses.values()} == {64}
    src_layers = sorted(lenses["g_shared"].jacobians)
    LSUB = src_layers
    print(f"[topk] all {len(LSUB)} layers: {LSUB}")
    J = {c: {L: lenses[c].jacobians[L].to(device=device, dtype=torch.float32) for L in LSUB}
         for c in lenses}

    gtext = model_g.model.language_model if hasattr(model_g.model, "language_model") else model_g.model
    sc_g = float(getattr(model_g.config.get_text_config(), "final_logit_softcapping", 0) or 0)
    sc_d = float(getattr(model_d.config.get_text_config(), "final_logit_softcapping", 0) or 0)
    def mk_unembed(norm, head, sc):
        def u(h):
            z = head(norm(h.to(head.weight.dtype))).float()
            return sc * torch.tanh(z / sc) if sc else z
        return u
    UNEMBED = {"g_shared": mk_unembed(gtext.norm, model_g.lm_head, sc_g),
               "dgc_shared": mk_unembed(enc_lm.norm, model_d.lm_head, sc_d),
               "dg": mk_unembed(model_d.model.decoder.norm, model_d.lm_head, sc_d)}
    UNEMBED["dgb_shared"] = UNEMBED["dg"]

    cap = {}
    def mk(L):
        def h(_m, _i, out): cap[L] = out[0] if isinstance(out, tuple) else out
        return h

    def _tpl_ids(text):
        x = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=True,
                                    add_generation_prompt=True, return_dict=True,
                                    return_tensors="pt")
        return [int(t) for t in x["input_ids"][0]]
    _a1, _a2 = _tpl_ids("Hello there"), _tpl_ids("Numbers: 17 42 99 end.")
    TPL_SUFFIX = 0
    while (TPL_SUFFIX < min(len(_a1), len(_a2))
           and _a1[-1 - TPL_SUFFIX] == _a2[-1 - TPL_SUFFIX]):
        TPL_SUFFIX += 1
    print(f"[topk] chat-template suffix = {TPL_SUFFIX} tokens")
    assert TPL_SUFFIX >= 1
    # constant prefix/suffix ids: inputs are CONSTRUCTED as prefix + text ids + suffix, so every
    # text token sits at a known position (re-tokenizing the templated string can merge boundary
    # tokens and shift interior positions — observed on multihop item 0)
    _h1 = tok("Hello there", add_special_tokens=False)["input_ids"]
    _st = next(o for o in range(len(_a1) - len(_h1) + 1) if _a1[o:o + len(_h1)] == _h1)
    PRE_IDS, SUF_IDS = _a1[:_st], _a1[_st + len(_h1):]
    assert len(SUF_IDS) == TPL_SUFFIX and _a2[:len(PRE_IDS)] == PRE_IDS

    @torch.no_grad()
    def read_paper(text, test, set_name):
        """[nL, 1, d] residuals at the paper-defined text-token position."""
        ids_l, pidx, _ = paper_readout(tok, set_name, text)
        T = len(ids_l)
        assert T <= MAX_TOK, (set_name, T, MAX_TOK)
        layers = g_layers if test == "g" else dec_layers
        hs = [layers[L].register_forward_hook(mk(L)) for L in LSUB]
        try:
            cap.clear()
            if test == "g":
                full = torch.tensor([PRE_IDS + ids_l + SUF_IDS], device=device)
                model_g(input_ids=full, attention_mask=torch.ones_like(full))
                off = len(PRE_IDS)                      # exact by construction
                mpos = [off + pidx]
            else:
                ids = torch.tensor([ids_l], device=device)
                p = torch.full((1, 1), bos, dtype=torch.long, device=device)
                model_d(input_ids=p, attention_mask=torch.ones_like(p, dtype=torch.bool),
                        decoder_input_ids=ids,
                        decoder_position_ids=torch.arange(1, 1 + T, device=device).unsqueeze(0))
                mpos = [pidx]
            h = torch.stack([torch.stack([cap[L][0, m].float() for m in mpos]) for L in LSUB])
        finally:
            for hh in hs:
                hh.remove()
        # rollout segments: [before, readout token, after] for the viewer
        segs = [tok.decode(ids_l[:pidx]), tok.decode([ids_l[pidx]]), tok.decode(ids_l[pidx + 1:])]
        return h, [pidx], [tok.decode([ids_l[pidx]])], T, segs

    # ---- eval sets (same sources/order as jlens_eval_2x2.py, so item idx == the page's unit id)
    sets = load_paper_sets(EVDIR)

    DATA.mkdir(parents=True, exist_ok=True)
    cell_cfg = {"gg": "g_shared|g", "gd": "g_shared|dg",
                "eg": "dgc_shared|g", "ed": "dgc_shared|dg",
                "pg": "dgb_shared|g", "pd": "dgb_shared|dg"}
    from datetime import datetime
    from zoneinfo import ZoneInfo
    META = {"script": "concept_probes/jlens_topk_capture.py",
            "captured": datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(timespec="minutes"),
            "models": {"g": rcp.MODEL_IDS["gemma4"], "dg": rcp.MODEL_IDS["diffusiongemma"]},
            "lenses": {**{c: LENS_PAT[c] for c in lenses},
                       "logitlens": "identity transport, test-side unembed"},
            "eval": "concept_probes/out/saeprobes/jlens/eval_2x2.json",
            # full-input rendering: the constant wrappers around the item text per test protocol
            "tpl_prefix": tok.decode(PRE_IDS), "tpl_suffix": tok.decode(SUF_IDS),
            "dg_prompt": tok.decode([bos])}
    for sname, eval_items in sets.items():
        toks, tidx = [], {}
        def ti(t):
            s = tok.decode([t])
            if s not in tidx:
                tidx[s] = len(toks); toks.append(s)
            return tidx[s]
        items = {}
        for ii, item in enumerate(tqdm(eval_items, desc=sname)):
            prompt = item["prompt"]
            rec = {"tops": {}, "probs": {}}
            for test in ("g", "dg"):
                h, pidx, ptoks, T, segs = read_paper(prompt, test, sname)
                if "pos" not in rec:
                    rec["pos"] = [{"i": i, "tok": s} for i, s in zip(pidx, ptoks)]
                    rec["T"] = T; rec["segs"] = segs; rec["trunc"] = False
                for c in (*lenses, "logitlens"):
                    u = UNEMBED[c] if c != "logitlens" else UNEMBED["g_shared" if test == "g" else "dg"]
                    gt, gp = [], []
                    for j, L in enumerate(LSUB):
                        hb = h[j] if c == "logitlens" else h[j] @ J[c][L].T   # [nPos, d]
                        top = u(hb).softmax(-1).topk(TOPK, dim=-1)            # [nPos, TOPK]
                        gt.append([[ti(int(t)) for t in row] for row in top.indices.tolist()])
                        gp.append([[int(round(v * 1000)) for v in row] for row in top.values.tolist()])
                    rec["tops"][f"{c}|{test}"] = gt
                    rec["probs"][f"{c}|{test}"] = gp
            items[str(ii)] = rec
        path = DATA / f"jlens_topk_{sname}.json"
        path.write_text(json.dumps({"layers": LSUB, "topk": TOPK, "cell_cfg": cell_cfg,
                                    "readout": "paper-defined single position", "meta": META, "toks": toks,
                                    "items": items}))
        print(f"[topk] wrote {path} ({len(items)} items, {len(toks)} unique tokens, "
              f"{path.stat().st_size // 1024} KiB)")
    print("[topk] done")


if __name__ == "__main__":
    main()
