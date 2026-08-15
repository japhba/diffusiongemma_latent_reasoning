"""The active Jacobian-lens comparison: three fitted Jacobians × two test models.

Fit configs (rows are grouped by fit model at report time):
  g_shared    : G causal fit on the shared 64-prompt WikiText corpus
  dgc_shared  : DG causal fit on the exact same prompts/tokens/positions
  dgb_shared  : DG bidirectional clean-canvas fit on the exact same prompts/tokens/positions
  logitlens   : identity transport (baseline, per test model)

Test protocols: gemma = causal forward and DG = decoder mode (text as clean canvas over a BOS
prompt). Both hook every lens layer at the paper-defined prompt-token position.
Cross cells decode with the FIT side's unembedding (the lens = its J + its unembed).

Evals: the 6 jacobian-lens paper sets (paper-defined single readout position; metric = normalized
pass@k AUC of min-over-layers rank of intermediates; AVBench dropped 2026-07-18).
-> out/saeprobes/jlens/eval_2x2.json

srun ... bash concept_probes/slurm/ensure_and_run.sh concept_probes/jlens_eval_2x2.py
"""
from __future__ import annotations

import functools
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

print = functools.partial(print, flush=True)
REPO = Path(os.environ.get("DGLR_ROOT", Path(__file__).resolve().parents[1]))
OUT = Path(os.environ.get("SAEP_OUT", REPO / "concept_probes/out/saeprobes")) / "jlens"
EVDIR = REPO / "third_party/jacobian-lens/data/evaluations"
sys.path.insert(0, str(REPO / "concept_probes"))
sys.path.insert(0, str(REPO / "third_party/jacobian-lens"))

import importlib.util
def _load(n, f):
    s = importlib.util.spec_from_file_location(n, REPO / f"concept_probes/{f}")
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

KS = [1, 3, 10, 32, 100, 316, 1000]
MAX_TOK = 380          # fail loudly above the shared context limit
from jlens_paper_eval import load_paper_sets, paper_readout, paper_reference_forms


def merged_lens(pattern):
    from jlens.lens import JacobianLens
    paths = sorted(glob.glob(str(OUT / pattern)))
    if not paths:
        return None
    lenses = [JacobianLens.load(p) for p in paths]
    return lenses[0] if len(lenses) == 1 else JacobianLens.merge(lenses)


def main():
    rcp = _load("rcp", "run_concept_probes.py")
    model_g, tok = rcp.load_model("gemma4", device_map={"": 0})
    model_d, _ = rcp.load_model("diffusiongemma", device_map={"": 0})
    _, g_layers = rcp.locate(model_g)
    enc_lm = model_d.model.encoder.language_model
    dec_layers = model_d.model.decoder.layers
    device = model_g.device
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.pad_token_id

    # ---- lenses ----
    lenses = {}
    for name in ("g_shared", "dgc_shared", "dgb_shared"):
        pat = f"{name}_lens_shard*.pt"
        L = merged_lens(pat)
        if L is not None:
            lenses[name] = L
            print(f"[jleval] lens {name}: {len(L.jacobians)} layers, n={L.n_prompts}")
    assert set(lenses) == {"g_shared", "dgc_shared", "dgb_shared"}, lenses
    assert {lens.n_prompts for lens in lenses.values()} == {64}
    src_layers = sorted(next(iter(lenses.values())).jacobians)

    # ---- unembeds (per FIT side) ----
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
    UNEMBED["g"] = UNEMBED["g_shared"]

    # ---- residual readers ----
    cap = {}
    def mk(L):
        def h(_m, _i, out): cap[L] = out[0] if isinstance(out, tuple) else out
        return h

    # chat-template geometry: the template SUFFIX after the user text is constant — measure it
    # by diffing two templated probes (prefix matching is fragile: tokenization is context-dep).
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
    print(f"[jleval] template tails: {_a1[-8:]} | {_a2[-8:]}")
    print(f"[jleval] chat-template suffix = {TPL_SUFFIX} tokens")
    assert TPL_SUFFIX >= 1, "template-suffix probe failed — readout position would be wrong"
    _h1 = tok("Hello there", add_special_tokens=False)["input_ids"]
    _st = next(o for o in range(len(_a1) - len(_h1) + 1) if _a1[o:o + len(_h1)] == _h1)
    PRE_IDS, SUF_IDS = _a1[:_st], _a1[_st + len(_h1):]
    assert len(SUF_IDS) == TPL_SUFFIX and _a2[:len(PRE_IDS)] == PRE_IDS

    @torch.no_grad()
    def read(text, test, set_name):
        """[nL, d] residuals at the paper-defined text-token readout position."""
        ids_l, pidx, _ = paper_readout(tok, set_name, text)
        assert len(ids_l) <= MAX_TOK, (set_name, len(ids_l), MAX_TOK)
        layers = g_layers if test == "g" else dec_layers
        hs = [layers[L].register_forward_hook(mk(L)) for L in src_layers]
        try:
            cap.clear()
            if test == "g":
                full = torch.tensor([PRE_IDS + ids_l + SUF_IDS], device=device)
                model_g(input_ids=full, attention_mask=torch.ones_like(full))
                pos = len(PRE_IDS) + pidx
            else:
                ids = torch.tensor([ids_l], device=device)
                T = ids.shape[1]
                p = torch.full((1, 1), bos, dtype=torch.long, device=device)
                model_d(input_ids=p, attention_mask=torch.ones_like(p, dtype=torch.bool),
                        decoder_input_ids=ids,
                        decoder_position_ids=torch.arange(1, 1 + T, device=device).unsqueeze(0))
                pos = pidx
            return torch.stack([cap[L][0, pos].float() for L in src_layers])
        finally:
            for h in hs:
                h.remove()

    def key_token_ids(word):
        w = word.strip()
        out = set()
        for v in (f" {w}", w, f" {w.capitalize()}", w.capitalize(), f" {w.lower()}", w.lower()):
            t = tok(v, add_special_tokens=False)["input_ids"]
            if len(t) == 1:
                out.add(t[0])
        return sorted(out)

    def reference_ranks(h_stack, lens_name, set_name, references):
        """One min-over-layers/single-token-synonyms vocabulary rank per paper reference."""
        candidates = [sorted({t for form in paper_reference_forms(set_name, ref)
                              for t in key_token_ids(form)}) for ref in references]
        u = UNEMBED[lens_name if lens_name != "logitlens" else reference_ranks._test_unembed]
        best = [None] * len(candidates)
        for j, L in enumerate(src_layers):
            hj = h_stack[j]
            if lens_name != "logitlens":
                J = lenses[lens_name].jacobians[L].to(device=device, dtype=torch.float32)
                hj = J @ hj
            logits = u(hj.unsqueeze(0))[0]
            for ci, cands in enumerate(candidates):
                if not cands:
                    continue
                rr = min(int((logits > logits[c]).sum().item()) + 1 for c in cands)
                best[ci] = rr if best[ci] is None else min(best[ci], rr)
        return best

    # ---- eval sets ----
    sets = load_paper_sets(EVDIR)

    configs = list(lenses) + ["logitlens"]
    res = {c: {t: {} for t in ("g", "dg")} for c in configs}
    per_item = {}          # (set, idx) -> {set, idx, name, tail, inter, ranks{cfg|test: r}}
    def item_rec(sname, ii, name, tail, inter):
        return per_item.setdefault(f"{sname}|{ii}", {
            "set": sname, "idx": ii, "name": name, "tail": tail[-110:], "inter": inter, "ranks": {}})
    examples = []
    for test in ("g", "dg"):
        reference_ranks._test_unembed = test
        for sname, items in sets.items():
            ranks = {c: [] for c in configs}
            for ii, it in enumerate(items):
                h = read(it["prompt"], test, sname)
                rec = item_rec(sname, ii, it.get("name"), it["prompt"], it["intermediates"])
                for c in configs:
                    r = reference_ranks(h, c, sname, it["intermediates"])
                    ranks[c].append(r)
                    rec["ranks"][f"{c}|{test}"] = r
            for c in configs:
                rs = ranks[c]
                item_ranks = [[x for x in r if x is not None] for r in rs]
                item_ranks = [r for r in item_ranks if r]
                pass_at = {str(k): float(np.mean([np.mean(np.asarray(r) <= k) for r in item_ranks]))
                           for k in KS}
                ys = np.asarray([pass_at[str(k)] for k in KS])
                logk = np.log(KS)
                auc = np.sum((ys[:-1] + ys[1:]) * np.diff(logk) / 2) / (logk[-1] - logk[0])
                res[c][test][sname] = {
                    "pass_at": pass_at,
                    "auc": float(auc),
                    "n": len(rs), "n_scorable_items": len(item_ranks),
                    "n_references": sum(len(r) for r in rs),
                    "n_scorable_references": sum(x is not None for r in rs for x in r)}
            print(f"[jleval] test={test} {sname}: " +
                  " ".join(f"{c}={res[c][test][sname]['auc']:.2f}" for c in configs))

    # small verbatim examples: top-10 lens tokens at 4 depths, 2 items per paper set, both tests
    for sname in ("multihop", "poetry"):
        for it in sets[sname][:2]:
            ex = {"set": sname, "name": it.get("name"), "prompt_tail": it["prompt"][-120:],
                  "intermediates": it["intermediates"], "tops": {}}
            for test in ("g", "dg"):
                h = read(it["prompt"], test, sname)
                reference_ranks._test_unembed = test
                for c in ("g_shared", "dgb_shared", "logitlens"):
                    if c != "logitlens" and c not in lenses:
                        continue
                    u = UNEMBED[c if c != "logitlens" else reference_ranks._test_unembed]
                    for j in (4, 12, 20, 27):
                        L = src_layers[j]
                        hj = h[j]
                        if c != "logitlens":
                            hj = lenses[c].jacobians[L].to(device=device, dtype=torch.float32) @ hj
                        top = u(hj.unsqueeze(0))[0].topk(10).indices.tolist()
                        ex["tops"][f"{test}|{c}|L{L}"] = [tok.decode([t]) for t in top]
            examples.append(ex)
    (OUT / "eval_2x2.json").write_text(json.dumps(
        {"results": res, "ks": KS, "src_layers": src_layers, "examples": examples,
         "configs": configs, "per_item": list(per_item.values())}, indent=1))
    print(f"[jleval] wrote eval_2x2.json ({len(per_item)} items)")


if __name__ == "__main__":
    main()
