"""Data for the rebuilt probe 2×2 sina-viewer (probe_matrix.json):
- concept-level AUC per cell for the canonical causal read and the genuinely distinct DG
  bidirectional last/mean reads — the aggregate matrix;
- per-exemplar cell scores (probe predict_proba) for N_EX held-out test examples per concept, the
  per-task column of sinas;
- per-exemplar per-token PROJECTIONS for the 4 cells (applied-model residual · trained-probe unit
  weight), the click-to-inspect token attribution.

Each probe uses its source stream's training-CV-selected layer and C, held fixed across the
corresponding transfer row (acts1024). Reads: gemma causal, DG encoder stack (the headline DG
read). -> out/saeprobes/probe_matrix.json

srun ... bash concept_probes/slurm/ensure_and_run.sh concept_probes/probe_matrix_data.py
"""
from __future__ import annotations

import functools, hashlib, json, os, sys
from pathlib import Path

import numpy as np

print = functools.partial(print, flush=True)
REPO = Path(os.environ.get("DGLR_ROOT", Path(__file__).resolve().parents[1]))
OUT = Path(os.environ.get("SAEP_OUT", REPO / "concept_probes/out/saeprobes"))
ACTS = OUT / "acts1024"
N_EX = 8                    # pos + neg exemplars per concept (each with token attribution)


def _load(n, f):
    import importlib.util
    s = importlib.util.spec_from_file_location(n, REPO / f"concept_probes/{f}")
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


def main():
    import torch
    from sklearn.linear_model import LogisticRegression
    sys.path.insert(0, str(REPO / "concept_probes"))
    rcp = _load("rcp", "run_concept_probes.py")
    jsg = _load("jsg", "judge_steer_gens.py")
    sd = _load("sd", "saeprobes_data.py")
    pp = _load("pp", "probe_protocol.py")

    keep = {t for t, v in json.loads((OUT / "concept_entity_filter.json").read_text()).items()
            if not v["single_entity"]}
    pr = {d["tag"]: d for d in json.loads((OUT / "probe_results.json").read_text())}
    fam_p = OUT / "dg_family_1024_w512.json"      # canonical 512-window family (96 era retired)
    fam = json.loads(fam_p.read_text()) if fam_p.exists() else None
    datasets = {d["tag"]: d for d in sd.load_datasets()}
    desc = jsg.concept_descriptions()
    tags = sorted((t for t in keep if (ACTS / f"{t}.npz").exists()), key=lambda t: int(t.split("_")[0]))
    print(f"[pm] {len(tags)} concepts")

    ARM = {"gg": "gemma_clean", "gd": "transfer_clean", "dg": "reverse_clean", "dd": "dgnative_clean"}
    # aggregate concept-level AUC per cell per DG-mode
    concept_auc = {"headline": {c: [] for c in ARM}}
    for t in tags:
        for c, a in ARM.items():
            concept_auc["headline"][c].append({"tag": t, "auc": pp.selected_value(pr[t], a)})
    if fam:
        FM = {"declast": "dec_last", "decmean": "dec_mean"}
        for mode, sub in FM.items():
            concept_auc[mode] = {c: [] for c in ARM}
            for t in tags:
                r = fam["per_concept"].get(t)
                if not r:
                    continue
                concept_auc[mode]["gg"].append({"tag": t, "auc": pp.selected_value(pr[t], "gemma_clean")})
                for c, pre in [("gd", "gd"), ("dg", "dg"), ("dd", "dd")]:
                    concept_auc[mode][c].append({"tag": t, "auc": r[f"{pre}|{sub}"]})

    model, tok = rcp.load_model("gemma4", device_map={"": 0})
    model_d, _ = rcp.load_model("diffusiongemma", device_map={"": 0})
    _, g_layers = rcp.locate(model)
    enc_lm = model_d.model.encoder.language_model
    enc_layers = enc_lm.layers
    device = model.device

    cap = {}
    def mk(store, L):
        def h(_m, _i, out): store[L] = out[0] if isinstance(out, tuple) else out
        return h

    @torch.no_grad()
    def read_tokens(text, Ls, which, max_len=512):
        """{L: [T,d] per-token residuals} for gemma ('g') or DG-encoder ('e')."""
        tok.truncation_side = "left"
        ids = tok(text, return_tensors="pt", truncation=True, max_length=max_len)["input_ids"].to(device)
        layers = g_layers if which == "g" else enc_layers
        st = {}
        hs = [layers[L].register_forward_hook(mk(st, L)) for L in Ls]
        try:
            (model if which == "g" else enc_lm)(input_ids=ids, attention_mask=torch.ones_like(ids))
        finally:
            for h in hs:
                h.remove()
        toks = [x.replace("▁", " ").replace("Ġ", " ") for x in tok.convert_ids_to_tokens(ids[0].tolist())]
        input_text = tok.decode(ids[0].tolist(), skip_special_tokens=True)
        return toks, {L: st[L][0].float().cpu().numpy() for L in Ls}, input_text

    def fit_probe(X, y, C):
        return LogisticRegression(C=C, solver="liblinear", max_iter=1000).fit(X, y)

    exemplars = {}
    for ci, tag in enumerate(tags):
        z = np.load(ACTS / f"{tag}.npz")
        lids = z["layer_ids"].tolist()
        rg = pp.selected_row(pr[tag], "gemma_clean")
        rd = pp.selected_row(pr[tag], "dgnative_clean")
        Lg, Ld = rg["layer"], rd["layer"]
        jg, jd = lids.index(Lg), lids.index(Ld)
        mg = fit_probe(z["g_train"][:, jg, :].astype(np.float32), z["y_train"], rg["C_gemma"])
        md = fit_probe(z["d_train"][:, jd, :].astype(np.float32), z["y_train"], rd["C_dg"])
        wg = mg.coef_[0] / (np.linalg.norm(mg.coef_[0]) + 1e-8)
        wd = md.coef_[0] / (np.linalg.norm(md.coef_[0]) + 1e-8)
        d = datasets[tag]
        eval_mask = pp.train_disjoint_test_mask(d)
        sem_pos = 0 if d["flipped"] else 1
        idx_pos = [i for i, y in enumerate(d["y_test"]) if eval_mask[i] and y == sem_pos][:N_EX]
        idx_neg = [i for i, y in enumerate(d["y_test"]) if eval_mask[i] and y != sem_pos][:N_EX]
        pts = []
        for idx, lbl in [(i, 1) for i in idx_pos] + [(i, 0) for i in idx_neg]:
            source_text = d["texts_test"][idx]
            gt, gh, g_input = read_tokens(source_text, sorted({Lg, Ld}), "g")
            et, eh, e_input = read_tokens(source_text, sorted({Lg, Ld}), "e")
            assert g_input == e_input
            def proj(arr, w):
                return [round(float(x), 4) for x in (arr @ w)]
            # cell = applied-model residual @ layer · trained-probe weight
            pts.append({
                "label": lbl, "src": idx, "text": g_input,
                "source_text_sha256": hashlib.sha256(source_text.encode()).hexdigest(),
                "gg": round(float(mg.predict_proba(z["g_test"][idx:idx+1, jg, :].astype(np.float32))[0, 1]), 4),
                "gd": round(float(mg.predict_proba(z["d_test"][idx:idx+1, jg, :].astype(np.float32))[0, 1]), 4),
                "dg": round(float(md.predict_proba(z["g_test"][idx:idx+1, jd, :].astype(np.float32))[0, 1]), 4),
                "dd": round(float(md.predict_proba(z["d_test"][idx:idx+1, jd, :].astype(np.float32))[0, 1]), 4),
                "tok_g": gt, "tok_e": et,
                "proj_gg": proj(gh[Lg], wg), "proj_dg": proj(gh[Ld], wd),
                "proj_gd": proj(eh[Lg], wg), "proj_dd": proj(eh[Ld], wd)})
        exemplars[tag] = {"desc": desc.get(tag, tag), "Lg": int(Lg), "Ld": int(Ld),
                          "flipped": bool(d["flipped"]), "pts": pts}
        print(f"[pm] {ci+1}/{len(tags)} {tag} (Lg{Lg} Ld{Ld}, {len(pts)} exemplars)")

    out = {"cells": list(ARM), "cell_meta": {"gg": ["gemma", "gemma"], "gd": ["gemma", "DG"],
           "dg": ["DG", "gemma"], "dd": ["DG", "DG"]},
           "dg_modes": list(concept_auc), "concept_auc": concept_auc, "exemplars": exemplars,
           "probe_protocol": {"layer_selection": "source-stream training-only 5-fold CV AUC",
                              "transfer_rule": "one source-selected layer held fixed across each matrix row",
                              "evaluation": "train-disjoint unique test strings",
                              "token_input": "same left-truncated 512-token input as the scored residual"}}
    (OUT / "probe_matrix.json").write_text(json.dumps(out))
    print(f"[pm] wrote probe_matrix.json ({len(tags)} concepts, {len(concept_auc)} DG modes)")


if __name__ == "__main__":
    main()
