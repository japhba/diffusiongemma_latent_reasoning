"""Per-EXAMPLE probe scores for the §2.2 interactive scatters: for each 2×2 cell and each
entity-filtered concept, refit the source probe at its training-CV-selected layer and C (same
recipe as fit_saeprobes, acts1024) and dump every source-test example's score
under the source read (x) and the target read (y).

Cells (probe → x-read / y-read):  gg: gemma probe → g clean / g NOISED (diagonal resolves the
noising testbed)   gd: gemma probe → g clean / d clean   dg: DG probe → d clean / g clean
dd: DG probe → d clean / d NOISED.

CPU:  SAEP_CPU=1 srun ... ensure_and_run.sh concept_probes/probe_example_scores.py
-> out/saeprobes/probe_example_scores.json
"""
from __future__ import annotations

import functools
import json
import os
from pathlib import Path

import numpy as np

print = functools.partial(print, flush=True)
REPO = Path(os.environ.get("DGLR_ROOT", Path(__file__).resolve().parents[1]))
OUT = Path(os.environ.get("SAEP_OUT", REPO / "concept_probes/out/saeprobes"))
ACTS = OUT / "acts1024"

# cell -> (train stream, x = (stream, cond), y = (stream, cond), probe_results arm for layer)
CELLS = {"gg": ("g", ("g", "clean"), ("g", "noised"), "gemma_clean"),
         "gd": ("g", ("g", "clean"), ("d", "clean"), "transfer_clean"),
         "dg": ("d", ("d", "clean"), ("g", "clean"), "reverse_clean"),
         "dd": ("d", ("d", "clean"), ("d", "noised"), "dgnative_clean")}


def _fit_one(tag, texts, source_indices, flipped, params_by_cell):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    z = np.load(ACTS / f"{tag}.npz")
    ytr = z["y_train"]
    yte_raw = z["y_test"][source_indices]
    yte = 1 - yte_raw if flipped else yte_raw
    lid = list(z["layer_ids"])

    def te(stream, cond, j):
        key = f"{stream}_test" + ("_noised" if cond == "noised" else "")
        return z[key][source_indices, j, :].astype(np.float32)

    rec = {"y": yte.tolist(), "texts": texts, "source_indices": source_indices.tolist(),
           "layers": {}, "cells": {}, "protocol_version": 2,
           "score_input": "left-truncated 512-token window; texts stores the full source string",
           "evaluation": "train-disjoint unique test strings", "semantic_orientation": True}
    for cell, (trs, xs, ys, arm) in CELLS.items():
        L, C = params_by_cell[cell]
        j = lid.index(L)
        mdl = LogisticRegression(C=C, random_state=1, solver="liblinear", max_iter=1000).fit(
            z[f"{trs}_train"][:, j, :].astype(np.float32), ytr)
        sx = mdl.predict_proba(te(*xs, j))[:, 1]
        sy = mdl.predict_proba(te(*ys, j))[:, 1]
        if flipped:
            sx, sy = 1 - sx, 1 - sy
        rec["layers"][cell] = int(L)
        rec["cells"][cell] = {
            "x": np.round(sx, 4).tolist(), "y": np.round(sy, 4).tolist(),
            "auc_x": round(float(roc_auc_score(yte, sx)), 4),
            "auc_y": round(float(roc_auc_score(yte, sy)), 4)}
    (OUT / "pex" / f"{tag}.json").write_text(json.dumps(rec))   # per-concept checkpoint (resume)
    print(f"[pex] {tag} done", flush=True)
    return tag, rec


def main():
    from joblib import Parallel, delayed
    import sys
    sys.path.insert(0, str(REPO / "concept_probes"))
    import importlib.util
    s = importlib.util.spec_from_file_location("sd", REPO / "concept_probes/saeprobes_data.py")
    sd = importlib.util.module_from_spec(s); s.loader.exec_module(sd)
    p = importlib.util.spec_from_file_location("pp", REPO / "concept_probes/probe_protocol.py")
    pp = importlib.util.module_from_spec(p); p.loader.exec_module(pp)

    keep = {t for t, v in json.loads((OUT / "concept_entity_filter.json").read_text()).items()
            if not v["single_entity"]}
    pr = {d["tag"]: d for d in json.loads((OUT / "probe_results.json").read_text())}
    datasets = {d["tag"]: d for d in sd.load_datasets()}
    tags = sorted((t for t in keep if (ACTS / f"{t}.npz").exists()),
                  key=lambda t: int(t.split("_")[0]))
    print(f"[pex] {len(tags)} concepts")

    def params_for(tag):
        out = {}
        for cell, (stream, _, _, arm) in CELLS.items():
            row = pp.selected_row(pr[tag], arm)
            out[cell] = (row["layer"], row["C_gemma" if stream == "g" else "C_dg"])
        return out

    (OUT / "pex").mkdir(exist_ok=True)
    def current(tag):
        path = OUT / "pex" / f"{tag}.json"
        return path.exists() and json.loads(path.read_text()).get("protocol_version") == 2
    todo = [t for t in tags if not current(t)]
    print(f"[pex] {len(todo)} to fit ({len(tags) - len(todo)} resumed)")
    jobs = []
    for t in todo:
        d = datasets[t]
        idx = np.flatnonzero(pp.train_disjoint_test_mask(d))
        jobs.append((t, [d["texts_test"][i] for i in idx], idx, d["flipped"], params_for(t)))
    Parallel(n_jobs=12, verbose=10)(delayed(_fit_one)(*j) for j in jobs)
    res = {t: json.loads((OUT / "pex" / f"{t}.json").read_text()) for t in tags}
    (OUT / "probe_example_scores.json").write_text(json.dumps(res))
    print(f"[pex] wrote probe_example_scores.json ({len(res)} concepts)")


if __name__ == "__main__":
    main()
