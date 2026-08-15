"""CPU probe fitting on the SAE-Probes activations (after run_saeprobes_gpu.py --phase extract).

SAE-Probes baseline methodology (utils_training.find_best_reg): logistic regression,
C tuned by stratified 5-fold CV on ROC-AUC, refit on the full train split.
Probes are TRAINED ON GEMMA-4 activations; we then measure zero-shot transfer of the
same probe (weights applied verbatim — the models share the residual basis hypothesis)
to DiffusionGemma activations on the same texts. DG-native probes give the ceiling.

Per (dataset, layer) we report AUC for:
  gemma_clean / gemma_noised          (native read, clean vs 10%-noised context)
  transfer_clean / transfer_noised    (gemma probe on DG activations)
  dgnative_clean / dgnative_noised    (DG-trained probe on DG activations)
plus cos(w_gemma, w_dg) — logreg-weight alignment — and the diff-of-means direction cosine.

Run on the workbench (no GPU):  python concept_probes/fit_saeprobes.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_protocol as pp
import saeprobes_data as sd

OUT = Path(os.environ.get("DGLR_ROOT", Path(__file__).resolve().parents[1])) / "concept_probes/out/saeprobes"
ACTS_DIR = os.environ.get("SAEP_ACTS_DIR", "acts")            # e.g. acts1024 for the paper regime
RESULTS = os.environ.get("SAEP_RESULTS", "probe_results.json")
C_GRID = [0.001, 0.01, 0.1, 1.0]


def cv_logreg(X, y, seed=1):
    """SAE-Probes find_best_reg, condensed: pick C by 5-fold CV AUC, refit on all."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = []
    for C in C_GRID:
        s = []
        for tr, va in skf.split(X, y):
            m = LogisticRegression(C=C, random_state=seed, solver="liblinear", max_iter=1000).fit(X[tr], y[tr])
            s.append(roc_auc_score(y[va], m.predict_proba(X[va])[:, 1]))
        scores.append(np.mean(s))
    best_C = C_GRID[int(np.argmax(scores))]
    return (LogisticRegression(C=best_C, random_state=seed, solver="liblinear", max_iter=1000).fit(X, y),
            best_C, float(np.max(scores)))


def unit(v):
    return v / (np.linalg.norm(v) + 1e-8)


def fit_one(f: Path, eval_mask: np.ndarray) -> dict:
    z = np.load(f)
    layer_ids = z["layer_ids"].tolist()
    y_tr, y_te = z["y_train"], z["y_test"][eval_mask]
    rows = []
    for j, L in enumerate(layer_ids):
        Xg_tr = z["g_train"][:, j, :].astype(np.float32)
        Xg_te = z["g_test"][eval_mask, j, :].astype(np.float32)
        Xg_te_n = z["g_test_noised"][eval_mask, j, :].astype(np.float32)
        Xd_tr = z["d_train"][:, j, :].astype(np.float32)
        Xd_te = z["d_test"][eval_mask, j, :].astype(np.float32)
        Xd_te_n = z["d_test_noised"][eval_mask, j, :].astype(np.float32)

        pg, C_g, val_g = cv_logreg(Xg_tr, y_tr)
        pd_, C_d, val_d = cv_logreg(Xd_tr, y_tr)

        def A(model, X):
            return float(roc_auc_score(y_te, model.predict_proba(X)[:, 1]))

        dom_g = unit(Xg_tr[y_tr == 1].mean(0) - Xg_tr[y_tr == 0].mean(0))
        dom_d = unit(Xd_tr[y_tr == 1].mean(0) - Xd_tr[y_tr == 0].mean(0))
        rows.append({
            "layer": int(L), "C_gemma": C_g, "C_dg": C_d,
            "val_auc_gemma": val_g, "val_auc_dg": val_d,
            "gemma_clean": A(pg, Xg_te), "gemma_noised": A(pg, Xg_te_n),
            "transfer_clean": A(pg, Xd_te), "transfer_noised": A(pg, Xd_te_n),
            "dgnative_clean": A(pd_, Xd_te), "dgnative_noised": A(pd_, Xd_te_n),
            # reverse transfer: DG-trained probe applied to gemma activations (probe 2x2)
            "reverse_clean": A(pd_, Xg_te), "reverse_noised": A(pd_, Xg_te_n),
            "cos_w": float(np.dot(unit(pg.coef_[0]), unit(pd_.coef_[0]))),
            "cos_dom": float(np.dot(dom_g, dom_d)),
        })
    print(f"[fit] {f.stem}: best gemma AUC "
          f"{max(r['gemma_clean'] for r in rows):.3f} | transfer "
          f"{max(r['transfer_clean'] for r in rows):.3f}")
    return {"tag": f.stem, "layers": rows, "n_test_source": int(len(eval_mask)),
            "n_test_eval": int(eval_mask.sum()),
            "test_rows_excluded_exact_duplicate": int((~eval_mask).sum()),
            "layer_selection": "source-stream training-only 5-fold CV AUC"}


def main():
    files = sorted((OUT / ACTS_DIR).glob("*.npz"))
    assert files, f"no activations under {OUT / ACTS_DIR} — run the GPU extract phase first"
    datasets = {d["tag"]: d for d in sd.load_datasets()}
    masks = {tag: pp.train_disjoint_test_mask(d) for tag, d in datasets.items()}
    results = Parallel(n_jobs=-1, verbose=5)(delayed(fit_one)(f, masks[f.stem]) for f in files)
    (OUT / RESULTS).write_text(json.dumps(results, indent=1))
    tc = [pp.selected_value(d, "transfer_clean") for d in results]
    gc = [pp.selected_value(d, "gemma_clean") for d in results]
    print(f"\n[fit] {len(results)} concepts  mean source-CV-layer AUC: "
          f"gemma {np.mean(gc):.3f}  transfer {np.mean(tc):.3f}  -> {RESULTS}")


if __name__ == "__main__":
    main()
