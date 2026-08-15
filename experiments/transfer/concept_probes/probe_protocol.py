"""Shared selection and evaluation rules for SAE-Probes activation probes.

Layers are selected using training-only cross-validation on the source stream and are then held
fixed when that fitted probe is applied to either target stream. Evaluation excludes exact text
duplicates of the probe-fit split and repeated test strings.
"""
from __future__ import annotations

import numpy as np


SOURCE_CV_KEY = {
    "gemma_clean": "val_auc_gemma", "gemma_noised": "val_auc_gemma",
    "transfer_clean": "val_auc_gemma", "transfer_noised": "val_auc_gemma",
    "reverse_clean": "val_auc_dg", "reverse_noised": "val_auc_dg",
    "dgnative_clean": "val_auc_dg", "dgnative_noised": "val_auc_dg",
}


def selected_row(result: dict, arm: str) -> dict:
    """The source-CV-selected layer row for an evaluation arm."""
    key = SOURCE_CV_KEY[arm]
    assert all(key in row for row in result["layers"]), (result["tag"], key)
    return max(result["layers"], key=lambda row: (row[key], -row["layer"]))


def selected_value(result: dict, arm: str) -> float:
    return float(selected_row(result, arm)[arm])


def train_disjoint_test_mask(dataset: dict, train_target: int = 1024) -> np.ndarray:
    """Keep one copy of each test string, excluding strings seen during probe fitting."""
    import saeprobes_data as sd

    train_texts, _ = sd.topup_train(dataset, target=train_target)
    seen = set(train_texts)
    keep = []
    for text in dataset["texts_test"]:
        keep.append(text not in seen)
        seen.add(text)
    mask = np.asarray(keep, dtype=bool)
    y = np.asarray(dataset["y_test"])[mask]
    assert len(y) and len(np.unique(y)) == 2, dataset["tag"]
    return mask
