"""SAE-Probes concept datasets (arXiv 2502.16681) as the NON-SYNTHETIC data source
for the Gemma-4 vs DiffusionGemma representational-similarity + probe-transfer study.

Replaces the Haiku-generated `stimuli.json`: every concept is one of the paper's
~113 real-text binary probing datasets (news topics, GLUE, historical figures,
code vs prose, sentiment, languages, ...), loaded from
third_party/SAE-Probes/data/cleaned_data/*.csv  (columns: prompt, target).

Split convention follows the paper (utils_data.get_train_test_indices):
balanced pos_ratio=0.5, seed=42, train then test disjoint.

Also provides the 10%-noising used by the robustness testbed: replace a fixed
fraction of token positions with random vocab tokens, identical corrupted ids
for both models (shared tokenizer).
"""
from __future__ import annotations

import re
import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(os.environ.get("DGLR_ROOT", Path(__file__).resolve().parents[1]))
SAEP = REPO / "third_party/SAE-Probes"
CLEANED = SAEP / "data/cleaned_data"

NUM_TRAIN = 512   # balanced train examples per concept (paper standard regime uses 1024; 512 keeps 2x26B extraction tractable)
NUM_TEST = 256    # held-out, balanced
SEED = 42


def _category(tag: str) -> str:
    """Coarse concept category from the dataset tag (for per-regime RSA aggregation)."""
    t = tag.lower()
    if any(k in t for k in ["hist_fig", "wikidata", "athlete_sport"]):
        return "entity"
    if any(k in t for k in ["nyc_borough", "us_state", "us_timezone", "world_country"]):
        return "geography"
    if any(k in t for k in ["glue", "cola", "qnli", "sst", "mrpc", "cm_isshort"]):
        return "linguistic"
    if any(k in t for k in ["code_", "aimade", "ai_gen", "art_type", "headline"]):
        return "code/format"
    if any(k in t for k in ["news", "agnews", "click_bait", "it_tick"]):
        return "news/topic"
    if any(k in t for k in ["movie_sent", "twt_emotion", "hate", "toxic", "spam", "amazon"]):
        return "sentiment/tone"
    if any(k in t for k in ["truthqa", "sciq", "phys_tf", "cs_tf", "reasoning", "arith",
                            "temp_sense", "temp_cat", "context_type", "cm_correct",
                            "deon", "just_is", "virtue_is"]):
        return "knowledge/reasoning"
    if any(k in t for k in ["cancer", "disease"]):
        return "medical"
    # 65-85: Gurnee sparse-probing bigram-context datasets (e.g. living-room, blood-pressure)
    num = int(t.split("_")[0])
    if 65 <= num <= 85:
        return "context-bigram"
    return "other"


def load_datasets(max_texts_chars: int = 4000) -> list[dict]:
    """All binary SAE-Probes datasets with balanced train/test splits.

    Returns [{tag, num, category, texts_train, y_train, texts_test, y_test}].
    Datasets too small for the balanced split get proportionally shrunk splits;
    below 64 train examples they are dropped (fail-loud print, not silent).
    """
    master = pd.read_csv(SAEP / "data/probing_datasets_MASTER.csv")
    binary = master[master["Data type"] == "Binary Classification"]
    out = []
    for save_name in binary["Dataset save name"]:
        path = SAEP / "data" / save_name
        tag = Path(save_name).stem  # e.g. "5_hist_fig_ismale"
        if not path.exists():
            print(f"[saeprobes_data] MISSING {path.name} (raw-text dropbox bundle incomplete?)")
            continue
        df = pd.read_csv(path)
        if "prompt" not in df.columns or "target" not in df.columns:
            print(f"[saeprobes_data] SKIP {tag}: no prompt/target columns ({list(df.columns)})")
            continue
        df = df.dropna(subset=["prompt", "target"])
        texts = [re.sub(r"\s+", " ", str(t)).strip()[:max_texts_chars] for t in df["prompt"]]
        # LabelEncoder semantics: sorted unique -> 0/1
        vals = sorted(df["target"].unique())
        if len(vals) != 2:
            print(f"[saeprobes_data] SKIP {tag}: {len(vals)} classes")
            continue
        y = (df["target"].values == vals[1]).astype(int)
        # y=1 is the alphabetically-LAST target value (LabelEncoder convention). For
        # these pairs that is the semantic NEGATIVE of the concept — probe metrics are
        # label-symmetric, but judge-based signs and example labels must correct for it.
        flipped = (str(vals[0]), str(vals[1])) in {
            ("bigram", "not_bigram"), ("clickbait", "factual"), ("correct", "incorrect")
        } or tag in {
            # These source CSVs use numeric labels whose `1` class is the semantic
            # complement named by the dataset tag/master-sheet target.
            "90_glue_qnli",       # 0 = entailment, 1 = not-entailment
            "100_news_fake",      # 0 = fake, 1 = real (Reuters)
        }
        n_pos, n_neg = int(y.sum()), int((1 - y).sum())
        half_tr = min(NUM_TRAIN // 2, int(0.8 * min(n_pos, n_neg)))
        half_te = min(NUM_TEST // 2, min(n_pos, n_neg) - half_tr)
        if half_tr < 32 or half_te < 16:
            print(f"[saeprobes_data] SKIP {tag}: too small (pos={n_pos} neg={n_neg})")
            continue
        rng = np.random.default_rng(SEED)
        pos_idx = rng.permutation(np.where(y == 1)[0])
        neg_idx = rng.permutation(np.where(y == 0)[0])
        tr = np.concatenate([pos_idx[:half_tr], neg_idx[:half_tr]])
        te = np.concatenate([pos_idx[half_tr : half_tr + half_te], neg_idx[half_tr : half_tr + half_te]])
        tr, te = rng.permutation(tr), rng.permutation(te)
        num = int(tag.split("_")[0])
        # everything OUTSIDE the canonical splits (splits stay byte-identical; the
        # held-out test set stays pure) — the FULL-data pool for vector derivation.
        # Datasets contain duplicate strings, so exclude string-matches to test too.
        EXTRA_CAP = 10**9  # uncapped: the entire remaining dataset
        test_strings = {texts[i] for i in te}
        extra_y1 = [i for i in pos_idx[half_tr + half_te:]
                    if texts[i] not in test_strings][:EXTRA_CAP]
        extra_y0 = [i for i in neg_idx[half_tr + half_te:]
                    if texts[i] not in test_strings][:EXTRA_CAP]
        out.append({
            "tag": tag, "num": num, "category": _category(tag), "flipped": flipped,
            "label_y1": str(vals[1]), "label_y0": str(vals[0]),
            "texts_train": [texts[i] for i in tr], "y_train": y[tr].tolist(),
            "texts_test": [texts[i] for i in te], "y_test": y[te].tolist(),
            "texts_extra_y1": [texts[i] for i in extra_y1],
            "texts_extra_y0": [texts[i] for i in extra_y0],
        })
    assert len(out) >= 80, f"only {len(out)} usable SAE-Probes datasets — raw-text bundle incomplete"
    print(f"[saeprobes_data] {len(out)} binary concept datasets "
          f"({sum(len(d['texts_train']) + len(d['texts_test']) for d in out)} texts)")
    return out


def topup_train(d: dict, target: int = 1024, seed: int = 123) -> tuple[list[str], list[int]]:
    """Extend a dataset's canonical train split toward `target` examples (paper standard regime)
    with balanced draws from the texts_extra pools. The canonical train texts stay FIRST and
    byte-identical, and the held-out test split is untouched (extras are already string-deduped
    against test; deduped against train here). Datasets without enough extras top up as far as
    they can — callers should report the achieved size."""
    have = len(d["texts_train"])
    need = (target - have) // 2
    if need <= 0:
        return list(d["texts_train"]), list(d["y_train"])
    tr_set = set(d["texts_train"])
    e1 = list(dict.fromkeys(t for t in d["texts_extra_y1"] if t not in tr_set))
    e0 = list(dict.fromkeys(t for t in d["texts_extra_y0"] if t not in tr_set))
    rng = np.random.default_rng(seed)
    k = min(need, len(e1), len(e0))
    add1 = [e1[i] for i in rng.permutation(len(e1))[:k]]
    add0 = [e0[i] for i in rng.permutation(len(e0))[:k]]
    return list(d["texts_train"]) + add1 + add0, list(d["y_train"]) + [1] * k + [0] * k


def noise_ids(ids: np.ndarray, attn: np.ndarray, frac: float, seed: int,
              vocab_size: int, protect_last: bool = True) -> np.ndarray:
    """Replace `frac` of real (non-BOS, optionally non-final) token positions with
    random vocab ids. ids/attn: [B, L] numpy. Returns a corrupted copy — the SAME
    corrupted ids are fed to both models (shared tokenizer)."""
    rng = np.random.default_rng(seed)
    out = ids.copy()
    for b in range(ids.shape[0]):
        real = np.where(attn[b] == 1)[0]
        cand = real[1:]  # never the BOS
        if protect_last and len(cand) > 1:
            cand = cand[:-1]  # keep the read position intact
        if len(cand) == 0:
            continue
        k = max(1, int(round(frac * len(cand))))
        pos = rng.choice(cand, size=min(k, len(cand)), replace=False)
        out[b, pos] = rng.integers(0, vocab_size, size=len(pos))
    return out
