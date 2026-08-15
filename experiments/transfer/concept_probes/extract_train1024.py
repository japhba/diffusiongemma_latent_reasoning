"""Incremental probe re-extraction at the paper's 1024-train regime.

The learning curve (§2.1) showed the probes still climbing at the 512 cap (≈+0.02/doubling), so we
move the headline to the SAE-Probes paper's standard 1024. Design: the canonical splits are kept
byte-identical — the train split is TOPPED UP with balanced draws from the unused texts_extra pool
(string-deduped against test AND train), and only the NEW texts are extracted here; test/noised
test arrays are copied verbatim from the existing acts/{tag}.npz. Datasets without enough extras
top up as far as they can (achieved sizes logged + stored).

-> out/saeprobes/acts1024/{tag}.npz   (same schema as acts/, g_train/d_train now up to 1024 rows)

srun ... bash concept_probes/slurm/ensure_and_run.sh concept_probes/extract_train1024.py
"""
from __future__ import annotations

import functools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

print = functools.partial(print, flush=True)
REPO = Path(os.environ.get("DGLR_ROOT", Path(__file__).resolve().parents[1]))
OUT = Path(os.environ.get("SAEP_OUT", REPO / "concept_probes/out/saeprobes"))
TARGET = int(os.environ.get("SAEP_TARGET", 1024))


def _load(n, f):
    import importlib.util
    s = importlib.util.spec_from_file_location(n, REPO / f"concept_probes/{f}")
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


def main():
    import torch
    sys.path.insert(0, str(REPO / "concept_probes"))
    rcp = _load("rcp", "run_concept_probes.py")
    rsg = _load("rsg", "run_saeprobes_gpu.py")
    sd = _load("sd", "saeprobes_data.py")

    model_g, tok = rcp.load_model("gemma4", device_map={"": 0})
    bb_g, Lg = rcp.locate(model_g)
    model_d, _ = rcp.load_model("diffusiongemma", device_map={"": 0})
    bb_d, Ld = rcp.locate(model_d)
    device = model_g.device
    layer_ids = rsg.cand_layers(len(Lg))

    datasets = sd.load_datasets()
    dst = OUT / os.environ.get("SAEP_DST_ACTS", "acts1024"); dst.mkdir(exist_ok=True)
    sizes = {}
    t0 = time.time()
    for di, d in enumerate(datasets):
        tag = d["tag"]
        f_old = OUT / "acts" / f"{tag}.npz"
        f_new = dst / f"{tag}.npz"
        if f_new.exists():
            sizes[tag] = int(np.load(f_new)["y_train"].shape[0]); continue
        old = np.load(f_old)
        assert old["layer_ids"].tolist() == layer_ids, f"{tag}: layer grid mismatch"
        texts, y = sd.topup_train(d, TARGET)
        n_old = old["y_train"].shape[0]
        assert y[:n_old] == list(old["y_train"]), f"{tag}: canonical train prefix changed"
        new_texts = texts[n_old:]
        if new_texts:
            g_new, d_new = rsg.extract_pair(bb_g, Lg, bb_d, Ld, tok, new_texts, layer_ids, device)
            g_tr = np.concatenate([old["g_train"], g_new.astype(np.float16)])
            d_tr = np.concatenate([old["d_train"], d_new.astype(np.float16)])
        else:
            g_tr, d_tr = old["g_train"], old["d_train"]
        np.savez(f_new, layer_ids=np.array(layer_ids),
                 y_train=np.array(y), y_test=old["y_test"],
                 g_train=g_tr, d_train=d_tr,
                 g_test=old["g_test"], d_test=old["d_test"],
                 g_test_noised=old["g_test_noised"], d_test_noised=old["d_test_noised"])
        sizes[tag] = len(y)
        print(f"[x1024 {di + 1:>3}/{len(datasets)}] {tag:<40} train {n_old}->{len(y)} "
              f"({(time.time() - t0) / 60:.0f} min)")
    (dst / "sizes.json").write_text(json.dumps(sizes, indent=1))
    full = sum(1 for v in sizes.values() if v >= TARGET)
    print(f"[x1024] done: {full}/{len(sizes)} at {TARGET}; min {min(sizes.values())}, "
          f"median {int(np.median(list(sizes.values())))}")


if __name__ == "__main__":
    main()
