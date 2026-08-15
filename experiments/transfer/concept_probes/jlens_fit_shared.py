"""Fit the three controlled J-Lens maps on one exact shared corpus.

Build the corpus once on CPU:
  SAEP_CPU=1 .../ensure_and_run.sh concept_probes/jlens_fit_shared.py --build-corpus

Fit on SLURM with JL_TARGET in {g_shared,dgc_shared,dgb_shared} and SAEP_SHARD=i/n.
Every target consumes the same manifest items and text-token offsets. Per-shard lens files are
merged by JacobianLens.merge in evaluation.
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import time
from pathlib import Path

import torch
from tqdm.auto import tqdm

print = functools.partial(print, flush=True)
REPO = Path(os.environ.get("DGLR_ROOT", Path(__file__).resolve().parents[1]))
OUT = REPO / "concept_probes/out/saeprobes/jlens"
sys.path.insert(0, str(REPO / "concept_probes"))
sys.path.insert(0, str(REPO / "third_party/jacobian-lens"))
sys.path.insert(0, str(REPO.parent / "worker"))  # DG worker (jlens_dg_common + server), vendored separately at ../worker

import importlib.util
def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, REPO / f"concept_probes/{filename}")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module

from jlens.lens import JacobianLens
from jlens.hf import Layout
import jlens
from jlens_shared import (D_MODEL, MAX_TEXT_TOKENS, SOURCE_LAYERS, assert_tokenizer_matches,
                          build_manifest, jacobian_for_input_ids, load_manifest)

TARGETS = ("g_shared", "dgc_shared", "dgb_shared")
DIM_BATCH = int(os.environ.get("JL_DIM_BATCH", 16))


def _atomic_save(state, path: Path) -> None:
    temporary = Path(f"{path}.tmp.{os.getpid()}")
    torch.save(state, temporary)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-corpus", action="store_true")
    args = parser.parse_args()
    if args.build_corpus:
        manifest = build_manifest()
        print(f"[jl/shared] wrote shared corpus n={manifest['n_prompts']} "
              f"sha256={manifest['corpus_sha256']}")
        return

    target = os.environ["JL_TARGET"]
    assert target in TARGETS, target
    shard_i, shard_n = (int(x) for x in os.environ["SAEP_SHARD"].split("/"))
    manifest = load_manifest()
    items = manifest["items"][shard_i::shard_n]
    assert items
    rcp = _load("rcp", "run_concept_probes.py")
    model_key = "gemma4" if target == "g_shared" else "diffusiongemma"
    model, tokenizer = rcp.load_model(model_key, device_map={"": 0})
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    assert_tokenizer_matches(tokenizer, manifest)

    if target == "g_shared":
        wrapped = jlens.from_hf(model, tokenizer)
        layers = wrapped.layers
    elif target == "dgc_shared":
        wrapped = jlens.from_hf(model, tokenizer,
                                layout=Layout("model.encoder.language_model"))
        layers = wrapped.layers
    else:
        wrapped = None
        layers = model.model.decoder.layers

    checkpoint = OUT / f"{target}_fit_ckpt_shard{shard_i}.pt"
    sums = {layer: torch.zeros(D_MODEL, D_MODEL) for layer in SOURCE_LAYERS}
    count = next_index = 0
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        assert state["target"] == target
        assert state["corpus_sha256"] == manifest["corpus_sha256"]
        assert state["shard"] == [shard_i, shard_n]
        sums, count, next_index = state["sums"], state["count"], state["next_index"]
        print(f"[jl/shared] resumed {target} shard {shard_i}/{shard_n} at {next_index}")

    def save_checkpoint(index: int) -> None:
        _atomic_save({"target": target, "corpus_sha256": manifest["corpus_sha256"],
                      "shard": [shard_i, shard_n], "sums": sums,
                      "count": count, "next_index": index}, checkpoint)

    t0 = time.time()
    for local_index, item in enumerate(tqdm(items, desc=f"{target} {shard_i}/{shard_n}")):
        if local_index < next_index:
            continue
        text_ids = item["text_input_ids"]
        valid_offsets = item["valid_text_offsets"]
        started = time.time()
        if target in ("g_shared", "dgc_shared"):
            bos = tokenizer.bos_token_id
            assert bos is not None
            input_ids = torch.tensor([[bos, *text_ids]], device=model.device)
            valid = torch.tensor([offset + 1 for offset in valid_offsets])
            jacobians = jacobian_for_input_ids(
                layers, wrapped.forward, input_ids, valid, dim_batch=DIM_BATCH)
        else:
            from jlens_dg_common import prefill_replay, replay_decoder
            bos = tokenizer.bos_token_id
            assert bos is not None
            encoder = {"input_ids": torch.tensor([[bos]], device=model.device)}
            past, attention, positions = prefill_replay(
                model, encoder, DIM_BATCH, MAX_TEXT_TOKENS)
            input_ids = torch.tensor([text_ids], device=model.device)
            valid = torch.tensor(valid_offsets)
            def forward(replicated_ids):
                return replay_decoder(model, past, attention, positions,
                                      replicated_ids, None, DIM_BATCH)
            jacobians = jacobian_for_input_ids(
                layers, forward, input_ids, valid, dim_batch=DIM_BATCH)
        for layer in SOURCE_LAYERS:
            sums[layer] += jacobians[layer]
        count += 1
        next_index = local_index + 1
        if next_index % 4 == 0 or next_index == len(items):
            save_checkpoint(next_index)
        print(f"[jl/shared] {target} global item {item['index']} "
              f"n_text={len(text_ids)} n_valid={len(valid_offsets)} "
              f"seconds={time.time() - started:.1f}")

    assert count == len(items), (count, len(items))
    lens_path = OUT / f"{target}_lens_shard{shard_i}.pt"
    JacobianLens({layer: sums[layer] / count for layer in SOURCE_LAYERS},
                 n_prompts=count, d_model=D_MODEL).save(str(lens_path))
    meta = {"target": target, "shard": [shard_i, shard_n],
            "global_prompt_indices": [item["index"] for item in items],
            "n_prompts": count, "n_text_tokens": MAX_TEXT_TOKENS,
            "valid_text_offsets": items[0]["valid_text_offsets"],
            "corpus_sha256": manifest["corpus_sha256"],
            "dim_batch": DIM_BATCH, "seconds": round(time.time() - t0, 1)}
    (OUT / f"{target}_fit_meta_shard{shard_i}.json").write_text(json.dumps(meta, indent=1))
    print(f"[jl/shared] wrote {lens_path} and metadata: {meta}")


if __name__ == "__main__":
    main()
