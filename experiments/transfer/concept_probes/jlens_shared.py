"""Shared fit corpus and Jacobian estimator for the controlled J-Lens comparison.

All fitted maps use the same 64 WikiText records, the same 127 text-token ids, and the same
text-token offsets. G and DG causal place the shared text after their BOS token; DG bidirectional
places the same text on its clean decoder canvas and uses BOS as the encoder prompt.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Callable

import torch

REPO = Path(os.environ.get("DGLR_ROOT", Path(__file__).resolve().parents[1]))
OUT = REPO / "concept_probes/out/saeprobes/jlens"
MANIFEST = OUT / "shared_fit_manifest.json"
MODEL_ID_G = "google/gemma-4-26b-a4b-it"
MODEL_ID_DG = "google/diffusiongemma-26B-A4B-it"
N_PROMPTS = 64
MAX_TEXT_TOKENS = 127
VALID_TEXT_START = 15
N_LAYERS = 30
D_MODEL = 2816
TARGET_LAYER = 29
SOURCE_LAYERS = list(range(TARGET_LAYER))


def _digest(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_manifest() -> dict:
    """Materialize the exact shared strings and token ids, checking both tokenizers agree."""
    import sys
    sys.path.insert(0, str(REPO / "third_party/jacobian-lens"))
    from jlens.examples import load_wikitext_prompts
    from transformers import AutoProcessor, AutoTokenizer

    candidates = load_wikitext_prompts(N_PROMPTS * 2)
    tok_g = AutoTokenizer.from_pretrained(MODEL_ID_G)
    tok_d = AutoProcessor.from_pretrained(MODEL_ID_DG).tokenizer
    items = []
    for text in candidates:
        ids_g = tok_g(text, add_special_tokens=False, truncation=True,
                      max_length=MAX_TEXT_TOKENS)["input_ids"]
        ids_d = tok_d(text, add_special_tokens=False, truncation=True,
                      max_length=MAX_TEXT_TOKENS)["input_ids"]
        assert ids_g == ids_d, f"shared tokenization diverged at WikiText candidate {len(items)}"
        if len(ids_g) < MAX_TEXT_TOKENS:
            continue
        index = len(items)
        valid = list(range(VALID_TEXT_START, len(ids_g) - 1))
        items.append({"index": index, "text": text, "text_input_ids": ids_g,
                      "valid_text_offsets": valid,
                      "text_sha256": hashlib.sha256(text.encode()).hexdigest()})
        if len(items) == N_PROMPTS:
            break
    assert len(items) == N_PROMPTS
    payload = {
        "source": "Salesforce/wikitext · wikitext-103-raw-v1 · train streaming",
        "selection": (f"first {N_PROMPTS} records with at least 600 characters and at least "
                      f"{MAX_TEXT_TOKENS} shared tokenizer tokens"),
        "models": [MODEL_ID_G, MODEL_ID_DG],
        "n_prompts": N_PROMPTS,
        "max_text_tokens": MAX_TEXT_TOKENS,
        "valid_text_offsets": f"{VALID_TEXT_START}…{MAX_TEXT_TOKENS - 2}",
        "items": items,
    }
    payload["corpus_sha256"] = _digest(payload)
    OUT.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(payload, indent=1))
    return payload


def load_manifest() -> dict:
    payload = json.loads(MANIFEST.read_text())
    claimed = payload.pop("corpus_sha256")
    actual = _digest(payload)
    payload["corpus_sha256"] = claimed
    assert actual == claimed, (actual, claimed)
    assert payload["n_prompts"] == N_PROMPTS == len(payload["items"])
    for index, item in enumerate(payload["items"]):
        assert item["index"] == index
        assert len(item["text_input_ids"]) == MAX_TEXT_TOKENS
        assert item["valid_text_offsets"] == list(range(VALID_TEXT_START, MAX_TEXT_TOKENS - 1))
    return payload


def assert_tokenizer_matches(tokenizer, manifest: dict) -> None:
    for item in manifest["items"]:
        ids = tokenizer(item["text"], add_special_tokens=False, truncation=True,
                        max_length=MAX_TEXT_TOKENS)["input_ids"]
        assert ids == item["text_input_ids"], item["index"]


def jacobian_for_input_ids(
    layers,
    forward: Callable[[torch.Tensor], object],
    input_ids: torch.Tensor,
    valid_positions: torch.Tensor,
    *,
    dim_batch: int,
) -> dict[int, torch.Tensor]:
    """The paper estimator on caller-supplied ids/positions for either attention mode."""
    import sys
    sys.path.insert(0, str(REPO / "third_party/jacobian-lens"))
    from jlens.hooks import ActivationRecorder

    jacobians = {layer: torch.zeros(D_MODEL, D_MODEL, dtype=torch.float32)
                 for layer in SOURCE_LAYERS}
    n_passes = math.ceil(D_MODEL / dim_batch)
    with ActivationRecorder(layers, at=[*SOURCE_LAYERS, TARGET_LAYER],
                            start_graph_at=min(SOURCE_LAYERS)) as recorder, torch.enable_grad():
        forward(input_ids.expand(dim_batch, -1))
        target = recorder.activations[TARGET_LAYER]
        sources = [recorder.activations[layer] for layer in SOURCE_LAYERS]
        valid = valid_positions.to(target.device)
        batch = torch.arange(dim_batch, device=target.device)
        cotangent = torch.zeros_like(target)
        for pass_index, dim_start in enumerate(range(0, D_MODEL, dim_batch)):
            n_dims = min(dim_batch, D_MODEL - dim_start)
            cotangent.zero_()
            cotangent[batch[:n_dims, None], valid[None, :],
                      dim_start + batch[:n_dims, None]] = 1.0
            grads = torch.autograd.grad(
                outputs=target, inputs=sources, grad_outputs=cotangent,
                retain_graph=pass_index < n_passes - 1)
            for layer, grad in zip(SOURCE_LAYERS, grads, strict=True):
                positions = valid.to(grad.device, non_blocking=True)
                jacobians[layer][dim_start:dim_start + n_dims] = (
                    grad[:n_dims, positions].float().mean(dim=1).cpu())
    return jacobians
