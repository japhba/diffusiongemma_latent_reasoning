"""Scan original J-Lens evals for tokens surfaced before their literal DG-canvas position.

The capture exactly matches the original clean-canvas bidirectional-DG protocol and uses the
original 64-WikiText-prompt ``dgb_shared`` lens.  Unlike the paper-position viewer, it reads every
text position so future-token surfacing can be located.  CPU analysis first finds exact future-token
hits; semantic/counterfactual controls are built from this resumable capture.
"""
from __future__ import annotations

import argparse
import functools
import glob
import json
import os
import re
import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm

print = functools.partial(print, flush=True)
REPO = Path(os.environ.get("DGLR_ROOT", Path(__file__).resolve().parents[1]))
OUT = REPO / "concept_probes/out/saeprobes/jlens"
EVDIR = REPO / "third_party/jacobian-lens/data/evaluations"
CAPTURE = OUT / "dg_original_future_scan"
TOPK = 50
WORKSPACE_LAYERS = tuple(range(8, 25))
sys.path[:0] = [str(REPO / "concept_probes"), str(REPO / "third_party/jacobian-lens")]

from jlens_paper_eval import load_paper_sets

STOP = {"the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "at", "for",
        "with", "is", "was", "are", "were", "be", "been", "it", "its", "he", "she",
        "his", "her", "they", "their", "this", "that", "as", "from", "by", "not"}

# Each control changes exactly one later token while leaving the probed prefix, sequence length,
# and every other suffix token fixed. Strings include their tokenizer-leading space.
CONTROL_CASES = (
    ("order-ops", 45, 3, 5, " minus", " plus"),
    ("order-ops", 46, 4, 7, " plus", " minus"),
    ("multihop", 8, 6, 9, " Topeka", " Denver"),
    ("multihop", 54, 6, 12, " Roman", " Greek"),
    ("multihop", 56, 10, 14, " Julius", " Augustus"),
    ("poetry", 4, 10, 22, " slumber", " rest"),
    ("poetry", 75, 9, 20, " grabbed", " dropped"),
    ("typo", 9, 3, 6, " city", " village"),
    ("association", 20, 30, 33, " twenty", " thirty"),
    ("association", 64, 6, 16, " silence", " noise"),
)


def capture_path(set_name: str, item_i: int) -> Path:
    return CAPTURE / f"{set_name}_{item_i:03d}.pt"


def merged_lens():
    from jlens.lens import JacobianLens

    paths = sorted(glob.glob(str(OUT / "dgb_shared_lens_shard*.pt")))
    assert paths
    lenses = [JacobianLens.load(path) for path in paths]
    lens = JacobianLens.merge(lenses)
    assert lens.n_prompts == 64
    return lens


def assigned(items: list[tuple[str, int, dict]]) -> list[tuple[str, int, dict]]:
    if "SAEP_SHARD" in os.environ:
        shard_i, shard_n = (int(value) for value in os.environ["SAEP_SHARD"].split("/"))
    elif "SLURM_PROCID" in os.environ:
        shard_i, shard_n = int(os.environ["SLURM_PROCID"]), int(os.environ["SLURM_NTASKS"])
    else:
        shard_i, shard_n = 0, 1
    result = items[shard_i::shard_n]
    print(f"[future/capture] shard {shard_i}/{shard_n}: {len(result)} items")
    return result


def stage_capture() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("rcp", REPO / "concept_probes/run_concept_probes.py")
    rcp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rcp)
    model, tok = rcp.load_model("diffusiongemma", device_map={"": 0})
    layers = model.model.decoder.layers
    device = model.device
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.pad_token_id
    lens = merged_lens()
    source_layers = sorted(lens.jacobians)
    assert source_layers == list(range(29))
    J = {layer: lens.jacobians[layer].to(device=device, dtype=torch.float32)
         for layer in source_layers}
    norm, head = model.model.decoder.norm, model.lm_head
    softcap = float(getattr(model.config.get_text_config(), "final_logit_softcapping", 0) or 0)

    def unembed(h):
        logits = head(norm(h.to(head.weight.dtype))).float()
        return softcap * torch.tanh(logits / softcap) if softcap else logits

    cap = {}

    def hook(layer):
        def save(_module, _inputs, output):
            cap[layer] = output[0] if isinstance(output, tuple) else output
        return save

    sets = load_paper_sets(EVDIR)
    items = assigned([(set_name, item_i, item) for set_name, rows in sets.items()
                      for item_i, item in enumerate(rows)])
    CAPTURE.mkdir(parents=True, exist_ok=True)
    todo = [(set_name, item_i, item) for set_name, item_i, item in items
            if not capture_path(set_name, item_i).exists()]
    print(f"[future/capture] {len(todo)} remaining")
    for set_name, item_i, item in tqdm(todo, desc="all-position original J-Lens"):
        token_ids = tok(item["prompt"], add_special_tokens=False)["input_ids"]
        ids = torch.tensor([token_ids], device=device)
        encoder = torch.full((1, 1), bos, dtype=torch.long, device=device)
        handles = [layers[layer].register_forward_hook(hook(layer)) for layer in source_layers]
        try:
            cap.clear()
            with torch.no_grad():
                model(input_ids=encoder, attention_mask=torch.ones_like(encoder, dtype=torch.bool),
                      decoder_input_ids=ids,
                      decoder_position_ids=torch.arange(1, 1 + len(token_ids),
                                                        device=device).unsqueeze(0))
                jlens_ids, jlens_probs, logit_ids, logit_probs = [], [], [], []
                for layer in source_layers:
                    h = cap[layer][0].float()
                    for destination_ids, destination_probs, transported in (
                            (jlens_ids, jlens_probs, h @ J[layer].T),
                            (logit_ids, logit_probs, h)):
                        top = torch.softmax(unembed(transported), dim=-1).topk(TOPK, dim=-1)
                        destination_ids.append(top.indices.to("cpu", torch.int32))
                        destination_probs.append(top.values.to("cpu", torch.float16))
        finally:
            for handle in handles:
                handle.remove()
        payload = {
            "set": set_name,
            "item_index": item_i,
            "item": item,
            "token_ids": torch.tensor(token_ids, dtype=torch.int32),
            "tokens": [tok.decode([token_id]) for token_id in token_ids],
            "layers": source_layers,
            "topk": TOPK,
            "jlens_ids": torch.stack(jlens_ids),
            "jlens_probs": torch.stack(jlens_probs),
            "logit_ids": torch.stack(logit_ids),
            "logit_probs": torch.stack(logit_probs),
            "protocol": "original dgb_shared lens; clean bidirectional DG canvas; every text position",
        }
        path = capture_path(set_name, item_i)
        tmp = path.with_suffix(f".tmp.{os.getpid()}")
        torch.save(payload, tmp)
        os.replace(tmp, path)
        print(f"[future/capture] {set_name} {item_i} {item['name']}\nPROMPT:\n{item['prompt']}")


def normalized(token: str) -> str:
    return re.sub(r"^[^\w]+|[^\w]+$", "", token, flags=re.UNICODE).casefold()


def stage_analyze() -> None:
    paths = sorted(CAPTURE.glob("*.pt"))
    expected = sum(len(rows) for rows in load_paper_sets(EVDIR).values())
    assert len(paths) == expected, (len(paths), expected)
    candidates = []
    for path in tqdm(paths, desc="exact future-token scan"):
        row = torch.load(path, map_location="cpu", weights_only=False)
        token_ids = row["token_ids"].long()
        tokens = row["tokens"]
        jlens = row["jlens_ids"][:, :, :20]
        logit = row["logit_ids"][:, :, :20]
        for source_position in range(len(tokens) - 2):
            prefix_ids = set(token_ids[:source_position + 1].tolist())
            future_by_id = {}
            for future_position in range(source_position + 2, len(tokens)):
                token_id = int(token_ids[future_position])
                word = normalized(tokens[future_position])
                if token_id in prefix_ids or len(word) < 2 or word in STOP:
                    continue
                future_by_id.setdefault(token_id, future_position)
            for token_id, future_position in future_by_id.items():
                layer_hits = []
                logit_hits = []
                for layer in WORKSPACE_LAYERS:
                    hit = (jlens[layer, source_position] == token_id).nonzero()
                    if len(hit):
                        layer_hits.append((layer, int(hit[0, 0]) + 1))
                    control = (logit[layer, source_position] == token_id).nonzero()
                    if len(control):
                        logit_hits.append((layer, int(control[0, 0]) + 1))
                if not layer_hits:
                    continue
                candidates.append({
                    "set": row["set"], "item_index": row["item_index"],
                    "name": row["item"]["name"], "prompt": row["item"]["prompt"],
                    "source_position": source_position, "source_token": tokens[source_position],
                    "future_position": future_position, "future_token": tokens[future_position],
                    "lead_tokens": future_position - source_position,
                    "jlens_best_rank": min(rank for _, rank in layer_hits),
                    "jlens_layers": layer_hits,
                    "logitlens_best_rank": min((rank for _, rank in logit_hits), default=None),
                    "logitlens_layers": logit_hits,
                })
    candidates.sort(key=lambda row: (row["logitlens_best_rank"] is not None,
                                     row["jlens_best_rank"], -len(row["jlens_layers"]),
                                     -row["lead_tokens"]))
    output = OUT / "dg_original_future_exact.json"
    output.write_text(json.dumps({"protocol": "exact later token appears in earlier-position top-20",
                                  "workspace_layers": list(WORKSPACE_LAYERS),
                                  "n_candidates": len(candidates),
                                  "candidates": candidates}, indent=1))
    print(output)
    print(json.dumps(candidates[:20], indent=1))


def stage_control() -> None:
    """Causally replace only the future token and measure the earlier original-lens readout."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("rcp", REPO / "concept_probes/run_concept_probes.py")
    rcp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rcp)
    model, tok = rcp.load_model("diffusiongemma", device_map={"": 0})
    layers = model.model.decoder.layers
    device = model.device
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.pad_token_id
    lens = merged_lens()
    source_layers = sorted(lens.jacobians)
    J = {layer: lens.jacobians[layer].to(device=device, dtype=torch.float32)
         for layer in source_layers}
    norm, head = model.model.decoder.norm, model.lm_head
    softcap = float(getattr(model.config.get_text_config(), "final_logit_softcapping", 0) or 0)

    def unembed(h):
        logits = head(norm(h.to(head.weight.dtype))).float()
        return softcap * torch.tanh(logits / softcap) if softcap else logits

    cap = {}

    def hook(layer):
        def save(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            cap[layer] = value
        return save

    handles = [layers[layer].register_forward_hook(hook(layer)) for layer in source_layers]

    @torch.no_grad()
    def probe(token_ids: torch.Tensor, source_position: int, target_id: int, foil_id: int) -> dict:
        cap.clear()
        decoder = token_ids.to(device).unsqueeze(0)
        encoder = torch.full((1, 1), bos, dtype=torch.long, device=device)
        model(input_ids=encoder, attention_mask=torch.ones_like(encoder, dtype=torch.bool),
              decoder_input_ids=decoder,
              decoder_position_ids=torch.arange(1, 1 + len(token_ids),
                                                device=device).unsqueeze(0))
        result = {"jlens": [], "logitlens": []}
        for layer in source_layers:
            h = cap[layer][0, source_position].float()
            for key, transported in (("jlens", J[layer] @ h), ("logitlens", h)):
                logits = unembed(transported)
                probs = torch.softmax(logits, dim=-1)
                top = probs.topk(20)
                target_logit, foil_logit = logits[target_id], logits[foil_id]
                result[key].append({
                    "layer": layer,
                    "target_rank": int((logits > target_logit).sum()) + 1,
                    "foil_rank": int((logits > foil_logit).sum()) + 1,
                    "target_probability": float(probs[target_id]),
                    "foil_probability": float(probs[foil_id]),
                    "target_minus_foil_logit": float(target_logit - foil_logit),
                    "top_tokens": [tok.decode([int(token_id)], skip_special_tokens=False)
                                   for token_id in top.indices],
                    "top_probabilities": [float(probability) for probability in top.values],
                })
        return result

    rows = []
    try:
        for set_name, item_i, source_position, future_position, target, foil in tqdm(
                CONTROL_CASES, desc="future-token counterfactuals"):
            original_capture = torch.load(capture_path(set_name, item_i), map_location="cpu",
                                          weights_only=False)
            token_ids = original_capture["token_ids"].long()
            assert original_capture["tokens"][future_position] == target
            target_tokens = tok(target, add_special_tokens=False)["input_ids"]
            foil_tokens = tok(foil, add_special_tokens=False)["input_ids"]
            assert target_tokens == [int(token_ids[future_position])] and len(foil_tokens) == 1
            target_id, foil_id = target_tokens[0], foil_tokens[0]
            counterfactual_ids = token_ids.clone()
            counterfactual_ids[future_position] = foil_id
            original = probe(token_ids, source_position, target_id, foil_id)
            counterfactual = probe(counterfactual_ids, source_position, target_id, foil_id)
            workspace = list(WORKSPACE_LAYERS)
            result = {
                "set": set_name, "item_index": item_i,
                "name": original_capture["item"]["name"],
                "prompt": original_capture["item"]["prompt"],
                "counterfactual_prompt": tok.decode(counterfactual_ids.tolist()),
                "source_position": source_position,
                "source_token": original_capture["tokens"][source_position],
                "future_position": future_position,
                "target_token": target, "foil_token": foil,
                "original": original, "counterfactual": counterfactual,
                "jlens_workspace_logodds_shift": float(sum(
                    original["jlens"][layer]["target_minus_foil_logit"] -
                    counterfactual["jlens"][layer]["target_minus_foil_logit"]
                    for layer in workspace) / len(workspace)),
                "logitlens_workspace_logodds_shift": float(sum(
                    original["logitlens"][layer]["target_minus_foil_logit"] -
                    counterfactual["logitlens"][layer]["target_minus_foil_logit"]
                    for layer in workspace) / len(workspace)),
                "jlens_original_best_target_rank": min(
                    original["jlens"][layer]["target_rank"] for layer in workspace),
                "jlens_counterfactual_best_foil_rank": min(
                    counterfactual["jlens"][layer]["foil_rank"] for layer in workspace),
            }
            rows.append(result)
            print(f"[future/control] {set_name}/{item_i} {target!r}->{foil!r} at r="
                  f"{future_position}, read q={source_position}: J shift="
                  f"{result['jlens_workspace_logodds_shift']:+.3f}, logit shift="
                  f"{result['logitlens_workspace_logodds_shift']:+.3f}")
    finally:
        for handle in handles:
            handle.remove()
    output = OUT / "dg_original_future_controls.json"
    output.write_text(json.dumps({
        "protocol": "one-token future counterfactual; prefix, length, and other suffix tokens fixed",
        "lens": "original dgb_shared, fit on 64 WikiText prompts",
        "workspace_layers": list(WORKSPACE_LAYERS),
        "rows": rows,
    }, indent=1))
    print(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("capture", "analyze", "control"))
    args = parser.parse_args()
    {"capture": stage_capture, "analyze": stage_analyze, "control": stage_control}[args.stage]()


if __name__ == "__main__":
    main()
