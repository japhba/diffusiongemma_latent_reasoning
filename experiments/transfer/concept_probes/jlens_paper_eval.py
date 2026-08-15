"""Canonical item and readout-position rules for the six Jacobian-Lens paper evals."""
from __future__ import annotations

import json
from pathlib import Path

PAPER_SETS = ("multihop", "multilingual", "order-ops", "poetry", "typo", "association")
_ONES = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
_TEENS = ("ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
          "seventeen", "eighteen", "nineteen")
_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
_OPS = {
    "addition": ("addition", "+", "add", "plus", "sum"),
    "division": ("division", "/", "divide", "divided", "quotient"),
    "mod": ("mod", "%", "modulo", "remainder"),
    "multiplication": ("multiplication", "*", "×", "multiply", "times", "product"),
    "squared": ("squared", "square", "²", "^2"),
    "subtraction": ("subtraction", "-", "minus", "subtract", "difference"),
}


def load_paper_sets(eval_dir: Path) -> dict[str, list[dict]]:
    return {name: json.loads((eval_dir / f"lens-eval-{name}.json").read_text())["items"]
            for name in PAPER_SETS}


def _number_word(n: int) -> str:
    assert 0 <= n < 100, n
    if n < 10:
        return _ONES[n]
    if n < 20:
        return _TEENS[n - 10]
    return _TENS[n // 10] if n % 10 == 0 else f"{_TENS[n // 10]}-{_ONES[n % 10]}"


def paper_reference_forms(set_name: str, reference: str) -> tuple[str, ...]:
    """Return the paper concept's accepted surface forms.

    The paper chose single-token intermediates. It explicitly accepts digit/word forms and operation
    synonyms for order-of-operations; numeric intermediates in the expanded multihop release get the
    same digit/word treatment so that a model tokenizer which splits digits is not scored on one
    meaningless digit fragment.
    """
    if set_name == "order-ops" and reference in _OPS:
        return _OPS[reference]
    if reference.isdigit():
        word = _number_word(int(reference))
        return tuple(dict.fromkeys((reference, word, word.replace("-", " "))))
    return (reference,)


def paper_readout(tokenizer, set_name: str, prompt: str) -> tuple[list[int], int, list[tuple[int, int]]]:
    """Tokenize an upstream prompt and return its paper-defined readout token.

    Five sets read the final prompt token (the token immediately before the held-out target for
    multihop/multilingual/order-ops). Poetry reads the token containing the final newline, i.e. the
    boundary after the cue line and before the unfinished rhyming line.
    """
    encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(encoded["input_ids"])
    offsets = [tuple(x) for x in encoded["offset_mapping"]]
    assert ids and len(ids) == len(offsets)
    if set_name != "poetry":
        assert set_name in PAPER_SETS, set_name
        return ids, len(ids) - 1, offsets
    newline = prompt.rfind("\n")
    assert newline >= 0, prompt
    hits = [i for i, (lo, hi) in enumerate(offsets) if lo <= newline < hi]
    assert len(hits) == 1, (newline, hits, offsets)
    return ids, hits[0], offsets
