"""Computational-battery datasets for the Part-3 commitment analysis (WildChat-like span
judging on the comparative-advantage tasks from the thinkfast/engels/constrained studies).

5 datasets x 16 rollouts, all answer-first/direct framing (the regime where DG's commitment
behaviour matters), single canvas C=64, T=48 (same step grid as the WildChat analysis):
  mult_3x3    - 3x3-digit product; DG steps load-bearing, DG BEATS Gemma 1.00 vs 0.88
  mult_4x4    - 4x4-digit product; DG 0.42 vs 0.19
  sum_squares - "perfect squares strictly between a,b" (Engels answer-first win family)
  n_hop       - depth-5 fictional lookup chain; DG steps INERT (the contrast case)
  univocalic  - only-one-vowel sentence; constrained-text win (large solution space)
"""
from pathlib import Path
import random
import re
import sys

_EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_EXP))  # experiments/ root -> thinkfast.*, constrained.* resolve
sys.path.insert(0, str(_EXP / "thinkfast"))  # money_tasks does `from battery import …`
from thinkfast.battery import first_choice, gen_n_hop, gen_square_count, last_int  # noqa: E402
from thinkfast.money_tasks import gen_mult  # noqa: E402
from constrained.battery import mk_lipogram, mk_self_count_words, mk_univocalic, mk_wordlen_seq  # noqa: E402

N_PER_DS = 16
ANSFIRST = ("This is a single-forward-pass evaluation. Respond immediately with your answer, "
            "before any reasoning. Respond in exactly this format: The answer is: <integer>")

BLURBS = {
    "mult_3x3": "3-digit × 3-digit product. DG's denoising steps are LOAD-BEARING here (T1 0.13 → T64 1.00) and DG beats Gemma-4 answer-first (1.00 vs 0.88) — refinable, parallelizable compute.",
    "mult_4x4": "4-digit × 4-digit product — the harder version (DG 0.42 vs Gemma 0.19). Same load-bearing-steps regime, near DG's capability edge.",
    "sum_squares": "Count perfect squares strictly between two bounds (the Engels answer-first win family: bounded multi-step enumeration where AR's up-front commitment fails).",
    "n_hop": "Depth-5 fictional lookup chain (MCQ). DG's steps are INERT here — data-dependent serial retrieval that denoising cannot unroll. The contrast case.",
    "univocalic": "Write a ≥7-word sentence whose only vowel is the given one. The constrained-text win (DG 0.94 vs Gemma 0.59): large solution space where greedy AR dead-ends.",
    "square_count": "The thinkfast paper task (DG 0.78 vs Gemma 0.02): count squares in a range, ANSWER FIRST THEN REASONING - the format that lets DG retro-correct its committed answer in-canvas. C=128 for reasoning room.",
    "lipogram": "≥10-word sentence avoiding one letter entirely (DG +0.17 in direct framing).",
    "piem": "Sentence whose first six word lengths encode the digits of pi or e (DG +0.19).",
    "self_count_words": "A sentence that truthfully states its own word count - self-referential fixed point (free-length instance DG +0.50).",
    "chain_sum_digits": "Add four numbers, then sum the digits of the total (Engels transform-chain win family, DG 0.88 vs 0.00).",
}

CANVAS = {"square_count": 128}  # reasoning room; all others C=64


def _sum_squares(rng):
    a = rng.randint(300, 900)
    b = a + rng.randint(250, 500)
    import math
    n = sum(1 for k in range(int(math.isqrt(a)) + 1, int(math.isqrt(b)) + 2) if a < k * k < b)
    q = f"How many perfect square numbers are there strictly between {a} and {b}?"
    return dict(user=q, fmt=ANSFIRST, answer=str(n),
                check=lambda t, n=n: last_int((t or "").replace(",", "")) == n)


def _chain_sum(rng):
    ns = [rng.randint(11, 49) for _ in range(4)]
    tot = sum(ns)
    ans = sum(int(d) for d in str(tot))
    q = (f"Add these four numbers: {ns[0]}, {ns[1]}, {ns[2]}, {ns[3]}. Then take the resulting total "
         f"and add up its digits. What is that final digit sum?")
    return dict(user=q, fmt=ANSFIRST, answer=str(ans),
                check=lambda t, a=ans: last_int((t or "").replace(",", "")) == a)


def build():
    rows = []
    # NOTE: append-only - ds_i feeds the instance seeds, so reordering/inserting would silently
    # change the already-captured datasets' instances.
    for ds_i, ds in enumerate(["mult_3x3", "mult_4x4", "sum_squares", "n_hop", "univocalic",
                               "square_count", "lipogram", "piem", "self_count_words", "chain_sum_digits"]):
        for i in range(N_PER_DS):
            rng = random.Random(7000 + 1000 * ds_i + i)
            if ds.startswith("mult_"):
                p = gen_mult(int(ds[-1]), rng)
                prompt = f"{p['system']}\n\n{p['user']}\n\n{p['fmt']}"
                answer, check = p["answer"], p["check"]
            elif ds == "sum_squares":
                p = _sum_squares(rng)
                prompt = f"{p['user']}\n\n{p['fmt']}"
                answer, check = p["answer"], p["check"]
            elif ds == "n_hop":
                p = gen_n_hop(5, rng)
                prompt = f"{p['system']}\n\n{p['user']}\n\n{p['fmt']}"
                answer, check = p["answer"], p["check"]
            elif ds == "univocalic":  # 4 vowels x 4 seeds
                v = ["a", "o", "e", "i"][i % 4]
                prompt = (f"Write one meaningful English sentence of at least 7 words in which the ONLY "
                          f"vowel that appears is '{v}' (none of the other vowels anywhere). "
                          f"Respond with ONLY the sentence.")
                answer, check = f"(any valid; vowel={v})", mk_univocalic(v)
            elif ds == "square_count":
                p = gen_square_count(2, rng)
                prompt = f"{p['system']}\n\n{p['user']}\n\n{p['fmt']}"
                answer, check = p["answer"], p["check"]
            elif ds == "lipogram":
                L = ["e", "a", "o", "t", "s"][i % 5]
                prompt = (f"Write one meaningful English sentence of at least 10 words that does NOT "
                          f"contain the letter '{L}' anywhere. Respond with ONLY the sentence.")
                answer, check = f"(any valid; no '{L}')", mk_lipogram(L)
            elif ds == "piem":
                name, seq = [("pi", (3, 1, 4, 1, 5, 9)), ("e", (2, 7, 1, 8, 2, 8))][i % 2]
                prompt = (f"Write a sentence whose first six words have lengths "
                          f"{', '.join(map(str, seq))} letters respectively (the first digits of {name}). "
                          f"Respond with ONLY the sentence.")
                answer, check = f"(word lengths {','.join(map(str, seq))})", mk_wordlen_seq(seq)
            elif ds == "self_count_words":
                prompt = ("Write a single sentence that correctly states, in words, how many words it "
                          "contains (count every word). Respond with ONLY the sentence.")
                answer, check = "(states own word count)", mk_self_count_words(4)
            else:  # chain_sum_digits
                p = _chain_sum(rng)
                prompt = f"{p['user']}\n\n{p['fmt']}"
                answer, check = p["answer"], p["check"]
            rows.append(dict(ds=ds, i=i, rid=f"{ds}_{i:02d}", prompt=prompt, answer=answer, check=check))
    return rows


if __name__ == "__main__":
    rows = build()
    import collections
    print(collections.Counter(r["ds"] for r in rows))
    for r in rows[::16]:
        print(f"--- {r['rid']} (ans {r['answer']}):\n{r['prompt'][:200]}\n")
