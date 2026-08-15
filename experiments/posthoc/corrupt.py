"""CoT corruption operators — the 'noise' knob for d(answer)/d(noise).

Given a chain-of-thought string and a strength rho in [0,1], return a corrupted version.
We keep token COUNT roughly stable so 'noise' varies content, not length.

  word_rand : replace a rho-fraction of whitespace tokens with random pool words (fluent-but-wrong).
  shuffle   : randomly permute a rho-fraction of token POSITIONS (destroys order/logic, keeps bag).
  drop      : delete a rho-fraction of tokens (missing reasoning).

word_rand is the primary clean knob: rho=0 -> original, rho=1 -> every word replaced.
Determinism: caller passes a seed so each (rho, corr_seed) is reproducible.
"""
import random

# generic, topic-neutral pool (avoids accidentally forming coherent alternative arithmetic)
POOL = ("the of and to in a is that it for on with as was at by an be this from or had "
        "but not are were they which you all can her has him his one our out day get has "
        "river table window forest yellow simple gentle morning paper silver garden cloud "
        "matter window basket purple letter summer corner pocket bottle mirror candle "
        "between number system because before through during without around almost across").split()


def corrupt(cot: str, rho: float, mode: str = "word_rand", seed: int = 0) -> str:
    rng = random.Random(seed)
    toks = cot.split()
    n = len(toks)
    if n == 0 or rho <= 0:
        return cot
    if mode == "word_rand":
        k = round(rho * n)
        idx = rng.sample(range(n), k)
        for i in idx:
            toks[i] = rng.choice(POOL)
        return " ".join(toks)
    if mode == "shuffle":
        k = max(2, round(rho * n))
        idx = rng.sample(range(n), k)
        vals = [toks[i] for i in idx]; rng.shuffle(vals)
        for i, v in zip(idx, vals):
            toks[i] = v
        return " ".join(toks)
    if mode == "drop":
        k = round(rho * n)
        idx = set(rng.sample(range(n), k))
        return " ".join(t for i, t in enumerate(toks) if i not in idx)
    raise ValueError(mode)
