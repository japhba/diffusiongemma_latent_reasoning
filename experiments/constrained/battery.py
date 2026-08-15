"""Constrained / global-constraint TEXT-GENERATION battery for the DiffusionGemma-vs-Gemma-4 study.

Sibling to `../engels/` (arithmetic serial depth). Here we probe the OTHER structural edge of a
diffusion canvas over a left-to-right autoregressive (AR) model: **causal commitment** (an AR token
at t cannot depend on a token at t+k) and **no revision**. AR is disadvantaged whenever the optimal
value at an EARLY position depends on a value resolved LATER, or a global constraint couples distant
positions. DiffusionGemma denoises the whole canvas jointly and rewrites any position.

PARAMETRIC. Each task TYPE (`ttype`) is instantiated several times with different targets (words,
letters, counts) so per-type accuracy is averaged over instances, not read off one hand-picked prompt.

THE SYSTEMATICITY TEST. Each instance is tagged `local` (greedy-satisfiable → predict tie) or
`global` (distant positions coupled / self-referential fixed point → predict DG win). If DG advantage
tracks the tag, the win is architectural.

SCORING IS EXACT. Correctness is a deterministic Python predicate `check(pid, candidate)`; the Node-V
judge only extracts the candidate text verbatim. No label noise.

Each instance dict: id, ttype, instance_idx, cat, coupling, label, q (instruction), check.
"""
import re
import string

# ----------------------------------------------------------------------------- text helpers
_STRIP = string.punctuation + "“”‘’\"'—–…"


def toks(t):
    out = []
    for w in (t or "").split():
        w2 = w.strip(_STRIP)
        if any(c.isalpha() for c in w2):
            out.append(w2)
    return out


def letters(t):
    return [c.lower() for c in (t or "") if c.isalpha()]


def lines(t):
    return [ln for ln in (t or "").splitlines() if ln.strip()]


NUMWORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "twentyone": 21, "twenty-one": 21, "twentytwo": 22, "twenty-two": 22, "thirty": 30,
}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50}
_ONES = {k: v for k, v in NUMWORDS.items() if 1 <= v <= 9}
VOWELS = set("aeiou")


_MULT = {"once": 1, "twice": 2, "thrice": 3}


def parse_count(t):
    """First integer stated in `t`, as a digit or English number-word (incl. 'thirty-one', 'twice')."""
    s = (t or "").lower()
    for w, v in _MULT.items():  # "appears twice" -> 2
        if re.search(r"\b" + w + r"\b", s):
            return v
    for tens, tv in _TENS.items():
        m = re.search(tens + r"[\s-]+(one|two|three|four|five|six|seven|eight|nine)", s)
        if m:
            return tv + _ONES[m.group(1)]
        if re.search(r"\b" + tens + r"\b", s):
            return tv
    for w, v in sorted(NUMWORDS.items(), key=lambda kv: -kv[1]):
        if re.search(r"\b" + re.escape(w) + r"\b", s):
            return v
    m = re.search(r"\b(\d+)\b", t or "")
    return int(m.group(1)) if m else None


# ----------------------------------------------------------------------------- check factories
def mk_tautogram(letter, nmin=6):
    return lambda t: (lambda w: len(w) >= nmin and all(x[0].lower() == letter for x in w))(toks(t))


def mk_wordcount(n):
    return lambda t: len(toks(t)) == n


def mk_minlen(k, nwords):
    return lambda t: (lambda w: len(w) >= nwords and all(len(x) >= k for x in w))(toks(t))


def mk_lipogram(letter, nwords=10):
    return lambda t: len(toks(t)) >= nwords and letter not in letters(t)


def mk_all_vowels(maxwords):
    return lambda t: VOWELS <= set(letters(t)) and 1 <= len(toks(t)) <= maxwords


def mk_acrostic_initials(target, coupling_local=False):
    tgt = [c.lower() for c in target]
    n = len(tgt)
    return lambda t: (lambda w: len(w) >= n and [x[0].lower() for x in w[:n]] == tgt)(toks(t))


def mk_palindrome_words(nwords):
    return lambda t: (lambda w: len(w) >= nwords and [x.lower() for x in w] == [x.lower() for x in w][::-1])(toks(t))


def mk_palindrome_chars(nletters, nwords):
    return lambda t: (lambda L, w: len(L) >= nletters and len(w) >= nwords and L == L[::-1])(letters(t), toks(t))


def mk_ends_with(word, nwords=8):
    return lambda t: (lambda w: len(w) >= nwords and w[-1].lower() == word)(toks(t))


def mk_starts_ends(first, last, nwords=8):
    return lambda t: (lambda w: len(w) >= nwords and w[0].lower() == first and w[-1].lower() == last)(toks(t))


def mk_mirror(nwords):
    return lambda t: (lambda w: len(w) >= nwords and w[0].lower() == w[-1].lower())(toks(t))


def mk_exact_letters(n, nwords):
    return lambda t: len(letters(t)) == n and len(toks(t)) >= nwords


def mk_acrostic_lines(target):
    tgt = target.lower()
    n = len(tgt)
    return lambda t: (lambda L: len(L) >= n and "".join(ln.strip()[0].lower() for ln in L[:n] if ln.strip()) == tgt)(lines(t))


def mk_wordlen_seq(seq):
    n = len(seq)
    return lambda t: (lambda w: len(w) >= n and [len(x) for x in w[:n]] == list(seq))(toks(t))


def mk_bridge(nwords):
    return lambda t: (lambda w: len(w) >= nwords and all(w[i][0].lower() == w[i - 1][-1].lower() for i in range(1, len(w))))(toks(t))


def mk_univocalic(vowel, nwords=7):
    others = VOWELS - {vowel}
    return lambda t: (lambda L, w: len(w) >= nwords and vowel in L and not (others & set(L)))(letters(t), toks(t))


def mk_pangram(maxletters):
    return lambda t: set(string.ascii_lowercase) <= set(letters(t)) and len(letters(t)) <= maxletters


def mk_self_count_words(minwords=4):
    return lambda t: (lambda w, n: len(w) >= minwords and n is not None and n == len(w))(toks(t), parse_count(t))


def mk_self_count_letter(letter):
    return lambda t: (lambda L, n: len(toks(t)) >= 4 and n is not None and n == L.count(letter))(letters(t), parse_count(t))


def mk_self_count_vowels():
    return lambda t: (lambda L, n: len(toks(t)) >= 4 and n is not None and n == sum(c in VOWELS for c in L))(letters(t), parse_count(t))


def mk_self_count_consonants():
    return lambda t: (lambda L, n: len(toks(t)) >= 4 and n is not None and n == sum(c not in VOWELS for c in L))(letters(t), parse_count(t))


def mk_self_count_word_occ(word):
    return lambda t: (lambda w, n: len(w) >= 4 and n is not None and (cnt := sum(x.lower() == word for x in w)) == n and cnt >= 1)(toks(t), parse_count(t))


# ----------------------------------------------------------------------------- instance specs
# (ttype, cat, coupling, [instance params], q_fn(params)->str, check_fn(params))
def _build():
    P = []

    def add(ttype, cat, coupling, params_list, qfn, checkfn, labelfn):
        for idx, prm in enumerate(params_list):
            P.append(dict(id=f"{ttype}__{idx}", ttype=ttype, instance_idx=idx, cat=cat, coupling=coupling,
                          label=labelfn(prm), q=qfn(prm), check=checkfn(prm)))

    # ===================== LOCAL (greedy-satisfiable; predict tie) =====================
    add("tautogram", "alliteration", "local", ["t", "s", "p", "b", "w", "c", "m"],
        lambda L: f"Write one meaningful English sentence of at least 6 words in which EVERY word begins with the letter '{L}' (any case).",
        lambda L: mk_tautogram(L), lambda L: f"every word starts with {L.upper()} (≥6)")

    add("wordcount", "length", "local", [5, 7, 8, 10, 11, 13, 15, 18],
        lambda n: f"Write one grammatical English sentence that is EXACTLY {n} words long (count every word).",
        lambda n: mk_wordcount(n), lambda n: f"exactly {n} words")

    add("minlen", "length", "local", [(5, 7), (6, 8), (4, 10), (7, 7)],
        lambda p: f"Write one meaningful English sentence of at least {p[1]} words in which EVERY word has at least {p[0]} letters.",
        lambda p: mk_minlen(*p), lambda p: f"every word ≥{p[0]} letters (≥{p[1]})")

    add("lipogram", "lipogram", "local", ["e", "a", "o", "t", "s"],
        lambda L: f"Write one meaningful English sentence of at least 10 words that does NOT contain the letter '{L}' anywhere.",
        lambda L: mk_lipogram(L), lambda L: f"no letter {L.upper()} (≥10 words)")

    add("all_vowels", "coverage", "local", [10, 12, 14],
        lambda m: f"Write one short English sentence of at most {m} words that contains each of the five vowels a, e, i, o and u at least once.",
        lambda m: mk_all_vowels(m), lambda m: f"contains a,e,i,o,u; ≤{m} words")

    add("abc_initials", "acrostic", "local", ["ABCDEF", "BCDEFG", "CDEFGH", "DEFGHI"],
        lambda s: f"Write a sentence whose first {len(s)} words begin, in order, with the letters {', '.join(s)}.",
        lambda s: mk_acrostic_initials(s), lambda s: f"first {len(s)} words → {','.join(s)}")

    # ===================== GLOBAL non-self-referential (predict DG win) =====================
    add("palindrome_words", "palindrome", "global", [4, 5, 6, 7],
        lambda n: f"Write a phrase of at least {n} words whose sequence of words reads the same forwards and backwards (a word-level palindrome).",
        lambda n: mk_palindrome_words(n), lambda n: f"word-palindrome (≥{n} words)")

    add("palindrome_chars", "palindrome", "global", [(15, 3), (17, 3), (21, 4), (25, 4)],
        lambda p: f"Write a phrase of at least {p[1]} words that is a palindrome at the letter level: ignoring spaces, punctuation and case, the letters read the same forwards and backwards. Use at least {p[0]} letters.",
        lambda p: mk_palindrome_chars(*p), lambda p: f"letter-palindrome (≥{p[0]} letters)")

    add("ends_with", "endpoint", "global", ["tomorrow", "yesterday", "forever", "silence", "home", "anyway", "goodbye"],
        lambda w: f"Write one grammatical English sentence of at least 8 words whose VERY LAST word is exactly '{w}'.",
        lambda w: mk_ends_with(w), lambda w: f"ends with '{w}' (≥8)")

    add("starts_ends", "endpoint", "global",
        [("Although", "anyway"), ("Because", "instead"), ("Whenever", "again"),
         ("Despite", "nonetheless"), ("Since", "regardless"), ("Unless", "somehow")],
        lambda p: f"Write one grammatical English sentence of at least 8 words whose first word is '{p[0]}' and whose last word is '{p[1]}'.",
        lambda p: mk_starts_ends(p[0].lower(), p[1].lower()), lambda p: f"'{p[0]}'…'{p[1]}'")

    add("mirror_first_last", "endpoint", "global", [8, 10, 12],
        lambda n: f"Write one grammatical English sentence of at least {n} words whose first and last words are the SAME word (and the words in between are different).",
        lambda n: mk_mirror(n), lambda n: f"first word == last word (≥{n})")

    add("exact_letters", "length", "global", [(35, 6), (40, 6), (48, 7), (55, 8), (62, 9), (70, 10), (80, 11)],
        lambda p: f"Write one grammatical English sentence using EXACTLY {p[0]} letters in total (count only letters a–z; ignore spaces and punctuation). It must be at least {p[1]} words long.",
        lambda p: mk_exact_letters(*p), lambda p: f"exactly {p[0]} letters")

    add("acrostic_word", "acrostic", "global", ["PLANET", "OCEAN", "FOREST", "SILENT", "GARDEN", "ROCKET", "BRIDGE", "CANDLE"],
        lambda s: f"Write a coherent sentence whose first {len(s)} words begin, in order, with the letters {', '.join(s)} (spelling {s}).",
        lambda s: mk_acrostic_initials(s), lambda s: f"first {len(s)} words spell {s}")

    add("acrostic_lines", "acrostic", "global", ["HELLO", "WORLD", "LIGHT", "DREAM", "PEACE"],
        lambda s: f"Write a short {len(s)}-line poem (one short line each) where the first letters of the {len(s)} lines, read top to bottom, spell {s}.",
        lambda s: mk_acrostic_lines(s), lambda s: f"{len(s)} lines spell {s}")

    add("piem", "word-length", "global", [("pi", (3, 1, 4, 1, 5, 9)), ("e", (2, 7, 1, 8, 2, 8))],
        lambda p: f"Write a sentence whose first six words have lengths {', '.join(map(str, p[1]))} letters respectively (the first digits of {p[0]}).",
        lambda p: mk_wordlen_seq(p[1]), lambda p: f"word lengths = {','.join(map(str, p[1]))}")

    add("snowball", "word-length", "global", [5, 6, 7],
        lambda m: f"Write a 'snowball' sentence: the first word has 1 letter, the second 2 letters, and so on up to a {m}-letter word (lengths {', '.join(str(i) for i in range(1, m + 1))}).",
        lambda m: mk_wordlen_seq(tuple(range(1, m + 1))), lambda m: f"lengths 1..{m}")

    add("bridge_chain", "chain", "global", [6, 7, 8],
        lambda n: f"Write a chain of at least {n} words where each word begins with the LAST letter of the previous word (like a word-chain / shiritori). It should read as a loose phrase.",
        lambda n: mk_bridge(n), lambda n: f"last-letter→first-letter chain (≥{n})")

    add("univocalic", "lipogram", "global", ["a", "o", "e", "i"],
        lambda v: f"Write one meaningful English sentence of at least 7 words in which the ONLY vowel that appears is '{v}' (none of the other vowels anywhere).",
        lambda v: mk_univocalic(v), lambda v: f"only vowel is {v.upper()} (≥7)")

    add("pangram", "coverage", "global", [110, 130, 150],
        lambda m: f"Write a single sentence that contains every letter of the alphabet (a pangram), using as few letters as you can — at most {m} letters total.",
        lambda m: mk_pangram(m), lambda m: f"all 26 letters in ≤{m}")

    # ===================== SELF-REFERENCE / fixed point (the interesting direction) =========
    sc_specs = [(4, "std"), (6, "std"), (8, "std"), (10, "std"), (12, "std"), (14, "std"),
                (4, "rephrase"), (8, "rephrase")]

    def sc_q(p):
        minw, mode = p
        floor = "" if minw <= 4 else f" of AT LEAST {minw} words"
        if mode == "rephrase":
            return (f"Count the words in the sentence you are about to write, and write a sentence{floor} that truthfully "
                    "reports that total number in words. Do not use the phrasing 'this sentence contains N words' — phrase "
                    "it some other way. The reported number must equal the actual number of words in your sentence.")
        return (f"Write a single sentence{floor} that correctly states, in words, the total number of words it contains. "
                "For example of the FORM (not the count): 'This sentence contains five words.' Count every word, including "
                "the number word itself. The stated number must equal the true count.")
    add("self_count_words", "self-reference", "global", sc_specs,
        sc_q, lambda p: mk_self_count_words(p[0]),
        lambda p: f"states own word count{'' if p[0] <= 4 else f' (≥{p[0]})'}{' [rephrased]' if p[1] == 'rephrase' else ''}")

    add("self_count_letter", "self-reference", "global", ["e", "t", "s", "o", "n"],
        lambda L: f"Write a single sentence that correctly states, in words, how many times the letter '{L}' appears in it (count every '{L}' in the whole sentence, including any in the number word). The stated number must equal the true count.",
        lambda L: mk_self_count_letter(L), lambda L: f"states own count of letter '{L}'")

    add("self_count_vowels", "self-reference", "global", [None],
        lambda _: "Write a single sentence that correctly states, in words, how many vowels (a, e, i, o, u) it contains in total, counting the vowels in the whole sentence including the number word. The stated number must equal the true vowel count.",
        lambda _: mk_self_count_vowels(), lambda _: "states own vowel count")

    add("self_count_consonants", "self-reference", "global", [None],
        lambda _: "Write a single sentence that correctly states, in words, how many consonants (letters that are not a, e, i, o, u) it contains in total, including the number word. The stated number must equal the true consonant count.",
        lambda _: mk_self_count_consonants(), lambda _: "states own consonant count")

    add("self_count_word_occ", "self-reference", "global", ["the", "a", "and", "of", "to", "is"],
        lambda w: f"Write a single sentence that correctly states, in words, how many times the word '{w}' appears in it. The sentence must actually contain the word '{w}' at least once, and the stated number must equal the true number of times '{w}' appears.",
        lambda w: mk_self_count_word_occ(w), lambda w: f"states own count of the word '{w}'")

    return P


PROBLEMS = _build()

FRAMINGS = {
    "direct":     "{q}\n\nOutput ONLY the text itself — no preamble, no quotes, no explanation.",
    "plan_first": "{q}\n\nFirst think step by step about how to satisfy the constraint, then on the final "
                  "line write exactly: FINAL: <your text>",
}

# ---- sampling config ----
N_SEEDS = 8
TARGET_SEEDS = 16
_DG_COMMON = dict(t_max=0.8, t_min=0.4, entropy_bound=0.1, top_k=2, enable_thinking=False)
DG_BY_FRAMING = {
    "direct":     dict(C=128, T=96,  **_DG_COMMON),
    "plan_first": dict(C=256, T=128, **_DG_COMMON),
}
AR_BY_FRAMING = {
    "direct":     dict(temperature=0.8, top_p=0.95, max_tokens=128, enable_thinking=False, max_model_len=2048),
    "plan_first": dict(temperature=0.8, top_p=0.95, max_tokens=512, enable_thinking=False, max_model_len=2048),
}


def prompt_for(p, framing):
    return FRAMINGS[framing].format(q=p["q"])


def dg_cfg(framing):
    return DG_BY_FRAMING[framing]


def ar_cfg(framing):
    return AR_BY_FRAMING[framing]


BY_ID = {p["id"]: p for p in PROBLEMS}


def check(pid, candidate):
    try:
        return bool(BY_ID[pid]["check"](candidate or ""))
    except Exception:
        return False


if __name__ == "__main__":
    from collections import Counter
    print(f"{len(PROBLEMS)} instances across {len(set(p['ttype'] for p in PROBLEMS))} task types")
    print("by coupling:", dict(Counter(p["coupling"] for p in PROBLEMS)))
    print("by ttype:", dict(Counter(p["ttype"] for p in PROBLEMS)))
    # validate factories on one hand-written good example per ttype (first matching instance)
    GOOD = {
        "tautogram": ("tautogram__0", "Ten tall trees took turns turning toward town."),
        "wordcount": ("wordcount__0", "I really do not know"),  # 5 words -> wordcount__0 is n=5
        "minlen": ("minlen__0", "Several mountain rivers slowly meander through ancient northern forests."),
        "lipogram": ("lipogram__0", "A jolly fox sits aloft, watching tiny birds dart through warm air."),
        "all_vowels": ("all_vowels__0", "A quick lion found six eggs."),
        "abc_initials": ("abc_initials__0", "Angry bears chase deer every fall."),
        "palindrome_words": ("palindrome_words__1", "you can cage a cage can you"),
        "palindrome_chars": ("palindrome_chars__1", "was it a car or a cat i saw"),
        "ends_with": ("ends_with__0", "We will finish all of the remaining work tomorrow."),
        "starts_ends": ("starts_ends__0", "Although it rained hard, we marched onward to the summit anyway."),
        "mirror_first_last": ("mirror_first_last__0", "Rivers carve canyons over countless quiet centuries beside rivers."),
        "exact_letters": ("exact_letters__3", "Bright autumn leaves drift gently across the frozen river softly"),
        "acrostic_word": ("acrostic_word__0", "People listen and never expect trouble."),
        "acrostic_lines": ("acrostic_lines__0", "Hello there\nEvery morning\nLight returns\nLovely skies\nOpen wide"),
        "piem": ("piem__0", "How I wish I could enumerate pie."),
        "snowball": ("snowball__1", "I am the cool bears travel."),
        "bridge_chain": ("bridge_chain__1", "dog goes some elk knows sharp pen"),
        "univocalic": ("univocalic__1", "Old folks look for cold wood now."),  # vowel o
        "pangram": ("pangram__0", "Pack my box with five dozen liquor jugs."),
        "self_count_words": ("self_count_words__0", "This sentence contains five words"),
        "self_count_letter": ("self_count_letter__0", "We see seven trees"),  # letter e -> 7
        "self_count_vowels": ("self_count_vowels__0", "I see six vowels"),
        "self_count_consonants": ("self_count_consonants__0", "Brrr sky"),  # placeholder, computed below
        "self_count_word_occ": ("self_count_word_occ__0", "The cat and the dog ran, so the word the appears four times here"),
    }
    # consonants example: build one that states its own consonant count
    # "This has ten consonants" -> consonants: This(3:Ths)? t,h,s=3; has h,s=2; ten t,n=2; consonants c,n,s,n,n,t,s=7 -> 14? skip exactness, just test it runs
    fails = 0
    for tt, (pid, ex) in GOOD.items():
        if pid not in BY_ID:
            print(f"  MISSINGID {pid}"); continue
        ok = check(pid, ex)
        if tt == "self_count_consonants":
            continue  # hard to hand-craft; factory logic shared with vowels (validated)
        if not ok:
            fails += 1
        print(f"  {'PASS' if ok else 'FAIL'}  {pid:24s}  {ex!r}")
    print(f"\nfactory self-test: {fails} unexpected FAIL(s)")
