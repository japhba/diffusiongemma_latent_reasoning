"""Shared pieces for the constraint trap-escape study: instance selection + graded margins.
margin(p, text) -> int violation count (0 = constraint satisfied), None = not yet parseable."""
from pathlib import Path
import re, sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "constrained"))
from battery import PROBLEMS, check, toks, letters, parse_count, NUMWORDS  # noqa: E402

# instance_idx chosen where prior DG accuracy is mid-range (traps + escapes both occur);
# tautogram/lipogram = local-coupling controls; ends_with = easy anchor-planning control.
SELECT = {
    "self_count_words": [0, 3, 6, 7],
    "wordcount": [1, 5, 6],
    "acrostic_word": [0, 3, 5, 6],
    "snowball": [0, 1, 2],
    "piem": [0, 1],
    "abc_initials": [1, 2, 3],
    "palindrome_words": [2, 3],
    "self_count_word_occ": [0, 4, 5],
    "tautogram": [1, 3, 4],
    "lipogram": [2, 3],
    "ends_with": [0],
}
CHOSEN = [p for p in PROBLEMS if p["instance_idx"] in SELECT.get(p["ttype"], [])
          and p["ttype"] in SELECT]

_WORDNUM = {w: n for w, n in NUMWORDS.items()}

def _stated_number(text):
    for w in toks(text):
        wl = w.lower().replace("-", "")
        if wl in _WORDNUM:
            return _WORDNUM[wl]
    return None

def _initials_margin(text, letters_seq):
    ws = toks(text)
    if len(ws) < len(letters_seq):
        return len(letters_seq) - len(ws) + sum(
            1 for w, L in zip(ws, letters_seq) if not w.lower().startswith(L.lower()))
    return sum(1 for w, L in zip(ws[:len(letters_seq)], letters_seq)
               if not w.lower().startswith(L.lower()))

def margin(p, text):
    tt, q, lab = p["ttype"], p["q"], p["label"]
    ws = toks(text)
    if tt == "wordcount":
        n = int(re.search(r"EXACTLY (\d+)", q).group(1))
        return abs(len(ws) - n)
    if tt == "self_count_words":
        st = _stated_number(text)
        return None if st is None else abs(st - len(ws))
    if tt == "self_count_word_occ":
        word = re.search(r"the word '([^']+)'", q).group(1)
        st = _stated_number(text)
        occ = sum(1 for w in ws if w.lower() == word.lower())
        return None if st is None else abs(st - occ)
    if tt in ("acrostic_word", "abc_initials"):
        seq = re.findall(r"letters ([A-Z](?:, [A-Z])+)", q)[0].split(", ")
        return _initials_margin(text, seq)
    if tt == "piem":
        lens = [int(x) for x in re.findall(r"lengths ([\d, ]+) letters", q)[0].split(",")]
        if len(ws) < len(lens):
            return len(lens) - len(ws) + sum(1 for w, L in zip(ws, lens) if len([c for c in w if c.isalpha()]) != L)
        return sum(1 for w, L in zip(ws[:len(lens)], lens) if len([c for c in w if c.isalpha()]) != L)
    if tt == "snowball":
        n = int(re.search(r"up to a (\d+)-letter", q).group(1))
        if len(ws) < n:
            return n - len(ws) + sum(1 for i, w in enumerate(ws[:n]) if len([c for c in w if c.isalpha()]) != i + 1)
        return sum(1 for i, w in enumerate(ws[:n]) if len([c for c in w if c.isalpha()]) != i + 1)
    if tt == "tautogram":
        L = re.search(r"letter '(\w)'", q).group(1)
        return sum(1 for w in ws if not w.lower().startswith(L.lower())) + max(0, 6 - len(ws))
    if tt == "lipogram":
        L = re.search(r"letter '(\w)'", q).group(1)
        return sum(1 for c in letters(text) if c == L.lower()) + max(0, 10 - len(ws))
    if tt == "palindrome_words":
        n = int(re.search(r"at least (\d+) words", q).group(1))
        low = [w.lower() for w in ws]
        mm = sum(1 for i in range(len(low) // 2) if low[i] != low[-1 - i])
        return mm + max(0, n - len(ws))
    if tt == "ends_with":
        word = re.search(r"exactly '(\w+)'", q).group(1)
        n = int(re.search(r"at least (\d+) words", q).group(1))
        return (0 if ws and ws[-1].lower() == word.lower() else 1) + max(0, n - len(ws))
    raise ValueError(tt)
