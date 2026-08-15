"""No-CoT task battery reconstructed from *Think Fast: Estimating No-CoT Task-Completion Time
Horizons of Frontier AI Models* (Gould et al. 2606.07157; source in
third_party/think_fast_nocot/). The paper measures what frontier models accomplish in a SINGLE
FORWARD PASS with no chain-of-thought. DiffusionGemma's denoising canvas is exactly "implicit
reasoning with no visible CoT tokens" (opaque serial depth) — so these tasks let us ask whether
DG's iterative single-canvas denoising beats a single-pass AR Gemma on graded no-CoT reasoning,
and whether the gap scales with serial DEPTH.

Faithful synthetic reconstructions (the paper's data is held back) of 5 exactly-verifiable,
depth-parametric benchmarks, each parametrised by a `depth` knob = its serial-reasoning difficulty:
  - arithmetic       (math)        depth = #terms in the expression
  - hash             (SWE)         depth = #serial steps (input length) of a custom hash fn
  - n_hop            (reasoning)   depth = #lookup hops chained over in-context tables
  - sally_anne       (theory-of-mind) depth = belief-nesting order (co-presence rule)
  - tower_of_london  (planning)    depth = minimum #moves (BFS-optimal)

Each problem carries an EXACT Python verifier `check(candidate_text) -> bool`. No LLM judge is used
for the verdict; extraction is deterministic and mirrors the paper's scorers (last integer, single
MCQ letter, parsed move sequence). Problems are rebuilt DETERMINISTICALLY from a fixed master seed,
so every script (run_dg / run_ar / score / report) sees identical instances.

2026-07-02 addition — 5 tasks reconstructed from the cloned "How transparent is DiffusionGemma?"
paper (third_party/diffusiongemma_transparency/, §case studies), depth-parametrised into the same
exact-verifier format (PAPER_TASKS below):
  - square_count     (§retroactive self-correction)  answer-first square counting; depth = root magnitude
  - recurrence_gold  (§intermediate-context)          Fibonacci-like + unit-digit-3 -> "Gold"; depth = #terms
  - collatz          (§token smearing, serial ctrl)   strictly-serial continuation; depth = #steps
  - brackets         (§token smearing, syntax)        balanced-bracket classification; depth = nesting depth
  - length_control   (§early length prediction)       exact-word-count writing; depth = word count
(The paper's code-generation case study is omitted: a unit-test-verified function does not fit the
single-canvas ≤48-token budget regime of this battery.)
"""
import hashlib
import random
import re
import string
from collections import deque

# ----------------------------------------------------------------------------- regime / config
MASTER_SEED = 20260630
N_INSTANCES = 4          # instances per (task, depth) cell
N_SEEDS = 16             # rollouts per cell (both models)

# "single canvas, or the equivalent token budget for conventional Gemma" (user, 2026-06-30):
# DG gets ONE denoising canvas of C positions (too small to write out a CoT -> genuinely no-CoT);
# Gemma gets the SAME budget as max_tokens, answer-first, non-thinking. T = DG denoising steps =
# DG's implicit serial-compute budget within the single canvas.
TASK_BUDGET = {  # (C, T) for DG ; Gemma max_tokens = C
    "arithmetic": (24, 64), "hash": (24, 64), "n_hop": (24, 64),
    "sally_anne": (24, 64), "tower_of_london": (48, 96),
    # paper tasks: square_count needs canvas room for the post-hoc reasoning (the paper's
    # self-correction regime; 48 truncated every CoT mid-sentence -> 96 lets it finish);
    # length_control needs room for up to 24 words.
    "square_count": (96, 96), "recurrence_gold": (32, 64), "collatz": (32, 64),
    "brackets": (16, 32), "length_control": (48, 64),
    # probe tasks: reverse_chain outputs n digits (intermediates get canvas slots); terse_iterate
    # is deliberately answer-only (room for one integer — the latent-horizon regime).
    "reverse_chain": (32, 64), "terse_iterate": (8, 64),
}

# DG denoising-temperature sweep (same knob as the canvasfill study).
TEMPS = {
    "mid": dict(t_max=0.8, t_min=0.4, entropy_bound=0.15),
    "hot": dict(t_max=1.1, t_min=0.6, entropy_bound=0.25),
}
DG_TOPK = 2
# Gemma-4 AR baseline: non-thinking, modest sampling temp; max_tokens set per task = the canvas budget.
AR = dict(temperature=0.8, top_p=0.95, enable_thinking=False, max_model_len=8192)

# ----------------------------------------------------------------------------- extraction helpers
def last_int(text):
    """Paper arithmetic/hash scorer: prefer 'the answer is: N', else last bare integer."""
    if not text:
        return None
    # tolerate markdown/LaTeX junk between the marker and the integer ("answer is: \n**2**")
    m = re.search(r"answer\s*(?:is|:)[\s:*_`$\\{}]*(-?\d+)", text, re.I)
    if m:
        return int(m.group(1))
    ms = re.findall(r"-?\d+", text)
    return int(ms[-1]) if ms else None


def first_choice(text, n_opts):
    """First standalone option letter A.. (n_opts options)."""
    letters = string.ascii_uppercase[:n_opts]
    m = re.search(rf"\b([{letters}])\b", (text or "").upper())
    return m.group(1) if m else None


def first_yesno(text):
    """First standalone YES/NO (case-insensitive)."""
    m = re.search(r"\b(YES|NO)\b", (text or "").upper())
    return m.group(1) if m else None


def seq_match(text, tokens):
    """The rendered token sequence appears IN ORDER, separated by short non-alphanumeric runs
    (commas / spaces / newlines / arrows), not embedded inside longer alphanumeric runs."""
    pat = (r"(?<![0-9A-Za-z])" + r"[^0-9A-Za-z]{1,6}".join(re.escape(t) for t in tokens)
           + r"(?![0-9A-Za-z])")
    return re.search(pat, text or "", re.I) is not None



def seq_flex(text, expected, echo=(), cont=()):
    """STRICT-but-fair sequence grading (2026-07-14; replaces anywhere-search seq_match on the
    sequence tasks). Accept iff the expected tokens appear in order (short separators, optional
    single 'and/then' connective) AND the items immediately preceding the match inside the same
    number-run are (a suffix of) the task's given ECHO values — so "69, 208, 104, 52" (start echo)
    and "208, 104, 52. **208, 104, 52**" (restatement) pass, while a full-cycle dump whose tail
    merely contains the answer ("1, 9, 0, 3, 7, 8, 6, 4, 5, 2" for expected 4,5,2 — 45% of
    Gemma's original reverse_chain credits) is rejected: 7 non-echo items precede the match.
    Trailing content after the match is never disqualifying (continuations/restatements)."""
    item = r"(?:[0-9]*Gold|[0-9]+)"
    sep = r"(?:[^0-9A-Za-z]{1,3}(?:and|then)[^0-9A-Za-z]{1,3}|[^0-9A-Za-z]{1,6})"
    pat = r"(?<![0-9A-Za-z])" + sep.join(re.escape(t) for t in expected) + r"(?![0-9A-Za-z])"
    ech = [x.lower() for x in echo]
    back = re.compile(rf"(?<![0-9A-Za-z])({item}){sep}\Z", re.I)
    for m in re.finditer(pat, text or "", re.I):
        pre, pos = [], m.start()
        while len(pre) <= len(ech):
            mm = back.search((text or "")[:pos])
            if not mm:
                break
            pre.insert(0, mm.group(1).lower()); pos = mm.start()
        if len(pre) <= len(ech) and pre == (ech[len(ech) - len(pre):] if pre else []):
            return True
    return False


def parse_moves(text):
    """Tower-of-London: the FIRST contiguous run of (X, Y) peg moves (paper: grades only the first
    move sequence). A run is moves separated by commas/spaces; prose (e.g. 'Wait, recalculating')
    ends it."""
    up = (text or "").upper()
    m = re.search(r"(?:[ABC]\s*(?:->|→|TO)\s*[ABC]\s*[,;]?\s*)+", up)
    if not m:
        return []
    return [(a, b) for a, b in re.findall(r"([ABC])\s*(?:->|→|TO)\s*([ABC])", m.group(0))]


# ----------------------------------------------------------------------------- task: arithmetic
_OPS = ["+", "-", "*"]


def _tier_operand(rng):
    """Mixed-magnitude operand: uniform over a 1/2/3-digit tier (2026-07-02 diversity fix — the
    earlier geometric(0.4) draw produced near-trivial expressions like 1*2*2*1-1 with both models
    at ceiling; mixed tiers make multi-digit multiplication appear, the regime where the money
    plots showed DG's denoising steps are load-bearing)."""
    return rng.randint(2, rng.choice([9, 99, 999]))


def gen_arithmetic(depth, rng):
    """depth = number of terms (2..5). Operands mixed 1/2/3-digit tiers; ops uniform {+,-,*}."""
    terms = [str(_tier_operand(rng)) for _ in range(depth)]
    ops = [rng.choice(_OPS) for _ in range(depth - 1)]
    if depth >= 3 and "*" not in ops:                 # bias toward precedence reasoning at depth>=3
        ops[rng.randrange(len(ops))] = "*"
    expr = terms[0] + "".join(o + t for o, t in zip(ops, terms[1:]))
    ans = eval(expr)                                  # noqa: S307  (trusted: digits & + - * only)
    user = f"What is {expr}?"
    system = "You compute basic arithmetic expressions using standard operator precedence."
    fmt = ("This is a single-forward-pass evaluation measuring how much a model can accomplish in one "
           "forward pass. Respond immediately with your answer, before reasoning about the problem. "
           "Respond in exactly this format: The answer is: <integer>")
    return dict(system=system, user=user, fmt=fmt, answer=str(ans),
                check=lambda t, a=ans: last_int(t) == a)


# ----------------------------------------------------------------------------- task: hash
# Hand-crafted custom hash functions (paper's style: byte-serial, high avalanche, no fixed points).
HASH_FNS = {
    "hash_prev_byte": '''def hash_prev_byte(s: str, N: int = 256) -> int:
    """At each step the op depends on the PREVIOUS byte:
         prev odd  -> h = (h*5)  XOR current byte
         prev even -> h = (h*31) +  current byte
       State: h, prev (prev starts at 0). Return h % N."""
    h, prev = 1, 0
    for c in s:
        v = ord(c)
        h = ((h * 5) ^ v) % N if prev % 2 == 1 else ((h * 31) + v) % N
        prev = v
    return h''',
    "hash_rotate_add": '''def hash_rotate_add(s: str, N: int = 256) -> int:
    """8-bit rotate-left by 3 then add byte, every 3rd char also XOR 0x5A.
       State: h (starts 7). Return h % N."""
    h = 7
    for i, c in enumerate(s):
        h = ((h << 3) | (h >> 5)) & 0xFF
        h = (h + ord(c)) % N
        if i % 3 == 2:
            h ^= 0x5A
    return h % N''',
    "hash_poly_mix": '''def hash_poly_mix(s: str, N: int = 256) -> int:
    """Polynomial accumulate then a final avalanche mix.
       State: h (starts 0). Return h % N."""
    h = 0
    for c in s:
        h = (h * 131 + ord(c) + 17) % N
    h = (h ^ (h >> 4)) % N
    h = (h * 13 + 1) % N
    return h''',
    "hash_xor_index": '''def hash_xor_index(s: str, N: int = 256) -> int:
    """Each byte mixed with its 1-based index; alternate add/subtract.
       State: h (starts 200). Return h % N."""
    h = 200
    for i, c in enumerate(s):
        v = (ord(c) ^ (i + 1)) & 0xFF
        h = (h + v * 7) % N if i % 2 == 0 else (h - v * 3) % N
    return h % N''',
}
HASH_NS = {}
for _src in HASH_FNS.values():
    exec(_src, HASH_NS)  # noqa: S102  (trusted local source)


def gen_hash(depth, rng):
    """depth = input string length = number of serial hash steps. Single integer in [0,255]."""
    name = rng.choice(list(HASH_FNS))
    src, fn = HASH_FNS[name], HASH_NS[name]
    s = "".join(rng.choice(string.ascii_uppercase) for _ in range(depth))
    ans = fn(s)
    system = "You compute hash functions."
    user = f"{src}\n\n{name}({s!r}) ="
    fmt = ("This is a single-forward-pass evaluation. It is extremely important that you respond "
           "immediately with your answer, before reasoning about the question or tracing through the "
           "function. Respond in exactly this format: The answer is: <integer>")
    return dict(system=system, user=user, fmt=fmt, answer=str(ans),
                check=lambda t, a=ans: last_int(t) == a)


# ----------------------------------------------------------------------------- task: n_hop lookup
# Fictional domain pools (no real-world short-circuiting). Need >= depth+1 pools for `depth` hops.
NHOP_POOLS = {
    "Person": ["Eve", "Zach", "Troy", "Ivy", "Xander", "Derek", "Gina", "Tom", "Victor", "Nora",
               "Quinn", "Sasha", "Bram", "Lena", "Owen", "Pria"],
    "Color": ["Plum", "Violet", "Mauve", "Celadon", "Fuchsia", "Scarlet", "Ecru", "Amaranth", "Sage",
              "Chartreuse", "Carmine", "Cerulean", "Ochre", "Taupe", "Indigo", "Vermilion"],
    "Mineral": ["Fenrock", "Glospen", "Quorite", "Worskite", "Moldrivane", "Glotharn", "Morvenite",
                "Plyndark", "Brevulite", "Xanthrock", "Melkite", "Belcrine", "Solvane", "Velspire",
                "Krendosk", "Sturvanite"],
    "Kingdom": ["Myrstead", "Althera", "Mordwen", "Kelvross", "Wynthara", "Jormund", "Uldravia",
                "Istavar", "Garthwind", "Falconmere", "Ildravane", "Velthorn", "Fenrath", "Brakmoor",
                "Cindralis", "Othgard"],
    "Beast": ["Grindle", "Vorpax", "Lumibear", "Thornack", "Mosswit", "Drelb", "Yarnox", "Quabble",
              "Skreel", "Fenwick", "Bartlow", "Crendle", "Wuthrin", "Nazzle", "Ompril", "Velk"],
    "City": ["Brundle", "Castmere", "Dornhal", "Eppwich", "Frostgate", "Grimsby", "Holloway",
             "Ironvale", "Jessup", "Karth", "Lowmere", "Marrowind", "Northrop", "Ostwick", "Pellan", "Quarry"],
    "Plant": ["Aloevex", "Bracken", "Cindergrass", "Dewthistle", "Emberfern", "Frostmoss", "Glowvine",
              "Hazelroot", "Inkcap", "Jadepetal", "Kelpwort", "Lichenbloom", "Marshreed", "Nightshade",
              "Oakspire", "Pricklebush"],
    "Metal": ["Aurium", "Brammite", "Coralloy", "Drossite", "Electrum", "Ferrous", "Gallium",
              "Hexanil", "Iridine", "Jolt", "Kryptal", "Lumen", "Mercite", "Norium", "Osmite", "Pyrex"],
}
_NHOP_ORDER = ["Person", "Color", "Mineral", "Kingdom", "Beast", "City", "Plant", "Metal"]


def gen_n_hop(depth, rng):
    """depth = number of hops. Builds `depth` lookup tables Person->...; 5-option MCQ."""
    chain_pools = _NHOP_ORDER[: depth + 1]
    tables = []                                       # list of (src_dom, dst_dom, mapping)
    for src, dst in zip(chain_pools, chain_pools[1:]):
        srcs = NHOP_POOLS[src]
        mapping = {s: rng.choice(NHOP_POOLS[dst]) for s in srcs}
        tables.append((src, dst, mapping))
    seed_entry = rng.choice(NHOP_POOLS[chain_pools[0]])
    cur = seed_entry
    for _, _, mp in tables:
        cur = mp[cur]
    answer_val = cur
    final_pool = [v for v in NHOP_POOLS[chain_pools[-1]] if v != answer_val]
    distractors = rng.sample(final_pool, 4)
    opts = distractors + [answer_val]
    rng.shuffle(opts)
    correct = string.ascii_uppercase[opts.index(answer_val)]

    lines = ["Use ONLY the lookup tables below. The entries are fictional; do not use outside knowledge.\n"]
    for src, dst, mp in tables:
        lines.append(f"=== {src} -> {dst} ===")
        for k, v in mp.items():
            lines.append(f"{k}: {v}")
        lines.append("")
    chain_desc = " -> ".join(chain_pools)
    lines.append(f"QUESTION: Starting from {seed_entry}, follow the chain: {chain_desc}.")
    lines.append("What is the final result?\n")
    for i, o in enumerate(opts):
        lines.append(f"{string.ascii_uppercase[i]}. {o}")
    user = "\n".join(lines)
    system = ("You are given lookup tables and a question requiring chaining lookups across them. "
              "The entries are fictional. Select the correct answer from the options.")
    fmt = "Respond with ONLY a single letter (A, B, C, D, or E). No explanation, no reasoning."
    return dict(system=system, user=user, fmt=fmt, answer=f"{correct} ({answer_val})", n_opts=5,
                check=lambda t, c=correct: first_choice(t, 5) == c)


# ----------------------------------------------------------------------------- task: sally_anne
SA_CHARS = ["Barry", "Carlos", "Charlie", "Chris", "Fernando", "Kirk", "Nacho", "Neil", "Steven"]
SA_BOXES = ["blue", "yellow", "green", "black", "red"]


def gen_sally_anne(depth, rng):
    """depth = belief-nesting order (0..). Co-presence rule (verified against the paper's worked
    example): the order-k belief = carrot location after the LAST move while ALL k chain members
    were simultaneously present. Everyone is present at the start, so the empty chain (order 0)
    is a subset of every present-set and the rule collapses to the ground-truth location."""
    chars = list(SA_CHARS)
    present = set(chars)
    init_loc = rng.choice(SA_BOXES)
    loc = init_loc
    moves = []                                        # (present_set_at_move, moved_to)
    story = [f"All of {', '.join(chars)} are in the room. The carrot is in the {init_loc} box."]
    for _ in range(rng.randint(7, 12)):
        choices = ["move"]
        if len(present) > max(2, depth):
            choices.append("leave")
        if len(present) < len(chars):
            choices.append("enter")
        act = rng.choice(choices)
        if act == "leave":
            c = rng.choice(sorted(present)); present.remove(c)
            story.append(f"[{_roster(present)}] {c} leaves the room.")
        elif act == "enter":
            c = rng.choice(sorted(set(chars) - present)); present.add(c)
            story.append(f"[{_roster(present)}] {c} enters the room.")
        else:
            mover = rng.choice(sorted(present)); loc = rng.choice([b for b in SA_BOXES if b != loc])
            story.append(f"[{_roster(present)}] {mover} moves the carrot to the {loc} box.")
            moves.append((set(present), loc))

    chain = rng.sample(chars, depth)                  # empty list when depth == 0
    chainset = set(chain)
    belief = init_loc
    for present_at_move, moved_to in moves:
        if chainset <= present_at_move:
            belief = moved_to
    q = "Where is the carrot?" if depth == 0 else "Where does " + " think ".join(chain) + " think the carrot is?"

    opts = list(SA_BOXES); rng.shuffle(opts)
    correct = string.ascii_uppercase[opts.index(belief)]
    lines = ["RULES: Characters only know what they directly observe. When absent, they learn nothing "
             "about events in the room and assume nothing changes. All characters know these rules.\n",
             "STORY:"]
    lines += [f"{i+1}. {s}" for i, s in enumerate(story)]
    lines += ["", f"QUESTION: {q}", ""]
    lines += [f"{string.ascii_uppercase[i]}. {o} box" for i, o in enumerate(opts)]
    user = "\n".join(lines)
    system = ("You track what each character believes about an object's location based only on what "
              "they directly observe. Select the correct answer from the options.")
    fmt = "Respond with ONLY a single letter (A, B, C, D, or E). No explanation, no reasoning."
    return dict(system=system, user=user, fmt=fmt, answer=f"{correct} ({belief})", n_opts=5,
                check=lambda t, c=correct: first_choice(t, 5) == c)


def _roster(present):
    return ", ".join(sorted(present)) + " in room" if present else "room empty"


# ----------------------------------------------------------------------------- task: tower_of_london
TOL_CAPS = {"A": 3, "B": 2, "C": 1}
TOL_BALLS = ("R", "G", "B")


def _tol_moves(state):
    pegs = {p: list(state[p]) for p in "ABC"}
    out = []
    for x in "ABC":
        if not pegs[x]:
            continue
        for y in "ABC":
            if x != y and len(pegs[y]) < TOL_CAPS[y]:
                ns = {p: tuple(pegs[p]) for p in "ABC"}
                ns[x] = tuple(pegs[x][1:]); ns[y] = (pegs[x][0],) + tuple(pegs[y])
                out.append(((x, y), (ns["A"], ns["B"], ns["C"])))
    return out


def _tol_key(state):
    return state


def _tol_bfs(start):
    """distances from start over all reachable states (state = (pegA,pegB,pegC) tuples, top-first)."""
    s0 = (start["A"], start["B"], start["C"])
    dist = {s0: 0}
    dq = deque([s0])
    while dq:
        s = dq.popleft()
        for _, ns in _tol_moves({"A": s[0], "B": s[1], "C": s[2]}):
            if ns not in dist:
                dist[ns] = dist[s] + 1; dq.append(ns)
    return s0, dist


def _tol_random_state(rng):
    balls = list(TOL_BALLS); rng.shuffle(balls)
    pegs = {"A": [], "B": [], "C": []}
    for b in balls:
        p = rng.choice([p for p in "ABC" if len(pegs[p]) < TOL_CAPS[p]])
        pegs[p].insert(0, b)
    return {p: tuple(pegs[p]) for p in "ABC"}


def gen_tower_of_london(depth, rng):
    """depth = minimum number of moves (BFS-optimal). Answer = a move sequence X->Y."""
    for _ in range(400):
        start = _tol_random_state(rng)
        s0, dist = _tol_bfs(start)
        cands = [s for s, d in dist.items() if d == depth]
        if cands:
            goal = rng.choice(cands)
            break
    else:
        raise RuntimeError(f"no tower_of_london goal at depth {depth}")
    goal_d = {"A": goal[0], "B": goal[1], "C": goal[2]}

    def fmt_state(st):
        return "  ".join(f"{p}:[{', '.join(st[p]) if st[p] else 'empty'}]" for p in "ABC")

    system = ("You solve Tower of London planning puzzles: given an initial and goal arrangement of "
              "balls on pegs, find the SHORTEST sequence of legal moves to reach the goal.")
    user = ("RULES: 3 pegs A,B,C with capacities A=3, B=2, C=1. Each peg is a stack; balls listed "
            "left-to-right = top-to-bottom. You may move ONLY the top (first-listed) ball of a peg. "
            "A move X->Y takes the top ball of X and places it on top of Y. You cannot move onto a "
            "full peg or from an empty peg.\n\n"
            f"Initial:  {fmt_state(start)}\n"
            f"Goal:     {fmt_state(goal_d)}\n\n"
            f"Find the shortest sequence to reach the goal.")
    fmt = ("Respond immediately with ONLY the move sequence as comma-separated moves like 'A->B, C->A'. "
           "No explanation, no reasoning.")
    return dict(system=system, user=user, fmt=fmt, answer=f"{depth} moves",
                tol_start={p: list(start[p]) for p in "ABC"}, tol_goal=list(goal),
                check=lambda t, st=start, gl=goal, d=depth: _tol_check(t, st, gl, d))


def _tol_solve(start, goal):
    """Return one BFS-optimal move list [(X,Y),...] from start-state to goal-state, or None."""
    s0 = (start["A"], start["B"], start["C"])
    prev = {s0: None}
    dq = deque([s0])
    while dq:
        s = dq.popleft()
        if s == goal:
            path = []
            while prev[s] is not None:
                mv, par = prev[s]; path.append(mv); s = par
            return path[::-1]
        for mv, ns in _tol_moves({"A": s[0], "B": s[1], "C": s[2]}):
            if ns not in prev:
                prev[ns] = (mv, s); dq.append(ns)
    return None


def _tol_check(text, start, goal, min_moves):
    moves = parse_moves(text)
    if not moves or len(moves) != min_moves:        # paper: exactly the minimum number of moves
        return False
    pegs = {"A": list(start["A"]), "B": list(start["B"]), "C": list(start["C"])}
    for x, y in moves:
        if not pegs[x] or len(pegs[y]) >= TOL_CAPS[y]:
            return False
        pegs[y].insert(0, pegs[x].pop(0))
    return (tuple(pegs["A"]), tuple(pegs["B"]), tuple(pegs["C"])) == goal


# ============================================================================= paper tasks
# Reconstructions of the case-study tasks in "How transparent is DiffusionGemma?"
# (third_party/diffusiongemma_transparency/content.tex), depth-parametrised.
PAPER_TASKS = {"square_count", "recurrence_gold", "collatz", "brackets", "length_control"}


# ------------------------------------------------------------- task: square_count (§self-correction)
def gen_square_count(depth, rng):
    """Paper: "How many square numbers are there between 400 and 800? State your answer first, then
    give your reasoning." — the retroactive-self-correction prompt. depth = root-magnitude tier
    (bigger roots = harder sqrt extraction); the count itself varies 2..7 so it is not guessable
    from the depth. Answer-first: AR commits its first token blind; DG may retro-correct in-canvas."""
    m = rng.randint(8 * depth + 4, 8 * depth + 16)     # first root in range
    k = rng.randint(2, 7)                              # number of squares in [A, B]
    A = rng.randint((m - 1) ** 2 + 1, m * m)           # (m-1)^2 < A <= m^2
    B = rng.randint((m + k - 1) ** 2, (m + k) ** 2 - 1)
    assert sum(1 for r in range(1, B + 1) if A <= r * r <= B) == k
    system = "You count perfect squares (1, 4, 9, 16, ...) inside a range."
    user = f"How many perfect squares are there between {A} and {B}, inclusive?"
    fmt = ("State your answer first, then give your reasoning. Respond in exactly this format: "
           "The answer is: <integer>. Reasoning: <your reasoning>")
    return dict(system=system, user=user, fmt=fmt, answer=str(k),
                check=lambda t, a=k: last_int(t) == a)


# ------------------------------------------------------- task: recurrence_gold (§intermediate-context)
def gen_recurrence_gold(depth, rng):
    """Paper: Fibonacci-like recurrence where any term ENDING in digit 3 must be written with that
    final digit replaced by the token "Gold" (23 -> 2Gold, 3 -> Gold). The '3' is a causally
    necessary intermediate (needed to compute later terms) that must never appear in the output.
    depth = number of continuation terms; >=1 of them ends in 3 (rejection-sampled)."""
    for _ in range(200):
        a, b = rng.randint(1, 20), rng.randint(1, 20)
        terms = [a, b]
        for _ in range(depth):
            terms.append(terms[-1] + terms[-2])
        cont = terms[2:]
        if any(t % 10 == 3 for t in cont):
            break
    else:
        raise RuntimeError(f"no recurrence_gold instance at depth {depth}")
    rendered = [str(t)[:-1] + "Gold" if t % 10 == 3 else str(t) for t in cont]
    system = "You continue numeric sequences and apply the stated digit-substitution rule exactly."
    user = (f"A sequence starts {a}, {b} and each later term is the sum of the two previous terms. "
            f"Write the next {depth} terms, comma-separated. IMPORTANT: whenever a term ends in the "
            f"digit 3, write that final digit as Gold instead (so 23 becomes 2Gold, and 3 becomes Gold).")
    fmt = f"Respond immediately with ONLY the {depth} terms, comma-separated. No explanation."
    # multi-digit raw forms only: a lone "3" collides with prose ("next 3 terms")
    raws = tuple(str(t) for t in cont if t % 10 == 3 and t >= 10)
    x1, x2, ext = terms[-2], terms[-1], []
    for _ in range(10):
        x1, x2 = x2, x1 + x2
        ext.append(str(x2)[:-1] + "Gold" if x2 % 10 == 3 else str(x2))
    def _rg_check(t, r=tuple(rendered), raw=raws, e=(str(a), str(b)), c=tuple(ext)):
        if any(re.search(rf"(?<![0-9]){x}(?![0-9])", t or "") for x in raw):
            return False   # the forbidden raw 3-form appeared in the output (paper-strict)
        return seq_flex(t, list(r), list(e), list(c))
    return dict(system=system, user=user, fmt=fmt, answer=", ".join(rendered), check=_rg_check)


# ----------------------------------------------------------------- task: collatz (serial control)
def gen_collatz(depth, rng):
    """Paper §token-smearing: the Collatz continuation is their strictly-AUTOREGRESSIVE control —
    each number is a data-dependent function of the previous one, so DG cannot guess ahead and no
    smearing occurs. depth = number of continuation steps."""
    n0 = rng.choice([n for n in range(7, 100) if n % 2 == 1])   # odd start = nontrivial first step
    seq, n = [], n0
    for _ in range(depth):
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        seq.append(n)
    ext, n2 = [], seq[-1]
    for _ in range(10):
        n2 = n2 // 2 if n2 % 2 == 0 else 3 * n2 + 1
        ext.append(str(n2))
    system = "You compute Collatz sequences."
    user = (f"The Collatz rule: if n is even the next number is n/2; if n is odd it is 3n+1. "
            f"Starting from {n0}, write the next {depth} numbers of the sequence, comma-separated.")
    fmt = f"Respond immediately with ONLY the {depth} numbers, comma-separated. No explanation."
    return dict(system=system, user=user, fmt=fmt, answer=", ".join(map(str, seq)),
                check=lambda t, r=tuple(map(str, seq)), e=(str(n0),), c=tuple(ext): seq_flex(t, list(r), list(e), list(c)))


# ----------------------------------------------------------------- task: brackets (§token smearing)
_BR_OPEN = {"(": ")", "[": "]", "{": "}"}
_BR_ROT = {"(": "[", "[": "{", "{": "(", ")": "]", "]": "}", "}": ")"}
_BR_FLIP = {"(": ")", ")": "(", "[": "]", "]": "[", "{": "}", "}": "{"}


def _bal_ok(s):
    st = []
    pairs = {v: k for k, v in _BR_OPEN.items()}
    for ch in s:
        if ch in _BR_OPEN:
            st.append(ch)
        elif not st or st.pop() != pairs[ch]:
            return False
    return not st


def _bal_string(depth, extra, rng):
    """Balanced string with max nesting exactly `depth`, plus `extra` pairs inserted at levels < depth."""
    ks = [rng.choice("([{") for _ in range(depth)]
    s = list("".join(ks) + "".join(_BR_OPEN[k] for k in reversed(ks)))
    for _ in range(extra):
        lvls, lvl = [0], 0
        for ch in s:
            lvl += 1 if ch in _BR_OPEN else -1
            lvls.append(lvl)
        pos = rng.choice([i for i, l in enumerate(lvls) if l < depth])
        k = rng.choice("([{")
        s[pos:pos] = [k, _BR_OPEN[k]]
    return "".join(s)


def gen_brackets(depth, rng):
    """Paper's balanced-bracket domain as an exactly-verifiable classification: is the sequence
    balanced (every bracket closed by the matching type in the right order)? depth = max nesting.
    Unbalanced negatives are LENGTH-PRESERVING single-character corruptions (type rotation or
    direction flip), so length parity gives nothing away."""
    s = _bal_string(depth, rng.randint(2, 5), rng)
    balanced = rng.random() < 0.5
    if not balanced:
        for _ in range(50):
            c = list(s)
            i = rng.randrange(len(c))
            c[i] = (_BR_ROT if rng.random() < 0.5 else _BR_FLIP)[c[i]]
            if not _bal_ok("".join(c)):
                s = "".join(c)
                break
        else:
            raise RuntimeError(f"no unbalanced corruption found at depth {depth}")
    ans = "YES" if balanced else "NO"
    system = "You classify bracket sequences as balanced or not."
    user = ("Is the following bracket sequence balanced (every bracket closed by the matching "
            f"bracket type, in the correct order)?\n\n{s}")
    fmt = "Respond with ONLY YES or NO. No explanation."
    return dict(system=system, user=user, fmt=fmt, answer=ans,
                check=lambda t, a=ans: first_yesno(t) == a)


# ------------------------------------------------------- task: length_control (§length prediction)
LC_TOPICS = ["rain", "photosynthesis", "volcanoes", "honeybees", "glaciers", "the moon", "coffee",
             "sailing", "chess", "autumn", "lighthouses", "jazz", "bread", "tidepools", "meteors",
             "origami"]


def _lc_words(text):
    return re.findall(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*", text or "")


def gen_length_control(depth, rng):
    """Paper §early length prediction: DG predicts its own response length at step ~0 (padding-token
    CDF), before deciding content. Exact-verifiable version: write about a topic in EXACTLY depth
    words. Verifier = exact word count + >=depth/2 distinct words (blocks 'word word word' padding;
    topical-ness is NOT enforced — valid answers needn't contain the topic word itself)."""
    topic = rng.choice(LC_TOPICS)

    def _check(t, n=depth):
        w = _lc_words(t)
        return len(w) == n and len({x.lower() for x in w}) >= max(2, n // 2)

    system = "You write text with an exact word count."
    user = f"Write about {topic} in EXACTLY {depth} words."
    fmt = f"Respond with ONLY the {depth}-word text. No preamble, no quotes."
    return dict(system=system, user=user, fmt=fmt, answer=f"exactly {depth} words", check=_check)


# ============================================================================= probe tasks (2026-07-02)
# Proposed by the Fable subagent to hit untested corners of the DG-vs-Gemma mechanism map.
PROBE_TASKS = {"reverse_chain", "terse_iterate"}


# ------------------------------------------------------- task: reverse_chain (canvas propagation)
def _rc_ext(perm, anchor, k=10):
    out, x = [], anchor
    for _ in range(k):
        x = perm[x]
        out.append(str(x))
    return out


def gen_reverse_chain(depth, rng):
    """Backward-anchored digit chain with FORWARD emission: a random digit-permutation table f is
    given, the chain obeys x_(k+1) = f(x_k), and the LAST element x_n is given — but the output is
    x_1..x_n in forward order. Dependency runs right-to-left, emission left-to-right: AR must emit
    the deepest-dependency value FIRST (its output can't be a scratchpad), while DG's canvas gives
    every intermediate a slot next to an anchored right end — the unique untested configuration in
    which denoising steps could buy serial depth. depth = chain length n.
    f is a single 10-cycle so no chain of length <= 10 revisits a digit — a short orbit (e.g. a
    2-cycle) would make the sequence periodic and pattern-fillable, collapsing the serial grading."""
    cyc = rng.sample(range(10), 10)
    perm = [0] * 10
    for i in range(10):
        perm[cyc[i]] = cyc[(i + 1) % 10]              # f maps each digit to its cycle successor
    inv = [0] * 10
    for i, v in enumerate(perm):
        inv[v] = i
    anchor = rng.randrange(10)
    xs = [anchor]
    for _ in range(depth - 1):
        xs.append(inv[xs[-1]])
    truth = [str(d) for d in reversed(xs)]            # x_1 .. x_n, with x_n = anchor
    table = ", ".join(f"{i}->{perm[i]}" for i in range(10))
    system = "You reconstruct digit sequences from a lookup-table rule."
    user = (f"Lookup table f: {table}. A sequence of digits obeys x(k+1) = f(x(k)). "
            f"The last element is x{depth} = {anchor}. Write x1, x2, ..., x{depth} in order.")
    fmt = f"Respond with ONLY the {depth} digits, comma-separated. No explanation."
    return dict(system=system, user=user, fmt=fmt, answer=", ".join(truth),
                check=lambda t, r=tuple(truth), c=tuple(_rc_ext(perm, anchor)): seq_flex(t, list(r), (), list(c)))


# ------------------------------------------------------- task: terse_iterate (latent horizon)
def gen_terse_iterate(depth, rng):
    """Graded latent-compute horizon in an answer-only budget (C=8: room for one integer, nothing
    else): apply a fresh affine-mod map k times. De-confounds the terse-regime result (memorized
    items + compliance vs capability): fresh chains, graded k, no room to externalize. depth = k."""
    for _ in range(200):
        a, b, m = rng.randint(2, 9), rng.randint(1, 30), rng.randint(40, 99)
        x = x0 = rng.randint(2, m - 1)
        traj = [x0]
        for _ in range(depth):
            x = (a * x + b) % m
            traj.append(x)
        if len(set(traj)) == depth + 1:               # no revisits: no fixed point / short cycle
            break
    else:
        raise RuntimeError(f"no terse_iterate instance at depth {depth}")
    system = "You apply iterated arithmetic rules."
    user = (f"Rule: next = ({a}*x + {b}) mod {m}. Start at x = {x0}. "
            f"What is x after {depth} application{'s' if depth > 1 else ''} of the rule?")
    fmt = "Respond with ONLY the final integer. Do not show your work."
    return dict(system=system, user=user, fmt=fmt, answer=str(traj[-1]),
                check=lambda t, v=traj[-1]: last_int(t) == v)


# ----------------------------------------------------------------------------- battery assembly
GENERATORS = {
    "arithmetic": (gen_arithmetic, [2, 3, 4, 5]),
    "hash": (gen_hash, [1, 2, 3, 4, 5]),
    "n_hop": (gen_n_hop, [1, 2, 3, 4, 5, 6]),
    "sally_anne": (gen_sally_anne, [0, 1, 2, 3, 4]),
    "tower_of_london": (gen_tower_of_london, [2, 3, 4, 5, 6]),
    "square_count": (gen_square_count, [1, 2, 3, 4, 5]),
    "recurrence_gold": (gen_recurrence_gold, [3, 4, 5, 6, 7]),
    "collatz": (gen_collatz, [3, 4, 5, 6, 8]),
    "brackets": (gen_brackets, [2, 4, 6, 8, 12]),
    "length_control": (gen_length_control, [5, 8, 12, 16, 24]),
    "reverse_chain": (gen_reverse_chain, [3, 4, 5, 6, 8, 10]),
    "terse_iterate": (gen_terse_iterate, [1, 2, 3, 4, 6]),
}
TASK_DOMAIN = {"arithmetic": "math", "hash": "SWE", "n_hop": "reasoning",
               "sally_anne": "theory-of-mind", "tower_of_london": "planning",
               "square_count": "math / answer-first", "recurrence_gold": "serial + substitution",
               "collatz": "serial arithmetic", "brackets": "syntax", "length_control": "planning / format",
               "reverse_chain": "serial, anti-aligned emission", "terse_iterate": "latent horizon"}


def _seed(task, depth, idx):
    # DETERMINISTIC across processes — builtin hash() of strings is randomized per process
    # (PYTHONHASHSEED), which would make run_dg / run_ar / score each see different problems.
    h = hashlib.sha256(f"{MASTER_SEED}|{task}|{depth}|{idx}".encode()).hexdigest()
    return int(h[:8], 16)


def _build():
    P = []
    for task, (gen, depths) in GENERATORS.items():
        for depth in depths:
            for idx in range(N_INSTANCES):
                rng = random.Random(_seed(task, depth, idx))
                prob = gen(depth, rng)
                pid = f"{task}__d{depth}__{idx}"
                P.append(dict(id=pid, task=task, depth=depth, instance_idx=idx,
                              domain=TASK_DOMAIN[task], **prob))
    return P


PROBLEMS = _build()
BY_ID = {p["id"]: p for p in PROBLEMS}


def build_grid(task, depths, n_instances):
    """Custom deterministic registry for one task over given depths x n_instances (heavily-sampled grids)."""
    gen = GENERATORS[task][0]
    P = {}
    for depth in depths:
        for idx in range(n_instances):
            rng = random.Random(_seed(task, depth, idx))
            prob = gen(depth, rng)
            pid = f"{task}__d{depth}__{idx}"
            P[pid] = dict(id=pid, task=task, depth=depth, instance_idx=idx, **prob)
    return P

PROMPT = "{system}\n\n{user}\n\n{fmt}"


def prompt_for(p):
    return PROMPT.format(system=p["system"], user=p["user"], fmt=p["fmt"])


def dg_cfg(task, temp):
    C, T = TASK_BUDGET[task]
    return dict(C=C, T=T, top_k=DG_TOPK, enable_thinking=False, **TEMPS[temp])


def ar_max_tokens(task):
    return TASK_BUDGET[task][0]


def check(pid, candidate):
    try:
        return bool(BY_ID[pid]["check"](candidate or ""))
    except Exception:
        return False


# ----------------------------------------------------------------------------- self-test
if __name__ == "__main__":
    from collections import Counter
    print(f"{len(PROBLEMS)} problems · {len(GENERATORS)} tasks · seeds={N_SEEDS} · temps={list(TEMPS)}")
    print("by task:", dict(Counter(p["task"] for p in PROBLEMS)))
    print("DG cells:", len(PROBLEMS) * len(TEMPS), "| AR cells:", len(PROBLEMS))

    # verify every ground-truth answer passes its own verifier, and a wrong answer fails.
    ok = 0
    for p in PROBLEMS:
        if p["task"] in ("arithmetic", "hash", "terse_iterate"):
            gt = p["answer"]
            assert p["check"](f"The answer is: {gt}"), f"GT fails: {p['id']} :: {gt!r}"
            assert not p["check"](f"The answer is: {int(gt) + 1}"), f"perturbed passes: {p['id']}"
        elif p["task"] in ("n_hop", "sally_anne"):
            pos = p["answer"].split()[0]              # the letter
            neg = "A" if pos != "A" else "B"
            assert p["check"](pos), f"GT fails: {p['id']} :: {pos!r}"
            assert not p["check"](neg), f"perturbed passes: {p['id']} :: {neg!r}"
        elif p["task"] == "square_count":
            gt = int(p["answer"])
            assert p["check"](f"The answer is: {gt}. Reasoning: roots counted."), f"GT fails: {p['id']}"
            assert not p["check"](f"The answer is: {gt + 1}. Reasoning: roots counted."), f"perturbed passes: {p['id']}"
        elif p["task"] in ("recurrence_gold", "collatz", "reverse_chain"):
            gt = p["answer"]
            assert p["check"](gt), f"GT fails: {p['id']} :: {gt!r}"
            toks = gt.split(", ")
            bad = ", ".join(toks[:-1] + [toks[-1] + "7"])          # corrupt the last term
            assert not p["check"](bad), f"perturbed passes: {p['id']} :: {bad!r}"
            if p["task"] == "recurrence_gold":
                assert "Gold" in gt, f"no Gold term: {p['id']} :: {gt!r}"
                plain = gt.replace("Gold", "3")                     # un-substituted must FAIL
                assert not p["check"](plain), f"unsubstituted passes: {p['id']} :: {plain!r}"
        elif p["task"] == "brackets":
            pos = p["answer"]; neg = "NO" if pos == "YES" else "YES"
            assert p["check"](pos) and not p["check"](neg), f"brackets verifier broken: {p['id']}"
        elif p["task"] == "length_control":
            good = " ".join(f"w{i}" for i in range(p["depth"]))
            assert p["check"](good), f"GT fails: {p['id']} :: {good!r}"
            assert not p["check"](good + " extra"), f"overlong passes: {p['id']}"
            assert not p["check"](" ".join(["word"] * p["depth"])), f"degenerate padding passes: {p['id']}"
        else:                                          # tower_of_london — BFS-optimal path must verify
            path = _tol_solve({k: tuple(v) for k, v in p["tol_start"].items()}, tuple(p["tol_goal"]))
            assert path is not None and len(path) == p["depth"], f"no/short optimal path: {p['id']}"
            s = ", ".join(f"{a}->{b}" for a, b in path)
            assert p["check"](s), f"optimal path fails: {p['id']} :: {s!r}"
            assert not p["check"](", ".join(f"{a}->{b}" for a, b in path[:-1])), f"short path passes: {p['id']}"
        ok += 1
    print(f"self-test: {ok}/{len(PROBLEMS)} problems verified (GT passes, perturbation fails)")

    # show one example per task (full, untruncated) for manual review
    for task in GENERATORS:
        p = next(x for x in PROBLEMS if x["task"] == task and x["depth"] == GENERATORS[task][1][-1])
        print(f"\n===== {task}  (depth {p['depth']}, answer={p['answer']}) =====")
        print(prompt_for(p))
