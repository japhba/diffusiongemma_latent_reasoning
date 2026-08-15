"""Serial-computation battery for the DiffusionGemma-vs-Gemma "where does diffusion win?" study.

Grounded in *How Transparent is DiffusionGemma?* (Engels et al., arXiv 2606.20560). The paper's
central concept is **opaque serial depth**: each denoising step is a full forward pass, so over T
steps DiffusionGemma (DG) gets ~T x the serial compute of a single autoregressive (AR) forward
pass. The paper shows DG exploits this for *retroactive self-correction* (its "square numbers"
case study) and *intermediate-context reasoning* (its Fibonacci 3->Gold case study): the canvas
acts as a latent scratchpad that an AR model, forced to commit each token left-to-right, lacks.

This battery turns that into a *performance* comparison. Every problem has a single verifiable
answer and requires genuine multi-step computation (counting/enumeration, multi-digit arithmetic,
combinatorics, multi-hop chains, compute-then-transform). We sweep three response FRAMINGS that
move along the serial-depth axis:

  answer_only      both models must emit the answer with no room to reason  -> serial-depth FLOOR
  answer_first     answer at the front, reasoning after. AR's answer token is causally fixed
                   BEFORE its reasoning (can't help it); DG's canvas reasoning can retro-correct
                   the answer -> DG's sweet spot (the paper's self-correction regime).
  reasoning_first  reason first, answer last. AR gets token-serial CoT depth, matching DG    -> CONTROL

Hypothesis (systematicity): DG's advantage (1-G)*D is concentrated in `answer_first`, small in
`answer_only`, and collapses in `reasoning_first` once AR is allowed token-serial depth.

`correct` is the canonical answer; `alts` are accepted equivalents (the judge normalizes anyway).
`cat` groups problems; `hard=True` marks problems an AR single forward pass should usually fail.
"""

PROBLEMS = [
    # --- A. CRT / intuition traps: AR can usually one-pass these; lure is the fast-but-wrong answer
    dict(id="bat_ball", cat="crt", hard=False,
         q="A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. "
           "How much does the ball cost, in cents?", correct="5", alts=["5 cents", "$0.05"], lure="10"),
    dict(id="lily", cat="crt", hard=False,
         q="A patch of lily pads doubles in size every day. If it takes 48 days to cover the whole "
           "lake, how many days does it take to cover half the lake?", correct="47", alts=[], lure="24"),
    dict(id="widgets", cat="crt", hard=False,
         q="If 5 machines take 5 minutes to make 5 widgets, how many minutes do 100 machines take "
           "to make 100 widgets?", correct="5", alts=[], lure="100"),
    dict(id="socks", cat="crt", hard=False,
         q="A drawer has 10 red and 10 blue socks, identical except for colour. Reaching in the dark, "
           "how many socks must you take out to be certain of a matching pair?", correct="3", alts=[], lure="11"),
    dict(id="months28", cat="crt", hard=False,
         q="How many months in a year have exactly 28 days?", correct="12", alts=["all 12", "twelve"], lure="1"),
    dict(id="monty", cat="crt", hard=False,
         q="A game show has three doors; one hides a car, the others goats. You pick door 1. The host, "
           "who knows the layout, opens door 3 to show a goat. To maximise your chance of the car, "
           "should you switch to door 2? Answer Yes or No.", correct="Yes", alts=["switch"], lure="No"),

    # --- B. Counting / enumeration: medium; one-pass is shaky
    dict(id="squares_400_800", cat="count", hard=False,
         q="How many perfect square numbers are there strictly between 400 and 800?",
         correct="8", alts=[], lure="9"),   # 21^2..28^2 -> 441,484,...,784  (8)
    dict(id="digit7_100", cat="count", hard=False,
         q="How many times does the digit 7 appear when you write out every integer from 1 to 100 "
           "inclusive?", correct="20", alts=[], lure="11"),
    dict(id="div3or5_100", cat="count", hard=False,
         q="How many integers from 1 to 100 inclusive are divisible by 3 or by 5?",
         correct="47", alts=[], lure="53"),
    dict(id="decagon", cat="count", hard=False,
         q="How many diagonals does a convex decagon (a 10-sided polygon) have?",
         correct="35", alts=[], lure="45"),

    # --- C. Hard enumeration / arithmetic: AR single forward pass should usually fail
    dict(id="sq4000", cat="hard", hard=True,
         q="How many perfect square numbers are there strictly between 4000 and 8000?",
         correct="26", alts=[], lure="27"),   # 64^2..89^2 (26)
    dict(id="sq1000", cat="hard", hard=True,
         q="How many perfect square numbers are there strictly between 1000 and 5000?",
         correct="39", alts=[], lure="40"),   # 32^2..70^2 (39)
    dict(id="digit1_200", cat="hard", hard=True,
         q="How many times does the digit 1 appear when you write out every integer from 1 to 200 "
           "inclusive?", correct="140", alts=[], lure="120"),
    dict(id="mult4753", cat="hard", hard=True,
         q="What is 47 multiplied by 53?", correct="2491", alts=[], lure="2391"),
    dict(id="mult6389", cat="hard", hard=True,
         q="What is 63 multiplied by 89?", correct="5607", alts=[], lure="5670"),
    dict(id="div_count_3and5", cat="hard", hard=True,
         q="How many integers from 1 to 300 inclusive are divisible by both 3 and 5?",
         correct="20", alts=[], lure="60"),

    # --- D. Multi-hop / compute-then-transform: the paper's intermediate-context flavour
    dict(id="chain_sum_digits", cat="transform", hard=True,
         q="Add these four numbers: 23, 19, 31, 17. Then take the resulting total and add up its "
           "digits. What is that final digit sum?", correct="9", alts=[]),   # 90 -> 9
    dict(id="subtract_chain", cat="transform", hard=True,
         q="Start at 100. Subtract 7, then subtract 7 again, and keep subtracting 7 until you reach "
           "a number below 10. What is that final number?", correct="9", alts=[]),   # 100,93,...,16,9 (<10)
    dict(id="reverse_then_add", cat="transform", hard=True,
         q="Take the number 4821, reverse its digits to make a new number, then add the new number "
           "to the original. What is the sum?", correct="6105", alts=[]),   # 4821+1284=6105
    dict(id="word_count_letter", cat="transform", hard=False,
         q="How many times does the letter 's' appear in the sentence "
           "'She sells seashells by the seashore'? Count all of them, lowercase and uppercase.",
         correct="8", alts=[]),  # S,s,s,s,s(seashells has 2? sea-s-hell-s) -> She(1) sells(2) seashells(2) seashore(1)... =8
]

# {q} -> the problem text. Each framing fixes WHERE the answer sits relative to the reasoning,
# which is what moves the comparison along the serial-depth axis.
FRAMINGS = {
    "answer_only":     "{q}\n\nReply with ONLY the final answer, nothing else.",
    "answer_first":    "{q}\n\nState your final answer on the very first line, then give your reasoning.",
    "reasoning_first": "{q}\n\nReason step by step. Then on the last line write 'Answer: <your answer>'.",
}

# ---- sampling config (shared so DG and AR see byte-identical prompts) -------------------------
N_SEEDS = 8

# DiffusionGemma canvas is PER-FRAMING. The canvas IS DG's scratchpad/serial workspace, so its size
# must match what the framing asks for: a 256-canvas under a terse "answer only" prompt collapses to
# all-pad (the model predicts an empty content region) — a format artifact, not a capability signal.
# answer_only therefore uses a short canvas matched to the short output (still many denoising steps =
# real serial depth); the reasoning framings get the full trained canvas.
_DG_COMMON = dict(t_max=0.8, t_min=0.4, entropy_bound=0.1, top_k=2, enable_thinking=False)
DG_BY_FRAMING = {
    "answer_only":     dict(C=64,  T=64,  **_DG_COMMON),
    "answer_first":    dict(C=256, T=128, **_DG_COMMON),
    "reasoning_first": dict(C=256, T=128, **_DG_COMMON),
}

# Autoregressive Gemma-4: non-thinking, matched single response, temperature for cross-seed spread.
AR = dict(temperature=0.8, top_p=0.95, max_tokens=640, enable_thinking=False, max_model_len=2048)


def prompt_for(p, framing):
    return FRAMINGS[framing].format(q=p["q"])


def dg_cfg(framing):
    return DG_BY_FRAMING[framing]
