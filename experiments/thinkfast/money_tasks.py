"""Extra tasks for the wide-extent money plot — need serial depth tunable much DEEPER than the main
battery, with a SHORT prompt (so DG stays fast even at high step counts) and no recall confound.

serial_recur: V=0; for each digit d in the sequence, V = (2*V + d) mod 100. depth = # steps.
Strictly serial (V_t depends on V_{t-1}); each step is trivial; Gemma-with-scratchpad can trace any
depth; a single no-CoT pass (AR) or a single frozen canvas (DG) can only fold a bounded number of
steps. Arbitrary depth, ~one short line of prompt.
"""
import hashlib
import random
from battery import last_int  # reuse the integer extractor

MASTER = 20260701
MOD = 100


def _seed(task, depth, idx):
    return int(hashlib.sha256(f"{MASTER}|{task}|{depth}|{idx}".encode()).hexdigest()[:8], 16)


def gen_serial_recur(depth, rng):
    digits = [rng.randint(0, 9) for _ in range(depth)]
    V = 0
    for d in digits:
        V = (2 * V + d) % MOD
    seq = " ".join(map(str, digits))
    system = (f"You track a value V, starting at V=0. Reading a sequence of digits left to right, for each "
              f"digit d you update V = (2*V + d) mod {MOD}.")
    user = f"Digit sequence ({depth} digits): {seq}\n\nWhat is the final value of V?"
    fmt = ("This is a single-forward-pass evaluation. Respond immediately with your answer, before reasoning. "
           "Respond in exactly this format: The answer is: <integer>")
    return dict(system=system, user=user, fmt=fmt, answer=str(V),
                reason_force="\n\nTherefore, the final value of V is: ",
                check=lambda t, a=V: last_int(t) == a)


def gen_mult(depth, rng):
    """Pure arithmetic (Ryan Greenblatt no-CoT-math-horizon style): D-digit x D-digit multiplication.
    depth = D = operand digit count (difficulty/serial-depth proxy). Answer = the exact product."""
    lo, hi = 10 ** (depth - 1), 10 ** depth - 1
    a, b = rng.randint(lo, hi), rng.randint(lo, hi)
    ans = a * b
    system = "You are a calculator. Compute the exact product of the two integers."
    user = f"What is {a} × {b}?"
    fmt = ("This is a single-forward-pass evaluation. Respond immediately with your answer, before reasoning. "
           "Respond in exactly this format: The answer is: <integer>")
    return dict(system=system, user=user, fmt=fmt, answer=str(ans),
                reason_force="\n\nTherefore, the product is: ",
                check=lambda t, a=ans: last_int((t or "").replace(",", "").replace(" ", "")) == a)


GRID = {"serial_recur": [2, 4, 6, 8, 10, 12, 16, 20]}
GENERATORS = {"serial_recur": gen_serial_recur, "mult": gen_mult}
N_INSTANCES = 3


def build():
    P = {}
    for task, depths in GRID.items():
        for depth in depths:
            for idx in range(N_INSTANCES):
                rng = random.Random(_seed(task, depth, idx))
                p = GENERATORS[task](depth, rng)
                pid = f"{task}__d{depth}__{idx}"
                P[pid] = dict(id=pid, task=task, depth=depth, instance_idx=idx, **p)
    return P


def build_grid(depths, n_instances, task="serial_recur"):
    """Custom registry: task x depths x n_instances (deterministic). For heavily-sampled small grids."""
    P = {}
    for depth in depths:
        for idx in range(n_instances):
            rng = random.Random(_seed(task, depth, idx))
            p = GENERATORS[task](depth, rng)
            pid = f"{task}__d{depth}__{idx}"
            P[pid] = dict(id=pid, task=task, depth=depth, instance_idx=idx, **p)
    return P


BY_ID = build()
PROMPT = "{system}\n\n{user}\n\n{fmt}"


def prompt_for(p):
    return PROMPT.format(system=p["system"], user=p["user"], fmt=p["fmt"])


def check(pid, cand):
    try:
        return bool(BY_ID[pid]["check"](cand or ""))
    except Exception:
        return False


if __name__ == "__main__":
    for pid, p in BY_ID.items():
        assert p["check"](f"The answer is: {p['answer']}")
        assert not p["check"](f"The answer is: {(int(p['answer'])+1) % MOD}")
    print(f"{len(BY_ID)} serial_recur problems, self-test OK")
    ex = BY_ID["serial_recur__d8__0"]
    print(prompt_for(ex), "\n-> answer", ex["answer"])
