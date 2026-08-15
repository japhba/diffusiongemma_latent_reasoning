"""Capability probe for the MULTIPLICATIVE uppercase-letter task: image = the letter whose
1-indexed alphabet position is k TIMES the operand's (A=1 ... Z=26), vs the additive shift used
so far. Two prompt phrasings x 3 multipliers x 6 seeds; report format compliance and arithmetic
correctness. Go/no-go for a full battery: if DG cannot compute x*k natively, a null diagonal
under injection would be uninterpretable. -> exp/dg_planning/xtask_mult_probe.json"""
import os
import json, string, time, urllib.request
from pathlib import Path

W = os.environ.get("DG_WORKER", "http://localhost:18711")
EXP = Path(os.environ.get("DG_PLANNING_DIR", str(Path(__file__).resolve().parent / "exp")))
FRAME = "{q}\n\nOutput ONLY the text itself — no preamble, no quotes, no explanation."
HOTR = dict(T=64, C=128, t_max=1.3, t_min=0.8, entropy_bound=0.3, early_stop=False, top_k=10)
UPP = list(string.ascii_uppercase)
SEEDS = list(range(6))
WORD = {2: "twice", 3: "three times", 4: "four times", 5: "five times"}

# pools: operand position i must satisfy i*k <= 26
POOLS = {k: [UPP[i - 1] for i in range(2, 27) if i * k <= 26] for k in (2, 3, 4, 5)}

PHRASINGS = {
    # A: explicit "position in the alphabet" arithmetic
    "pos": ("Pick any uppercase letter from A to {hi}, write it, then write the uppercase letter "
            "whose position in the alphabet is exactly {word} the position of the letter you picked, "
            "separated by a comma. (A=1, B=2, ..., Z=26.) Begin your answer with 'Letters:'."),
    # B: terser, no worked key
    "mul": ("Pick any uppercase letter from A to {hi}, write it, then multiply its alphabet index "
            "by {k} and write the uppercase letter at that index, separated by a comma. "
            "Begin your answer with 'Letters:'."),
}


def post(path, req, timeout=1800):
    for a in range(6):
        try:
            r = urllib.request.urlopen(urllib.request.Request(
                f"{W}/{path}", json.dumps(req).encode(), {"Content-Type": "application/json"}), timeout=timeout)
            return json.loads(r.read())
        except Exception as e:
            print(f"  retry {a}: {type(e).__name__}", flush=True)
            time.sleep(15 * (a + 1))
    raise RuntimeError("worker unreachable")


def parse(text):
    """-> (operand, answer) uppercase single letters, or None."""
    line = text.splitlines()[-1]
    if "Letters:" in line:
        line = line.split("Letters:")[-1]
    parts = [p.strip().strip(".").strip() for p in line.split(",")]
    if len(parts) < 2:
        return None
    a, b = parts[0], parts[1]
    if len(a) != 1 or len(b) != 1 or a not in UPP or b not in UPP:
        return None
    return a, b


def main():
    out, f = {}, EXP / "xtask_mult_probe.json"
    if f.exists():
        out = json.load(open(f))
    for ph, tmpl in PHRASINGS.items():
        for k in (2, 3, 4):
            pool = POOLS[k]
            q = tmpl.format(hi=pool[-1], word=WORD[k], k=k)
            ok = comply = 0
            for s in SEEDS:
                key = f"{ph}|k{k}|s{s}"
                if key not in out:
                    d = post("sample", dict(prompt=FRAME.format(q=q), seed=s, **HOTR))
                    out[key] = dict(q=q, text=d["final_text"].split("<channel|>")[-1].strip())
                    json.dump(out, open(f, "w"))
                pr = parse(out[key]["text"])
                out[key]["parsed"] = pr
                if pr:
                    comply += 1
                    i = UPP.index(pr[0]) + 1
                    good = i * k <= 26 and UPP[i * k - 1] == pr[1]
                    out[key]["correct"] = bool(good)
                    ok += good
                    print(f"  {ph} k={k} s{s}: {out[key]['text']!r} -> {pr[0]}({i}) x{k} = "
                          f"{UPP[i*k-1] if i*k<=26 else 'OOR'}, model said {pr[1]}  {'OK' if good else 'WRONG'}",
                          flush=True)
                else:
                    out[key]["correct"] = None
                    print(f"  {ph} k={k} s{s}: {out[key]['text']!r}  UNPARSEABLE", flush=True)
            json.dump(out, open(f, "w"))
            print(f"{ph} k={k} (pool {pool[0]}..{pool[-1]}, {len(pool)}): "
                  f"parsed {comply}/{len(SEEDS)}, correct {ok}/{len(SEEDS)}", flush=True)
    print("MULT PROBE DONE", flush=True)


if __name__ == "__main__":
    main()
