"""Capability probe for NON-TRANSLATION letter operations, to separate two readings of the
+k-transfers / xk-doesn't result: (E1) S^t carries the operand in a representation where
TRANSLATION specifically is cheap, vs (E2) harder computations simply transfer worse.

Operations (all UPPER->UPPER, one comma-separated pair, same frame as the +k / xk tasks):
  copy   identity                      -- ceiling anchor
  minus  x' = x-3 (translation)        -- letters subtraction control (only numbers had one)
  refl   x' = 27-pos(x), A<->Z         -- AFFINE slope -1: not a translation, but full 26-letter
                                          pool, bijective, image set = whole alphabet (so unlike
                                          xk there is no sparse image lattice to prime)
  mod3   x' = 3*pos(x) mod 26          -- multiplication WITHOUT xk's confounds (3 is coprime
                                          with 26 => bijection, full pool, no lattice)
  kbd    next key right on QWERTY      -- a TRANSLATION in a different ordering: discriminates
                                          "successor in any learned order" from "alphabet index"
-> exp/dg_planning/xtask_ops_probe.json"""
import os
import json, string, time, urllib.request
from pathlib import Path

W = os.environ.get("DG_WORKER", "http://localhost:18711")
EXP = Path(os.environ.get("DG_PLANNING_DIR", str(Path(__file__).resolve().parent / "exp")))
FRAME = "{q}\n\nOutput ONLY the text itself — no preamble, no quotes, no explanation."
HOTR = dict(T=64, C=128, t_max=1.3, t_min=0.8, entropy_bound=0.3, early_stop=False, top_k=10)
UPP = list(string.ascii_uppercase)
SEEDS = list(range(8))
ROWS = ["QWERTYUIOP", "ASDFGHJKL", "ZXCVBNM"]
KBD = {r[i]: r[i + 1] for r in ROWS for i in range(len(r) - 1)}

OPS = {
    "copy": dict(
        pool=UPP,
        q="Pick any uppercase letter from A to Z, write it, then write the same letter again, "
          "separated by a comma. Begin your answer with 'Letters:'.",
        f=lambda c: c),
    "minus": dict(
        pool=[c for c in UPP if UPP.index(c) >= 3],
        q="Pick any uppercase letter between D and Z, write it, then write the letter three "
          "positions earlier in the alphabet, also in uppercase, separated by a comma. "
          "Begin your answer with 'Letters:'.",
        f=lambda c: UPP[UPP.index(c) - 3]),
    "refl": dict(
        pool=UPP,
        q="Pick any uppercase letter from A to Z, write it, then write the uppercase letter at "
          "the mirrored position in the alphabet (A pairs with Z, B with Y, C with X, and so on), "
          "separated by a comma. Begin your answer with 'Letters:'.",
        f=lambda c: UPP[25 - UPP.index(c)]),
    "mod3": dict(
        pool=UPP,
        q="Pick any uppercase letter from A to Z, write it, then multiply its alphabet index by 3 "
          "and, if the result is larger than 26, keep subtracting 26 until it lies between 1 and 26; "
          "write the uppercase letter at that index, separated by a comma. (A=1, B=2, ..., Z=26.) "
          "Begin your answer with 'Letters:'.",
        f=lambda c: UPP[((UPP.index(c) + 1) * 3 - 1) % 26]),
    "kbd": dict(
        pool=sorted(KBD),
        q="Pick any uppercase letter that is not P, L or M, write it, then write the uppercase "
          "letter immediately to its right on a standard QWERTY keyboard, separated by a comma. "
          "Begin your answer with 'Letters:'.",
        f=lambda c: KBD.get(c)),
}


def post(path, req, timeout=1800):
    for a in range(8):
        try:
            r = urllib.request.urlopen(urllib.request.Request(
                f"{W}/{path}", json.dumps(req).encode(), {"Content-Type": "application/json"}), timeout=timeout)
            return json.loads(r.read())
        except Exception as e:
            print(f"  retry {a}: {type(e).__name__}", flush=True)
            time.sleep(20 * (a + 1))
    raise RuntimeError("worker unreachable")


def parse(text):
    line = text.splitlines()[-1]
    if "Letters:" in line:
        line = line.split("Letters:")[-1]
    parts = [p.strip().strip(".").strip() for p in line.split(",")]
    if len(parts) < 2 or any(len(p) != 1 or p not in UPP for p in parts[:2]):
        return None
    return parts[0], parts[1]


def main():
    f = EXP / "xtask_ops_probe.json"
    out = json.load(open(f)) if f.exists() else {}
    summary = {}
    for op, cfg in OPS.items():
        ok = comply = inpool = 0
        for s in SEEDS:
            key = f"{op}|s{s}"
            if key not in out:
                d = post("sample", dict(prompt=FRAME.format(q=cfg["q"]), seed=s, **HOTR))
                out[key] = dict(q=cfg["q"], text=d["final_text"].split("<channel|>")[-1].strip())
                json.dump(out, open(f, "w"))
            pr = parse(out[key]["text"])
            out[key]["parsed"] = pr
            if pr:
                comply += 1
                good = pr[0] in cfg["pool"] and cfg["f"](pr[0]) == pr[1]
                inpool += pr[0] in cfg["pool"]
                out[key]["correct"] = bool(good)
                ok += good
                print(f"  {op:6s} s{s}: {out[key]['text']!r} -> {pr[0]} => expect "
                      f"{cfg['f'](pr[0]) if pr[0] in cfg['pool'] else 'OUT-OF-POOL'}, got {pr[1]}  "
                      f"{'OK' if good else 'WRONG'}", flush=True)
            else:
                out[key]["correct"] = None
                print(f"  {op:6s} s{s}: {out[key]['text']!r}  UNPARSEABLE", flush=True)
        json.dump(out, open(f, "w"))
        summary[op] = (comply, inpool, ok, len(SEEDS))
        print(f"{op}: parsed {comply}/{len(SEEDS)}, in-pool {inpool}/{len(SEEDS)}, "
              f"correct {ok}/{len(SEEDS)}", flush=True)
    print("\n=== OPS PROBE SUMMARY (correct/total) ===", flush=True)
    for op, (c, ip, k, n) in summary.items():
        verdict = "GO" if k >= 0.75 * n else ("MARGINAL" if k >= 0.4 * n else "NO-GO")
        print(f"  {op:6s} parsed {c}/{n}  correct {k}/{n}  -> {verdict}", flush=True)
    print("OPS PROBE DONE", flush=True)


if __name__ == "__main__":
    main()
