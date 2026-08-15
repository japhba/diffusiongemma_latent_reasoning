"""Battery for NON-TRANSLATION letter operations (same paired one-step S^t probe as
xtask_uu_10x / xtask_mult, so NE and placebo-corrected specificity are directly comparable
across every task family on this page).

Usage:  python3 xtask_ops.py RF MM3 KB      (op tags; default = all that probed GO)

Each op supplies a prompt, an operand pool, and an image map x -> x'. The map need not be a
translation; what the analysis needs is the INVERSE map (the preimage of an output), which is
emitted into the json as `preimage` so the report can build map rows generically instead of
hardcoding x'-k / x'/k arithmetic per family.
-> exp/dg_planning/xtask_ops{,_nsweep}.json
"""
import os
import json, string, sys, time, urllib.request
from pathlib import Path

import numpy as np

W = os.environ.get("DG_WORKER", "http://localhost:18711")
EXP = Path(os.environ.get("DG_PLANNING_DIR", str(Path(__file__).resolve().parent / "exp")))
FRAME = "{q}\n\nOutput ONLY the text itself — no preamble, no quotes, no explanation."
HOTR = dict(T=64, C=128, t_max=1.3, t_min=0.8, entropy_bound=0.3, early_stop=False, top_k=10)
VOCAB = 262144
DRAWS = 8
T = 2
NREP = 3
NLVL = [1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18]
EPS0 = 0.04
EPSM = [float(x) for x in __import__("os").environ.get("OPS_EPS", "0.45").split(",")]
SEEDS = list(range(int(__import__("os").environ.get("OPS_SEEDS", "10"))))

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("google/gemma-4-26b-a4b-it")
def tid(s):
    t = tok.encode(s, add_special_tokens=False)
    assert len(t) == 1
    return t[0]

LOW = list(string.ascii_lowercase); UPP = list(string.ascii_uppercase)
F52 = LOW + UPP
ROWS = ["QWERTYUIOP", "ASDFGHJKL", "ZXCVBNM"]
KBD = {r[i]: r[i + 1] for r in ROWS for i in range(len(r) - 1)}

OPS = {
    "RF": dict(  # affine slope -1: NOT a translation, but full pool + bijective + dense image set
        pool=UPP, f=lambda c: UPP[25 - UPP.index(c)],
        q="Pick any uppercase letter from A to Z, write it, then write the uppercase letter at "
          "the mirrored position in the alphabet (A pairs with Z, B with Y, C with X, and so on), "
          "separated by a comma. Begin your answer with 'Letters:'."),
    "MM3": dict(  # multiplication without xk's confounds: 3 coprime with 26 => bijection, no lattice
        pool=UPP, f=lambda c: UPP[((UPP.index(c) + 1) * 3 - 1) % 26],
        q="Pick any uppercase letter from A to Z, write it, then multiply its alphabet index by 3 "
          "and, if the result is larger than 26, keep subtracting 26 until it lies between 1 and 26; "
          "write the uppercase letter at that index, separated by a comma. (A=1, B=2, ..., Z=26.) "
          "Begin your answer with 'Letters:'."),
    "KB": dict(  # a translation in a DIFFERENT ordering (keyboard, not alphabet)
        pool=sorted(KBD), f=lambda c: KBD.get(c),
        q="Pick any uppercase letter that is not P, L or M, write it, then write the uppercase "
          "letter immediately to its right on a standard QWERTY keyboard, separated by a comma. "
          "Begin your answer with 'Letters:'."),
    "CP": dict(  # identity: ceiling anchor
        pool=UPP, f=lambda c: c,
        q="Pick any uppercase letter from A to Z, write it, then write the same letter again, "
          "separated by a comma. Begin your answer with 'Letters:'."),
    # Phrasing variants. The sheet SEED barely moves the T=2 draft (all 10 seeds of an op can
    # share one draft), so seeds are near-replicates and per-cell error bars pseudo-replicate.
    # Re-wording the same operation DOES produce a different canvas, which is how this study gets
    # genuinely independent states. Same map as RF/KB, so they pool as extra states of that op.
    "RFB": dict(
        pool=UPP, f=lambda c: UPP[25 - UPP.index(c)],
        q="Pick any uppercase letter from A to Z, write it, then write the letter that is the same "
          "distance from the END of the alphabet as your letter is from the start, in uppercase, "
          "separated by a comma. Begin your answer with 'Letters:'."),
    "RFC": dict(
        pool=UPP, f=lambda c: UPP[25 - UPP.index(c)],
        q="Pick any uppercase letter from A to Z, write it, then reverse the alphabet (so A becomes "
          "Z, B becomes Y, C becomes X) and write what your letter becomes, separated by a comma. "
          "Begin your answer with 'Letters:'."),
    "KBB": dict(
        pool=sorted(KBD), f=lambda c: KBD.get(c),
        q="Pick any uppercase letter that is not P, L or M, write it, then write the uppercase "
          "letter you would type next if you moved one key to the right on a QWERTY keyboard, "
          "separated by a comma. Begin your answer with 'Letters:'."),
    # Pool-restricted variants. Re-wording alone does NOT change the committed answer (every
    # phrasing of reflection yields "G, T", so the T=2 draft coincides); restricting the pool
    # forces a DIFFERENT natural operand, which is how the +k family got its state diversity
    # (different k => different answer => different draft).
    "RFD": dict(
        pool=[c for c in UPP if c >= "N"], f=lambda c: UPP[25 - UPP.index(c)],
        q="Pick any uppercase letter from N to Z, write it, then write the uppercase letter at "
          "the mirrored position in the alphabet (A pairs with Z, B with Y, C with X, and so on), "
          "separated by a comma. Begin your answer with 'Letters:'."),
    "RFE": dict(
        pool=[c for c in UPP if "F" <= c <= "R"], f=lambda c: UPP[25 - UPP.index(c)],
        q="Pick any uppercase letter from F to R, write it, then write the uppercase letter at "
          "the mirrored position in the alphabet (A pairs with Z, B with Y, C with X, and so on), "
          "separated by a comma. Begin your answer with 'Letters:'."),
    "MN3": dict(  # letters subtraction: translation control (only the numbers had one)
        pool=[c for c in UPP if UPP.index(c) >= 3], f=lambda c: UPP[UPP.index(c) - 3],
        q="Pick any uppercase letter between D and Z, write it, then write the letter three "
          "positions earlier in the alphabet, also in uppercase, separated by a comma. "
          "Begin your answer with 'Letters:'."),
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


def main():
    tags = [t for t in sys.argv[1:] if t in OPS] or list(OPS)
    fop, fns = EXP / "xtask_ops.json", EXP / "xtask_ops_nsweep.json"
    dop = json.load(open(fop)) if fop.exists() else {}
    dns = json.load(open(fns)) if fns.exists() else {}
    temp = 1.3 + (0.8 - 1.3) * T / 63
    ids = {w: tid(" " + w) for w in F52}
    probe = sorted(set(ids.values()))
    pix = {v: i for i, v in enumerate(probe)}
    ncap = nsw = nmp = nskip = 0
    print(f"ops: {tags}", flush=True)
    for tag in tags:
        cfg = OPS[tag]
        pool = cfg["pool"]
        # emit the inverse map once per op: for each of the 52 field letters, the pool letter
        # whose image it is (None => 'other' row in the report's transfer map)
        inv = {}
        for w in pool:
            im = cfg["f"](w)
            if im:
                inv[im] = w
        dop[f"{tag}|meta"] = dict(pool=pool, prompt=cfg["q"],
                                  image={w: cfg["f"](w) for w in pool},
                                  preimage=[inv.get(c) for c in F52])
        json.dump(dop, open(fop, "w"))
        for s in SEEDS:
            fp = EXP / f"nego2/{tag}__s{s}.json"
            if not fp.exists():
                d = post("sample", dict(prompt=FRAME.format(q=cfg["q"]), seed=s, **HOTR, s_topk_record=32))
                slim = {kk: d[kk] for kk in ("final_ids", "final_text", "pad_token_id", "eos_token_ids",
                                             "id2str", "canvas_length", "s_rec")}
                slim["steps_argmax"] = [st["argmax"] for st in d["steps"]]
                slim["tag"] = tag; slim["seed"] = s; slim["q"] = cfg["q"]
                json.dump(slim, open(fp, "w")); ncap += 1
                print(f"cap {tag} s{s}: {d['final_text'].splitlines()[-1][:70]!r}", flush=True)
            d = json.load(open(fp))
            i2s, fin = d["id2str"], d["final_ids"]
            dead = set(d["eos_token_ids"]) | {d["pad_token_id"]}
            live = [p for p, x in enumerate(fin) if x not in dead]
            txt = lambda x: i2s.get(str(x), "?").replace("▁", " ")
            pc = next((p for p in live if txt(fin[p]).strip() == ","), None)
            if pc is None:
                print(f"{tag} s{s}: no comma — SKIP", flush=True); continue
            A = max(p for p in live if p < pc)
            B = next(p for p in live if p > pc)
            nat = txt(fin[A]).strip()
            if nat not in pool:
                print(f"{tag} s{s}: nat {nat!r} outside pool — SKIP", flush=True); continue
            draft = d["steps_argmax"][T]
            sheet0 = {"ids": d["s_rec"]["ids"][T], "lp": d["s_rec"]["lp"][T]}
            rng0 = np.random.default_rng(83 + s)
            noise = rng0.integers(0, VOCAB, size=(DRAWS, 2))
            canvases = []
            for dd in range(DRAWS):
                cv = list(draft)
                cv[A] = int(noise[dd][0]); cv[B] = int(noise[dd][1])
                canvases.append(cv)

            def brows(sheet):
                rows = []
                for i in range(0, DRAWS, 2):
                    rows += post("energy", {"prompt": FRAME.format(q=cfg["q"]),
                                            "canvases": canvases[i:i + 2], "probe_ids": probe,
                                            "s_sparse": sheet, "temperature": temp})["probe"]
                return [[round(float(np.exp(r[B][pix[ids[w]]])), 6) for w in F52] for r in rows]

            bk = f"{tag}|s{s}|base"
            if bk not in dop:
                dop[bk] = dict(nat=nat, A=A, B=B, rows=brows(sheet0))
                json.dump(dop, open(fop, "w"))
            if bk not in dns:
                dns[bk] = dict(nat_op=nat, A=A, B=B, rows=dop[bk]["rows"])
                json.dump(dns, open(fns, "w"))
            ops_ = [o for o in pool if o != nat]

            def inject(subset, eps):
                ids2 = [list(r) for r in sheet0["ids"]]
                lp2 = [list(r) for r in sheet0["lp"]]
                p = np.exp(np.array(lp2[A], dtype=float)) * (1.0 - eps * len(subset))
                row = list(ids2[A])
                for w in subset:
                    tk_ = ids[w]
                    if tk_ in row:
                        p[row.index(tk_)] += eps
                    else:
                        j = int(np.argmin(p)); row[j] = tk_; p[j] = eps
                ranks = {w: int(np.sum(p > p[row.index(ids[w])])) for w in subset}
                ids2[A] = row
                lp2[A] = [float(x) for x in np.log(np.maximum(p, 1e-12))]
                return {"ids": ids2, "lp": lp2}, ranks

            for n in [x for x in NLVL if x <= len(ops_)]:
                seedstr = f"{tag}|{s}|opn|{n}"
                rng = np.random.default_rng(sum(ord(c) * (i + 7) for i, c in enumerate(seedstr)))
                for rep in range(NREP):
                    key = f"{tag}|s{s}|ext|n{n}|e{rep}"
                    if key in dns:
                        continue
                    subset = list(rng.choice(ops_, size=n, replace=False))
                    sheet, ranks = inject(subset, EPS0)
                    dns[key] = dict(subset=subset, ranks=ranks, rows=brows(sheet))
                    json.dump(dns, open(fns, "w")); nsw += 1
            for eps in EPSM:
                for b in ops_:
                    key = f"{tag}|s{s}|e{eps}|b1|{b}"
                    if key in dop:
                        continue
                    sheet, ranks = inject([b], eps)
                    if ranks[b] == 0:
                        dop[key] = dict(skipped=True); json.dump(dop, open(fop, "w")); nskip += 1; continue
                    dop[key] = dict(eps0=eps, subset=[b], ranks=ranks, rows=brows(sheet))
                    json.dump(dop, open(fop, "w")); nmp += 1
            print(f"{tag} s{s}: done (nat {nat}, {len(ops_)} ops) [nsweep {nsw}, map {nmp}, rank-0 {nskip}]",
                  flush=True)
    print(f"OPS DONE: {ncap} captures, {nsw} nsweep cells, {nmp} map cells, {nskip} rank-0", flush=True)


if __name__ == "__main__":
    main()
