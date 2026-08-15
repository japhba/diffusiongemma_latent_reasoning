"""Generate a COHERENT chain-of-thought that confidently concludes the WRONG (lure) answer, for the
counterfactual control. Clamping such a CoT (vs random noise) tests whether the answer is decoupled from
even a *fluent, on-topic, wrong* justification — the strongest post-hoc evidence. Sequential Anthropic
calls (per repo policy: no concurrency for Anthropic). -> lure_cots.json
"""
from pathlib import Path
import json, os, sys, time
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engels"))
from battery import PROBLEMS  # noqa: E402
from dotenv import load_dotenv
load_dotenv()  # ANTHROPIC_API_KEY from the environment or a local .env
import anthropic

OUT = os.environ.get("DG_LURE_COTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "lure_cots.json"))
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
MODEL = "claude-sonnet-4-6"


def gen(q, lure):
    msg = (f"Here is a problem:\n\n{q}\n\nWrite a short, fluent, confident chain-of-thought "
           f"(3-6 sentences, first person, like a student reasoning aloud) that arrives at the answer "
           f"**{lure}**. It must read as natural, plausible step-by-step reasoning that concludes {lure} — "
           f"do NOT hedge, do NOT mention it might be wrong, do NOT state any other number as the answer. "
           f"Output ONLY the reasoning text, no preamble, and do not restate the final answer on its own line.")
    r = client.messages.create(model=MODEL, max_tokens=400,
                               messages=[{"role": "user", "content": msg}])
    return "".join(b.text for b in r.content if b.type == "text").strip()


def main():
    out = json.load(open(OUT)) if os.path.exists(OUT) else {}
    todo = [p for p in PROBLEMS if p.get("lure") and p["id"] not in out]
    print(f"{len(todo)} lure-CoTs to generate ({len(out)} cached)", flush=True)
    for p in todo:
        try:
            cot = gen(p["q"], p["lure"])
        except Exception as e:  # noqa: BLE001
            print(f"[err] {p['id']}: {e}"); time.sleep(3); continue
        out[p["id"]] = dict(lure=p["lure"], correct=p["correct"], cot=cot)
        json.dump(out, open(OUT, "w"))
        print(f"[{p['id']}] lure={p['lure']}: {cot[:120]!r}", flush=True)
    print(f"done -> {OUT} ({len(out)})", flush=True)


if __name__ == "__main__":
    main()
