"""Full-grid film capture (scope C, 2026-07-15): for EVERY battery cell (task x depth x T rung),
capture ONE companion rollout per instance via /sample — per-step argmax + per-position entropy —
for the popup denoising animation (viridis entropy shading).

COMPANION rollouts: /sample is single-row, the graded grid was batched (/rollouts, N=8), and RNG
consumption differs — so these are FRESH rollouts under the SAME conditions (TEMPS[mid], same
C/T/prompts), graded on the fly and labeled 'companion film (fresh seed)' in the UI. Seeds 1000+idx.

Output: one JSON per cell at exp/.../thinkfast/films/<task>__d<d>__T<T>.json
  {task, depth, T, C, rolls: [{idx, seed, ok, text, ids, film[[T x C argmax]], ent100[[T x C]]}],
   id2str} — ent100 = entropy in centinats (int) to keep files small. Resumable (skips existing
   files). ~2200 samples, ~2-4 h through the Node V tunnel.
"""
import functools, json, os, sys, time, urllib.request
print = functools.partial(print, flush=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from battery import BY_ID, GENERATORS, TASK_BUDGET, TEMPS, prompt_for  # noqa: E402

WORKER = os.environ.get("DG_WORKER", "http://127.0.0.1:8711")
STEPS_T = [1, 2, 4, 8, 16, 32, 64, 128, 256]
D = os.environ.get("DG_FILMS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "films"))
INSTANCES = 4


def sample(prompt, C, T, seed):
    body = dict(prompt=prompt, T=T, seed=seed, C=C, max_new_tokens=C, top_k=1,
                enable_thinking=False, **TEMPS["mid"])
    req = urllib.request.Request(WORKER + "/sample", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=3000))


def main():
    os.makedirs(D, exist_ok=True)
    cells = [(t, d, T) for t, (_, depths) in GENERATORS.items() for d in depths for T in STEPS_T]
    todo = [c for c in cells if not os.path.exists(f"{D}/{c[0]}__d{c[1]}__T{c[2]}.json")]
    print(f"[films] {len(todo)}/{len(cells)} cells to capture ({INSTANCES} companion rollouts each)")
    t00 = time.time()
    for n, (task, depth, T) in enumerate(cells):
        path = f"{D}/{task}__d{depth}__T{T}.json"
        if os.path.exists(path):
            continue
        C = TASK_BUDGET[task][0]
        rolls, id2str = [], {}
        for idx in range(INSTANCES):
            p = BY_ID[f"{task}__d{depth}__{idx}"]
            r = sample(prompt_for(p), C, T, 1000 + idx)
            id2str.update(r["id2str"])
            rolls.append(dict(
                idx=idx, seed=1000 + idx, ok=bool(p["check"](r["final_text"])),
                text=r["final_text"], ids=r["final_ids"],
                film=[s["argmax"] for s in r["steps"]],
                ent100=[[int(round(e * 100)) for e in s["entropy"]] for s in r["steps"]]))
        json.dump(dict(task=task, depth=depth, T=T, C=C, rolls=rolls, id2str=id2str),
                  open(path, "w"))
        done = sum(os.path.exists(f"{D}/{c[0]}__d{c[1]}__T{c[2]}.json") for c in cells)
        el = (time.time() - t00) / 60
        print(f"[{done}/{len(cells)}] {task}__d{depth}__T{T} ({el:.0f}m elapsed)")
    print(f"FILMS_DONE -> {D}")


if __name__ == "__main__":
    main()
