"""Capture DG rollouts for poem prompts (direction-indifferent panel of figA11).

Runs against the DG worker (experiments/worker/server.py; DG_WORKER env or localhost:8711,
originally captured 2026-08-07 on the worker pod, same sampler as capture_bench_order.py).
Output vendored as src_data/planning/poem_order.json."""
import json, os, urllib.request, time
from pathlib import Path
W = os.environ.get("DG_WORKER", "http://localhost:8711")
TOPICS = ["the sea", "autumn leaves", "a city at night", "an old friendship", "a thunderstorm",
          "morning coffee", "distant mountains", "a lost key", "the full moon", "night trains",
          "a walled garden", "the passage of time"]
GRID = dict(T=48, C=256, t_max=0.8, t_min=0.4, entropy_bound=0.1, top_k=3,
            enable_thinking=False, early_stop=False)
out = []
for i, topic in enumerate(TOPICS):
    prompt = f"Write a short rhyming poem (four lines) about {topic}."
    req = urllib.request.Request(f"{W}/sample",
                                 json.dumps(dict(prompt=prompt, seed=17, **GRID)).encode(),
                                 {"Content-Type": "application/json"})
    d = json.loads(urllib.request.urlopen(req, timeout=1800).read())
    steps = [s["argmax"] for s in d["steps"]]
    dead = set(d.get("eos_token_ids", [1, 106])) | {d.get("pad_token_id", 0)}
    fin = d["final_ids"]
    content = [p for p, x in enumerate(fin) if x not in dead]
    T = len(steps)
    def lock(p):
        t = T - 1
        while t > 0 and steps[t][p] == steps[t - 1][p]:
            t -= 1
        return t
    out.append(dict(bench="poem", pid=f"poem_{i:02d}", T=T, content=content,
                    lock=[lock(p) for p in content], final_text=d["final_text"][:300]))
    print(f"{i+1}/12 {topic} n={len(content)}", flush=True)
    json.dump(out, open(Path(__file__).resolve().parent.parent / "src_data" / "planning" / "poem_order.json", "w"))
print("DONE")
