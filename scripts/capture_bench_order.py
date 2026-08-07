"""Capture DG denoising trajectories for classic-benchmark prompts (GPQA / MATH / HumanEval /
WildChat, 12 each; default sampler T=48 t 0.8->0.4 eb 0.1, C=256, thinking off) and reduce to
per-position final-commit steps (argmax lock-in).

NOT bare-clone-rerunnable: runs ON the DG-worker pod against localhost:8711 (captured
2026-08-07, worker /workspace/dg/server.py); jobs built from commit_ds problem files
(bench_capture_jobs.json). Output vendored as src_data/planning/bench_order.json."""
import json, urllib.request, time

W = "http://localhost:8711"
jobs = json.load(open("/root/bench_capture_jobs.json"))
GRID = dict(T=48, C=256, t_max=0.8, t_min=0.4, entropy_bound=0.1, top_k=3,
            enable_thinking=False, early_stop=False)
out = []
for i, j in enumerate(jobs):
    t0 = time.time()
    req = urllib.request.Request(f"{W}/sample",
                                 json.dumps(dict(prompt=j["prompt"], seed=17, **GRID)).encode(),
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
    out.append(dict(bench=j["bench"], pid=j["pid"], T=T,
                    content=content, lock=[lock(p) for p in content],
                    final_text=d["final_text"][:300]))
    print(f"{i+1}/{len(jobs)} {j['bench']}/{j['pid']} n={len(content)} {time.time()-t0:.0f}s", flush=True)
    json.dump(out, open("/root/bench_order.json", "w"))
print("DONE")
