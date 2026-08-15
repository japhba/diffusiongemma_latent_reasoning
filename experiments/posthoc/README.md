# posthoc/ — post-hoc-justification suite (pod final state, n=40)

The exact suite (pod's final capture code) behind the vendored n=40 post-hoc data.
Worker-driven (set `DG_WORKER`, default `http://127.0.0.1:8711`), CPU locally.

## Files

- `suscept.py` — the three phases; resumable JSONs into `$DG_POSTHOC_DIR` (default
  `posthoc/out/`):
  - **clean**: answer-first rollouts; per-position lock-in localization
    (`lockin_and_localize`), answer-flip trajectory.
  - **suscept**: surgical-clamp susceptibility — pin the CoT positions, randomize a
    rho-fraction to real-but-wrong word pieces, let the free answer slot re-denoise;
    flat match(rho) = post-hoc, steep drop = load-bearing.
  - **counterfactual**: clamp a fluent lure CoT (`clamp_text`+`clamp_offset`), read the
    free front answer.
- `battery.py` — the 20-problem EXTENSION battery (2026-08-08); `battery20.py` — the
  original 20 problems (a staged copy of `../engels/battery.py`). n=40 = both.
- `corrupt.py` — corruption operators (`word_rand` primary; `shuffle`, `drop`).
- `ext_anim.py` / `ext_anim_batch.py` — entropy-resolution captures (one exemplar pair /
  whole n=40), writing `ext_anim.json` / `ext_anim_curves.json` into the cwd (run from
  this directory).
- `chain.sh` — the pod's actual chain: `suscept.py --phase both --n-clean 5
  --rhos 0,0.25,0.5,0.75,1.0 --corr-seeds 5`, then the anchor follow-up if present.
- `gen_lure_cots.py` — generates the counterfactual lure CoTs (sequential Anthropic calls;
  `ANTHROPIC_API_KEY` via env/.env) into `$DG_LURE_COTS` (default `posthoc/lure_cots.json`,
  read back by `suscept.py --phase counterfactual`).
- `ext_difficulty.json` — per-problem difficulty for the extension battery: **3 blind LLM
  raters, 0-1, from the problem text (+ correct answer) only** — no model behavior, no
  category labels shown; there is no generating script (protocol documented in the file's
  `note` field).

## Protocol constants

- `GRID = dict(C=256, T=128, top_k=3, t_max=0.9, t_min=0.5, entropy_bound=0.15,
  enable_thinking=False)` — all phases, all captures.
- `ANSWER_FIRST` framing: "State your final answer on the very first line, then give your
  reasoning."
- **Scaffold-parsing fix**: DG-it (pod state since 2026-08-04) opens every canvas with
  `<|channel>thought\n<channel|>`; `strip_scaffold` treats that scaffold as neither answer
  nor CoT (old-format texts pass through unchanged). Lock-in localization skips it too.

## Run

```bash
bash chain.sh                                   # full clean+suscept battery
python3 suscept.py --phase counterfactual       # after gen_lure_cots.py
python3 ext_anim_batch.py                       # entropy curves for the whole n=40
```

## Feeds

`src_data/posthoc/{clean,suscept,counterfactual,anim_curves,difficulty}.json`
(via `scripts/extract_posthoc_ext.py`).
