# Experiments

The code that generated every study output vendored under `../src_data/`. The blog figures
regenerate on CPU from `src_data/` alone (`../README.md`); this tree is for rerunning the
experiments themselves. Directory names mirror the original research repos.

## The two execution modes

- **Worker-driven** (planning, thinkfast, posthoc, and the `scripts/capture_*` jobs): a Flask
  worker (`worker/server.py`) holds `google/diffusiongemma-26B-A4B-it` on one GPU and exposes
  `/sample`, `/energy`, etc.; capture scripts are CPU clients that POST to it
  (`DG_WORKER` env, default `http://localhost:8711`; the planning captures default to
  `:18711`, the historical SSH-tunnel port — `ssh -N -L 18711:localhost:8711 <gpu-host>`).
- **Direct-GPU** (lockin, transfer): the scripts load the model(s) themselves.

Every hardcoded path from the original environment has been replaced by an env var with an
in-tree default; running a script with no env set writes under its own directory
(`data/`, `exp/`, `figs/`, `out/`, `films/` — gitignored).

## Map: post section → experiment → vendored data

| post section | experiment | key entry point | feeds |
|---|---|---|---|
| top-k truncation ladder | `lockin/` | `MODES=... python ds_ablate_bench.py`, `STAB_MODES=... python ds_ablate_stab.py`, `ds_paper_sweep.py` | `src_data/commit_ds/` |
| letter arithmetic (transfer map, example) | `planning/` | `xtask_samecase.py` → `xtask_samecase_nsweep.py` → `build_superpos.py` | `src_data/symbol_arithmetic_payload.json` |
| parallel computation | `planning/` | `xtask_par3.py` → `build_par_ladderfrac.py` | `src_data/planning/xtask_par3.json` |
| autonomous usage (seasonal/idiom) | `planning/` | `ember_kill2.py`, `ember_preserve_fig.py` | `src_data/ember_*.json` |
| commit causality (reverse_chain) | `thinkfast/` | `DG_WORKER=... python grid_films.py` | films → `scripts/extract_films_order.py` |
| commit causality (GPQA / poem) | `../scripts/` | `capture_bench_order.py`, `capture_poem_order.py`, `judge_logical_order.py` | `src_data/planning/{bench,poem}_order.json` |
| post-hoc vs load-bearing CoT (appendix) | `posthoc/` | `chain.sh` (= `suscept.py --phase both ...`), `ext_anim_batch.py` | `src_data/posthoc/` |
| RSA / probes / steering / J-Lens | `transfer/` | see `transfer/README.md` | `src_data/saeprobes/`, `src_data/jlens_future_rows.json` |

`constrained/` and `engels/` hold the task batteries the above import (palindrome/ember
instances; the post-hoc n=20 problem set).

## Requirements

- **Models** (HF, gated — accept terms + `HF_TOKEN`): `google/diffusiongemma-26B-A4B-it`
  (worker + all DG runs), `google/gemma-4-26b-a4b-it` (transfer suite counterpart; its
  tokenizer is also used CPU-side by the planning captures). One ≥80 GB GPU suffices for the
  worker; the transfer suite loads both models on one GPU (originally a 141 GB H200).
- **Benchmark inputs**: GPQA problems ship encrypted (`../src_data/gated/`, anti-contamination
  norm — password in that README); `lockin/stage_{math,humaneval,lcb,imo}.py` rebuild the
  other problem files from public HF datasets (math + humaneval verified byte-identical to
  the originals; lcb best-effort, see `lockin/README.md`).
- **Judges**: only two scripts call an LLM judge — `transfer/` uses OpenRouter
  `google/gemini-3-flash-preview` (`OPENROUTER_API_KEY`), and `../scripts/judge_logical_order.py`
  any OpenAI-compatible endpoint (`NODEV_URL`/`NODEV_KEY`). Everything else grades with exact
  Python verifiers (`thinkfast/battery.py check()`, `lockin/fig2_report.py norm_record`).
- **Third-party**: the transfer suite clones
  [SAE-Probes](https://github.com/JoshEngels/SAE-Probes),
  [representation-engineering](https://github.com/andyzoujm/representation-engineering) and
  [jacobian-lens](https://github.com/anthropics/jacobian-lens) into `transfer/third_party/`
  (details in `transfer/README.md`); `worker/jlens_dg_common.py` finds jacobian-lens via
  `JLENS_REPO` (point it at the same clone).

## Referenced task sources

Think Fast battery tasks after Gould et al. ([2606.07157](https://arxiv.org/abs/2606.07157))
plus transparency-paper case studies (Engels et al.); GPQA
([Rein et al.](https://arxiv.org/abs/2311.12022)); MATH problems from AI-MO
AMC/AIME validation sets; HumanEval; LiveCodeBench; WildChat; RepE tasks after
Zou et al. ([2310.01405](https://arxiv.org/abs/2310.01405)); probing concepts after
Kantamneni et al. ([2502.16681](https://arxiv.org/abs/2502.16681)).
