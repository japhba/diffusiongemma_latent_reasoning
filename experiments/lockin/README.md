# lockin/ — GPQA/MATH/HumanEval/LCB/IMO top-k truncation ladder

Direct-GPU capture suite (each `ds_*.py` loads DiffusionGemma **in-process** — no worker):
the paper's f_k / f_p self-conditioning truncation ladders, the decoding-stabilizer suite,
and the full paper-protocol sweep with denoising films. Needs 1 GPU (>= 80 GB); the report
builder is CPU-only.

## Data dir

All scripts read problem files from and write manifests/acts under
`$DG_LOCKIN_DIR` (default: `lockin/data/`). Films from `ds_paper_sweep.py` go to
`$DG_RB_DIR` (default `$DG_LOCKIN_DIR/rb_steps`); `fig2_report.py` writes
`$DG_FIG2_HTML` (default `$DG_LOCKIN_DIR/fig2_trunc.html`).

## Staging the problem files

```bash
python stage_math.py        # AI-MO/aimo-validation-{amc,aime}: first 48 amc + first 24 aime
                            #   rows (verified byte-identical to the original study file)
python stage_humaneval.py   # openai/openai_humaneval: exact 40-task-id ordered selection
                            #   pinned in the script (verified byte-identical)
python stage_imo.py         # OpenEvals/IMO-AnswerBench: 32 integer-answer problems,
                            #   seeded + stratified (deterministic)
python stage_lcb.py         # livecodebench/code_generation_lite release_v1: exact 32-qid
                            #   ordered selection pinned in the script, first 8 tests each.
                            #   BEST-EFFORT restage (not diffed against the 168MB original);
                            #   note: needs datasets<3 for the loading-script path.
```

`gpqa_problems.json` is **not** restageable from public HF (gated benchmark; verbatim text
must not be posted in plaintext). It arrives as a password-protected zip — see
`../../src_data/gated/README.md` for the password and the `unzip ... -d` command that
extracts it into `$DG_LOCKIN_DIR`.

## Entry points

```bash
python ds_ablate.py                       # word-battery f_k ladder (k1/k2/k8/onehot), C=64 T=48
python ds_ablate_math.py                  # AMC/AIME paper regime (thinking + adaptive stopping)
MODES=soft,k1,k2,k4,k8 TASKS=gpqa_nt,gpqa,humaneval python ds_ablate_bench.py
STAB_MODES=k1_rep,k1_slow,k1_ema,soft_rep python ds_ablate_stab.py   # stabilizer suite
python ds_ablate_lcb.py                   # LCB soft/k1 +- slow3 closure pair
TASKS=gpqa,amc_aime,lcb,imo ARMS_OVERRIDE=... WORKER=0 NWORKERS=1 LIMIT_PER_ARM=0 \
  python ds_paper_sweep.py                # full ladder + films (idx-mod sharding)
python fig2_report.py                     # CPU; builds the Fig-2 truncation report HTML
```

`ds_battery.py` is a library (the 5x16 word/arith battery; it imports
`thinkfast.battery`, `thinkfast.money_tasks` and `constrained.battery` from the sibling
vendored dirs). Everything is resumable — existing manifest entries/shards are skipped.

## Sampler arms

| arm | entropy_bound | steps/canvas T | temperature t_max -> t_min | token budget |
|---|---|---|---|---|
| standard (`soft`, `k*`, `p*`) | 0.1 | 48 | 0.8 -> 0.4 | 8192 |
| gentle (`*_slow3`) | 0.02 | 96 | 1.0 -> 0.5 | 12288 |
| step-matched gentle (`*_slow3m`) | 0.02 | 48 | 1.0 -> 0.5 | 8192 |

`k<N>` = paper f_k (keep top-N token probs, spread the rest uniformly), `p<val>` = paper
f_p (keep tokens with prob > val), `soft` = untruncated baseline, `onehot` = k=0 argmax.

## Grading

Per-rollout normalization/grading rules live in `fig2_report.py::norm_record` (boxed-int
extraction for math/IMO, letter extraction for GPQA, execution grading for code, plus
failure-mode classification: degenerate-loop `dup8`, unfinished, wrong).

## Feeds

Manifests + films are the sources behind `../../src_data/commit_ds/{acts_bench,
acts_psweep, acts_stab}` (consumed by `scripts/plot_gpqa_trunc_failures.py` and the Fig-2
material in the post).
