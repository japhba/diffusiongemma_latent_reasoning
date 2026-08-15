# Encrypted inputs

These two files contain verbatim GPQA problem text (GPQA is a gated benchmark whose authors
ask that examples not be posted in plaintext, to keep them out of web-scraped training
corpora), so they are committed as password-protected zips. The password is the one GPQA
itself uses for distribution: `deserted-untie-orchid`.

- `gpqa_problems.json.zip` — the 64-problem GPQA-diamond subset used by the truncation
  ladder (`experiments/lockin/`); schema `{pid, problem, answer, domain}`. Extract into the
  lockin data dir: `unzip -P deserted-untie-orchid gpqa_problems.json.zip -d ../../experiments/lockin/data/`
- `bench_capture_jobs.json.zip` — the 48 prompts (GPQA / MATH / HumanEval / WildChat, 12
  each) behind `scripts/capture_bench_order.py`; schema `{bench, pid, prompt}`. Extract next
  to this file: `unzip -P deserted-untie-orchid bench_capture_jobs.json.zip -d .`
