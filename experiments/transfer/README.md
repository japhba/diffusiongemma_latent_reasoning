# Transfer suite — RSA / probes / steering / J-Lens (gemma-4 vs DiffusionGemma)

Vendored, runnable copy of the `concept_probes/` experiments that produced the blog's
transfer-section artifacts. A reader with a GPU, Hugging Face access, and an OpenRouter key can
regenerate:

- `src_data/saeprobes/dg_rsa_cka_curves.json` — RSA/CKA layer curves
- `src_data/saeprobes/probe_matrix.json`, `probe_example_scores.json` — probe transfer 2×2 + per-example scores
- `src_data/saeprobes/dom_gens.json`, `dom_carriers.json`, `judged_dom_gens_paired_gemini.json` — RepE dominant-direction steering generations + blind paired Gemini judging
- `src_data/saeprobes/jlens/eval_2x2.json`, `jlens/judged_jlens_percepts.json` — controlled J-Lens 2×2 eval + judged percepts
- `src_data/jlens_future_rows.json` — J-Lens future-token counterfactual rows

`src_data/saeprobes/carrier_additions_sonnet5.json` is a retired hand-authored artifact with no
rerun path (currently unused: its `additions` lists are empty).

## Layout / path convention

Every script resolves paths relative to `REPO = $DGLR_ROOT` (default: this directory,
`experiments/transfer/`). Outputs land under `concept_probes/out/saeprobes/`, third-party code is
expected under `third_party/`. Set `DGLR_ROOT` only if you run from elsewhere.

## Setup

1. **Third-party repos** — clone into `experiments/transfer/third_party/`:

   ```bash
   cd third_party
   git clone https://github.com/JoshEngels/SAE-Probes.git
   git clone https://github.com/andyzoujm/representation-engineering.git
   git clone https://github.com/anthropics/jacobian-lens
   ```

   - `SAE-Probes`: `data/probing_datasets_MASTER.csv` ships with the repo. The raw text datasets
     do not — download `sae_probes_raw_text.zip` (~33 GB) via the Dropbox link in the SAE-Probes
     README ("Raw Text Datasets"), unzip it to `third_party/SAE-Probes/data/dropbox_raw/`, then
     `ln -s dropbox_raw/cleaned_data third_party/SAE-Probes/data/cleaned_data` (the loaders read
     `data/cleaned_data/*.csv`).
   - `representation-engineering`: `data/emotions/*.json` + `data/facts/facts_true_false.csv`
     ship with the repo (used by the steering stimuli).
   - `jacobian-lens`: `data/evaluations/lens-eval-*.json` ship with the repo; the `jlens` package
     is imported straight from the checkout.

2. **DG worker** (only for the `dgb_shared`/`dgc_shared` J-Lens fits): `jlens_fit_shared.py`
   imports `jlens_dg_common` (prefill replay / decoder replay for DiffusionGemma) from
   `../worker` relative to `DGLR_ROOT`, i.e. `experiments/worker/` — vendored separately.

3. **Models** (Hugging Face, both gated → set `HF_TOKEN` / `huggingface-cli login`):
   `google/gemma-4-26b-a4b-it` and `google/diffusiongemma-26B-A4B-it`. Both load in bf16
   (`dtype="auto"`, `device_map="auto"`, ~50 GB each) and most GPU stages need them co-resident:
   originally run on a single 141 GB H200; a single 80 GB card only fits single-model stages.

4. **Keys**: put `OPENROUTER_API_KEY=...` in `experiments/transfer/.env` (the judge scripts read
   `REPO/.env`; judge model `google/gemini-3-flash-preview`). Node V is not needed:
   `judge_steer_gens.py` is imported only for its concept descriptions — its own judge endpoint
   (env `NODEV_URL`/`NODEV_KEY`) is required only if you run that script standalone.

5. **Environment**: `slurm/ensure_and_run.sh` builds a node-local venv with `uv`
   (python 3.12, `torch==2.9.1` cu128, `transformers>=5.10` — ships both `gemma4` and
   `diffusion_gemma` — accelerate, numpy, pillow, sentencepiece, datasets, plus scikit-learn,
   scipy, joblib, flask, tiktoken, protobuf) and then execs the given script:

   ```bash
   bash concept_probes/slurm/ensure_and_run.sh concept_probes/<script>.py [args...]
   ```

   `VENV_DIR`, `HF_HOME`, `UV_CACHE_DIR` are env-overridable. Any venv with those packages works
   equally: `python concept_probes/<script>.py`. On shared clusters the GPU stages were submitted
   as Slurm jobs wrapping exactly this runner (`srun ... bash concept_probes/slurm/ensure_and_run.sh ...`);
   the commands below are given bare. `SAEP_CPU=1` marks CPU-only stages (skips the runner's CUDA
   assert). Sharded stages take `SAEP_SHARD=i/n` (0-based) as independent parallel jobs.

## Pipeline

Run from `experiments/transfer/` (or export `DGLR_ROOT`).

**Probes** (→ `probe_matrix.json`, `probe_example_scores.json`):

```bash
python concept_probes/run_saeprobes_gpu.py --phase extract       # GPU: acts/ per concept
python concept_probes/extract_train1024.py                       # GPU: acts1024/ (1024/class train)
SAEP_ACTS_DIR=acts1024 python concept_probes/fit_saeprobes.py    # CPU: probe fits
python concept_probes/classify_entity_filter.py                  # OpenRouter: concept_entity_filter.json
SAEP_FAM_MAXLEN=512 python concept_probes/dg_family_full.py      # GPU (SAEP_SHARD-able): family1024_w512/
SAEP_FAM_MAXLEN=512 python concept_probes/dg_family_full.py --fit  # CPU: dg_family_1024_w512.json
SAEP_CPU=1 python concept_probes/probe_matrix_data.py            # CPU: probe_matrix.json
SAEP_CPU=1 python concept_probes/probe_example_scores.py         # CPU: probe_example_scores.json
```

**RSA** (→ `dg_rsa_cka_curves.json`):

```bash
python concept_probes/dg_rsa_cka.py        # GPU: paired activation extract
python concept_probes/dg_rsa_cka.py --fit  # CPU: dg_rsa_cka_curves.json
```

**Steering** (→ `dom_gens.json`, `dom_carriers.json`, `judged_dom_gens_paired_gemini.json`):

```bash
python concept_probes/steer_dom_repe.py --derive   # GPU: dominant directions per (concept, model)
python concept_probes/steer_dom_repe.py            # GPU: ±steered generations; SAEP_SHARD=i/n writes
                                                   #      dom_gens_shard{i}.json — merge the shard dicts
                                                   #      into dom_gens.json before judging
python concept_probes/judge_steer_pairs_openrouter.py dom_gens   # OpenRouter paired judge
```

(Default `SAEP_STEER_SET=repe` is the vendored path; the `aemo`/`aemoall` variants need
`emotion_stories_steer.py`, which is not vendored.)

**J-Lens** (→ `jlens/eval_2x2.json`, `jlens/judged_jlens_percepts.json`, future rows):

```bash
SAEP_CPU=1 python concept_probes/jlens_fit_shared.py --build-corpus   # shared corpus manifest
for i in 0 1 2 3; do   # per target; dgc_shared/dgb_shared need the DG worker (../worker)
  JL_TARGET=g_shared   SAEP_SHARD=$i/4 python concept_probes/jlens_fit_shared.py
  JL_TARGET=dgc_shared SAEP_SHARD=$i/4 python concept_probes/jlens_fit_shared.py
  JL_TARGET=dgb_shared SAEP_SHARD=$i/4 python concept_probes/jlens_fit_shared.py
done
python concept_probes/jlens_eval_2x2.py           # GPU: jlens/eval_2x2.json
python concept_probes/jlens_topk_capture.py       # GPU: reports/concept_probes/data/jlens_topk_{set}.json
python concept_probes/judge_jlens_percepts.py     # OpenRouter: jlens/judged_jlens_percepts.json

python concept_probes/jlens_dg_future_scan.py capture   # GPU: per-item jlens/logitlens stacks
python concept_probes/jlens_dg_future_scan.py analyze   #      dg_original_future_exact.json
python concept_probes/jlens_dg_future_scan.py control   # GPU: dg_original_future_controls.json
mkdir -p reports/concept_probes                          # analyze_jlens_future writes its HTML here
python concept_probes/analyze_jlens_future.py            # CPU: summary + jlens_future.html
```

## Outputs → blog `src_data/`

All outputs land under `concept_probes/out/saeprobes/` (relative to `DGLR_ROOT`). To refresh the
blog figures, copy:

| output | destination |
|---|---|
| `out/saeprobes/dg_rsa_cka_curves.json` | `../../src_data/saeprobes/` |
| `out/saeprobes/probe_matrix.json` | `../../src_data/saeprobes/` |
| `out/saeprobes/probe_example_scores.json` | `../../src_data/saeprobes/` |
| `out/saeprobes/dom_gens.json` | `../../src_data/saeprobes/` |
| `out/saeprobes/dom_carriers.json` | `../../src_data/saeprobes/` |
| `out/saeprobes/judged_dom_gens_paired_gemini.json` | `../../src_data/saeprobes/` |
| `out/saeprobes/jlens/eval_2x2.json` | `../../src_data/saeprobes/jlens/` |
| `out/saeprobes/jlens/judged_jlens_percepts.json` | `../../src_data/saeprobes/jlens/` |

`../../src_data/jlens_future_rows.json` is the `rows` array of
`out/saeprobes/jlens/dg_original_future_controls.json` after `analyze_jlens_future.py` adds the
`jlens_top20_switch_layers` / `logitlens_top20_switch_layers` fields — extract it verbatim from
the `const rows=` literal in `reports/concept_probes/jlens_future.html`.
