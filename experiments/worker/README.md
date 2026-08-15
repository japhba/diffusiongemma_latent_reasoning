# worker/ — DiffusionGemma trajectory worker

The HTTP worker every worker-driven capture in `../planning/`, `../thinkfast/` and
`../posthoc/` POSTs to. Loads **google/diffusiongemma-26B-A4B-it** (HF-gated: accept the
license and export `HF_TOKEN`) once and serves full per-denoising-step trajectories.
Needs **1 GPU with >= 80 GB** (26B bf16 + trajectory buffers).

## Files

- `server.py` — Flask worker (this is the pod's final copy: includes the `/energy` route,
  `s_topk_record`, and the `no_commit` op used by `planning/ember_kill2.py`). Routes:
  `/health`, `/tasks`, `/embed_tokens`, `/sample`, `/rollouts`, `/logitlens`, `/scope`,
  `/resample`, `/barrier`, `/steer`, `/energy`, `/cloze`.
- `jlens_dg_common.py` — shared Jacobian-lens machinery (prefill/decoder replay, per-step
  capture, `jacobian_for_step`). It does `from server import ...`, so it must sit next to
  `server.py`; the transfer suite's dgb lens fit imports it as
  `from jlens_dg_common import prefill_replay, replay_decoder`. Its `jlens` imports need a
  checkout of the jacobian-lens repo: set `JLENS_REPO=/path/to/jacobian-lens` (default:
  `worker/jacobian-lens`).
- `run_worker.sh` — venv bootstrap (pinned: torch 2.9.1 cu128, transformers 5.12.1, flask)
  + launch. Env-var defaults: `DG_VENV` (venv path, default `/opt/dgvenv`), `HF_HOME`,
  `DG_WORKER_PORT` (default 8711).

## Run

```bash
python server.py --port 8711            # or: bash run_worker.sh
```

Env knobs: `DIFFUSIONGEMMA_MODEL` (model override), `DG_MEM_FRACTION` (cap this worker's
GPU share when co-resident with another model), `DG_LENS_DIR` (see below).

## Optional j-lens artifact

The `/logitlens` j-lens routes load fitted transports from `$DG_LENS_DIR/lens_<name>.pt`
(e.g. `lens_pooled.pt`). These are **not vendored** — they are produced by the
`experiments/transfer/` lens fits. Everything else works without them.

## Driving a remote worker

Captures talk plain HTTP; to drive a worker on a remote GPU box, tunnel the port and point
`DG_WORKER` at it:

```bash
ssh -N -L 18711:localhost:8711 <user>@<gpu-host>
DG_WORKER=http://localhost:18711 python ../planning/xtask_compute8.py
```
