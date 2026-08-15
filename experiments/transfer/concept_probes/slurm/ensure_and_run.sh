#!/bin/bash
# Node-independent runner: ensures a node-local `cprobe` venv (transformers>=5.10
# ships BOTH gemma4 and diffusion_gemma), then execs the given python script with it.
# Usage: bash ensure_and_run.sh <script.py> [args...]
set -eu
REPO=${DGLR_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
VENV=${VENV_DIR:-/var/tmp/$USER/venvs/cprobe}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_XET_HIGH_PERFORMANCE=1
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false
export UV_CACHE_DIR=${UV_CACHE_DIR:-$HOME/.cache/uv}
export UV_LINK_MODE=copy
cd "$REPO"

# Serialize concurrent builds on the same node (two jobs racing `uv venv --clear` on one
# /var/tmp corrupts both — observed 2026-07-15). Lock is held only while (re)building.
mkdir -p "$(dirname "$VENV")"
exec 9>"${VENV}.lock"
flock 9
if ! "$VENV/bin/python" -c "import transformers.models.diffusion_gemma, transformers.models.gemma4, torch, numpy, datasets" 2>/dev/null; then
    echo "[ensure] building $VENV on $(hostname)"
    rm -rf "$VENV"                          # a corrupted half-built venv makes `uv venv --clear` itself fail
    uv venv --clear "$VENV" --python 3.12   # --clear: recreate even if an outdated venv dir exists
    # Pinned torch cu128 stack from the trusted pytorch index: its nvidia-* wheels are UNDATED on
    # download.pytorch.org, so ANY exclude-newer cutoff (incl. the CLI override) silently filters
    # e.g. nvidia-cuda-nvrtc-cu12==12.8.93 → "torch cannot be used". --no-config drops the global
    # 7-day exclude-newer for this ONE pinned install; the transformers/... install below runs
    # normally and KEEPS the 7-day supply-chain protection.
    uv pip install --no-config --python "$VENV/bin/python" torch==2.9.1 torchvision --index-url https://download.pytorch.org/whl/cu128
    uv pip install --python "$VENV/bin/python" "transformers>=5.10" accelerate numpy pillow sentencepiece datasets
fi
# CPU-fit extras (--fit modes: sklearn CV probes, scipy spearman, joblib parallel) +
# flask (imported transitively by diffusiongemma/server.py for the jlens DG fit) +
# tiktoken/protobuf (transformers 5.x slow-tokenizer conversion for Llama-2 mirrors)
if ! "$VENV/bin/python" -c "import sklearn, scipy, joblib, flask, tiktoken, google.protobuf" 2>/dev/null; then
    uv pip install --python "$VENV/bin/python" scikit-learn scipy joblib flask tiktoken protobuf
fi
flock -u 9
"$VENV/bin/python" -c "import torch;assert torch.cuda.is_available() or '${SAEP_CPU:-}','no CUDA';print('[ensure]',torch.__version__,'on',__import__('socket').gethostname())"
exec "$VENV/bin/python" "$@"
