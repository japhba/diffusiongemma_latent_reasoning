#!/bin/bash
# DiffusionGemma trajectory worker. Needs 1 GPU (>=80GB for the 26B model at bf16 with headroom);
# if co-resident with another model on the same GPU, cap this worker's share via DG_MEM_FRACTION.
set -u
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} HF_XET_HIGH_PERFORMANCE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
VENV=${DG_VENV:-/opt/dgvenv}
PORT=${DG_WORKER_PORT:-8711}
if ! "$VENV/bin/python" -c "import transformers.models.diffusion_gemma, flask, torch" 2>/dev/null; then
  echo "[run_worker] building venv at $VENV (torch cu128 + transformers 5.12.1)"
  python3.12 -m pip install -q uv
  python3.12 -m uv venv "$VENV" --python 3.12
  python3.12 -m uv pip install --python "$VENV/bin/python" --exclude-newer 2026-06-17 torch==2.9.1 torchvision --index-url https://download.pytorch.org/whl/cu128
  python3.12 -m uv pip install --python "$VENV/bin/python" --exclude-newer 2026-06-17 transformers==5.12.1 accelerate flask pillow sentencepiece
fi
"$VENV/bin/python" -c "import torch;print('[run_worker] torch',torch.__version__,'cuda_ok',torch.cuda.is_available())"
cd "$(dirname "$0")"
exec "$VENV/bin/python" server.py --port "$PORT" --idle-timeout 0
