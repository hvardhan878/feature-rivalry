#!/usr/bin/env bash
# One-shot environment repair for Paperspace Gradient (CUDA 12.x driver, cu121 PyTorch wheel).
#
# Hugging Face token: set HF_TOKEN in the environment, or create a .env file (gitignored):
#   cp .env.example .env
#   # edit .env and set HF_TOKEN=hf_...

set -euo pipefail
cd "$(dirname "$0")"

if [[ -f .env ]]; then
  set -a
  # shellcheck source=/dev/null
  source .env
  set +a
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN is not set."
  echo "  Either export HF_TOKEN=hf_... before running this script, or"
  echo "  copy .env.example to .env and add your token there."
  exit 1
fi

echo "=== Hugging Face login (gated Gemma models) ==="
huggingface-cli login --token "$HF_TOKEN"

echo "=== Installing PyTorch (cu121) ==="
pip install --force-reinstall \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121

echo "=== Installing Python dependencies (may pull a newer torch via sae-lens) ==="
pip install -r requirements.txt

echo "=== Re-pinning PyTorch to cu121 (undo any torch upgrade from sae-lens) ==="
pip install --force-reinstall \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121

echo "=== NumPy / SciPy stack (torch may have upgraded numpy) ==="
pip install --force-reinstall \
  "numpy<2" scipy scikit-learn pandas pyarrow \
  'fsspec[http]<=2025.3.0,>=2023.1.0' \
  sentencepiece protobuf

echo "=== CUDA library path (fix libnvJitLink / libcusparse symbol errors) ==="
SITE="$(python -c "import site; print(site.getsitepackages()[0])")"
CUDA_LIB_PATH="${SITE}/nvidia/nvjitlink/lib:${SITE}/nvidia/cusparse/lib:${SITE}/nvidia/cublas/lib:${SITE}/nvidia/cuda_runtime/lib:${SITE}/nvidia/cudnn/lib:${SITE}/nvidia/cufft/lib:${SITE}/nvidia/curand/lib:${SITE}/nvidia/cusolver/lib:${SITE}/nvidia/nccl/lib"
export LD_LIBRARY_PATH="${CUDA_LIB_PATH}:${LD_LIBRARY_PATH:-}"

cat > .paperspace_env <<EOF
# Source this in Paperspace terminals before running Python commands:
#   source .paperspace_env
export LD_LIBRARY_PATH="${CUDA_LIB_PATH}:\${LD_LIBRARY_PATH:-}"
EOF

MARKER="# feature_rivalry CUDA libs (setup_paperspace.sh)"
if ! grep -qF "$MARKER" ~/.bashrc 2>/dev/null; then
  {
    echo ""
    echo "$MARKER"
    echo "source \"$(pwd)/.paperspace_env\""
  } >> ~/.bashrc
fi

echo "=== Verify ==="
python - <<'PY'
import torch
import transformers
import numpy as np
print("torch      ", torch.__version__, "cuda:", torch.cuda.is_available(), "cuda_tag:", torch.version.cuda)
print("transformers", transformers.__version__)
print("numpy      ", np.__version__)
PY

echo ""
echo "Done."
echo "IMPORTANT: because ./setup_paperspace.sh runs as a subprocess, run this in the same terminal:"
echo "  source .paperspace_env"
echo "Then:"
echo "  python sanity_check.py"
