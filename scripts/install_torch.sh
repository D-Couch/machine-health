#!/usr/bin/env bash
set -euo pipefail

# Install the project-local PyTorch build for the hardware visible to this host.
# Override PYTHON or PYTORCH_CUDA_INDEX_URL when using a non-default environment.
PYTHON="${PYTHON:-.venv/bin/python}"
CUDA_INDEX_URL="${PYTORCH_CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu130}"
CPU_INDEX_URL="https://download.pytorch.org/whl/cpu"
TORCH_VERSION="torch==2.14.1"

if [[ ! -x "$PYTHON" ]]; then
  echo "Python executable not found: $PYTHON" >&2
  echo "Create the project environment first, or set PYTHON=/path/to/python." >&2
  exit 2
fi

if "$PYTHON" -c 'import torch; assert torch.cuda.is_available()' >/dev/null 2>&1; then
  "$PYTHON" -c 'import torch; print(f"PyTorch {torch.__version__}: CUDA device {torch.cuda.get_device_name(0)} already available")'
  exit 0
fi

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -qE '^GPU [0-9]+:'; then
  echo "NVIDIA GPU detected; installing the PyTorch CUDA 13.0 build into $PYTHON."
  if "$PYTHON" -m pip install --upgrade --force-reinstall "$TORCH_VERSION" --index-url "$CUDA_INDEX_URL" \
      && "$PYTHON" -c 'import torch; assert torch.cuda.is_available(); print(f"PyTorch {torch.__version__}: CUDA {torch.version.cuda}, {torch.cuda.get_device_name(0)}")'; then
    exit 0
  fi
  echo "CUDA PyTorch could not access the visible GPU; installing CPU PyTorch as fallback." >&2
else
  # A restricted Codex command may hide /dev/nvidia* even though the project
  # already has CUDA PyTorch. Do not silently downgrade that environment.
  if "$PYTHON" -c 'import torch; assert torch.version.cuda is not None' >/dev/null 2>&1; then
    echo "A CUDA PyTorch build is installed, but this command cannot access a CUDA device. Keeping it unchanged; rerun with GPU access enabled if you intended to install CUDA PyTorch." >&2
    exit 0
  fi
  echo "No NVIDIA GPU is visible; ensuring the CPU-only PyTorch build is installed."
fi

"$PYTHON" -m pip install --upgrade --force-reinstall "${TORCH_VERSION}+cpu" --index-url "$CPU_INDEX_URL"
"$PYTHON" -c 'import torch; print(f"PyTorch {torch.__version__}: CPU fallback (CUDA available: {torch.cuda.is_available()})")'
