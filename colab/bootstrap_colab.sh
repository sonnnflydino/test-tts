#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "Root dir: $ROOT_DIR"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python not found: $PYTHON_BIN" >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found. Switch Colab to GPU runtime first, then rerun this script." >&2
  exit 1
fi

nvidia-smi

"$PYTHON_BIN" -m pip install --upgrade pip "setuptools<82" wheel
"$PYTHON_BIN" -m pip install --index-url https://download.pytorch.org/whl/cu126 \
  torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0
"$PYTHON_BIN" -m pip install -r "$ROOT_DIR/requirements-colab.txt"

"$PYTHON_BIN" - <<'PY'
import importlib.util
import torch

print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print("gpu_name", props.name)
    print("gpu_vram_gb", round(props.total_memory / (1024**3), 2))

mods = ["voxcpm", "argbind", "tensorboardX", "transformers", "librosa"]
for mod in mods:
    print(mod, bool(importlib.util.find_spec(mod)))
PY
