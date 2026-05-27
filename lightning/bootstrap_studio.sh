#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv-lightning}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "Root dir: $ROOT_DIR"
echo "Venv dir: $VENV_DIR"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python not found: $PYTHON_BIN" >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found. Start a GPU Studio first." >&2
  exit 1
fi

nvidia-smi

if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "$ROOT_DIR/requirements-lightning.txt"

python - <<'PY'
import importlib.util
mods = ["voxcpm", "torch", "torchaudio", "argbind", "tensorboardX", "transformers", "librosa"]
for mod in mods:
    print(mod, bool(importlib.util.find_spec(mod)))
PY

echo
echo "Bootstrap completed."
echo "Activate with: source \"$VENV_DIR/bin/activate\""
