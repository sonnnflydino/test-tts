#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv-lightning}"
CONFIG_PATH="${1:-$ROOT_DIR/finetune/configs/voxcpm2_north_female_full.yaml}"

if [ ! -d "$VENV_DIR" ]; then
  echo "Missing venv at $VENV_DIR. Run lightning/bootstrap_studio.sh first." >&2
  exit 1
fi

source "$VENV_DIR/bin/activate"
cd "$ROOT_DIR"

python build_voxcpm_manifests.py
python run_voxcpm_finetune.py --config "$CONFIG_PATH"
