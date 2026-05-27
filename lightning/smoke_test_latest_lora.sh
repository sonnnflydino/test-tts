#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv-lightning}"
LORA_PATH="${1:-$ROOT_DIR/finetune/checkpoints/north_female_lora/latest}"
OUTPUT_PATH="${2:-$ROOT_DIR/output_north_female_lora.wav}"
TEXT="${3:-Xin chao, day la ban test giong nu mien Bac da fine-tune.}"

if [ ! -d "$VENV_DIR" ]; then
  echo "Missing venv at $VENV_DIR. Run lightning/bootstrap_studio.sh first." >&2
  exit 1
fi

source "$VENV_DIR/bin/activate"
cd "$ROOT_DIR"

python main.py \
  --lora-weights "$LORA_PATH" \
  --text "$TEXT" \
  --output "$OUTPUT_PATH"
