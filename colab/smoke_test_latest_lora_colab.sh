#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LORA_PATH="${1:-$ROOT_DIR/finetune/checkpoints/north_female_lora/latest}"
OUTPUT_PATH="${2:-$ROOT_DIR/output_north_female_lora_colab.wav}"
TEXT="${3:-Xin chao, day la ban test giong nu mien Bac da fine-tune tren Colab.}"

cd "$ROOT_DIR"
python3 main.py \
  --lora-weights "$LORA_PATH" \
  --text "$TEXT" \
  --output "$OUTPUT_PATH"
