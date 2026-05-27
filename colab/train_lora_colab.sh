#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONFIG_PATH="${1:-$ROOT_DIR/finetune/configs/voxcpm2_north_female_lora.yaml}"
ALLOW_LOW_VRAM="${ALLOW_LOW_VRAM:-0}"

cd "$ROOT_DIR"
if [ "$ALLOW_LOW_VRAM" = "1" ]; then
  python3 colab/preflight_check.py --mode lora --allow-low-vram
  python3 prepare_north_female_dataset.py
  python3 build_voxcpm_manifests.py
  python3 run_voxcpm_finetune.py --config "$CONFIG_PATH" --allow-low-vram
else
  python3 colab/preflight_check.py --mode lora
  python3 prepare_north_female_dataset.py
  python3 build_voxcpm_manifests.py
  python3 run_voxcpm_finetune.py --config "$CONFIG_PATH"
fi
