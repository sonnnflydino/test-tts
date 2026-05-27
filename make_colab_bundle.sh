#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_PATH="${1:-$ROOT_DIR/test-tts-colab-bundle.zip}"

cd "$ROOT_DIR"
rm -f "$OUT_PATH"

zip -r "$OUT_PATH" . \
  -x ".git/*" \
  -x ".agents/*" \
  -x ".codex/*" \
  -x ".venv/*" \
  -x ".venv-lightning/*" \
  -x ".cache/*" \
  -x "__pycache__/*" \
  -x "*/__pycache__/*" \
  -x "datasets/*" \
  -x "finetune/checkpoints/*" \
  -x "finetune/tensorboard/*" \
  -x "finetune/.resolved/*" \
  -x "finetune/upstream/*" \
  -x "output*.wav" \
  -x "test-tts-colab-bundle.zip"

echo "Wrote bundle: $OUT_PATH"
