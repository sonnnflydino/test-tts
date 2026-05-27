#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="${HOME}/.lightning_studio"
TARGET_FILE="${TARGET_DIR}/on_stop.sh"

mkdir -p "$TARGET_DIR"
cp "$ROOT_DIR/lightning/on_stop.sh.template" "$TARGET_FILE"
chmod +x "$TARGET_FILE"

echo "Installed Lightning on-stop action at: $TARGET_FILE"
