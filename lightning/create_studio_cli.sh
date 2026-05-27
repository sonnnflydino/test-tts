#!/usr/bin/env bash
set -euo pipefail

if ! command -v lightning >/dev/null 2>&1; then
  echo "Lightning CLI not found. Install with: pip install lightning-sdk -U" >&2
  exit 1
fi

STUDIO_NAME="${STUDIO_NAME:-voxcpm-north-female}"
TEAMSPACE="${TEAMSPACE:-}"
MACHINE="${MACHINE:-L4}"

CREATE_CMD=(lightning studio create --name "$STUDIO_NAME")
START_CMD=(lightning studio start --name "$STUDIO_NAME" --machine "$MACHINE")

if [ -n "$TEAMSPACE" ]; then
  CREATE_CMD+=(--teamspace "$TEAMSPACE")
  START_CMD+=(--teamspace "$TEAMSPACE")
fi

echo "Creating Studio: $STUDIO_NAME"
"${CREATE_CMD[@]}"

echo "Starting Studio on machine: $MACHINE"
"${START_CMD[@]}"
