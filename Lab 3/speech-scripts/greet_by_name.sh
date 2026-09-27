#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB_DIR="$(dirname "$SCRIPT_DIR")"
VOICES_DIR="$LAB_DIR/voices"
VOICE_MODEL="en_US-lessac-medium"
# VOICE_MODEL="fr_FR-tom-medium"

if [[ $# -gt 0 ]]; then
  name="$*"
else
  read -r -p "What is your name? " name
fi

if [[ -z "$name" ]]; then
  printf 'Please provide a name.\n' >&2
  exit 1
fi

if [[ -x "$LAB_DIR/.venv/bin/python" ]]; then
  python="$LAB_DIR/.venv/bin/python"
else
  python="python3"
fi

if ! "$python" -c 'import piper' >/dev/null 2>&1; then
  printf 'Piper is unavailable. Activate Lab 3/.venv and run setup.sh first.\n' >&2
  exit 1
fi

"$python" -m piper \
  --model "$VOICE_MODEL" \
  --data-dir "$VOICES_DIR" \
  --output-raw \
  -- "Hello, $name." \
  | aplay -r 22050 -f S16_LE -t raw -