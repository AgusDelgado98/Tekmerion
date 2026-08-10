#!/usr/bin/env bash
# Tekmérion portfolio demo (no secrets required)
# Usage: ./scripts/run_demo.sh [--no-fake]
set -euo pipefail
cd "$(dirname "$0")/.."

export TEKMERION_DATA_MODE=showroom
if [[ "${1:-}" == "--no-fake" ]]; then
  export TEKMERION_LLM_PROVIDER="${TEKMERION_LLM_PROVIDER:-disabled}"
  echo "Dataset: showroom | LLM: $TEKMERION_LLM_PROVIDER"
else
  export TEKMERION_LLM_PROVIDER=fake
  echo "Dataset: showroom | LLM: fake (deterministic DEMO, not a real model call)"
fi

echo "Starting Flask at http://127.0.0.1:5000 ..."
exec python -m app
