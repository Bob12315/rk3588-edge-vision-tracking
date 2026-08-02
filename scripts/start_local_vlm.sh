#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ollama_bin="${project_root}/artifacts/ollama-runtime/bin/ollama"

if [[ ! -x "${ollama_bin}" ]]; then
  echo "local Ollama runtime is missing; run scripts/install_local_vlm.sh" >&2
  exit 2
fi

export OLLAMA_MODELS="${project_root}/artifacts/ollama-models"
export OLLAMA_HOST="127.0.0.1:11434"
export OLLAMA_CONTEXT_LENGTH="4096"
export OLLAMA_MAX_LOADED_MODELS="1"
export OLLAMA_NUM_PARALLEL="1"
export OLLAMA_NO_CLOUD="1"

exec "${ollama_bin}" serve
