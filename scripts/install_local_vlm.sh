#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_dir="${project_root}/artifacts/ollama-runtime"
model_dir="${project_root}/artifacts/ollama-models"
ollama_version="${OLLAMA_VERSION:-v0.32.5}"

case "$(uname -m)" in
  x86_64) archive="ollama-linux-amd64.tar.zst" ;;
  aarch64|arm64) archive="ollama-linux-arm64.tar.zst" ;;
  *) echo "unsupported architecture: $(uname -m)" >&2; exit 2 ;;
esac

mkdir -p "${runtime_dir}" "${model_dir}"
if [[ ! -x "${runtime_dir}/bin/ollama" ]]; then
  curl --fail --location --progress-bar \
    "https://github.com/ollama/ollama/releases/download/${ollama_version}/${archive}" \
    | tar --zstd -x -C "${runtime_dir}"
fi

echo "Ollama installed at ${runtime_dir}/bin/ollama"
echo "Next: make vlm-serve"
