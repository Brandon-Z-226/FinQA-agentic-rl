#!/usr/bin/env bash
# Run the local environment from any working directory. No credentials are set.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PROJECT_ROOT/.cache/uv}"
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.cache/huggingface}"
export RLLM_HOME="${RLLM_HOME:-$PROJECT_ROOT/.rllm}"
export FINQA_DATA_DIR="${FINQA_DATA_DIR:-$PROJECT_ROOT/data}"
export TMPDIR="${FINQA_TMP_DIR:-$PROJECT_ROOT/.cache/tmp}"
mkdir -p "$TMPDIR"
exec uv run --locked "$@"
