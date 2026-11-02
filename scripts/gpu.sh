#!/usr/bin/env bash
# Linux GPU environment only. Does not change the system driver or shell files.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
ACTION="${1:-help}"
if [[ "$ACTION" == help || "$ACTION" == --help ]]; then
    cat <<'USAGE'
Usage: FINQA_GPU_ROOT=/root/autodl-tmp/finqa bash scripts/gpu.sh COMMAND [args]
Commands: setup, download, prepare, check, lora-check, config, smoke, train, serve, eval
Choose a CUDA 13.0 devel / Linux x86_64 image with nvcc and uv installed.
setup installs only the independent uv GPU environment. No job starts automatically.
config composes Hydra without starting training; smoke runs one optimizer step.
USAGE
    exit 0
fi
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
    echo "GPU commands require Linux x86_64; use local pytest/static checks on macOS." >&2
    exit 2
fi
: "${FINQA_GPU_ROOT:?Set FINQA_GPU_ROOT to an absolute path on the data disk}"
[[ "$FINQA_GPU_ROOT" == /* ]] || { echo "FINQA_GPU_ROOT must be absolute." >&2; exit 2; }
command -v uv >/dev/null || { echo "Install uv first (see environments/cuda/README.md)." >&2; exit 2; }
shift
export FINQA_PROJECT_ROOT="$PROJECT_ROOT"
export UV_PROJECT_ENVIRONMENT="$FINQA_GPU_ROOT/venv"
export UV_CACHE_DIR="$FINQA_GPU_ROOT/uv-cache"
export UV_PYTHON_INSTALL_DIR="$FINQA_GPU_ROOT/python"
export UV_LINK_MODE=hardlink
export HF_HOME="$FINQA_GPU_ROOT/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export FINQA_DATA_DIR="$FINQA_GPU_ROOT/data"
export RLLM_HOME="$FINQA_GPU_ROOT/rllm"
export TMPDIR="$FINQA_GPU_ROOT/tmp"
# Keep Ray's socket paths short; the big datasets/model/checkpoints stay on data disk.
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/finqa-ray}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MAX_JOBS="${MAX_JOBS:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$TMPDIR"
if [[ "$ACTION" == setup ]]; then
    for program in python3 git g++ make nvcc nvidia-smi; do
        command -v "$program" >/dev/null || { echo "Missing build prerequisite: $program (see CUDA README)." >&2; exit 2; }
    done
    python3 -c 'import os, shutil, sys; free = shutil.disk_usage(os.environ["FINQA_GPU_ROOT"]).free / 2**30; print(f"Data disk free before installation: {free:.1f} GiB"); sys.exit(0 if free >= 40 else "Need at least 40 GiB free before installing; expand the data disk.")'
    command -v nvcc >/dev/null || { echo "CUDA devel image with nvcc 13.0 is required." >&2; exit 2; }
    nvcc --version | grep -q 'release 13.0' || { echo "This environment targets CUDA 13.0." >&2; exit 2; }
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    if [[ ! -f environments/cuda/uv.lock ]]; then
        uv lock --project environments/cuda --python 3.11
    fi
    uv sync --project environments/cuda --locked --python 3.11
    uv pip check --python "$UV_PROJECT_ENVIRONMENT/bin/python"
    echo "GPU environment installed. Run download, prepare, check and lora-check. Keep the generated CUDA uv.lock."
    exit 0
fi
[[ -f environments/cuda/uv.lock ]] || { echo "Run setup first to generate the CUDA lock." >&2; exit 2; }
ENV_ARGS=()
if [[ -f "$PROJECT_ROOT/.env" ]]; then
    ENV_ARGS=(--env-file "$PROJECT_ROOT/.env")
fi
exec uv run --project environments/cuda --locked "${ENV_ARGS[@]}" python -m finqa_agentic_rl.gpu "$ACTION" "$@"
