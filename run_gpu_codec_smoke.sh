#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
UV="${UV:-}"
if [ -z "$UV" ]; then
  if command -v uv >/dev/null 2>&1; then UV=uv; else UV=/home/jl_fs/bin/uv; fi
fi
SYSTEM_PYTHON="${SYSTEM_PYTHON:-python3}"
VENV="${VENV:-$ROOT/.venv-gpu}"
EXLLAMA="$ROOT/upstream/exllamav3-v0.0.43"

command -v "$UV" >/dev/null 2>&1 || {
  echo "uv is required; install it before running the smoke" >&2
  exit 2
}
if [ ! -x "$VENV/bin/python" ]; then
  "$UV" venv --system-site-packages --python "$SYSTEM_PYTHON" "$VENV"
fi
PYTHON="$VENV/bin/python"
"$UV" pip install --python "$PYTHON" "ninja==1.13.0"
EXLLAMA_NOCOMPILE=1 "$UV" pip install --python "$PYTHON" --no-deps --editable "$EXLLAMA"

"$PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print("torch", torch.__version__, "cuda", torch.version.cuda, "gpu", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))'
"$PYTHON" -c 'import ninja; print("ninja", ninja.__version__)'
TORCH_CUDA="$("$PYTHON" -c 'import torch; print(torch.version.cuda)')"
if [ -z "${CUDA_HOME:-}" ]; then
  for candidate in "/usr/local/cuda-$TORCH_CUDA" "/usr/local/cuda-${TORCH_CUDA%%.*}" /usr/local/cuda; do
    if [ -x "$candidate/bin/nvcc" ]; then CUDA_HOME="$candidate"; break; fi
  done
fi
test -n "${CUDA_HOME:-}" && test -x "$CUDA_HOME/bin/nvcc" || {
  echo "no nvcc matching torch CUDA $TORCH_CUDA" >&2
  exit 2
}
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"

export B300_DISK_RESERVE_BYTES="${B300_DISK_RESERVE_BYTES:-0}"
export EXL3_B300_BUILD_DIR="${EXL3_B300_BUILD_DIR:-$ROOT/work/torch_extensions/exllamav3_ext}"
export MAX_JOBS="${MAX_JOBS:-16}"
exec "$PYTHON" "$ROOT/tools/run_codec_smoke.py" --root "$ROOT" --bits 3 4 "$@"
