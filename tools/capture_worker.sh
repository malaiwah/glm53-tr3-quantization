#!/usr/bin/env bash
set -Eeuo pipefail

REPO="${1:?usage: capture_worker.sh REPO REVISION}"
REVISION="${2:?usage: capture_worker.sh REPO REVISION}"
[[ "$REVISION" =~ ^[0-9a-f]{40}$ ]] || { echo "revision must be a full SHA" >&2; exit 2; }
ROOT="${GLM53_CAPTURE_ROOT:-/workspace/glm53}"
ADAPTER="$ROOT/tr3/uniform-adapter"
PYTHON="$ROOT/capture-venv/bin/python"
HF="$ROOT/hf-venv/bin/hf"
BF16_SRC="$ROOT/source"
CAPTURE_DIR="/dev/shm/glm53-tr3-capture"
PLAN="$ROOT/tr3/capture_plan.json"
EXPORT="$ROOT/capture-export"
ACKS="$ROOT/capture-acks"
LOG="$ROOT/logs/capture-worker.log"
WINDOWS=(3-10 11-18 19-26 27-34 35-42 43-50 51-58 59-66 67-74 75-77)
mkdir -p "$ROOT" "$ROOT/tr3" "$ROOT/receipts" "$EXPORT" "$ACKS" "$(dirname "$LOG")"
cd "$ROOT"
sha256sum -c uniform-adapter.tar.gz.sha256
if [[ ! -d "$ADAPTER" ]]; then
  tar -xzf uniform-adapter.tar.gz -C "$ROOT/tr3"
fi
exec > >(tee -a "$LOG") 2>&1

printf 'CAPTURE_WORKER_START repo=%s revision=%s utc=%s\n' "$REPO" "$REVISION" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
export DEBIAN_FRONTEND=noninteractive
apt update -y
apt dist-upgrade -y
apt install -y tmux nvtop rsync git curl jq ca-certificates build-essential pciutils ninja-build
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
[[ "$GPU_COUNT" == 8 ]] || { echo "expected 8 GPUs, found $GPU_COUNT" >&2; exit 3; }
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader | tee "$ROOT/receipts/gpus.txt"
nvidia-smi topo -m | tee "$ROOT/receipts/topology.txt"
nvidia-smi topo -p2p r | tee "$ROOT/receipts/p2p-read.txt"
nvidia-smi topo -p2p w | tee "$ROOT/receipts/p2p-write.txt"

uv venv --python python3 "$ROOT/hf-venv"
uv pip install --python "$ROOT/hf-venv/bin/python" 'huggingface_hub[hf_xet]'
uv venv --python python3 --system-site-packages "$ROOT/capture-venv"
uv pip install --python "$PYTHON" ninja
"$PYTHON" -c 'import vllm; version=str(vllm.__version__); assert "gilded.gnosis" in version.lower(), version; print("GILDED_GNOSIS", version)'
if [[ ! -d "$ROOT/code/exllamav3/.git" ]]; then
  git clone https://github.com/turboderp-org/exllamav3 "$ROOT/code/exllamav3"
fi
git -C "$ROOT/code/exllamav3" fetch --depth 1 origin c5d9c657966ffeeaa9353f0cc899f18629da4a13
git -C "$ROOT/code/exllamav3" checkout --detach c5d9c657966ffeeaa9353f0cc899f18629da4a13
EXLLAMA_NOCOMPILE=1 uv pip install --python "$PYTHON" --no-deps --editable "$ROOT/code/exllamav3"

[[ -s "$HOME/.hf_token" ]] || { echo "HF token missing" >&2; exit 4; }
export HF_TOKEN="$(cat "$HOME/.hf_token")"
export HF_XET_HIGH_PERFORMANCE=1
export HF_XET_CHUNK_CACHE_SIZE_BYTES=0
export HF_HOME="$ROOT/hf-cache"
mkdir -p "$BF16_SRC" "$HF_HOME"
"$HF" download "$REPO" --revision "$REVISION" --local-dir "$BF16_SRC" --max-workers 32
printf '%s\n' "$REVISION" > "$BF16_SRC/revision.txt"

export SCRIPT_DIR="$ADAPTER"
export WORK_ROOT="$ROOT/tr3"
export WORK_DIR="$ROOT/tr3/unused-encode-work"
export BF16_SRC OWNER_CORPUS="$ADAPTER/calibration/reap_recall_calib.jsonl"
export BASE_ENCODER_PY="$ADAPTER/encode_tr3_v31.py"
export CAPTURE_PLAN="$PLAN" CAPTURE_DIR
export PYTHON CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export EXLLAMAV3_EXT_SOURCE="$ROOT/code/exllamav3/exllamav3/exllamav3_ext"
export EXL3_ARCH_LIST="${EXL3_ARCH_LIST:-10.0}"
export TORCH_CUDA_ARCH_LIST="$EXL3_ARCH_LIST"
export TR3_BITS=3 MAX_JOBS="${MAX_JOBS:-32}"
export B300_DISK_RESERVE_BYTES="${B300_DISK_RESERVE_BYTES:-214748364800}"
export B300_RAMFS_RESERVE_BYTES="${B300_RAMFS_RESERVE_BYTES:-68719476736}"
"$ADAPTER/convert_b300.sh" preflight
"$ADAPTER/convert_b300.sh" ext
"$ADAPTER/convert_b300.sh" plan

for WINDOW in "${WINDOWS[@]}"; do
  READY="$EXPORT/$WINDOW/READY.json"
  if [[ -s "$ACKS/$WINDOW.ack" ]]; then
    echo "window $WINDOW already acknowledged"
    continue
  fi
  rm -rf "$CAPTURE_DIR" "$EXPORT/$WINDOW.new" "$EXPORT/$WINDOW"
  mkdir -p "$CAPTURE_DIR" "$EXPORT/$WINDOW.new"
  LAYERS="$WINDOW" "$ADAPTER/convert_b300.sh" capture-window
  cp "$PLAN" "$EXPORT/$WINDOW.new/capture_plan.json"
  rsync -a "$CAPTURE_DIR/" "$EXPORT/$WINDOW.new/capture/"
  WINDOW="$WINDOW" EXPORT_DIR="$EXPORT/$WINDOW.new" "$PYTHON" - <<'PY'
import hashlib, json, os, time
from pathlib import Path
root = Path(os.environ["EXPORT_DIR"])
files = {}
for path in sorted(root.rglob("*")):
    if not path.is_file():
        continue
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    files[path.relative_to(root).as_posix()] = {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}
body = {
    "schema": "glm53-capture-window-export/1",
    "window": os.environ["WINDOW"],
    "files": files,
    "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
body["receipt_sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
(root / "READY.json").write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
PY
  mv "$EXPORT/$WINDOW.new" "$EXPORT/$WINDOW"
  echo "CAPTURE_WINDOW_READY $WINDOW"
  DEADLINE=$((SECONDS + 14400))
  while [[ ! -s "$ACKS/$WINDOW.ack" ]]; do
    (( SECONDS < DEADLINE )) || { echo "ack timeout for $WINDOW" >&2; exit 5; }
    sleep 10
  done
  rm -rf "$EXPORT/$WINDOW" "$CAPTURE_DIR"
  echo "CAPTURE_WINDOW_ACKNOWLEDGED $WINDOW"
done
printf 'CAPTURE_WORKER_COMPLETE utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
