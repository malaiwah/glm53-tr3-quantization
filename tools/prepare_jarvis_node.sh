#!/usr/bin/env bash
set -euo pipefail
ROLE="${1:-worker}"
SHARED_ROOT="${SHARED_ROOT:-/home/jl_fs}"
export DEBIAN_FRONTEND=noninteractive

if command -v sudo >/dev/null 2>&1; then APT=(sudo apt); else APT=(apt); fi
"${APT[@]}" update -y
"${APT[@]}" dist-upgrade -y
"${APT[@]}" install -y \
  build-essential ca-certificates curl git jq ninja-build nvtop rsync skopeo tmux umoci

mkdir -p "$SHARED_ROOT"/{bin,release,cache,models,captures,work,outputs,receipts/nodes,locks}
test -x "$SHARED_ROOT/bin/uv" || {
  echo "shared uv binary missing at $SHARED_ROOT/bin/uv" >&2
  exit 2
}

if command -v nvidia-smi >/dev/null 2>&1 && python3 -c 'import torch' >/dev/null 2>&1; then
  TORCH_CUDA="$(python3 -c 'import torch; print(torch.version.cuda or "")')"
  MATCHING_NVCC="/usr/local/cuda-$TORCH_CUDA/bin/nvcc"
  if [ -n "$TORCH_CUDA" ] && [ ! -x "$MATCHING_NVCC" ]; then
    CUDA_MAJOR="${TORCH_CUDA%%.*}"
    "${APT[@]}" install -y "cuda-toolkit-${CUDA_MAJOR}-0"
  fi
fi

NODE_ROLE="$ROLE" SHARED_ROOT="$SHARED_ROOT" python3 - <<'PY'
import hashlib, json, os, pathlib, platform, shutil, socket, subprocess, time

def command(*args):
    try:
        return subprocess.run(args, check=True, text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT).stdout.strip()
    except Exception as exc:
        return f"unavailable: {exc}"

root = pathlib.Path(os.environ["SHARED_ROOT"])
host = socket.gethostname()
body = {
    "schema": "glm53-jarvis-node-preparation/1",
    "host": host,
    "role": os.environ["NODE_ROLE"],
    "platform": platform.platform(),
    "kernel": platform.release(),
    "python": platform.python_version(),
    "uv": command(str(root / "bin/uv"), "--version"),
    "gpu": command("nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader")
           if shutil.which("nvidia-smi") else None,
    "topology": command("nvidia-smi", "topo", "-m") if shutil.which("nvidia-smi") else None,
    "filesystem": command("findmnt", "-T", str(root), "-o", "TARGET,SOURCE,FSTYPE", "-n"),
    "utilities": {name: shutil.which(name) for name in ("tmux", "nvtop", "rsync", "skopeo", "umoci")},
    "prepared_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
body["receipt_sha256"] = hashlib.sha256(json.dumps(
    body, sort_keys=True, separators=(",", ":")
).encode()).hexdigest()
out = root / "receipts/nodes" / f"{host}.json"
tmp = out.with_suffix(".json.tmp")
tmp.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
tmp.replace(out)
print(out)
PY
