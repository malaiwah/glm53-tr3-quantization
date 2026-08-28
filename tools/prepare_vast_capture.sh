#!/usr/bin/env bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt update -y
apt dist-upgrade -y
apt install -y tmux nvtop rsync git curl jq ca-certificates openssh-client build-essential pciutils
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
install -d -m 700 /workspace/glm53/{source,capture,receipts,logs,code}
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader \
  | tee /workspace/glm53/receipts/gpus.txt
nvidia-smi topo -m | tee /workspace/glm53/receipts/topology.txt
nvidia-smi topo -p2p r | tee /workspace/glm53/receipts/p2p-read.txt
nvidia-smi topo -p2p w | tee /workspace/glm53/receipts/p2p-write.txt
date -u +%Y-%m-%dT%H:%M:%SZ | tee /workspace/glm53/receipts/prepared-utc.txt
printf 'VAST_CAPTURE_READY\n'
