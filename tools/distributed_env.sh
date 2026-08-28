#!/usr/bin/env bash
# Source before torchrun. DISTRIBUTED_SCOPE=single|multi (default single).
set -euo pipefail
DISTRIBUTED_SCOPE="${DISTRIBUTED_SCOPE:-single}"
DATA_IFACE="${DATA_IFACE:-$(ip route get 1.1.1.1 2>/dev/null | sed -n 's/.* dev \([^ ]*\).*/\1/p' | head -1)}"
[ -n "$DATA_IFACE" ] || { echo "cannot determine data interface" >&2; return 2 2>/dev/null || exit 2; }
if [ "$DISTRIBUTED_SCOPE" = single ]; then
  export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo,$DATA_IFACE}"
  export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
elif [ "$DISTRIBUTED_SCOPE" = multi ]; then
  export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-$DATA_IFACE}"
  export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-$DATA_IFACE}"
else
  echo "DISTRIBUTED_SCOPE must be single or multi" >&2
  return 2 2>/dev/null || exit 2
fi
export MASTER_PORT="${MASTER_PORT:-$((29500 + RANDOM % 2000))}"
printf 'DISTRIBUTED_ENV scope=%s iface=%s nccl=%s gloo=%s port=%s\n' \
  "$DISTRIBUTED_SCOPE" "$DATA_IFACE" "$NCCL_SOCKET_IFNAME" "$GLOO_SOCKET_IFNAME" "$MASTER_PORT"
