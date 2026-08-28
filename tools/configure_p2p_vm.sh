#!/usr/bin/env bash
set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then exec sudo -E "$0" "$@"; fi
if [ "$(systemd-detect-virt --container 2>/dev/null || true)" != none ] && systemd-detect-virt --container --quiet; then
  echo "P2P kernel-module configuration must run on a VM host, not a container" >&2
  exit 2
fi
GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
if [ "$GPU_COUNT" -lt 2 ]; then
  echo "single-GPU node: P2P module gate not applicable"
  exit 0
fi
ACTIVE="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^$/d' || true)"
if [ -n "$ACTIVE" ]; then
  echo "GPU processes are active; stop them before reloading NVIDIA modules: $ACTIVE" >&2
  exit 3
fi

cat > /etc/modprobe.d/nvidia-p2p-override.conf <<'EOF'
options nvidia NVreg_RegistryDwords="ForceP2P=0x11;RMForceP2PType=1;RMPcieP2PType=2;GrdmaPciTopoCheckOverride=1;EnableResizableBar=1"
EOF
cat > /etc/modprobe.d/uvm.conf <<'EOF'
options nvidia_uvm uvm_disable_hmm=1
EOF

systemctl stop nvidia-persistenced 2>/dev/null || true
if ! modprobe -r nvidia_uvm nvidia_drm nvidia_modeset nvidia; then
  echo "module options are installed but the driver could not be unloaded; reboot before use" >&2
  exit 4
fi
modprobe nvidia
modprobe nvidia_uvm
modprobe nvidia_modeset 2>/dev/null || true
systemctl start nvidia-persistenced 2>/dev/null || true

PARAMS="$(cat /proc/driver/nvidia/params)"
for token in 'ForceP2P=0x11' 'RMForceP2PType=1' 'RMPcieP2PType=2' \
             'GrdmaPciTopoCheckOverride=1' 'EnableResizableBar=1'; do
  grep -Fq "$token" <<<"$PARAMS" || {
    echo "loaded NVIDIA parameters lack $token" >&2
    exit 5
  }
done
UVM_VALUE="$(cat /sys/module/nvidia_uvm/parameters/uvm_disable_hmm 2>/dev/null || true)"
case "$UVM_VALUE" in 1|Y|y) ;; *) echo "uvm_disable_hmm is not active: $UVM_VALUE" >&2; exit 5;; esac

nvidia-smi topo -m
nvidia-smi topo -p2p r || true
cat /proc/driver/nvidia/params | grep -E \
  'RegistryDwords|ForceP2P|RMForceP2PType|RMPcieP2PType|GrdmaPciTopoCheckOverride|EnableResizableBar|DmaRemapPeerMmio'
echo "P2P_MODULE_GATE_OK GPUs=$GPU_COUNT"
