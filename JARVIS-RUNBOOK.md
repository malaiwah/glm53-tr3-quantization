# JarvisLabs release-race operations

Validated against `jarvislabs` CLI 0.2.17 and the live account on 2026-08-28. Official reference: <https://docs.jarvislabs.ai/cli/>.

## Authentication and CLI

Install and upgrade with `uv`:

```bash
uv tool install jarvislabs
uv tool upgrade jarvislabs
export JL_API_KEY='<temporary key>'
jl status
```

Do not commit or echo the key. `JL_API_KEY` overrides the saved config at `~/.config/jl/config.toml`. Revoke temporary keys after the campaign.

Use `--json` for starts and machine-readable reads, and `--yes` for commands that would prompt. Start managed runs detached with JSON, check once after 15 seconds, then poll every 60–120 seconds with bounded log tails:

```bash
jl run . --script job.py --on "$MACHINE" --yes --json
jl run logs "$RUN_ID" --tail 30
jl run status "$RUN_ID" --json
```

Never use unbounded logs or `--follow` from an agent. `jl exec ID -- command` is for short diagnostics; `jl run` is for supervised jobs with PID/log state.

## Billing and persistence

- `jl pause ID` stops CPU/GPU compute billing. Instance-storage billing continues.
- `/home` persists across pause/resume. Treat system packages and files outside `/home` as ephemeral; rerun node preparation after resume.
- Destroying an instance deletes its instance-local `/home`. A managed filesystem survives instance pause, resume, and destroy.
- The current CLI's `cost` field is accrued session cost for running instances or accrued storage cost for paused instances, not an hourly rate. Get rates from `jl gpus --json` / `jl cpus --json`.
- Paused instances do not reserve a GPU. Resume is still capacity-dependent.

## Shared filesystem

Managed filesystems are region-bound but multi-attach within a region. This was exercised concurrently from two RTX containers and one CPU VM.

Provisioned release storage:

```text
fs_id:   3423
name:    glm53-main-store
region:  IN1
size:    10240 GB
mount:   /home/jl_fs
```

The published docs allow 50–10,240 GB. CLI 0.2.17 incorrectly caps validation at 2,048 GB; the documented API accepted pre-use expansions 2,048 → 5,120 → 10,240 GB and rotated `fs_id` 3421 → 3422 → 3423. The volume was verified after each expansion. Never resize it while campaign nodes are attached.

CLI 0.2.17 also rejects `--fs-id` for CPU VMs although the web/backend supports it. The runner was attached through the CPU resume API and verified at `/home/jl_fs`.

Use atomic writes and per-work-unit locks: simultaneous mount access does not make two writers to the same output safe.

Persistent layout:

```text
/home/jl_fs/bin        shared uv and immutable utilities
/home/jl_fs/release    pinned code/bundles
/home/jl_fs/cache      environment-keyed compile/download caches
/home/jl_fs/models     immutable source checkpoints
/home/jl_fs/captures   calibration captures
/home/jl_fs/work       resumable work units
/home/jl_fs/outputs    materialized candidates
/home/jl_fs/receipts   machine-readable gates
/home/jl_fs/locks      cross-node claims/locks
```

## Current release nodes

| Role | ID | Shape | Region | Local disk | Shared FS | Normal state |
|---|---:|---|---|---:|---:|---|
| runner | 485036 | CPU VM, 4 vCPU / 16 GB | IN1 | 100 GB | 3423 | running for release watch/downloads |
| quant worker A | 485056 | 1× RTX PRO 6000 96 GB spot container | IN1 | 100 GB | 3423 | paused until encode |
| quant worker B | 485057 | 1× RTX PRO 6000 96 GB spot container | IN1 | 100 GB | 3423 | paused until encode |

Worker spot rate at this snapshot: $0.99/GPU-hour. Runner rate: $0.0992/hour. Two workers halve wall clock when work units are independent; total GPU cost should remain approximately constant. CPU/seal contention must be measured rather than assumed.

Authenticated HF download measurements (`HF_XET_HIGH_PERFORMANCE=1`, distinct
5.36 GB GLM-5.2 shards, destination filesystem 3422 (same volume, now expanded as 3423):

| Runner | hf-xet throughput | Projected 1.507 TB source | Runner compute |
|---|---:|---:|---:|
| 2 vCPU / 8 GB | 223.3 MB/s | 112.4 min | $0.09 |
| **4 vCPU / 16 GB** | **243.7 MB/s** | **103.1 min** | **$0.17** |
| 8 vCPU / 32 GB | 233.3 MB/s | 107.6 min | $0.36 |

Use 4 vCPU / 16 GB: it was the fastest observed and leaves CPU capacity for
hashing/manifests. Network plus hf-xet plus the managed filesystem plateaus at
roughly 1.8–1.95 Gbit/s; allocating 8 vCPUs did not improve it. Single-stream
HTTP range tests were noisy (33–77 MB/s) and are not the release sizing metric.
Receipt: `receipts/hf-bandwidth.json`.

The large original-weight capture node is intentionally not provisioned yet. GLM-5.2 BF16 is about 1.507 TB. Current IN1 availability tops out at 8×96 GB = 768 GB, and current IN2 8×H200 totals about 1.128 TB; neither cleanly fits a BF16 744B model plus runtime overhead. Wait for a fitting B300/GB300-class shape or define and qualify an explicit CPU-offload/multi-node plan before spending.

## Node preparation

Every new/resumed node starts with:

```bash
sudo apt update -y
sudo apt dist-upgrade -y
```

Then run `tools/prepare_jarvis_node.sh`. It installs `tmux`, `nvtop`, `rsync`, `skopeo`, `umoci`, and build tools, verifies shared `uv`, and writes a sealed node receipt. Python environments use `uv` and live under `/home` or the shared filesystem.

Jarvis containers cannot run nested containers. To consume a published image there, use `skopeo copy` plus `umoci unpack` and run against the extracted rootfs. Jarvis VMs may run Docker normally.

## SSH turnaround

Use ControlMaster for all repeated VM/container supervision:

```sshconfig
ControlMaster auto
ControlPath ~/.ssh/cm-%C
ControlPersist 15m
ServerAliveInterval 30
ServerAliveCountMax 4
```

Use `rsync -e ssh`; it automatically rides the established control socket. Current aliases are `glm53-main-runner`, `glm53-main-worker-a`, and `glm53-main-worker-b`.

## Multi-GPU VM P2P gate

Before any multi-GPU RTX PRO 6000 VM workload, run `tools/configure_p2p_vm.sh` with no GPU processes active. It applies and verifies:

```text
options nvidia NVreg_RegistryDwords="ForceP2P=0x11;RMForceP2PType=1;RMPcieP2PType=2;GrdmaPciTopoCheckOverride=1;EnableResizableBar=1"
options nvidia_uvm uvm_disable_hmm=1
```

The file on disk is not proof; reload the NVIDIA modules or reboot, then verify `/proc/driver/nvidia/params`, the UVM parameter, `nvidia-smi topo -m`, and a P2P latency test. The ForceP2P override is specifically critical on direct-attach/NODE topologies; switch topologies still require measurement because ACS can change the best path. Reference: <https://github.com/local-inference-lab/rtx6kpro/blob/master/hardware/pcie-bandwidth.md#nvidia-p2p-driver-override-forcep2p>.

## Current shape snapshot

Availability changes continuously; always rerun `jl gpus --json` before provisioning.

| Region | Shape | VRAM | Workload | Free now | On-demand | Spot |
|---|---|---:|---|---:|---:|---:|
| IN1 | RTX PRO 6000 | 96 GB | VM | 2 | $1.89/h | $0.99/h |
| IN1 | RTX PRO 6000 | 96 GB | container | 8 | $1.89/h | $0.99/h |
| IN2 | A30 | 24 GB | container | 4 | $0.41/h | $0.29/h |
| IN2 | L4 | 24 GB | VM | 8 | $0.44/h | $0.29/h |
| IN2 | L4 | 24 GB | container | 6 | $0.44/h | $0.29/h |
| IN2 | A100 | 40 GB | container | 1 | $0.89/h | $0.79/h |
| IN2 | A100 80 GB | 80 GB | VM/container | 1 each | $1.49/h | $0.89/h |
| IN2 | H100 | 80 GB | VM | 8 | $2.69/h | $1.19/h |
| IN2 | H100 | 80 GB | container | 0 | $2.69/h | $1.19/h |
| IN2 | H200 | 141 GB | VM | 8 | $3.99/h | $1.99/h |
| IN2 | H200 | 141 GB | container | 4 | $3.99/h | $1.99/h |
| EU1 | H100 | 80 GB | unspecified | 1 | $2.99/h | — |
| EU1 | H200 | 141 GB | unspecified | 1 | $3.99/h | — |

CPU VMs:

| vCPU | RAM | Region availability | Rate |
|---:|---:|---|---:|
| 2 | 8 GB | IN1, IN2 | $0.0496/h |
| 4 | 16 GB | IN1, IN2 | $0.0992/h |
| 8 | 32 GB | IN1, IN2 | $0.1984/h |
| 16 | 64 GB | IN1, IN2 | $0.3968/h |
| 32 | 128 GB | IN2 only | $0.7936/h |
