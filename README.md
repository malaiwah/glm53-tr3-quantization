# GLM-5.3 TR3 quantization race

Reproducible tooling for producing and evaluating GLM-5.3 `glm_moe_dsa` routed-expert quants:

- flat K3 and flat K4 TR3 parts bins;
- topology-neutral offline assembly, including the GLM-5.2-style 3.42-bpw budget;
- fresh calibration activation capture;
- BF16/FP8/NVFP4/quant hidden-state and full-vocabulary KLD comparison;
- MTP K3/K4/K5/K6 overlays;
- attention K6/K8 offline or vLLM-GG online-cache variants;
- receipt-gated Hugging Face publication.

The repository contains no credentials or model weights. Operators provide their own JarvisLabs, GitHub, and Hugging Face keys.

## Current status

Green:

- GLM-5.2 topology and published 3.42 tier budget reproduced from pinned metadata.
- exllamav3 0.0.43 CUDA extension compiled on RTX PRO 6000 SM120 with Torch/CUDA 13.0.
- Calibrated K3 and K4 core smoke passed both GLM geometries (`6144×512`, `512×6144`).
- Pack/unpack/reconstruct exactness passed.
- Sequential v2, lockstep v3, and pooled-GSS v3.1 were byte-identical.
- Authenticated hf-xet benchmark selected a 4-vCPU/16-GB download runner.
- A 10-TB IN1 shared filesystem, CPU runner, and two paused one-GPU RTX PRO 6000 workers are prepared.
- A four-repository watcher polls GLM-5.2, GLM-5.2-FP8, GLM-5.3, and GLM-5.3-FP8 and runs baseline full-download smoke tests.

Open gates:

- GLM-5.3 weights are not released yet.
- The public historical B300 adapter is flat K3 and pre-shared-H; the production plan uses independent uniform K3/K4 parts bins and offline tier assembly rather than a mixed live campaign.
- A full real-weight prep → encode → seal work-unit rehearsal remains required.
- A fitting original-BF16 capture node is not currently listed on JarvisLabs; 8×H200 and 8×RTX PRO 6000 do not cleanly fit the roughly 1.5-TB BF16 checkpoint plus runtime overhead.

No public model is called qualified until its sealed KLD and runtime receipts are green.

## Layout

```text
smoke-contract.json          pinned GLM-5.2 reference and micro-smoke contract
tools/preflight.py           topology, tier-budget, artifact, and source checks
tools/run_codec_smoke.py     K3/K4 calibrated codec proof on one GPU
run_gpu_codec_smoke.sh       uv-based GPU launcher
tools/watch_matrix.py        release/baseline poll and concurrent job supervisor
tools/on_release.py          authenticated pin, topology preflight, and full download
tools/jl_resume_ssh.py       resume → IP → SSH alias → ControlMaster → SSH_OK
tools/prepare_jarvis_node.sh idempotent OS/tool/node receipt preparation
tools/configure_p2p_vm.sh    multi-GPU RTX VM NVIDIA P2P gate
JARVIS-RUNBOOK.md            lifecycle, shapes, persistence, cost, and P2P runbook
```

## Local preflight

```bash
python3 tools/preflight.py
python3 -m py_compile tools/*.py
bash -n run_gpu_codec_smoke.sh tools/*.sh
```

## One-GPU codec smoke

Requires a CUDA-enabled Python environment with Torch and a matching CUDA toolkit. Python dependency management uses `uv`.

```bash
UV=/path/to/uv ./run_gpu_codec_smoke.sh
```

The receipt is written to `receipts/gpu-codec-smoke.json`.

## JarvisLabs operation

See [JARVIS-RUNBOOK.md](JARVIS-RUNBOOK.md). Core rules:

- update the OS before use;
- keep persistent state on the managed filesystem;
- use `uv`, `tmux`, `nvtop`, and ControlMaster-backed `rsync`;
- use `skopeo` + `umoci` in containers instead of nested Docker;
- apply and verify NVIDIA P2P module settings before multi-GPU VM work;
- supervise active rentals every ten minutes;
- pause nodes immediately after their role ends.

## Attribution

- Z.ai: GLM-5.2 and GLM-5.3 source models.
- Brandon Music: GLM-5.2 calibrated v3.1 TR3 encoder and corpus lineage.
- willfalco: GLM-5.2 3.42-bpw coder-aligned reference checkpoint.
- turboderp-org: exllamav3 and TR3/MCG codec.
- local-inference-lab: vLLM-GG/B12X runtime and RTX PRO 6000 operational documentation.
- malaiwah: campaign orchestration, fidelity/KLD, MTP capture, assembly, and publication tooling.

See the upstream model and code licenses before publishing derived weights.
