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
- Calibrated K3/K4 codec exactness passed both GLM geometries; v2, lockstep v3, and pooled-GSS v3.1 were byte-identical.
- Real GLM-5.2 BF16 layer-3 uniform K3/K4 work units encoded all 256 experts and rehashed exactly.
- Full BF16 (1.507 TB) and official FP8 (755.7 GB) downloads completed exact file censuses with serialized four-worker hf-xet on an 8-vCPU/32-GB runner.
- The two-pass `shared_h_v1` writer emits the published 9,228-tensor schema. All 76 qualified GLM-5.2 sign rows are range-fetched and sealed; template-bound real-BF16 K3/K4 parts produced the same shared-profile SHA, then passed a 206/50 whole-expert mixed materialization.
- A 10-TB IN1 filesystem, always-on CPU runner, and two paused RTX PRO 6000 workers are prepared. P2P settings were proven to survive VM pause/resume.
- RunPod and Vast inventory watchers, release/topology/funding/hourly guards, capture streaming, quant dispatch, mixed-tier materialization, private upload, and receipt-bound publication gates are implemented.

Open gates:

- GLM-5.3 main weights are not released yet.
- Template-bound K3/K4 shared-H outputs still require production-capture KLD and Gilded Gnosis runtime qualification. Synthetic-fixture weight NMSE is explicitly not a quality gate.
- RunPod currently has no count-8 B300 stock. Vast has a guarded 8×B300 path, but its account is below the $420 eight-hour campaign funding gate.
- Jarvis balance is $53.25; campaign GPUs are paused, but the user's separate active VMs may exhaust that account before release unless they finish or it is topped up.
- Real K3/K4 parts passed the 206/50 mixed layer materialization and trellis-shape audit. Full checkpoint assembly awaits the released source tree.
- MTP and attention variants follow the flat K3/K4/3.42 publication path.

No public model is called qualified until its sealed KLD and runtime receipts are green.

## Layout

```text
smoke-contract.json                    pinned GLM-5.2 reference and micro-smoke contract
tools/preflight.py                     topology, tier-budget, artifact, and source checks
tools/watch_matrix.py                  release/baseline poll and download supervisor
tools/on_release.py                    authenticated pin, topology preflight, full download
tools/build_uniform_adapter.py         hash-pinned K3/K4/K5/K6 shared-H adapter builder
tools/shared_h_overlay.py              two-pass shared-H profile and writer
tools/fetch_shared_h_sign_template.py  ranged qualified-sign extraction
tools/capture_controller.py            external capture streaming and provider cleanup
tools/quant_dispatch.py                capture-driven K3/K4/3.42 assembly and private upload
tools/work_queue.py                    atomic resumable cross-node claims
tools/publication_gate.py              artifact/KLD/Gilded five-run public gate
tools/jl_resume_ssh.py                 resume → IP → ControlMaster → SSH_OK
JARVIS-RUNBOOK.md                      lifecycle, persistence, cost, and P2P runbook
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
- willfalco: GLM-5.2 3.42-bpw reference, tier plan, and qualified shared-H sign rows.
- turboderp-org: exllamav3 and TR3/MCG codec.
- local-inference-lab: public shared-H recipe, vLLM-GG/B12X runtime, and RTX PRO 6000 operational documentation.
- malaiwah: campaign orchestration, fidelity/KLD, MTP capture, assembly, and publication tooling.

See the upstream model and code licenses before publishing derived weights.
