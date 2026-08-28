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
- A 10-TB IN1 filesystem and the release source state are preserved. The CPU runner, two 4×RTX PRO 6000 workers, and 8×H200 capture node are paused.
- GLM-5.3 was pinned at the first observed release SHA `30333038ada1f1dacb294a93270305a890b50c14`. Its topology is unchanged: 78 base layers plus MTP-78, 256 routed experts, natural top-8 routing, 6,144 hidden width, and 2,048 expert width.
- `zai-org/GLM-5.3` is the 755.7-GB official FP8 checkpoint. The quant source is the separate `zai-org/GLM-5.3-BF16`, pinned at `304b8051cfb2b260b61ce0cbe330e02a98e73639`: 282 shards and 1,506,687,604,850 bytes.
- The official BF16 metadata preflight is release-ready with 59,585 tensors, 57,600 main routed-expert tensors, and 768 MTP routed-expert tensors. IN1 and H200 each closed an exact 291-file / 1,506,693,048,122-byte census.
- The Jarvis-only route passed its 8×H200 TP8 worker-extension and full-BF16 rehearsal gates. The production capture was stopped during model loading before any layer manifest was sealed.
- Flat K3/K4 prioritization and fail-loud unmeasured publication gates remain implemented but were not executed.

Campaign stopped:

- The operator stopped the K3/K4 race at `2026-08-28T17:10:53Z` after competing releases moved ahead.
- No GLM-5.3 capture, K3/K4 work unit, assembled checkpoint, upload, or public model was completed.
- Machines `485730`, `485732`, `485743`, and runner `485098` were paused; all watchers, bridges, dispatchers, and budget processes were stopped.
- Source checkpoints, receipts, prepared adapters, and repository tooling remain on managed storage for an explicit later restart. Managed-filesystem storage billing continues while compute is paused.

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
tools/capture_controller.py            external fallback capture streaming and cleanup
tools/build_h200_offload_capture.py    H200-compatible TP8 capture adapter
tools/h200_release_rearm.py            scheduled resume, runtime cap, bridge wait, pause
tools/h200_release_capture.py          official-BF16 capture and sealed window export
tools/jarvis_capture_bridge.py         exact IN2→IN1 window transfer and verification
tools/quant_dispatch.py                official-BF16 flat K3/K4 first, then mixed 3.42
tools/write_unmeasured_card.py         fail-loud first-payload disclosure
tools/unmeasured_publication_gate.py   exact-artifact public gate without fidelity claims
tools/publication_gate.py              strict KLD/Gilded five-run qualification gate
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
