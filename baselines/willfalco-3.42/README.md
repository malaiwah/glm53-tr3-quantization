---
language:
- en
- zh
license: mit
library_name: transformers
pipeline_tag: text-generation
base_model: zai-org/GLM-5.2
tags:
- glm
- exl3
- trellis
- vllm
- blackwell
- mixture-of-experts
- quantization
- compression
inference: false
---

# GLM-5.2 EXL3 TR3 3.42 bpw Coder

with Coding expert allignments from [3.25bpw](https://huggingface.co/willfalco/GLM-5.2-EXL3-TR3-3.25bpw)/[NF3](https://huggingface.co/madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid)

This is a TP4, rank-sliced EXL3 Trellis build of
[zai-org/GLM-5.2](https://huggingface.co/zai-org/GLM-5.2), optimized for
four NVIDIA Blackwell workstation GPUs. Routed MoE experts in layers 3-78 use
EXL3 Trellis weights targeting 3.0/4.0 bits per weight, including the MTP (layer 78) routed experts using [malaiwah's calibration-capture](https://huggingface.co/datasets/malaiwah/GLM-5.2-MTP78-calibration-capture).
Accuracy-sensitive and dense components remain in BF16 but can be used in mxfp8 or EXL3 Trellis 6bpw format (see below).
The repository payload is 327 GiB. This format requires the
custom vLLM + Sparkinfer runtime below; it is not a drop-in Transformers model.
The routed weights are EXL3 Trellis and the required launch
flag is `--quantization exl3`. NVFP4 in the supplied runtime refers to the KV
cache, not the routed-expert weight format.

```
Weights        | KV format                 | KLD
───────────────────────────────────────────────────────────────────────
NF3            | Dynamic NVFP4 + RoPE8     | 0.139036 ± 0.002010
NF3            | Standard FP8 + BF16 RoPE  | 0.1263†
EXL3 3.0-bpw   | Dynamic NVFP4 + RoPE8     | 0.119525
EXL3 3.0-bpw   | Standard FP8 + BF16 RoPE  | 0.102508
EXL3 3.25-bpw  | Dynamic NVFP4 + RoPE8     | 0.095971
EXL3 3.25-bpw  | Standard FP8 + BF16 RoPE  | 0.087711
EXL3 3.36-bpw  | Dynamic NVFP4 + RoPE8     | 0.077767
EXL3 3.36-bpw  | Standard FP8 + BF16 RoPE  | 0.068458
EXL3 3.40-bpw  | Dynamic NVFP4 + RoPE8     | ...
EXL3 3.40-bpw  | Standard FP8 + BF16 RoPE  | ...
EXL3 3.40-bpw  | FP8 + Dynamic EXL3 6bpw   | ...
EXL3 3.42-bpw  | Dynamic NVFP4 + RoPE8     | ...
EXL3 3.42-bpw  | Standard FP8 + BF16 RoPE  | ...
EXL3 3.42-bpw  | FP8 + Dynamic EXL3 6bpw   | ...
```

FP8 Context 454,656 tok with partial online MXFP8 quant of dense layers, trading more KV for a bit of accuracy:

  KLD 0.06862 `- '--quantization-config={"linear":{"weight":"mxfp8"},"ignore":["re:.*\\.q_a_proj$$","re:.*kv_a_proj_with_mqa"]}'`

  KLD 0.06958 `- '--quantization-config={"linear":{"weight":"mxfp8"},"shared_experts":{"weight":"mxfp8"},"ignore":["re:.*\\.fused_qkv_a_proj$","re:.*\\.q_a_proj$","re:.*kv_a_proj_with_mqa","re:.*\\.mlp\\.gate$","model.layers.78.eh_proj","lm_head"]}'`

  KLD ....... `ONLINE_QUANT=exl3-b6`

Mind that reasoning_effort:high, set to reasoning_effort:max

```
services:
  g52h:
    image: voipmonitor/vllm:gilded-gnosis-v20-vllme1e9426-si200c1db-fi801d57a-cu132-20260804-r28
    container_name: g52h
    ports:
      - "0.0.0.0:8000:8000"
    gpus: all
    shm_size: "32g"
    ipc: "host"
    ulimits:
      memlock: -1
      nofile: 1048576
    environment:
      - CUDA_VISIBLE_DEVICES=0,1,2,3
      - CUDA_DEVICE_MAX_CONNECTIONS=32
      - CUTE_DSL_ARCH=sm_120a
      - OMP_NUM_THREADS=16
      - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
      - SAFETENSORS_FAST_GPU=1
      - NCCL_IB_DISABLE=1
      - NCCL_P2P_LEVEL=SYS
      - NCCL_PROTO=LL,LL128,Simple
      - VLLM_USE_FLASHINFER_SAMPLER=1
      - VLLM_USE_B12X_FP8_GEMM=0  # +kld
      - VLLM_USE_B12X_SPARSE_INDEXER=1
      - VLLM_USE_V2_MODEL_RUNNER=1
      - VLLM_ENABLE_PCIE_ALLREDUCE=1
      - VLLM_PCIE_ALLREDUCE_BACKEND=b12x
      - VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE=64KB
      - VLLM_PCIE_ONESHOT_FUSED_ADD_RMS_NORM_MAX_SIZE=84KB
      - B12X_PCIE_DMA_FP8=0  # +kld
      - B12X_DENSE_SPLITK_TURBO=1
      - B12X_W4A16_TC_DECODE=1
      - B12X_MOE_FORCE_A16=1
      - VLLM_USE_AOT_COMPILE=1
      - VLLM_USE_BREAKABLE_CUDAGRAPH=0
      - VLLM_USE_FUSED_MOE_GROUPED_TOPK=1
      - VLLM_USE_B12X_MHC=1
      - B12X_MHC_MAX_TOKENS=16384
      - VLLM_USE_B12X_WO_PROJECTION=1
      - B12X_MLA_SM120_UNIFIED=1
      - VLLM_CACHE_DIR=/cache/jit/vllm
      - TRITON_CACHE_DIR=/cache/jit/triton
      - TORCH_EXTENSIONS_DIR=/cache/jit/torch_extensions
      - TORCHINDUCTOR_CACHE_DIR=/cache/jit/torchinductor
      - FLASHINFER_WORKSPACE_BASE=/cache/jit/flashinfer
      - XDG_CACHE_HOME=/cache/jit
      - TVM_FFI_CACHE_DIR=/cache/jit/tvm-ffi
      - GLOO_SOCKET_IFNAME=lo
      - NCCL_SOCKET_IFNAME=lo
      - VLLM_WORKER_MULTIPROC_METHOD=spawn
      - VLLM_PCIE_DMA_MIN_BYTES=6MB
      - VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE=0  # +pp +kld
      - VLLM_B12X_MLA_SPEC_DECODE_MAX_Q=8
      - VLLM_USE_B12X_DCP_A2A=1
      - VLLM_DCP_A2A_MAX_TOKENS=16
      - VLLM_DCP_A2A_LARGE_BACKEND=ag_rs
      - VLLM_B12X_MLA_CKV_GATHER=1
      - VLLM_B12X_MLA_CKV_GATHER_MIN_TOKENS=512  # for VLLM_B12X_MLA_CKV_GATHER=1
      - VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS=16384  # for VLLM_B12X_MLA_CKV_GATHER=1
      - VLLM_DCP_QUERY_SPLIT=1  # r14
      - VLLM_MEMORY_PROFILE_INCLUDE_ATTN=1
      - VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1
      - TORCH_CUDA_ARCH_LIST=12.0a
      - FLASHINFER_CUDA_ARCH_LIST=12.0f
      - FLASHINFER_DISABLE_VERSION_CHECK=1
      - VLLM_USE_B12X_MOE=1
      - VLLM_CPP_AR_1STAGE_NCCL_CUTOFF=56KB
      - VLLM_CPP_AR_IGNORE_CUTOFF_MAX_ROWS=0
      - VLLM_RTX6K_FUSED_ALLREDUCE_ADD=0
      - VLLM_RTX6K_FUSED_ALLREDUCE_ADD_END_BARRIER=0
      - VLLM_DISABLE_SHARED_EXPERTS_STREAM=0  # v20
      - VLLM_DISABLED_KERNELS=MarlinFP8ScaledMMLinearKernel
      - VLLM_DCP_GLOBAL_TOPK=1
      - VLLM_DCP_SHARD_DRAFT=1
      - VLLM_DCP_QUERY_SPLIT=0
      - VLLM_EXL3_TRELLIS_MIN_M=1
      - VLLM_EXL3_TRELLIS_MAX_M=48
      - VLLM_EXL3_TRELLIS_BLOCK_M=8
      - VLLM_EXL3_PREFILL_CHUNK=128
      - KV_FP8_ROPE=0  # +kld
      - VLLM_B12X_ABSORB_BMM=0
      - ONLINE_QUANT=exl3-b6
      - VLLM_EXL3_ONLINE_TRELLIS_BITS=6
      - VLLM_EXL3_ENCODER_SOURCE=/opt/exllamav3-python/exllamav3
      - VLLM_EXL3_ONLINE_CACHE_DIR=/cache/exl3-online
      - VLLM_EXL3_ONLINE_CACHE_MODE=readwrite
    volumes:
      - /data1/GLM-5.2-EXL3-TR3-3.42bpw:/model:ro
      - /data1/GLM-5.2-EXL3-TR3-3.42bpw.cache:/cache:rw
      - /data1/GLM-5.2-EXL3-TR3-3.42bpw.cache:/root/.cache:rw
      - /data1/GLM-5.2-EXL3-TR3-3.42bpw.cache:/container-tmp:rw
    entrypoint:
      - /bin/sh
      - -c
      - "unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS && exec vllm serve \"$@\""
      - --
    command:
      - /model
      - --served-model-name=g52h
      - --trust-remote-code
      - --tensor-parallel-size=4
      - --decode-context-parallel-size=4
      - --dcp-comm-backend=a2a
      - --dcp-kv-cache-interleave-size=1
      - --quantization=exl3
      - --kv-cache-dtype=fp8
      - --attention-backend=B12X_MLA_SPARSE
      - --moe-backend=b12x
      - --load-format=safetensors
      - '--compilation-config={"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"],"pass_config":{"fuse_allreduce_rms":true}}'
      - --gpu-memory-utilization=0.971
      - '--quantization-config={"linear":{"weight":"mxfp8"},"ignore":["re:.*\\.q_a_proj$$","re:.*kv_a_proj_with_mqa"]}'  # KLD 0.06862
      - --max-model-len=128128
      - --max-num-seqs=16
      - --max-num-batched-tokens=2048
      - --max-cudagraph-capture-size=64
      - --enable-auto-tool-choice
      - --tool-call-parser=glm47
      - --reasoning-parser=glm45
      - --enable-prefix-caching
      - --enable-chunked-prefill
      - --no-async-scheduling
      - --enable-flashinfer-autotune
      - '--default-chat-template-kwargs={"reasoning_effort":"high"}'
      - '--hf-overrides={"use_index_cache":true,"index_topk_pattern":"FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS"}'
      - '--speculative-config={"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton","draft_sample_method":"greedy"}'
#      - '--override-generation-config={"top_p":0.95,"repetition_penalty":1.18}'  # for temp=0.1 MMLU-Pro
      - --host=0.0.0.0
      - --port=8000
```

## Source

- [vLLM EXL3 integration PR](https://github.com/local-inference-lab/vllm/pull/139)
- [Sparkinfer EXL3 Trellis PR](https://github.com/local-inference-lab/b12x/pull/49)
- [Upstream GLM-5.2 model](https://huggingface.co/zai-org/GLM-5.2)
- [GLM-5 technical report](https://arxiv.org/abs/2602.15763)
- [brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw](https://huggingface.co/brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw)
- [madeby561's NF3](https://huggingface.co/madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid)

## License

The model and this derivative are released under the MIT license. See
`LICENSE` and the upstream model card for attribution and usage terms.
