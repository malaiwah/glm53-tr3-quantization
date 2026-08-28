# GLM-5.3 TR3 quantization campaign plan

## Mission and release order

1. Publish flat K3 and flat K4 as independently sealed, KLD-measured TR3 models.
2. Preserve both uniform campaigns as a topology-neutral parts bin.
3. Assemble the GLM-5.2-style 3.42-bpw budget offline from whole K3/K4 experts.
4. Publish fresh calibration activations and fidelity/KLD datasets.
5. Add MTP K3/K4/K5/K6 and attention K6/K8 variants after the core three models are public.

Weights upload privately as soon as materialized. Public visibility requires at least a sealed static/roundtrip gate and the quick same-panel KLD receipt. Five-cold-run and task/runtime results update the card afterward. No unmeasured output is labelled qualified.

## Reasonableness of the quality objective

The process is reasonable; the requested numeric outcome is a stretch target, not something tooling can guarantee.

Historical GLM-5.2 evidence says flat K4 can approach official FP8, while flat K3 and a 3.42-bpw model may not beat a strong NVFP4 checkpoint. Therefore:

- **Hard requirement:** measure every candidate under the same tokens, head, score window, runtime geometry, and numerical reducer.
- **Hard requirement:** do not publish a false “beats FP8/NVFP4” claim.
- **Release gate:** finite complete receipts, static integrity, KLD below the declared safety ceiling, and no material regression versus the corresponding GLM-5.2 rate baseline.
- **Stretch target:** flat K4 or optimized 3.42 beats NVFP4 and comes within the official FP8 confidence interval.
- **Diagnostic product:** flat K3 remains useful as the minimum-size point even if it cannot beat NVFP4.

Report weight-only/shared-head KLD separately from as-served KLD with online dense/attention quantization and compressed KV. Otherwise a runtime overlay can be incorrectly attributed to routed weights.

## Architecture

```text
4-vCPU runner, IN1, shared FS 3423
    ├── authenticated release/competitor polling
    ├── BF16 then official-FP8 download
    ├── immutable remote/local manifests
    ├── topology gate and calibration plan
    ├── work queue, assembly, uploads, cards
    └── ten-minute heartbeat/receipts

original-weight capture node, provider TBD
    ├── BF16 model-card vLLM capture
    ├── optional vLLM-GG control capture
    ├── sealed per-layer activation windows
    └── teacher hidden states/logits

worker A: 4× RTX PRO 6000 spot container
    └── uniform K3 campaign

worker B: 4× RTX PRO 6000 spot container
    └── uniform K4 campaign

qualification fleet
    ├── official FP8 / K3 / K4 / 3.42 / NVFP4 panel
    └── vLLM-GG runtime and task checks
```

Workers A/B use the same source revision, calibration, transform seed, and layer receipts. They never run a mixed live campaign. The CPU runner assembles any whole-expert tier map later.

## Streaming schedule and resource utilization

| Resource | Keep busy with | Backpressure boundary |
|---|---|---|
| Runner network | BF16 first, FP8 second, competitor metadata | one hf-xet process on 4 vCPUs; concurrent processes OOMed |
| Runner CPU | Xet reconstruction, manifests, hashing, assembly | hashing may not delay release download |
| Shared filesystem | immutable source, sealed captures, work units, outputs | 1.5-TB reserve; atomic rename only |
| Capture GPU | next calibration batch/layer | never overwrite an unconsumed layer |
| Capture RAM | bounded layer window | spill sealed layer to shared FS |
| Worker A GPU/CPU | K3 prep/encode/seal claims | consume only sealed capture+source inputs |
| Worker B GPU/CPU | K4 prep/encode/seal claims | independent queue, same seed contract |
| Qualification GPUs | one loaded candidate and repeated panel runs | preserve teacher panel; avoid reload between cold-run groups where methodology permits |
| Publisher | private resumable upload | flip public only on receipt gate |

Each capture layer becomes eligible only after its payload, routed-ID census, hashes, and capture-contract receipt are atomic. It remains retained until both K3 and K4 layer receipts are green. Preparation and encode use cross-node claims; no two workers write the same unit.

## Topology-neutral/shared-H output

The durable parts bin stores canonical whole-expert choices and source/hash bindings, not a hard-wired TP4 checkpoint. Materialization selects TP at the edge when the runtime supports it. The first vLLM-GG serving artifact may still be a disclosed TP4 materialization while the parts bin remains topology-neutral.

Shared-H must be part of the uniform K3 and K4 preparation contract. It cannot be produced by deleting per-expert scales after encoding unless equality is proven. The published historical B300 adapter is pre-shared-H; a release adapter or upstream source is still required.

## Fidelity matrix

Minimum operands:

| Operand | Weight-only KLD | As-served GG KLD | Five cold runs | Card gate |
|---|---:|---:|---:|---|
| BF16 original | teacher | capture floor | yes | reference |
| official FP8 | yes | yes | yes | comparison |
| flat K3 | yes | yes | yes | required |
| flat K4 | yes | yes | yes | required |
| assembled 3.42 | yes | yes | yes | required |
| first credible NVFP4 | yes | yes | preferably | comparison |
| MTP variants | target logits + acceptance | yes | targeted | card update |
| attention K6/K8 | ablation KLD | yes | targeted | card update |

Quick publication may use the sealed 25-window panel once per operand. “Qualified” requires the complete cold-run contract and task/runtime gates. Use float64 KLD reduction over fp32 logits and preserve every tokenwise vector.

## Tabletop timeline from release detection

Times are ranges until a real full GLM-5.2 work-unit benchmark and a fitting capture node are completed.

| Clock | Phase | Best case | Conservative |
|---|---|---:|---:|
| T+0 | detect, pin, metadata, topology/license | 2–10 min | 15–30 min |
| T+0 | provision/prepare fitting capture node | 20–45 min | 1–2 h |
| T+0 | authenticated BF16 download (1.507-TB analogue) | 1.7 h | 2.5 h |
| T+1.7 h | original load and capture startup | 30–60 min | 1–2 h |
| T+2.5 h | first sealed capture layers | 30–90 min | 2–4 h |
| T+3 h | K3/K4 workers begin streaming | — | — |
| T+3–10 h | full calibration capture | 4–7 h | 8–14 h |
| T+3–13 h | parallel uniform K3/K4 encode | 6–10 h | 12–20 h |
| T+10–15 h | topology-neutral assembly/materialization | 1–2 h | 3–4 h |
| T+11–18 h | quick KLD, private uploads, card gates | 2–4 h | 5–8 h |
| T+14–20 h | likely public K3/K4 | target | T+24–32 h |
| T+16–22 h | likely public 3.42 | target | T+28–36 h |
| T+20–40 h | MTP and attention variants | 8–18 h | 18–36 h |
| T+24–48 h | five-run/task/runtime card closure | 8–20 h | 24–48 h |

An early-access publisher can still win the timestamp. The controllable objective is zero avoidable post-release setup delay and the first reproducible, measured model—not an unqualified empty repo.

## Instance choices

| Phase | Preferred shape | Reason |
|---|---|---|
| Poll/download/assembly | CPU VM 4 vCPU / 16 GB | measured 243.7 MB/s hf-xet; 8 vCPU was slower |
| K3 encode | 4× RTX PRO 6000 spot container | 112 CPU cores + SM120; CPU/seal-bound |
| K4 encode | separate 4× RTX PRO 6000 spot container | independent failure domain and equal-rate parallelism |
| Original BF16 capture | 8× B300/GB300-class, ≥2 TB aggregate VRAM | current 8×H200/RTX offerings do not cleanly fit |
| Jarvis fallback capture | 16×H200 multi-node | fits, but cross-node/runtime risk and region mismatch |
| KLD EP fleet | 8×H200 spot container if available | decoded candidates fit; memory/EP problem |
| GG runtime | 4× RTX PRO 6000 | target serving topology |

A two-GPU RTX VM was used only to prove P2P configuration persistence and is paused.

## Budget

Current Jarvis rates observed on 2026-08-28:

- 10,240-GB filesystem: $0.00014/GB-hour = **$1.4336/hour**, $34.41/day.
- runner 4 vCPU: **$0.0992/hour**.
- RTX PRO 6000 spot container: **$0.99/GPU-hour**.
- H200 spot container: **$1.99/GPU-hour**.
- H200 on-demand VM: **$3.99/GPU-hour**.

| Cost center | Best case | Conservative |
|---|---:|---:|
| 10-TB FS for 48–72 h | $69 | $103 |
| runner for 48–72 h | $5 | $7 |
| two 4×RTX workers, 8–16 h | $63 | $127 |
| original capture node/provider | $150 | $500 |
| EP8 KLD fleet | $64 | $255 |
| 4×RTX GG runtime checks | $16 | $48 |
| retries, freight, publication | $25 | $100 |
| **Total** | **about $390** | **about $1,140** |

Recommended available balance before release work: **$1,000 minimum**, with a $150 stop floor. The capture-node price dominates uncertainty. If only the 16×H200 multi-node fallback works, reserve the conservative budget.

Delete the 10-TB filesystem promptly after durable remote closure; every idle day costs $34.41.

## Gaps found by smoke/tabletop

1. Public B300 adapter lacks uniform K4 orchestration and the final shared-H writer.
2. No fitting original-BF16 capture node is currently listed in Jarvis IN1/IN2.
3. Two concurrent high-performance hf-xet jobs OOM-killed the 4-vCPU/16-GB runner; downloads are now serialized and GLM-5.3 preempts baseline smoke.
4. Editable exllamav3 source discovery failed until explicitly bound to the pinned tree.
5. Torch CUDA 13.0 initially had only CUDA 12.6 nvcc; matching toolkit installation is now gated.
6. Multi-NIC distributed jobs need explicit NCCL/GLOO interface selection and randomized torchrun ports.
7. Rebuilding a shared extension can invalidate sealed preparation receipts; build only when import fails and key caches by environment digest.
8. Notification credentials are still missing from the IN1 runner, so protected ten-minute phone updates are not active.
9. A real BF16 prep → encode → seal work unit remains unbenchmarked.
10. Publication cards need the final upstream license and exact source revision after release.

## Confidence

- **0.90**: release detection, authenticated download, manifests, storage, worker resume, codec K3/K4 core, and private repo staging.
- **0.75**: flat K3/K4 encode once a fresh capture and shared-H uniform adapter exist.
- **0.65**: public K3/K4 within 24 hours of release; capture hardware is the critical dependency.
- **0.55**: first public 3.42 with KLD before all competitors.
- **<0.30**: guarantee that K3 or 3.42 beats official FP8/NVFP4; historical evidence does not support a guarantee.

Operational oversight is high-confidence if the session remains connected: active-rental state, logs, GPU/CPU/disk, balance, and receipts can be checked every ten minutes. Truly autonomous phone updates require the protected notification configuration.
