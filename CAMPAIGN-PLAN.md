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
8-vCPU runner, IN1, shared FS 3423
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
| Runner network | BF16 first, FP8 second, competitor metadata | one hf-xet process, four file workers, 32-GB RAM |
| Runner CPU | Xet reconstruction, manifests, hashing, assembly | hashing may not delay release download |
| Shared filesystem | immutable source, sealed captures, work units, outputs | 1.5-TB reserve; atomic rename only |
| Capture GPU | next calibration batch/layer | never overwrite an unconsumed layer |
| Capture RAM | bounded layer window | spill sealed layer to shared FS |
| Worker A GPU/CPU | K3 prep/encode/seal claims | consume only sealed capture+source inputs |
| Worker B GPU/CPU | K4 prep/encode/seal claims | independent queue, same seed contract |
| Qualification GPUs | one loaded candidate and repeated panel runs | preserve teacher panel; avoid reload between cold-run groups where methodology permits |
| Publisher | private resumable upload | flip public only on receipt gate |

Each capture layer becomes eligible only after its payload, routed-ID census, hashes, and capture-contract receipt are atomic. It remains retained until both K3 and K4 layer receipts are green. Preparation and encode use cross-node claims; no two workers write the same unit.

`tools/work_queue.py` implements the shared-filesystem claim protocol: sealed
contract-bound state, POSIX-locked claims, profile filters, nonce-checked
completion receipts, and age-gated stale-claim recovery. A 20-process
concurrency smoke produced 20 distinct claims with no duplicate or lost unit;
receipt: `evidence/work-queue-smoke.json`.

## Topology-neutral/shared-H output

The durable parts bin stores canonical whole-expert choices and source/hash bindings, not a hard-wired TP4 checkpoint. Materialization selects TP at the edge when the runtime supports it. The first vLLM-GG serving artifact may still be a disclosed TP4 materialization while the parts bin remains topology-neutral.

Shared-H is a two-pass encode, never a deletion pass. The public [r28 recipe](https://github.com/local-inference-lab/rtx6kpro/blob/master/models/glm5.2_exl3_shared_h_quantization.md) forms one H-side profile per layer/projection/rank, moves gate/up `g_scale` into expert-local `SV`, and emits 9,228 EXL3 tensors/layer. `tools/shared_h_overlay.py` independently implements that contract because the referenced kquant PR is no longer public. Its real-BF16 K4 work unit exactly matched the published physical key census and artifact rehash, but the deliberately synthetic 256-token fixture raised mean weight NMSE from 0.00468 to 0.00648. That is not production fidelity evidence. The release path reuses sealed sign rows range-extracted from the qualified GLM-5.2 shared-H checkpoint, then still requires fresh production KLD and Gilded Gnosis closure. Receipt: `evidence/shared-h-k4-work-unit.json`.

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
| Poll/download/assembly | CPU VM 8 vCPU / 32 GB | full-repo memory headroom; serialized high-performance hf-xet |
| K3 encode | 4× RTX PRO 6000 spot container | 112 CPU cores + SM120; CPU/seal-bound |
| K4 encode | separate 4× RTX PRO 6000 spot container | independent failure domain and equal-rate parallelism |
| Original BF16 capture | 8× B300/GB300-class, ≥2 TB aggregate VRAM | current 8×H200/RTX offerings do not cleanly fit |
| Jarvis fallback capture | 16×H200 multi-node | fits, but cross-node/runtime risk and region mismatch |
| KLD EP fleet | 8×H200 spot container if available | decoded candidates fit; memory/EP problem |
| GG runtime | 4× RTX PRO 6000 | target serving topology |

A two-GPU RTX VM was used only to prove P2P configuration persistence and is paused.

## External original-model providers

The capture pipeline is standard Linux/Python/CUDA and is portable to any
provider offering root/SSH, matching Torch/CUDA, enough local/network storage,
and a single NVLink domain. Jarvis-specific lifecycle helpers are not required
on the capture host; captures are sealed and transferred to filesystem 3423.

| Provider | B300 status | Fit | Decision |
|---|---|---|---|
| RunPod | B300 288 GB advertised at $6.94/h community or $7.89/h secure per GPU; authenticated count-8 inventory is currently `NONE` | 8× = 2.304 TB; SSH/pods/network storage | Preferred on-demand path if count-8 stock returns; $55.52–$63.12/h |
| Vast.ai | One verified Canadian 8×B300 offer currently rentable (two earlier): 2.200 TB reported GPU RAM, ~956 GB/s NVLink, CUDA 13.2 | 2-TB disk adds $1.39/h; 7.1+ Gbit/s down, 4.4+ Gbit/s up | **Live fallback**: bid floor totals $41.39/h; cap campaign bid at $50 GPU + storage. On-demand $101.39/h is outside the safe runtime envelope |
| Lambda | B300/GB300 listed, but pricing/capacity commonly sales or reserved | Technically compatible | Too slow to procure unless the account already has capacity |
| Spheron/other bare metal | Some 1–8× B300 listings with root SSH | Technically compatible | Fallback after topology/network verification |

RunPod capture allowance at current public rates:

| Runtime | Community 8× B300 | Secure 8× B300 |
|---:|---:|---:|
| 4 h | $222 | $252 |
| 6 h | $333 | $379 |
| 8 h | $444 | $505 |

Vast live offers at a 2,000-GB disk allocation:

| Runtime | Interruptible floor ($41.39/h) | Campaign bid cap ($51.39/h) | On-demand ($101.39/h) |
|---:|---:|---:|---:|
| 4 h | $166 | $206 | $406 |
| 6 h | $248 | $308 | $608 |
| 8 h | $331 | $411 | $811 |

RunPod's current v2 API supports count-specific inventory before placement.
`tools/runpod_b300_preflight.py` requests eight GPUs, CUDA ≥12.8, exact cloud
type, and a $64/hour cap; it writes a sealed no-secret receipt. Authenticated
checks found secure and community count-8 availability `NONE` at 2026-08-28
04:51 UTC. Receipts: `evidence/runpod-b300-secure.json` and
`evidence/runpod-b300-community.json`. The older v1 OpenAPI enum omits B300;
use v2 catalog data rather than guessing the placement identifier.

Current executable order is therefore: recheck both inventories at release,
then rent the faster-disk Vast offer interruptibly with a **$50/hour GPU bid**
plus $1.39/hour storage. Layer-atomic capture streaming makes preemption
recoverable. RunPod remains cheaper, but its official B300 template is not the
pinned Gilded Gnosis runtime and custom-image SSH is unqualified; autonomous
launch stays disabled rather than spending on a host the controller may not
reach. Do not use Vast on-demand unless a separate runtime cap is explicitly
authorized.

The launch funding floor is **$420 USD**, not merely one billable hour: eight
hours at the capped bid plus storage costs about $411. Current Vast available
credit is about $25, so the watcher must remain launch-blocked until topped up.

The Vast pod image is pinned to Gilded Gnosis r34 by manifest digest
`voipmonitor/vllm@sha256:820181fbbc975cd5291c411cda9771d58fecee1636d916f508f47230df20592b`.
`capture_worker.sh` refuses to load BF16 unless the installed vLLM version
identifies `gilded.gnosis`; the adapter's vLLM worker extension then performs
the model-card-style TP8 capture and engine-output smoke.

The 1,000-CAD ceiling supports roughly eight hours at the capped Vast bid under
the best-case remainder, but only about four on Vast on-demand. Qualification
must use spot capacity. Download source directly
on the capture provider, then stream sealed layer captures to IN1 with
ControlMaster-backed rsync. Do not copy the full 1.5-TB source across providers
unless the capture host cannot download it from Hugging Face itself.

## Budget

Current Jarvis rates observed on 2026-08-28:

- 10,240-GB filesystem: $0.00014/GB-hour = **$1.4336/hour**, $34.41/day.
- runner 8 vCPU: **$0.1984/hour**.
- RTX PRO 6000 spot container: **$0.99/GPU-hour**.
- H200 spot container: **$1.99/GPU-hour**.
- H200 on-demand VM: **$3.99/GPU-hour**.

| Cost center | Best case | Conservative |
|---|---:|---:|
| 10-TB FS for 48–72 h | $69 | $103 |
| runner for 48–72 h | $10 | $14 |
| two 4×RTX workers, 8–16 h | $63 | $127 |
| original 8×B300 capture node | $166 (Vast floor, 4 h) | $411 (Vast $50 bid + storage, 8 h) |
| EP8 KLD fleet | $64 | $255 |
| 4×RTX GG runtime checks | $16 | $48 |
| retries, freight, publication | $25 | $100 |
| **Total** | **about $413 USD** | **about $1,058 USD** |

The operator ceiling is approximately **1,000 CAD**. Use **$700 USD as the
automatic hard cap** to preserve exchange-rate/tax margin, with a $150 stop
floor. At $550 spent, defer MTP K5/K6, attention K8, extra cold runs, and
nonessential runtime sweeps; finish K3, K4, 3.42, quick KLD, durable uploads,
and cards first. The conservative scenario exceeds the ceiling and is not
authorized without a new decision.

Jarvis balance was $53.25 at 07:43 UTC. All campaign GPUs are paused and only
the $0.1984/hour runner remains active; the user's separate VMs remain
untouched. Their current burn can exhaust the account before release, so
Jarvis must be topped up or those user-owned jobs must finish before the
four-GPU quant workers resume. Recheck both Jarvis and Vast funding before
capture provisioning.

Delete the 10-TB filesystem promptly after durable remote closure; every idle day costs $34.41.

## Gaps found by smoke/tabletop

1. **FIXED (physical contract):** template-bound shared-H K3 and K4 each passed 9,228-tensor validation, exact rehash, and byte-identical shared-profile SHA. Offline materialization then selected 206 K3 + 50 K4 whole experts and audited every mixed trellis shape (`evidence/shared-h-mixed-work-unit.json`). Production-capture KLD/runtime qualification remains a fidelity gate, not an encoder-format gap.
2. **MITIGATED:** Jarvis still has no fitting capture node. The authenticated watcher currently sees one verified Canadian 8×B300/2.2-TB-NVLink Vast offer at $41.39/hour interruptible or $101.39/hour on-demand with 2-TB storage (two were available earlier); RunPod has no count-8 stock. Provider funding and release-time recheck remain gates.
3. **FIXED:** concurrent/full-worker hf-xet OOMs — runner is 8 vCPU/32 GB, repositories are serialized, file workers are capped at four, and GLM-5.3 preempts smoke.
4. **FIXED:** editable exllamav3 source is explicitly bound to the pinned tree.
5. **FIXED:** Torch/CUDA compiler mismatch is gated; a sealed prebuilt SM120 extension survives container pause.
6. **FIXED:** `tools/distributed_env.sh` pins NCCL/GLOO interfaces and chooses a randomized torchrun port for single- or multi-node launches.
7. **FIXED:** shared extension rebuild clobber — load the sealed prebuilt `.so`; rebuild only when import fails.
8. **OPEN:** OMP's internal notification transport exposes no reusable runner endpoint, so protected ten-minute phone updates need an explicit webhook/ntfy credential.
9. **FIXED:** independent real GLM-5.2 BF16 layer-3 K3/K4 prep → encode → seal units passed. K3 took 959.5 s and K4 941.1 s for all 256 experts/768 source tensors; exact artifact rehashes matched. Receipt: `evidence/real-work-unit-smoke.json`.
10. **OPEN:** publication cards need the final upstream license and exact source revision after release.

## Confidence

- **0.90**: release detection, authenticated download, manifests, storage, worker resume, codec core, and a real BF16 K3/K4 work unit.
- **0.85**: uniform shared-H K3/K4 physical encode once a fresh production capture exists; the real one-layer K3/K4/mixed chain passed.
- **0.65**: final shared-H fidelity closure; physical schema is proven, but synthetic-fixture NMSE regressed and cannot authorize publication.
- **0.70**: public K3/K4 within 24 hours of release; fitting Vast hardware is live but provider availability and funding remain critical.
- **0.55**: first public 3.42 with KLD before all competitors.
- **<0.30**: guarantee that K3 or 3.42 beats official FP8/NVFP4; historical evidence does not support a guarantee.

Operational oversight is high-confidence if the session remains connected: active-rental state, logs, GPU/CPU/disk, balance, and receipts can be checked every ten minutes. Truly autonomous phone updates require the protected notification configuration.
