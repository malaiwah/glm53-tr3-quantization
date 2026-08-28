#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

RUNTIME = "verdictai/glm52-exl3-sparkinfer:v31-gg-v20-sic3828fd-vllm0c79e41-cu132-sm120a@sha256:0433ae94665b769b78dd301f952d907508a3ba80bce47a1630ec20ade8812dff"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rewrite_manifest(root: Path) -> None:
    names = sorted(path.name for path in root.iterdir() if path.is_file() and path.name != "MANIFEST.sha256")
    with (root / "MANIFEST.sha256").open("w") as handle:
        for name in names:
            handle.write(f"{sha256_file(root / name)}  {name}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Write a fail-loud UNMEASURED first-release model card")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--profile", choices=["flat-k3", "flat-k4", "mixed-3.42"], required=True)
    parser.add_argument("--repo", required=True)
    args = parser.parse_args()
    if len(args.source_revision) != 40:
        raise SystemExit("source revision must be a full SHA")
    license_text = (args.source / "LICENSE").read_text(errors="replace")
    if not license_text.startswith("GLM-5.3 License"):
        raise RuntimeError("upstream license is not recognized as the GLM-5.3 License")
    config = json.loads((args.artifact / "config.json").read_text())
    metadata = config.get("hybrid_tr3_tail", {})
    if metadata.get("rotation_layout") != "shared_h_v1":
        raise RuntimeError("artifact is not shared_h_v1")
    if metadata.get("source_format") != "BF16":
        raise RuntimeError("artifact is not encoded directly from official BF16 weights")
    calibration = json.loads((args.artifact / "calibration_manifest.json").read_text())
    tokens = int(calibration.get("total_tokens") or 0)
    corpus_sha = calibration.get("corpus_sha256")
    if tokens <= 0 or not isinstance(corpus_sha, str):
        raise RuntimeError("calibration manifest is incomplete")
    title = {
        "flat-k3": "GLM-5.3 TR3 flat K3",
        "flat-k4": "GLM-5.3 TR3 flat K4",
        "mixed-3.42": "GLM-5.3 TR3 mixed 3.42 bpw",
    }[args.profile]
    card = f'''---
license: other
license_name: glm-5.3
library_name: transformers
pipeline_tag: text-generation
base_model: zai-org/GLM-5.3-BF16
tags:
- glm
- exl3
- trellis
- vllm
- blackwell
- quantization
- unmeasured
inference: false
---

# {title}

> [!WARNING]
> **UNMEASURED FIRST RELEASE — NOT QUALIFIED.** This artifact has passed physical
> integrity and exact manifest gates only. It has not yet passed official-FP8
> KLD, NVFP4 comparison, five cold runs, or task evaluation. Do not infer
> that it matches or beats any other quantization.

Source: official `zai-org/GLM-5.3-BF16@{args.source_revision}`. Repository: `{args.repo}`.

## Format

- EXL3/MCG Trellis routed experts, TP4 rank-sliced.
- `shared_h_v1` physical rotations; 9,228 EXL3 tensors per routed MoE layer.
- Profile: `{args.profile}`.
- Calibration: {tokens:,} tokens/layer from corpus `{corpus_sha}`, captured from
  the pinned official BF16 checkpoint.
- Attention, shared experts, dense layers, embeddings, router, MTP, and head
  retain their official BF16 tensors byte-for-byte unless the config explicitly
  states otherwise.

This format is not a drop-in Transformers checkpoint. Use the pinned runtime:

```text
{RUNTIME}
```

## Integrity

Verify `MANIFEST.sha256` before loading. The artifact was assembled only from
sealed uniform K3/K4 parts with one bitrate per whole expert.

## Qualification status

`qualification_status: unmeasured`

KLD and Gilded Gnosis cold-run receipts will be added after the first payload
release. Public visibility is speed-of-availability, not a fidelity claim.

## License and attribution

GLM-5.3 License, matching the upstream source and its conditions. Credits:
Z.ai; Brandon Music; willfalco; turboderp-org/exllamav3;
local-inference-lab; and malaiwah. Reproducible tooling:
https://github.com/malaiwah/glm53-tr3-quantization
'''
    (args.artifact / "README.md").write_text(card)
    rewrite_manifest(args.artifact)
    print(json.dumps({"repo": args.repo, "profile": args.profile, "status": "unmeasured"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
