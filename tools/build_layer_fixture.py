#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a disclosed one-layer GLM-5.2 BF16 I/O/encode fixture"
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--fixture-source", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--reuse-fixture-source", action="store_true")
    args = parser.parse_args()
    source = args.source.resolve()
    fixture = args.fixture_source.resolve()
    capture = args.capture_dir.resolve()
    if args.tokens < 64:
        raise SystemExit("fixture needs at least 64 tokens")
    index = json.loads((source / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    layer_prefix = f"model.layers.{args.layer}.mlp.experts."
    required = {
        weight_map[name]
        for name in weight_map
        if name.startswith(layer_prefix) and name.endswith(".weight")
    }
    if not required:
        raise SystemExit(f"source index has no layer {args.layer} expert weights")
    missing_required = sorted(name for name in required if not (source / name).is_file())
    if missing_required:
        raise SystemExit(f"real layer shards missing: {missing_required}")

    all_shards = sorted(set(weight_map.values()))
    if args.reuse_fixture_source:
        if not (fixture / "config.json").is_file() or not (
            fixture / "model.safetensors.index.json"
        ).is_file():
            raise SystemExit("reused fixture source is incomplete")
        for name in required:
            if not (fixture / name).is_file():
                raise SystemExit(f"reused fixture source lacks {name}")
    else:
        if fixture.exists():
            shutil.rmtree(fixture)
        fixture.mkdir(parents=True)
        for name in (
            "config.json", "model.safetensors.index.json", "tokenizer.json",
            "tokenizer_config.json", "generation_config.json", "chat_template.jinja",
            "LICENSE", "README.md",
        ):
            path = source / name
            if path.is_file():
                shutil.copy2(path, fixture / name)
        for name in all_shards:
            destination = fixture / name
            real = source / name
            if name in required:
                destination.symlink_to(real)
            else:
                destination.write_bytes(b"S")

    layer_dir = capture / f"layer_{args.layer:03d}"
    if layer_dir.exists():
        shutil.rmtree(layer_dir)
    layer_dir.mkdir(parents=True)
    rng = np.random.default_rng(20260828 + args.layer)
    x = (rng.standard_normal((args.tokens, 6144), dtype=np.float32) * 0.02).astype(np.float32)
    bf16_bits = (x.view(np.uint32) >> 16).astype(np.uint16)
    x_path = layer_dir / "x.bin"
    bf16_bits.tofile(x_path)
    ids = np.asarray(
        [[(token * 8 + offset) % 256 for offset in range(8)] for token in range(args.tokens)],
        dtype=np.uint8,
    )
    ids_path = layer_dir / "ids.bin"
    ids.tofile(ids_path)
    counts = np.bincount(ids.reshape(-1), minlength=256).tolist()

    source_identity = {
        "config_sha256": sha256_file(fixture / "config.json"),
        "index_sha256": sha256_file(fixture / "model.safetensors.index.json"),
    }
    plan = {
        "schema": "glm52-smoke-capture-plan-v1",
        "selection_policy": "deterministic-synthetic-I/O-work-unit-smoke",
        "selection_note": "Synthetic activations and routing; never calibration or quality evidence.",
        "corpus_sha256": hashlib.sha256(b"glm52-layer-work-unit-smoke-v1").hexdigest(),
        "corpus_rows": args.tokens,
        "axis_rows": {"synthetic_smoke": args.tokens},
        "passes": [{
            "name": "synthetic_smoke",
            "axis": "synthetic_smoke",
            "samples": [{"line": 0, "ntok": args.tokens}],
            "tokens": args.tokens,
        }],
        "total_tokens": args.tokens,
        "capture_tp": 8,
        "output_tp": 4,
        "owner_corpus_only": False,
        "calibration_baseline": False,
        "routing": {
            "natural": True,
            "forced_expert_activation": False,
            "scoring_func": "sigmoid",
            "top_k": 8,
            "n_group": 1,
            "topk_group": 1,
        },
        "source": source_identity,
        "smoke_fixture": True,
    }
    plan["capture_fingerprint"] = canonical_sha256(plan)
    args.plan.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.plan, plan)
    layer_manifest = {
        "schema": "glm52-smoke-layer-capture/1",
        "capture_fingerprint": plan["capture_fingerprint"],
        "layer": args.layer,
        "tokens": args.tokens,
        "hidden": 6144,
        "x_dtype": "bfloat16",
        "ids_dtype": "uint8",
        "routed_counts": counts,
        "sha256_x": sha256_file(x_path),
        "sha256_ids": sha256_file(ids_path),
        "synthetic": True,
        "quality_evidence_permitted": False,
    }
    atomic_json(layer_dir / "layer_manifest.json", layer_manifest)
    receipt = {
        "schema": "glm52-real-weight-work-unit-fixture/1",
        "layer": args.layer,
        "tokens": args.tokens,
        "real_source_shards": sorted(required),
        "stub_source_shards": len(all_shards) - len(required),
        "source": source_identity,
        "capture_fingerprint": plan["capture_fingerprint"],
        "synthetic_capture": True,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    atomic_json(capture / "FIXTURE.json", receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
