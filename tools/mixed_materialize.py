#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

from release_gate import canonical_sha256

EXPECTED_TENSORS = 9_228
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_layers(value: str) -> list[int]:
    result = []
    for part in value.split(","):
        if "-" in part:
            start, stop = map(int, part.split("-", 1))
            result.extend(range(start, stop + 1))
        else:
            result.append(int(part))
    return sorted(set(result))


def load_adapter(adapter_dir: Path, base_encoder: Path):
    os.environ["TR3_BITS"] = "3"
    module_path = adapter_dir / "encode_b300.py"
    spec = importlib.util.spec_from_file_location("_glm53_mixed_adapter", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import campaign adapter")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(adapter_dir))
    spec.loader.exec_module(module)
    module.load_base_encoder(base_encoder)
    return module


def layer_paths(work: Path, layer: int) -> tuple[Path, Path]:
    return work / f"tr3-layer-{layer:03d}.safetensors", work / f"layer-{layer:03d}.done.json"


def validate_source_layer(adapter, work: Path, layer: int, bits: int) -> tuple[object, dict]:
    shard, receipt_path = layer_paths(work, layer)
    receipt = json.loads(receipt_path.read_text())
    if (
        int(receipt.get("bits", -1)) != bits
        or receipt.get("rotation_layout") != "shared_h_v1"
        or int(receipt.get("tensor_count", -1)) != EXPECTED_TENSORS
        or int(receipt.get("source_expert_tensor_count", -1)) != 768
        or receipt.get("keep_nvfp4") != []
        or receipt.get("file_sha256") != sha256_file(shard)
    ):
        raise RuntimeError(f"K{bits} layer {layer} receipt is invalid")
    reader = adapter.BASE.STReader(str(shard))
    if len(reader.tensors) != EXPECTED_TENSORS:
        raise RuntimeError(f"K{bits} layer {layer} tensor count differs")
    return reader, receipt


def expected_keys(layer: int) -> tuple[set[str], set[str]]:
    shared = {
        f"model.layers.{layer}.mlp.experts.shared_h.{projection}.rank{rank}."
        f"{'svh' if projection == 'down_proj' else 'suh'}"
        for projection in PROJECTIONS
        for rank in range(4)
    }
    local = set()
    for expert in range(256):
        for projection in PROJECTIONS:
            local_side = "suh" if projection == "down_proj" else "svh"
            for rank in range(4):
                prefix = f"model.layers.{layer}.mlp.experts.{expert}.{projection}.rank{rank}"
                local.update((f"{prefix}.{local_side}", f"{prefix}.trellis", f"{prefix}.mcg"))
    return shared, local



def audit_mixed_shard(adapter, shard: Path, layer: int, tiers: list[int]) -> None:
    reader = adapter.BASE.STReader(str(shard))
    shared, local = expected_keys(layer)
    if set(reader.tensors) != shared | local:
        raise RuntimeError(f"layer {layer}: materialized key census differs")
    for expert, bits in enumerate(tiers):
        for projection in PROJECTIONS:
            for rank in range(4):
                key = (
                    f"model.layers.{layer}.mlp.experts.{expert}."
                    f"{projection}.rank{rank}.trellis"
                )
                dtype, shape, _, _ = reader.tensors[key]
                if dtype != "I16" or int(shape[-1]) != 16 * bits:
                    raise RuntimeError(f"{key}: materialized K{bits} shape differs")


def materialize_layer(adapter, k3_work: Path, k4_work: Path, out: Path, layer: int, tiers: list[int], plan_sha: str) -> dict:
    out_shard, out_done = layer_paths(out, layer)
    if out_shard.is_file() and out_done.is_file():
        current = json.loads(out_done.read_text())
        if (
            current.get("tier_plan_sha256") == plan_sha
            and current.get("file_sha256") == sha256_file(out_shard)
            and current.get("tensor_count") == EXPECTED_TENSORS
        ):
            audit_mixed_shard(adapter, out_shard, layer, tiers)
            return current
    k3, done3 = validate_source_layer(adapter, k3_work, layer, 3)
    k4, done4 = validate_source_layer(adapter, k4_work, layer, 4)
    for field in (
        "capture", "source_expert_payload_sha256", "source_expert_tensor_count",
        "shared_h_profile_sha256", "shared_h_sign_template_sha256",
    ):
        if done3.get(field) != done4.get(field):
            raise RuntimeError(f"layer {layer}: K3/K4 {field} differs")
    shared, local = expected_keys(layer)
    expected = shared | local
    if set(k3.tensors) != expected or set(k4.tensors) != expected:
        raise RuntimeError(f"layer {layer}: uniform key census differs")
    entries = []
    for key in sorted(shared):
        dtype3, shape3, _, _ = k3.tensors[key]
        dtype4, shape4, _, _ = k4.tensors[key]
        payload = k3.read_bytes(key)
        if (dtype3, tuple(shape3)) != (dtype4, tuple(shape4)) or payload != k4.read_bytes(key):
            raise RuntimeError(f"layer {layer}: shared profile differs between K3/K4: {key}")
        entries.append((key, dtype3, tuple(shape3), payload))
    selected_errors = []
    for expert, bits in enumerate(tiers):
        reader = k3 if bits == 3 else k4
        source_done = done3 if bits == 3 else done4
        selected_errors.append(source_done["expert_rel_rt_mse"][expert])
        prefix = f"model.layers.{layer}.mlp.experts.{expert}."
        for key in sorted(name for name in local if name.startswith(prefix)):
            dtype, shape, _, _ = reader.tensors[key]
            entries.append((key, dtype, tuple(shape), reader.read_bytes(key)))
    if len(entries) != EXPECTED_TENSORS:
        raise RuntimeError(f"layer {layer}: mixed entry count differs")
    out.mkdir(parents=True, exist_ok=True)
    temporary = out_shard.with_suffix(out_shard.suffix + ".new")
    _, file_sha = adapter.BASE.write_safetensors(str(temporary), entries, metadata={"format": "pt"})
    temporary.replace(out_shard)
    audit_mixed_shard(adapter, out_shard, layer, tiers)
    body = {
        "schema": "glm53-shared-h-mixed-layer/1",
        "layer": layer,
        "bits": "mixed",
        "k": tiers,
        "k3_experts": tiers.count(3),
        "k4_experts": tiers.count(4),
        "tier_plan_sha256": plan_sha,
        "rotation_layout": "shared_h_v1",
        "tensor_count": EXPECTED_TENSORS,
        "source_expert_payload_sha256": done3["source_expert_payload_sha256"],
        "source_expert_tensor_count": 768,
        "capture": done3["capture"],
        "shared_h_profile_sha256": done3["shared_h_profile_sha256"],
        "shared_h_sign_template_sha256": done3["shared_h_sign_template_sha256"],
        "experts_layer_h_fallback": done3.get("experts_layer_h_fallback", []),
        "q_fallback_slices": done3.get("q_fallback_slices", []),
        "expert_rel_rt_mse": selected_errors,
        "file": out_shard.name,
        "file_sha256": file_sha,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    body["recipe_fingerprint"] = canonical_sha256({
        "tier_plan_sha256": plan_sha,
        "rotation_layout": "shared_h_v1",
        "shared_h_sign_template_sha256": done3["shared_h_sign_template_sha256"],
    })
    body["receipt_sha256"] = canonical_sha256(body)
    temporary_done = out_done.with_suffix(out_done.suffix + ".new")
    temporary_done.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    temporary_done.replace(out_done)
    return body


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize arbitrary whole-expert K3/K4 choices from uniform shared-H parts")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--base-encoder", type=Path, required=True)
    parser.add_argument("--k3-work", type=Path, required=True)
    parser.add_argument("--k4-work", type=Path, required=True)
    parser.add_argument("--tier-bitmap", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--layers", default="3-77")
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    adapter = load_adapter(args.adapter.resolve(), args.base_encoder.resolve())
    tier_source = json.loads(args.tier_bitmap.read_text())
    layers = parse_layers(args.layers)
    selected = {}
    plan_material = {}
    for layer in layers:
        tiers = list(map(int, tier_source[str(layer)]["k"]))
        if len(tiers) != 256 or set(tiers) - {3, 4}:
            raise RuntimeError(f"layer {layer}: tier plan must contain 256 K3/K4 values")
        plan_material[str(layer)] = tiers
    plan_sha = canonical_sha256(plan_material)
    recipe_fingerprints = set()
    for layer in layers:
        receipt = materialize_layer(
            adapter, args.k3_work, args.k4_work, args.out, layer,
            plan_material[str(layer)], plan_sha,
        )
        recipe_fingerprints.add(receipt["recipe_fingerprint"])
        selected[str(layer)] = {
            "k": receipt["k"],
            "k3_experts": receipt["k3_experts"],
            "k4_experts": receipt["k4_experts"],
            "expert_rel_rt_mse": receipt["expert_rel_rt_mse"],
            "layer_receipt_sha256": receipt["receipt_sha256"],
        }
    if len(recipe_fingerprints) != 1:
        raise RuntimeError("mixed layers have different recipe fingerprints")
    body = {
        "schema": "glm53-shared-h-mixed-parts/1",
        "rotation_layout": "shared_h_v1",
        "tier_plan_sha256": plan_sha,
        "recipe_fingerprint": next(iter(recipe_fingerprints)),
        "layers": selected,
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    receipt = {**body, "receipt_sha256": canonical_sha256(body)}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "layers": len(layers), "tier_plan_sha256": plan_sha,
        "receipt_sha256": receipt["receipt_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
