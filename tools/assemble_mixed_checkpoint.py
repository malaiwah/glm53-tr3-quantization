#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

from mixed_materialize import EXPECTED_TENSORS, load_adapter, parse_layers, sha256_file
from release_gate import canonical_sha256


def layer_paths(work: Path, layer: int) -> tuple[Path, Path]:
    return work / f"tr3-layer-{layer:03d}.safetensors", work / f"layer-{layer:03d}.done.json"


def validate_mixed_layer(adapter, work: Path, layer: int, plan_sha: str, recipe: str) -> bool:
    shard, done_path = layer_paths(work, layer)
    try:
        done = json.loads(done_path.read_text())
        if (
            done.get("schema") != "glm53-shared-h-mixed-layer/1"
            or done.get("tier_plan_sha256") != plan_sha
            or done.get("recipe_fingerprint") != recipe
            or done.get("rotation_layout") != "shared_h_v1"
            or done.get("tensor_count") != EXPECTED_TENSORS
            or done.get("file_sha256") != sha256_file(shard)
        ):
            return False
        reader = adapter.BASE.STReader(str(shard))
        return len(reader.tensors) == EXPECTED_TENSORS
    except Exception:
        return False


def audit_mixed_output(adapter, output: Path, tier_bitmap: dict, layers: list[int]) -> dict:
    layer_results = []
    for layer in layers:
        reader = adapter.BASE.STReader(str(output / f"model-layer-{layer:03d}.safetensors"))
        tiers = list(map(int, tier_bitmap[str(layer)]["k"]))
        shared = [name for name in reader.tensors if ".experts.shared_h." in name]
        if len(shared) != 12:
            raise RuntimeError(f"layer {layer}: expected 12 shared-H tensors")
        exl3_count = sum(
            ".mlp.experts." in name
            and name.rsplit(".", 1)[-1] in {"trellis", "mcg", "suh", "svh"}
            for name in reader.tensors
        )
        if exl3_count != EXPECTED_TENSORS:
            raise RuntimeError(f"layer {layer}: EXL3 tensor count differs")
        for expert, bits in enumerate(tiers):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                for rank in range(4):
                    key = f"model.layers.{layer}.mlp.experts.{expert}.{projection}.rank{rank}.trellis"
                    _, shape, _, _ = reader.tensors[key]
                    if int(shape[-1]) != 16 * bits:
                        raise RuntimeError(f"{key}: tier shape differs")
        layer_results.append({
            "layer": layer,
            "k3_experts": tiers.count(3),
            "k4_experts": tiers.count(4),
            "exl3_tensor_count": exl3_count,
        })
    return {"layers": layer_results, "layer_count": len(layers)}


def rewrite_manifest(output: Path) -> None:
    names = sorted(path.name for path in output.iterdir() if path.is_file() and path.name != "MANIFEST.sha256")
    with (output / "MANIFEST.sha256").open("w") as handle:
        for name in names:
            handle.write(f"{sha256_file(output / name)}  {name}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Assemble a full shared-H mixed K3/K4 checkpoint from materialized layer parts")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--base-encoder", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--capture-manifest", type=Path, required=True)
    parser.add_argument("--mixed-work", type=Path, required=True)
    parser.add_argument("--mixed-receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--io-workers", type=int, default=8)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    mixed = json.loads(args.mixed_receipt.read_text())
    claimed = mixed.get("receipt_sha256")
    mixed_body = {key: value for key, value in mixed.items() if key != "receipt_sha256"}
    if mixed.get("schema") != "glm53-shared-h-mixed-parts/1" or claimed != canonical_sha256(mixed_body):
        raise RuntimeError("mixed parts receipt seal differs")
    layers = sorted(map(int, mixed["layers"]))
    if layers != list(range(3, 78)):
        raise RuntimeError("main mixed assembly requires layers 3-77")
    tier_bitmap = {str(layer): mixed["layers"][str(layer)] for layer in layers}
    adapter = load_adapter(args.adapter.resolve(), args.base_encoder.resolve())
    capture_plan = adapter.read_capture_plan(args.capture_manifest.resolve(), args.source.resolve())
    plan_sha = mixed["tier_plan_sha256"]
    recipe = mixed["recipe_fingerprint"]

    def mixed_done(work: str, layer: int, expected_recipe: str | None = None) -> bool:
        return validate_mixed_layer(
            adapter, Path(work), layer, plan_sha, expected_recipe or recipe
        )

    adapter.layer_done = mixed_done
    adapter.BASE.layer_done = mixed_done
    adapter.CURRENT_EXPECTED_RECIPE = recipe
    assembly_args = SimpleNamespace(
        out=str(args.out.resolve()),
        work=str(args.mixed_work.resolve()),
        src=str(args.source.resolve()),
        io_workers=args.io_workers,
        capture_plan=capture_plan,
        expected_recipe=recipe,
    )
    adapter.assemble(assembly_args)

    config_path = args.out / "config.json"
    config = json.loads(config_path.read_text())
    metadata = config["hybrid_tr3_tail"]
    total_experts = len(layers) * 256
    total_bits = sum(sum(row["k"]) for row in tier_bitmap.values())
    metadata.update({
        "producer": "assemble_mixed_checkpoint.py",
        "producer_version": "shared-h-mixed-v1",
        "bits": "mixed",
        "bits_per_expert": "tier_bitmap.json:k",
        "k_values": [3, 4],
        "expert_bpw_mean": total_bits / total_experts,
        "k_plan_sha256": plan_sha,
        "recipe_fingerprint": recipe,
        "rotation_layout": "shared_h_v1",
        "shared_h_tensor_schema": "model.layers.{L}.mlp.experts.shared_h.{proj}.rank{r}.{suh|svh}",
    })
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    tier_path = args.out / "tier_bitmap.json"
    tier_path.write_text(json.dumps(tier_bitmap, indent=2, sort_keys=True) + "\n")
    audit = audit_mixed_output(adapter, args.out, tier_bitmap, layers)
    rewrite_manifest(args.out)
    body = {
        "schema": "glm53-shared-h-mixed-checkpoint/1",
        "source_revision": (args.source / "revision.txt").read_text().strip(),
        "rotation_layout": "shared_h_v1",
        "tier_plan_sha256": plan_sha,
        "recipe_fingerprint": recipe,
        "expert_bpw_mean": total_bits / total_experts,
        "audit": audit,
        "output": str(args.out.resolve()),
        "manifest_sha256": sha256_file(args.out / "MANIFEST.sha256"),
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    receipt = {**body, "receipt_sha256": canonical_sha256(body)}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "expert_bpw_mean": body["expert_bpw_mean"],
        "manifest_sha256": body["manifest_sha256"],
        "receipt_sha256": receipt["receipt_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
