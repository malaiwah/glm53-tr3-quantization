#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def config_view(config: dict) -> dict:
    text = config.get("text_config")
    return text if isinstance(text, dict) else config


def git_head(path: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-c", f"safe.directory={path}", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def check_file(path: Path, expected: str, errors: list[str], files: dict) -> None:
    if not path.is_file():
        errors.append(f"missing pinned file: {path}")
        return
    observed = sha256_file(path)
    files[str(path)] = {"bytes": path.stat().st_size, "sha256": observed}
    if observed != expected:
        errors.append(f"pinned file hash differs: {path}: {observed} != {expected}")


def tensor_key(layer: int, expert: int, projection: str) -> str:
    return f"model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).parents[1])
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--require-release-ready", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    contract_path = root / "smoke-contract.json"
    contract = load_json(contract_path)
    source_contract = contract["reference_source"]
    quant_contract = contract["reference_quant"]
    encoder_contract = contract["encoder"]
    source = (args.candidate or root / source_contract["directory"]).resolve()
    baseline = (root / quant_contract["directory"]).resolve()
    encoder = (root / encoder_contract["directory"]).resolve()
    exllama = (root / "upstream/exllamav3-v0.0.43").resolve()
    out = (args.out or root / "receipts/preflight.json").resolve()

    errors: list[str] = []
    blockers: list[str] = []
    files: dict[str, dict] = {}
    check_file(contract_path, sha256_file(contract_path), errors, files)
    for name, expected in (
        ("config.json", quant_contract["config_sha256"]),
        ("model.safetensors.index.json", quant_contract["index_sha256"]),
        ("tier_bitmap.json", quant_contract["tier_bitmap_sha256"]),
        ("calibration_manifest.json", quant_contract["calibration_manifest_sha256"]),
    ):
        check_file(baseline / name, expected, errors, files)
    check_file(
        encoder / "encode_tr3_v31.py",
        encoder_contract["production_encoder_sha256"],
        errors,
        files,
    )
    check_file(
        encoder / "calibration/reap_recall_calib.jsonl",
        encoder_contract["corpus_sha256"],
        errors,
        files,
    )

    candidate_config_path = source / "config.json"
    candidate_index_path = source / "model.safetensors.index.json"
    if not candidate_config_path.is_file() or not candidate_index_path.is_file():
        errors.append(f"candidate metadata incomplete: {source}")
        candidate_config, candidate_index = {}, {"weight_map": {}}
    else:
        candidate_config = load_json(candidate_config_path)
        candidate_index = load_json(candidate_index_path)
        files[str(candidate_config_path)] = {
            "bytes": candidate_config_path.stat().st_size,
            "sha256": sha256_file(candidate_config_path),
        }
        files[str(candidate_index_path)] = {
            "bytes": candidate_index_path.stat().st_size,
            "sha256": sha256_file(candidate_index_path),
        }

    topology = config_view(candidate_config)
    observed_topology = {}
    for key, expected in contract["topology"].items():
        if key == "tp":
            continue
        observed = topology.get(key, candidate_config.get(key))
        observed_topology[key] = observed
        if observed != expected:
            errors.append(f"candidate topology {key}={observed!r}, expected {expected!r}")

    weight_map = candidate_index.get("weight_map")
    if not isinstance(weight_map, dict):
        errors.append("candidate index has no weight_map")
        weight_map = {}
    first = int(contract["topology"]["first_k_dense_replace"])
    layers = int(contract["topology"]["num_hidden_layers"])
    experts = int(contract["topology"]["n_routed_experts"])
    missing_main = []
    for layer in range(first, layers):
        for expert in range(experts):
            for projection in PROJECTIONS:
                key = tensor_key(layer, expert, projection)
                if key not in weight_map:
                    missing_main.append(key)
                    if len(missing_main) >= 16:
                        break
            if len(missing_main) >= 16:
                break
        if len(missing_main) >= 16:
            break
    if missing_main:
        errors.append(f"candidate routed tensor census incomplete: {missing_main[:4]}")
    mtp_missing = [
        tensor_key(layers, expert, projection)
        for expert in range(experts)
        for projection in PROJECTIONS
        if tensor_key(layers, expert, projection) not in weight_map
    ]
    if mtp_missing:
        errors.append(f"candidate MTP tensor census incomplete: {mtp_missing[:4]}")

    baseline_config = load_json(baseline / "config.json") if (baseline / "config.json").is_file() else {}
    baseline_index = load_json(baseline / "model.safetensors.index.json") if (baseline / "model.safetensors.index.json").is_file() else {"weight_map": {}}
    tiers = load_json(baseline / "tier_bitmap.json") if (baseline / "tier_bitmap.json").is_file() else {}
    expected_layers = {str(layer) for layer in range(3, 79)}
    if set(tiers) != expected_layers:
        errors.append("3.42 tier bitmap layer census is not exactly 3..78")
    tier_counts = {}
    for layer in range(3, 79):
        values = tiers.get(str(layer), {}).get("k")
        if not isinstance(values, list) or len(values) != experts or set(values) - {3, 4}:
            errors.append(f"layer {layer}: invalid K-tier vector")
            continue
        counts = {"k3": values.count(3), "k4": values.count(4)}
        tier_counts[str(layer)] = counts
        expected = (
            contract["bit_budget"]["layer_3"]
            if layer == 3
            else contract["bit_budget"]["mtp_layer_78"]
            if layer == 78
            else contract["bit_budget"]["layers_4_77"]
        )
        if counts != expected:
            errors.append(f"layer {layer}: tier budget {counts}, expected {expected}")

    hybrid = baseline_config.get("hybrid_tr3_tail", {})
    reference_mode = files.get(str(candidate_config_path), {}).get("sha256") == source_contract["config_sha256"]
    if reference_mode:
        if files[str(candidate_index_path)]["sha256"] != source_contract["index_sha256"]:
            errors.append("GLM-5.2 rehearsal source index differs from pinned reference")
        if hybrid.get("source_config_sha256") != source_contract["config_sha256"]:
            errors.append("3.42 config does not bind the pinned GLM-5.2 config")
        if hybrid.get("source_index_sha256") != source_contract["index_sha256"]:
            errors.append("3.42 config does not bind the pinned GLM-5.2 index")

    micro = []
    baseline_map = baseline_index.get("weight_map", {})
    for row in contract["micro_payload"]:
        layer, expert = int(row["layer"]), int(row["expert"])
        source_shards = sorted({
            weight_map.get(tensor_key(layer, expert, projection))
            for projection in PROJECTIONS
        })
        prefix = f"model.layers.{layer}.mlp.experts.{expert}."
        quant_keys = [name for name in baseline_map if name.startswith(prefix)]
        quant_shards = sorted({baseline_map[name] for name in quant_keys})
        if reference_mode and source_shards != [row["source_shard"]]:
            errors.append(f"micro source shard drift for layer {layer} expert {expert}: {source_shards}")
        if quant_shards != [row["reference_shard"]] or len(quant_keys) != 36:
            errors.append(f"micro reference schema drift for layer {layer} expert {expert}")
        micro.append({
            **row,
            "candidate_source_shards": source_shards,
            "reference_quant_shards": quant_shards,
            "reference_quant_tensor_count": len(quant_keys),
        })

    encoder_text = (encoder / "encode_b300.py").read_text() if (encoder / "encode_b300.py").is_file() else ""
    public_mixed_adapter = "--tier-bitmap" in encoder_text or "bits_per_expert" in encoder_text
    public_shared_h_adapter = "shared-h-v1" in encoder_text or "shared_h_v1" in encoder_text
    campaign_adapter_contract = root / "work/uniform-adapter/UNIFORM_ADAPTER.txt"
    campaign_adapter_text = (
        campaign_adapter_contract.read_text() if campaign_adapter_contract.is_file() else ""
    )
    campaign_shared_h_adapter = (
        "rotation_layout=shared_h_v1" in campaign_adapter_text
        and "supported_bits=3,4,5,6" in campaign_adapter_text
    )
    mixed_evidence_path = root / "evidence/shared-h-mixed-work-unit.json"
    try:
        mixed_evidence = load_json(mixed_evidence_path)
        campaign_mixed_adapter = (
            mixed_evidence.get("checks", {}).get("passed_physical_contract") is True
            and mixed_evidence.get("mixed", {}).get("tensor_count") == 9_228
            and (root / "tools/mixed_materialize.py").is_file()
            and (root / "tools/assemble_mixed_checkpoint.py").is_file()
        )
    except Exception:
        campaign_mixed_adapter = False
    if not public_mixed_adapter and not campaign_mixed_adapter:
        blockers.append("offline mixed assembly has no sealed physical proof")
    if not public_shared_h_adapter and not campaign_shared_h_adapter:
        blockers.append("no sealed shared-H campaign adapter exists")
    gpu_harness = root / "tools/run_codec_smoke.py"
    if not gpu_harness.is_file():
        blockers.append("K3/K4 GPU codec harness has not been authored")

    exllama_head = git_head(exllama)
    if exllama_head != encoder_contract["exllamav3_commit"]:
        errors.append(f"exllamav3 pin differs: {exllama_head!r}")

    body = {
        "schema": "glm53-mainline-release-preflight/1",
        "contract_sha256": sha256_file(contract_path),
        "candidate": str(source),
        "candidate_mode": "glm52-rehearsal" if reference_mode else "glm53-release",
        "metadata_passed": not errors,
        "release_ready": not errors and not blockers,
        "errors": errors,
        "blockers": blockers,
        "observed_topology": observed_topology,
        "candidate_index": {
            "tensors": len(weight_map),
            "shards": len(set(weight_map.values())),
            "main_routed_expected": (layers - first) * experts * len(PROJECTIONS),
            "mtp_routed_expected": experts * len(PROJECTIONS),
        },
        "tier_counts": tier_counts,
        "micro_payload": micro,
        "pins": {
            "encoder_sha256": files.get(str(encoder / "encode_tr3_v31.py"), {}).get("sha256"),
            "corpus_sha256": files.get(str(encoder / "calibration/reap_recall_calib.jsonl"), {}).get("sha256"),
            "exllamav3_commit": exllama_head,
            "runtime_image_digest": contract["runtime"]["image_digest"],
        },
        "tooling": {
            "published_mixed_adapter": public_mixed_adapter,
            "published_shared_h_adapter": public_shared_h_adapter,
            "campaign_shared_h_adapter": campaign_shared_h_adapter,
            "campaign_mixed_adapter": campaign_mixed_adapter,
            "gpu_codec_harness": gpu_harness.is_file(),
            "docker_available": shutil.which("docker") is not None,
        },
        "files": files,
        "checked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    body["receipt_sha256"] = canonical_sha256(body)
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = out.with_suffix(out.suffix + ".tmp")
    staging.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    staging.replace(out)
    print(json.dumps({
        "metadata_passed": body["metadata_passed"],
        "release_ready": body["release_ready"],
        "errors": len(errors),
        "blockers": blockers,
        "receipt": str(out),
    }, sort_keys=True))
    if errors or (args.require_release_ready and blockers):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
