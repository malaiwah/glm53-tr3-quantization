#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path, PurePosixPath

from release_gate import canonical_sha256


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_sealed(path: Path, schema: str) -> dict:
    receipt = json.loads(path.read_text())
    claimed = receipt.get("receipt_sha256")
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if receipt.get("schema") != schema:
        raise RuntimeError(f"{path}: schema is {receipt.get('schema')!r}, expected {schema!r}")
    if not isinstance(claimed, str) or canonical_sha256(body) != claimed:
        raise RuntimeError(f"{path}: receipt seal differs")
    return receipt


def verify_artifacts(root: Path, manifest: dict) -> None:
    expected = manifest.get("files")
    if not isinstance(expected, dict) or not expected:
        raise RuntimeError("artifact manifest has no files")
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"artifact tree contains symlink: {path}")
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    if actual != set(expected):
        raise RuntimeError(f"artifact census differs: missing={set(expected)-actual}, extra={actual-set(expected)}")
    for name, metadata in expected.items():
        pure = PurePosixPath(name)
        if pure.is_absolute() or ".." in pure.parts:
            raise RuntimeError(f"unsafe artifact path: {name}")
        path = root / pure
        if path.stat().st_size != int(metadata["bytes"]):
            raise RuntimeError(f"artifact size differs: {name}")
        if sha256_file(path) != metadata["sha256"]:
            raise RuntimeError(f"artifact SHA-256 differs: {name}")


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".new-{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Authorize a public flip only after artifact, KLD, and Gilded Gnosis closure")
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--artifact-manifest", type=Path, required=True)
    parser.add_argument("--kld-receipt", type=Path, required=True)
    parser.add_argument("--runtime-receipt", type=Path, required=True)
    parser.add_argument("--hf-repo", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    artifact = load_sealed(args.artifact_manifest, "glm53-publish-artifact-manifest/1")
    kld = load_sealed(args.kld_receipt, "glm53-kld-gate/1")
    runtime = load_sealed(args.runtime_receipt, "glm53-runtime-gate/1")
    source_revision = artifact.get("source_revision")
    if not isinstance(source_revision, str) or len(source_revision) != 40:
        raise RuntimeError("artifact source revision is not a full SHA")
    if artifact.get("complete") is not True:
        raise RuntimeError("artifact manifest is not complete")
    verify_artifacts(args.artifact_root.resolve(), artifact)
    artifact_sha = artifact["receipt_sha256"]
    for label, receipt in (("KLD", kld), ("runtime", runtime)):
        if receipt.get("source_revision") != source_revision:
            raise RuntimeError(f"{label} source revision differs")
        if receipt.get("artifact_manifest_sha256") != artifact_sha:
            raise RuntimeError(f"{label} artifact binding differs")
        if receipt.get("passed") is not True:
            raise RuntimeError(f"{label} gate did not pass")
    metrics, thresholds = kld.get("metrics", {}), kld.get("thresholds", {})
    sample_count = int(metrics.get("sample_count") or 0)
    mean = float(metrics.get("bf16_kld_mean", float("inf")))
    p95 = float(metrics.get("bf16_kld_p95", float("inf")))
    if sample_count < int(thresholds.get("minimum_samples") or 0):
        raise RuntimeError("KLD sample floor not met")
    if mean > float(thresholds.get("maximum_bf16_kld_mean", float("-inf"))):
        raise RuntimeError("mean BF16 KLD threshold not met")
    if p95 > float(thresholds.get("maximum_bf16_kld_p95", float("-inf"))):
        raise RuntimeError("p95 BF16 KLD threshold not met")
    if runtime.get("engine") != "vllm-gilded-gnosis":
        raise RuntimeError("runtime gate did not use vLLM Gilded Gnosis")
    if int(runtime.get("cold_runs") or 0) < 5:
        raise RuntimeError("runtime gate has fewer than five cold runs")
    body = {
        "schema": "glm53-publication-authorization/1",
        "hf_repo": args.hf_repo,
        "source_revision": source_revision,
        "profile": artifact.get("profile"),
        "artifact_manifest_sha256": artifact_sha,
        "kld_receipt_sha256": kld["receipt_sha256"],
        "runtime_receipt_sha256": runtime["receipt_sha256"],
        "authorized_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "public_flip_authorized": True,
    }
    receipt = {**body, "receipt_sha256": canonical_sha256(body)}
    atomic_json(args.out, receipt)
    print(json.dumps({"authorized": True, "receipt_sha256": receipt["receipt_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
