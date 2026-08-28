#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from release_gate import release_revision, topology_gate
from provider_funding import runpod_balance
from runpod_b300_preflight import canonical_sha256

API = "https://api.runpod.io"
GPU_ID = "NVIDIA B300 SXM6 AC"
TEMPLATE_ID = "runpod-torch-v280"


def request_json(token: str, method: str, path: str, body: dict | None = None) -> dict:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        API + path,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "glm53-campaign-launcher/1.0",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read()
            return {} if not payload else json.loads(payload)
    except urllib.error.HTTPError as exc:
        detail = exc.read(4096).decode("utf-8", "replace").strip()
        raise RuntimeError(f"RunPod HTTP {exc.code}: {detail}") from None


def atomic_json(path: Path, value: dict, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".new-{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.chmod(tmp, mode)
    tmp.replace(path)




def main() -> int:
    parser = argparse.ArgumentParser(description="Capacity-, budget-, SSH-, and release-gated RunPod 8xB300 launcher")
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--watch-state", type=Path, required=True)
    parser.add_argument("--topology-receipt", type=Path)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--private-state", type=Path, required=True)
    parser.add_argument("--cloud", choices=["SECURE", "COMMUNITY"], default="SECURE")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--storage-gb", type=int, default=2000)
    parser.add_argument("--max-hourly-usd", type=float, default=64.0)
    parser.add_argument("--minimum-funding-usd", type=float, default=64.0)
    parser.add_argument("--allow-unreleased-dry-run", action="store_true")
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    token = args.token_file.read_text().strip()
    revision = release_revision(args.watch_state)
    if revision is None and (args.launch or not args.allow_unreleased_dry_run):
        raise SystemExit("GLM-5.3 release revision is not sealed in watcher state")
    topology_sha256 = None
    if args.topology_receipt and args.topology_receipt.is_file():
        topology_sha256 = topology_gate(args.topology_receipt)
    elif args.launch:
        raise SystemExit("sealed GLM-5.3 topology receipt is required for launch")
    query = urllib.parse.urlencode({
        "include": "AVAILABILITY",
        "product": "POD",
        "count": args.count,
        "cloud": args.cloud,
        "minCudaVersion": "13.0",
    })
    catalog = request_json(token, "GET", f"/v2/catalog/gpus?{query}")
    candidate = next((row for row in catalog.get("gpus", []) if row.get("id") == GPU_ID), None)
    price = None if candidate is None else candidate.get("price", {}).get(args.cloud.lower())
    hourly = None if price is None else float(price) * args.count
    data_centers = [] if candidate is None else [
        row["id"] for row in candidate.get("dataCenters", [])
        if row.get("availability") not in (None, "NONE")
    ]
    capacity = bool(
        candidate
        and candidate.get("availability") not in (None, "NONE")
        and int(candidate.get("maxCount", {}).get(args.cloud.lower()) or 0) >= args.count
        and data_centers
        and hourly is not None
        and hourly <= args.max_hourly_usd
    )
    keys = request_json(token, "GET", "/v2/account/ssh-keys").get("keys", [])
    account = runpod_balance(args.token_file)
    balance = float(account.get("clientBalance") or 0)
    spend_limit = float(account.get("spendLimit") or 0)
    body = {
        "name": "glm53-b300-capture",
        "templateId": TEMPLATE_ID,
        "gpu": {"id": GPU_ID, "count": args.count, "minCudaVersion": "13.0"},
        "cloud": args.cloud,
        "dataCenterIds": data_centers,
        "disk": 100,
        "mounts": {"persistent": {"size": args.storage_gb, "path": "/workspace"}},
        "ports": ["22/tcp"],
        "startSsh": True,
        "startJupyter": False,
    }
    plan = {
        "schema": "glm53-runpod-b300-launch-plan/1",
        "planned_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "release_revision": revision,
        "topology_receipt_sha256": topology_sha256,
        "cloud": args.cloud,
        "gpu_id": GPU_ID,
        "gpu_count": args.count,
        "memory_gb_per_gpu": None if candidate is None else candidate.get("memory"),
        "hourly_usd": hourly,
        "storage_gb": args.storage_gb,
        "capacity": capacity,
        "data_centers": data_centers,
        "ssh_key_count": len(keys),
        "funding_sufficient": balance >= args.minimum_funding_usd,
        "spend_limit_sufficient": spend_limit >= args.max_hourly_usd,
        "minimum_funding_usd": args.minimum_funding_usd,
        "max_hourly_usd": args.max_hourly_usd,
        "launched": False,
    }
    if not args.launch:
        receipt = {**plan, "receipt_sha256": canonical_sha256(plan)}
        atomic_json(args.receipt, receipt)
        print(json.dumps({key: receipt[key] for key in (
            "capacity", "funding_sufficient", "spend_limit_sufficient", "hourly_usd", "ssh_key_count", "receipt_sha256"
        )}, sort_keys=True))
        return 0 if capacity else 3
    if not capacity:
        raise SystemExit("RunPod has no count-8 B300 capacity within the hourly cap")
    if not keys:
        raise SystemExit("RunPod account has no registered SSH key")
    if not plan["funding_sufficient"]:
        raise SystemExit("RunPod balance is below one guarded B300 hour")
    if not plan["spend_limit_sufficient"]:
        raise SystemExit("RunPod account hourly spend limit is below campaign cap")
    pod = request_json(token, "POST", "/v2/pods", body)
    pod_id = pod.get("id")
    if not pod_id:
        raise RuntimeError(f"RunPod create returned no pod id: {pod}")
    plan["launched"] = True
    plan["pod_ref_sha256"] = hashlib.sha256(pod_id.encode()).hexdigest()
    plan["launched_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    receipt = {**plan, "receipt_sha256": canonical_sha256(plan)}
    atomic_json(args.receipt, receipt)
    atomic_json(args.private_state, {
        "schema": "glm53-runpod-b300-instance-private/1",
        "pod_id": pod_id,
        "release_revision": revision,
        "hourly_usd": hourly,
    }, mode=0o600)
    print(json.dumps({
        "launched": True,
        "pod_ref_sha256": plan["pod_ref_sha256"],
        "receipt_sha256": receipt["receipt_sha256"],
        "hourly_usd": hourly,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
