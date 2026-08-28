#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.parse
import urllib.error
import urllib.request
from pathlib import Path

API = "https://api.runpod.io/v2/catalog/gpus"


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".new-{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Query authenticated RunPod v2 inventory for an eight-GPU B300 capture host")
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cloud", choices=["SECURE", "COMMUNITY"], default="SECURE")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--max-hourly-usd", type=float, default=64.0)
    args = parser.parse_args()
    token = args.token_file.read_text().strip()
    if not token:
        raise SystemExit("empty RunPod token")
    query = urllib.parse.urlencode({
        "include": "AVAILABILITY",
        "product": "POD",
        "count": args.count,
        "cloud": args.cloud,
        "minCudaVersion": "12.8",
    })
    request = urllib.request.Request(
        f"{API}?{query}",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "User-Agent": "glm53-campaign-preflight/1.0",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read(2048).decode("utf-8", "replace").strip()
        raise SystemExit(f"RunPod catalog HTTP {exc.code}: {detail}") from None
    candidates = []
    for row in payload.get("gpus", []):
        identity = f"{row.get('id', '')} {row.get('name', '')}".upper()
        if "B300" not in identity and "GB300" not in identity:
            continue
        price = row.get("price", {}).get(args.cloud.lower())
        hourly = None if price is None else float(price) * args.count
        candidates.append({
            "id": row.get("id"),
            "name": row.get("name"),
            "memory_gb_per_gpu": row.get("memory"),
            "availability": row.get("availability"),
            "max_count": row.get("maxCount", {}).get(args.cloud.lower()),
            "price_per_gpu_hour_usd": price,
            "price_for_requested_count_usd": hourly,
            "within_hourly_cap": hourly is not None and hourly <= args.max_hourly_usd,
            "data_centers": row.get("dataCenters", []),
            "cuda_versions": row.get("cudaVersions", []),
        })
    body = {
        "schema": "glm53-runpod-b300-preflight/1",
        "checked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cloud": args.cloud,
        "requested_count": args.count,
        "max_hourly_usd": args.max_hourly_usd,
        "candidates": candidates,
        "deployable_within_cap": any(
            row["availability"] not in (None, "NONE")
            and (row["max_count"] or 0) >= args.count
            and row["within_hourly_cap"]
            for row in candidates
        ),
    }
    receipt = {**body, "receipt_sha256": canonical_sha256(body)}
    atomic_json(args.out, receipt)
    print(json.dumps({
        "candidate_count": len(candidates),
        "deployable_within_cap": receipt["deployable_within_cap"],
        "receipt_sha256": receipt["receipt_sha256"],
    }, sort_keys=True))
    return 0 if receipt["deployable_within_cap"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
