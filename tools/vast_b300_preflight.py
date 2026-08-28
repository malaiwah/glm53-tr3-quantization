#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

QUERY = "gpu_name=B300 num_gpus>=8 gpu_total_ram>=2000 disk_space>=2000 cuda_vers>=12.8"


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".new-{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def query(vastai: str, price_type: str, storage_gb: int) -> list[dict]:
    command = [
        vastai,
        "search",
        "offers",
        QUERY,
        "--raw",
        "--limit",
        "50",
        "--storage",
        str(storage_gb),
        "--type",
        price_type,
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def public_row(row: dict, price_type: str) -> dict:
    identity = f"{row.get('id')}:{row.get('machine_id')}:{row.get('host_id')}"
    return {
        "offer_ref_sha256": hashlib.sha256(identity.encode()).hexdigest(),
        "price_type": price_type,
        "total_hourly_usd": row.get("dph_total"),
        "minimum_gpu_bid_usd": row.get("min_bid"),
        "num_gpus": row.get("num_gpus"),
        "gpu_name": row.get("gpu_name"),
        "gpu_ram_mb": row.get("gpu_ram"),
        "gpu_total_ram_mb": row.get("gpu_total_ram"),
        "nvlink_bandwidth": row.get("bw_nvlink"),
        "cuda_max": row.get("cuda_max_good"),
        "driver_version": row.get("driver_version"),
        "cpu_cores": row.get("cpu_cores"),
        "cpu_ram_mb": row.get("cpu_ram"),
        "disk_space_gb": row.get("disk_space"),
        "disk_bandwidth_mbps": row.get("disk_bw"),
        "download_mbps": row.get("inet_down"),
        "upload_mbps": row.get("inet_up"),
        "reliability": row.get("reliability"),
        "verification": row.get("verification"),
        "country": str(row.get("geolocation", "")).split(",")[-1].strip(),
        "rentable": row.get("rentable"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Query Vast for verified rentable 8xB300 capture hosts")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--private-state", type=Path)
    parser.add_argument("--vastai", default="vastai")
    parser.add_argument("--storage-gb", type=int, default=2000)
    parser.add_argument("--max-bid-total-usd", type=float, default=52.0)
    args = parser.parse_args()
    bid = query(args.vastai, "bid", args.storage_gb)
    on_demand = query(args.vastai, "on-demand", args.storage_gb)
    rows = [public_row(row, "bid") for row in bid] + [
        public_row(row, "on-demand") for row in on_demand
    ]
    rows.sort(key=lambda row: (row["price_type"], row["total_hourly_usd"], -row["disk_bandwidth_mbps"]))
    deployable = any(
        row["price_type"] == "bid"
        and row["rentable"]
        and row["verification"] == "verified"
        and float(row["total_hourly_usd"]) <= args.max_bid_total_usd
        for row in rows
    )
    body = {
        "schema": "glm53-vast-b300-preflight/1",
        "checked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "query": QUERY,
        "storage_gb": args.storage_gb,
        "max_bid_total_usd": args.max_bid_total_usd,
        "offers": rows,
        "deployable_bid_within_cap": deployable,
    }
    receipt = {**body, "receipt_sha256": canonical_sha256(body)}
    atomic_json(args.out, receipt)
    if args.private_state:
        private = {
            "schema": "glm53-vast-b300-private-state/1",
            "checked_utc": body["checked_utc"],
            "bid": bid,
            "on_demand": on_demand,
        }
        atomic_json(args.private_state, private)
        os.chmod(args.private_state, 0o600)
    print(json.dumps({
        "bid_offers": len(bid),
        "on_demand_offers": len(on_demand),
        "deployable_bid_within_cap": deployable,
        "receipt_sha256": receipt["receipt_sha256"],
    }, sort_keys=True))
    return 0 if deployable else 3


if __name__ == "__main__":
    raise SystemExit(main())
