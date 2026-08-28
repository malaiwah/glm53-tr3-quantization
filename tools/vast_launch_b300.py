#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from release_gate import release_revision, topology_gate
from vast_b300_preflight import canonical_sha256, public_row, query

DEFAULT_IMAGE = "voipmonitor/vllm@sha256:820181fbbc975cd5291c411cda9771d58fecee1636d916f508f47230df20592b"


def atomic_json(path: Path, value: dict, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".new-{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.chmod(tmp, mode)
    tmp.replace(path)


def funding(vastai: str) -> dict:
    result = subprocess.run(
        [vastai, "show", "user", "--raw"], check=True, capture_output=True, text=True
    )
    row = json.loads(result.stdout)
    return {
        "available": float(row.get("balance") or 0) + float(row.get("credit") or 0),
        "has_billing": bool(row.get("has_billing")),
        "can_pay": bool(row.get("can_pay")),
    }




def main() -> int:
    parser = argparse.ArgumentParser(description="Budget- and release-gated Vast 8xB300 launcher")
    parser.add_argument("--watch-state", type=Path, required=True)
    parser.add_argument("--topology-receipt", type=Path)
    parser.add_argument("--onstart", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--private-state", type=Path, required=True)
    parser.add_argument("--vastai", default="vastai")
    parser.add_argument("--storage-gb", type=int, default=2000)
    parser.add_argument("--gpu-bid-usd", type=float, default=50.0)
    parser.add_argument("--max-total-hourly-usd", type=float, default=52.0)
    parser.add_argument("--minimum-funding-usd", type=float, default=420.0)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--allow-unreleased-dry-run", action="store_true")
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    revision = release_revision(args.watch_state)
    if revision is None and (args.launch or not args.allow_unreleased_dry_run):
        raise SystemExit("GLM-5.3 release revision is not sealed in watcher state")
    topology_sha256 = None
    if args.topology_receipt and args.topology_receipt.is_file():
        topology_sha256 = topology_gate(args.topology_receipt)
    elif args.launch:
        raise SystemExit("sealed GLM-5.3 topology receipt is required for launch")
    account = funding(args.vastai)
    offers = query(args.vastai, "bid", args.storage_gb)
    eligible = []
    for row in offers:
        storage_hourly = float(row.get("storage_total_cost") or 0)
        total_hourly = args.gpu_bid_usd + storage_hourly
        if (
            row.get("rentable") is True
            and row.get("verification") == "verified"
            and int(row.get("num_gpus") or 0) == 8
            and int(row.get("gpu_total_ram") or 0) >= 2_000_000
            and float(row.get("min_bid") or float("inf")) <= args.gpu_bid_usd
            and total_hourly <= args.max_total_hourly_usd
        ):
            eligible.append((row, total_hourly))
    if not eligible:
        raise SystemExit("no verified rentable 8xB300 offer within bid/hourly guards")
    eligible.sort(key=lambda pair: (-float(pair[0].get("disk_bw") or 0), pair[1]))
    selected, total_hourly = eligible[0]
    public = public_row(selected, "bid")
    public["campaign_gpu_bid_usd"] = args.gpu_bid_usd
    public["campaign_total_hourly_usd"] = total_hourly
    plan = {
        "schema": "glm53-vast-b300-launch-plan/1",
        "planned_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "release_revision": revision,
        "topology_receipt_sha256": topology_sha256,
        "selected": public,
        "image": args.image,
        "storage_gb": args.storage_gb,
        "funding_sufficient": account["available"] >= args.minimum_funding_usd,
        "minimum_funding_usd": args.minimum_funding_usd,
        "launched": False,
    }
    if not args.launch:
        receipt = {**plan, "receipt_sha256": canonical_sha256(plan)}
        atomic_json(args.receipt, receipt)
        print(json.dumps({
            "eligible": len(eligible),
            "funding_sufficient": plan["funding_sufficient"],
            "release_revision": revision,
            "selected_offer_ref_sha256": public["offer_ref_sha256"],
            "total_hourly_usd": total_hourly,
        }, sort_keys=True))
        return 0
    if not plan["funding_sufficient"]:
        raise SystemExit(
            f"Vast available credit ${account['available']:.2f} is below "
            f"${args.minimum_funding_usd:.2f} funding gate"
        )
    command = [
        args.vastai,
        "create",
        "instance",
        str(selected["id"]),
        "--image",
        args.image,
        "--disk",
        str(args.storage_gb),
        "--bid_price",
        str(args.gpu_bid_usd),
        "--ssh",
        "--direct",
        "--cancel-unavail",
        "--label",
        "glm53-b300-capture",
        "--onstart",
        str(args.onstart.resolve()),
        "--raw",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    response = json.loads(result.stdout)
    contract = response.get("new_contract")
    if not response.get("success") or contract is None:
        raise RuntimeError(f"Vast create returned no contract: {response}")
    plan["launched"] = True
    plan["contract_ref_sha256"] = hashlib.sha256(str(contract).encode()).hexdigest()
    plan["launched_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    receipt = {**plan, "receipt_sha256": canonical_sha256(plan)}
    atomic_json(args.receipt, receipt)
    atomic_json(args.private_state, {
        "schema": "glm53-vast-b300-instance-private/1",
        "contract_id": contract,
        "offer_id": selected["id"],
        "release_revision": revision,
        "total_hourly_usd": total_hourly,
    }, mode=0o600)
    print(json.dumps({
        "launched": True,
        "contract_ref_sha256": plan["contract_ref_sha256"],
        "receipt_sha256": receipt["receipt_sha256"],
        "total_hourly_usd": total_hourly,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
