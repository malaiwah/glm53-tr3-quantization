#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path


def runpod_balance(token_file: Path) -> dict:
    token = token_file.read_text().strip()
    query = "query { myself { clientBalance currentSpendPerHr spendLimit underBalance minBalance } }"
    url = "https://api.runpod.io/graphql?" + urllib.parse.urlencode({"api_key": token})
    request = urllib.request.Request(
        url,
        data=json.dumps({"query": query}).encode(),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "glm53-campaign-funding/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.load(response)
    if payload.get("errors"):
        raise RuntimeError(f"RunPod GraphQL error: {payload['errors']}")
    return payload["data"]["myself"]


def vast_balance(vastai: str) -> dict:
    result = subprocess.run(
        [vastai, "show", "user", "--raw"], check=True, capture_output=True, text=True
    )
    payload = json.loads(result.stdout)
    return {
        "balance": payload.get("balance"),
        "credit": payload.get("credit"),
        "has_billing": payload.get("has_billing"),
        "can_pay": payload.get("can_pay"),
        "autobill_threshold": payload.get("autobill_threshold"),
        "autobill_amount": payload.get("autobill_amount"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Print secret-free RunPod/Vast funding readiness")
    parser.add_argument("--runpod-token-file", type=Path, required=True)
    parser.add_argument("--vastai", default="vastai")
    args = parser.parse_args()
    print(json.dumps({
        "runpod": runpod_balance(args.runpod_token_file),
        "vast": vast_balance(args.vastai),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
