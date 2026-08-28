#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import urllib.request
from pathlib import Path


def runpod(token_file: Path, public_key: str) -> tuple[int, bool]:
    token = token_file.read_text().strip()
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": "glm53-campaign-key-registration/1.0",
    }
    request = urllib.request.Request("https://api.runpod.io/v2/account/ssh-keys", headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        current = json.load(response).get("keys", [])
    if public_key in current:
        return len(current), False
    updated = current + [public_key]
    request = urllib.request.Request(
        "https://api.runpod.io/v2/account/ssh-keys",
        data=json.dumps({"keys": updated}).encode(),
        method="PUT",
        headers=headers,
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        result = json.load(response).get("keys", [])
    if sorted(result) != sorted(updated):
        raise RuntimeError("RunPod returned a different SSH key set")
    return len(result), True


def vast(vastai: str, public_key: str) -> tuple[int, bool]:
    listed = subprocess.run(
        [vastai, "show", "ssh-keys", "--raw"], check=True, capture_output=True, text=True
    )
    current = json.loads(listed.stdout)
    if any(row.get("public_key") == public_key for row in current):
        return len(current), False
    subprocess.run(
        [vastai, "create", "ssh-key", public_key, "--yes", "--raw"],
        check=True,
        capture_output=True,
        text=True,
    )
    listed = subprocess.run(
        [vastai, "show", "ssh-keys", "--raw"], check=True, capture_output=True, text=True
    )
    result = json.loads(listed.stdout)
    if not any(row.get("public_key") == public_key for row in result):
        raise RuntimeError("Vast did not retain the campaign SSH key")
    return len(result), True


def main() -> int:
    parser = argparse.ArgumentParser(description="Idempotently register one campaign SSH public key")
    parser.add_argument("--public-key", type=Path, required=True)
    parser.add_argument("--runpod-token-file", type=Path, required=True)
    parser.add_argument("--vastai", default="vastai")
    args = parser.parse_args()
    public_key = args.public_key.read_text().strip()
    if not public_key.startswith(("ssh-", "ecdsa-", "sk-")):
        raise SystemExit("public key is not OpenSSH authorized_keys format")
    runpod_count, runpod_added = runpod(args.runpod_token_file, public_key)
    vast_count, vast_added = vast(args.vastai, public_key)
    print(json.dumps({
        "runpod": {"key_count": runpod_count, "added": runpod_added},
        "vast": {"key_count": vast_count, "added": vast_added},
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
