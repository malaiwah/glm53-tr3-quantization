#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from jarvislabs import Client


def main() -> int:
    parser = argparse.ArgumentParser(description="Pause one campaign Jarvis instance at a hard wall-clock budget")
    parser.add_argument("--machine-id", type=int, required=True)
    parser.add_argument("--max-seconds", type=float, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=60)
    args = parser.parse_args()
    if not os.environ.get("JL_API_KEY"):
        raise SystemExit("JL_API_KEY is required")
    started = time.time()
    client = Client()
    reason = "runtime-cap"
    try:
        while True:
            instance = client.instances.get(args.machine_id)
            if instance.status != "Running":
                reason = f"instance-{instance.status.lower()}"
                break
            elapsed = time.time() - started
            print(json.dumps({
                "elapsed_seconds": round(elapsed, 1),
                "machine_id": args.machine_id,
                "status": instance.status,
            }), flush=True)
            if elapsed >= args.max_seconds:
                client.instances.pause(args.machine_id)
                reason = "runtime-cap-paused"
                break
            time.sleep(min(args.poll_seconds, max(1, args.max_seconds - elapsed)))
    finally:
        client.close()
    body = {
        "schema": "glm53-jarvis-budget-watch/1",
        "machine_id": args.machine_id,
        "max_seconds": args.max_seconds,
        "elapsed_seconds": time.time() - started,
        "reason": reason,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    print(json.dumps(body, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
