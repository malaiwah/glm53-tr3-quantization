#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

from capture_controller import verify_window
from release_gate import canonical_sha256

WINDOWS = ("3-10", "11-18", "19-26", "27-34", "35-42", "43-50", "51-58", "59-66", "67-74", "75-77")


def main() -> int:
    parser = argparse.ArgumentParser(description="Pull sealed H200 capture windows from IN2 into the IN1 quant filesystem")
    parser.add_argument("--alias", required=True)
    parser.add_argument("--remote-root", default="/home/jl_fs/capture-export")
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--poll-seconds", type=float, default=20)
    parser.add_argument("--timeout-hours", type=float, default=8)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    deadline = time.time() + args.timeout_hours * 3600
    completed = []
    for window in WINDOWS:
        final = args.local_root / args.revision / window
        if (final / "READY.json").is_file():
            ready = verify_window(final)
        else:
            remote = f"{args.remote_root}/{args.revision}/{window}"
            while time.time() < deadline:
                status = subprocess.run(
                    ["ssh", args.alias, f"test -s {remote}/READY.json"],
                    check=False,
                )
                if status.returncode == 0:
                    break
                time.sleep(args.poll_seconds)
            else:
                raise SystemExit(f"capture bridge timed out waiting for {window}")
            incoming = final.with_name(f"{window}.new")
            if incoming.exists():
                shutil.rmtree(incoming)
            incoming.mkdir(parents=True)
            subprocess.run([
                "rsync", "-a", "--partial", "--info=progress2",
                f"{args.alias}:{remote}/", str(incoming) + "/",
            ], check=True)
            ready = verify_window(incoming)
            final.parent.mkdir(parents=True, exist_ok=True)
            incoming.replace(final)
        completed.append({"window": window, "receipt_sha256": ready["receipt_sha256"]})
        print(json.dumps(completed[-1]), flush=True)
    body = {
        "schema": "glm53-jarvis-capture-bridge/1",
        "revision": args.revision,
        "windows": completed,
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    body["receipt_sha256"] = canonical_sha256(body)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
