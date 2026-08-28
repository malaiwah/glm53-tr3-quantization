#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def api_model(repo: str, token: str) -> tuple[int, dict | None]:
    request = urllib.request.Request(
        f"https://huggingface.co/api/models/{repo}",
        headers={"Authorization": f"Bearer {token}", "User-Agent": "glm53-release-watcher/1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        return exc.code, None


def notify(message: str, title: str, priority: str = "default") -> None:
    url, token = os.environ.get("NTFY_URL"), os.environ.get("NTFY_TOKEN")
    if not url or not token:
        return
    request = urllib.request.Request(
        url,
        data=message.encode(),
        method="POST",
        headers={"Authorization": f"Bearer {token}", "Title": title, "Priority": priority},
    )
    with urllib.request.urlopen(request, timeout=15):
        pass


def released(payload: dict | None) -> bool:
    if not isinstance(payload, dict):
        return False
    siblings = payload.get("siblings")
    if not isinstance(siblings, list):
        return False
    names = {row.get("rfilename") for row in siblings if isinstance(row, dict)}
    return (
        isinstance(payload.get("sha"), str)
        and len(payload["sha"]) == 40
        and "config.json" in names
        and "model.safetensors.index.json" in names
        and any(isinstance(name, str) and name.endswith(".safetensors") for name in names)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="zai-org/GLM-5.3")
    parser.add_argument("--token-file", type=Path, default=Path.home() / ".hf_token")
    parser.add_argument("--state-dir", type=Path, default=Path("/home/jl_fs/receipts/release-watch"))
    parser.add_argument("--release-epoch", type=int, default=1787929200)  # 2026-08-28 15:00 UTC
    parser.add_argument("--far-interval", type=int, default=1800)
    parser.add_argument("--near-interval", type=int, default=60)
    parser.add_argument("--near-window", type=int, default=3600)
    parser.add_argument("--heartbeat", type=int, default=600)
    parser.add_argument("--on-release", type=Path, required=True)
    parser.add_argument("--hf", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("/home/jl_fs"))
    args = parser.parse_args()

    if args.token_file.is_symlink() or not args.token_file.is_file() or args.token_file.stat().st_mode & 0o777 != 0o600:
        raise SystemExit(f"{args.token_file} must be a regular mode-600 file")
    token = args.token_file.read_text().strip()
    if not token:
        raise SystemExit("Hugging Face token is empty")
    state_dir = args.state_dir.resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    last_heartbeat = 0.0
    polls = 0
    notify(f"Watching {args.repo} for open weights.", "GLM-5.3 release watcher started")

    while True:
        now = time.time()
        status, payload = api_model(args.repo, token)
        polls += 1
        remaining = args.release_epoch - now
        state = {
            "schema": "glm53-release-watch-state/1",
            "repo": args.repo,
            "http_status": status,
            "released": released(payload),
            "observed_sha": payload.get("sha") if isinstance(payload, dict) else None,
            "polls": polls,
            "remaining_seconds": remaining,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
            "checked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        }
        state["receipt_sha256"] = canonical_sha256(state)
        atomic_json(state_dir / "state.json", state)
        with (state_dir / "watch.log").open("a") as handle:
            handle.write(json.dumps(state, sort_keys=True) + "\n")

        if state["released"]:
            event = {
                "schema": "glm53-release-detected/1",
                "repo": args.repo,
                "revision": payload["sha"],
                "api_payload_sha256": canonical_sha256(payload),
                "detected_utc": state["checked_utc"],
                "polls": polls,
            }
            event["receipt_sha256"] = canonical_sha256(event)
            atomic_json(state_dir / "RELEASE_DETECTED.json", event)
            notify(f"{args.repo} @ {payload['sha']} is available. Authenticated download starting.",
                   "GLM-5.3 WEIGHTS RELEASED", "urgent")
            command = [
                str(args.on_release), "--repo", args.repo, "--revision", payload["sha"],
                "--token-file", str(args.token_file), "--hf", str(args.hf),
                "--root", str(args.root),
            ]
            return subprocess.run(command, check=False).returncode

        if now - last_heartbeat >= args.heartbeat:
            interval = args.near_interval if remaining <= args.near_window else args.far_interval
            notify(
                f"polls={polls}; HTTP={status}; remaining={max(0, int(remaining))}s; next={interval}s",
                "GLM-5.3 release watch heartbeat",
            )
            last_heartbeat = now
        interval = args.near_interval if remaining <= args.near_window else args.far_interval
        time.sleep(max(1, interval))


if __name__ == "__main__":
    raise SystemExit(main())
