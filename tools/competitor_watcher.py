#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from watch_matrix import api_competitors, atomic_json, notify


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-file", type=Path, default=Path.home() / ".hf_token")
    parser.add_argument("--state-dir", type=Path, default=Path("/home/jl_fs/receipts/competitors"))
    parser.add_argument("--release-epoch", type=int, default=1787929200)
    parser.add_argument("--far-interval", type=int, default=1800)
    parser.add_argument("--near-interval", type=int, default=60)
    parser.add_argument("--near-window", type=int, default=3600)
    args = parser.parse_args()
    if args.token_file.is_symlink() or not args.token_file.is_file() or args.token_file.stat().st_mode & 0o777 != 0o600:
        raise SystemExit(f"{args.token_file} must be a regular mode-600 file")
    token = args.token_file.read_text().strip()
    state_dir = args.state_dir.resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    known: set[str] = set()
    while True:
        now = time.time()
        try:
            rows = api_competitors(token)
            for row in rows:
                if row["repo"] not in known:
                    known.add(row["repo"])
                    notify(
                        f"{row['repo']} @ {row.get('revision')} detected",
                        "GLM-5.3 FP4/NVFP4 competitor detected",
                        "high",
                    )
            state = {
                "schema": "glm53-competitor-watch/1",
                "competitors": rows,
                "checked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
            }
            state["receipt_sha256"] = canonical_sha256(state)
            atomic_json(state_dir / "state.json", state)
            with (state_dir / "watch.log").open("a") as handle:
                handle.write(json.dumps(state, sort_keys=True) + "\n")
        except Exception as exc:
            with (state_dir / "watch.log").open("a") as handle:
                handle.write(json.dumps({
                    "error": f"{type(exc).__name__}: {exc}",
                    "checked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
                }, sort_keys=True) + "\n")
        remaining = args.release_epoch - time.time()
        interval = args.near_interval if remaining <= args.near_window else args.far_interval
        time.sleep(max(1, interval))


if __name__ == "__main__":
    raise SystemExit(main())
