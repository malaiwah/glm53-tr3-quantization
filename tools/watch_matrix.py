#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import signal
import time
import urllib.error
import urllib.request
from pathlib import Path

TARGETS = {
    "zai-org/GLM-5.3": "models/glm53/source",
    "zai-org/GLM-5.3-FP8": "models/glm53/fp8",
    "zai-org/GLM-5.2": "models/smoke/zai-org--GLM-5.2",
    "zai-org/GLM-5.2-FP8": "models/smoke/zai-org--GLM-5.2-FP8",
}


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def api_model(repo: str, token: str) -> tuple[int, dict | None]:
    request = urllib.request.Request(
        f"https://huggingface.co/api/models/{repo}",
        headers={"Authorization": f"Bearer {token}", "User-Agent": "glm53-watch-matrix/1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        return exc.code, None


def is_released(payload: dict | None) -> bool:
    if not isinstance(payload, dict) or not isinstance(payload.get("siblings"), list):
        return False
    names = {row.get("rfilename") for row in payload["siblings"] if isinstance(row, dict)}
    return (
        isinstance(payload.get("sha"), str) and len(payload["sha"]) == 40
        and "config.json" in names and "model.safetensors.index.json" in names
        and any(isinstance(name, str) and name.endswith(".safetensors") for name in names)
    )


def notify(message: str, title: str, priority: str = "default") -> None:
    url, token = os.environ.get("NTFY_URL"), os.environ.get("NTFY_TOKEN")
    if not url or not token:
        return
    request = urllib.request.Request(
        url, data=message.encode(), method="POST",
        headers={"Authorization": f"Bearer {token}", "Title": title, "Priority": priority},
    )
    with urllib.request.urlopen(request, timeout=15):
        pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-file", type=Path, default=Path.home() / ".hf_token")
    parser.add_argument("--root", type=Path, default=Path("/home/jl_fs"))
    parser.add_argument("--hf", type=Path, required=True)
    parser.add_argument("--downloader", type=Path, required=True)
    parser.add_argument("--release-epoch", type=int, default=1787929200)
    parser.add_argument("--far-interval", type=int, default=1800)
    parser.add_argument("--near-interval", type=int, default=60)
    parser.add_argument("--near-window", type=int, default=3600)
    parser.add_argument("--heartbeat", type=int, default=600)
    parser.add_argument("--max-downloads", type=int, default=1)
    args = parser.parse_args()
    if args.token_file.is_symlink() or not args.token_file.is_file() or args.token_file.stat().st_mode & 0o777 != 0o600:
        raise SystemExit(f"{args.token_file} must be a regular mode-600 file")
    token = args.token_file.read_text().strip()
    root = args.root.resolve()
    state_root = root / "receipts/watch-matrix"
    state_root.mkdir(parents=True, exist_ok=True)
    next_poll = {repo: 0.0 for repo in TARGETS}
    first_revision: dict[str, str] = {}
    children: dict[str, subprocess.Popen] = {}
    child_logs = {}
    status = {repo: {"http": None, "released": False, "revision": None} for repo in TARGETS}
    download_retry_at = {repo: 0.0 for repo in TARGETS}
    last_heartbeat = 0.0
    notify("Watching GLM-5.2/5.2-FP8/5.3/5.3-FP8; released baselines will download as a full-path smoke.",
           "GLM release matrix started")

    while True:
        now = time.time()
        remaining = args.release_epoch - now
        interval = args.near_interval if remaining <= args.near_window else args.far_interval
        for repo, relative_dest in TARGETS.items():
            if now < next_poll[repo]:
                continue
            code, payload = api_model(repo, token)
            available = is_released(payload)
            revision = payload.get("sha") if available else None
            status[repo] = {"http": code, "released": available, "revision": revision}
            next_poll[repo] = now + interval
            if available and repo not in first_revision:
                first_revision[repo] = revision
                notify(f"{repo}@{revision} detected", f"{repo} available", "high")
            if (
                available
                and repo in {"zai-org/GLM-5.3", "zai-org/GLM-5.3-FP8"}
                and repo not in children
            ):
                for active in list(children):
                    if active not in {"zai-org/GLM-5.2", "zai-org/GLM-5.2-FP8"}:
                        continue
                    os.killpg(children[active].pid, signal.SIGTERM)
                    try:
                        children[active].wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        children[active].kill()
                        os.killpg(children[active].pid, signal.SIGKILL)
                    child_logs[active].close()
                    children.pop(active, None)
                    child_logs.pop(active, None)
                    download_retry_at[active] = now + 300
                    next_poll[active] = now + 300
                    notify(
                        f"Preempted {active} baseline download for {repo}",
                        "GLM-5.3 release download priority",
                        "high",
                    )
            if (
                available
                and repo not in children
                and now >= download_retry_at[repo]
                and len(children) < args.max_downloads
            ):
                slug = repo.replace("/", "--")
                complete = root / f"receipts/downloads/{slug}/SOURCE_DOWNLOAD_COMPLETE.json"
                if complete.is_file():
                    continue
                log_path = state_root / f"{slug}.launcher.log"
                log = log_path.open("a")
                command = [
                    str(args.downloader), "--repo", repo, "--revision", first_revision[repo],
                    "--token-file", str(args.token_file), "--hf", str(args.hf),
                    "--root", str(root), "--dest", str(root / relative_dest),
                ]
                children[repo] = subprocess.Popen(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=os.environ,
                    start_new_session=True,
                )
                child_logs[repo] = log

        finished = []
        for repo, child in children.items():
            rc = child.poll()
            if rc is not None:
                child_logs[repo].close()
                status[repo]["download_rc"] = rc
                notify(f"{repo} downloader exited rc={rc}", f"{repo} download {'complete' if rc == 0 else 'failed'}",
                       "high" if rc == 0 else "urgent")
                if rc != 0:
                    download_retry_at[repo] = time.time() + 300
                    next_poll[repo] = download_retry_at[repo]
                finished.append(repo)
        for repo in finished:
            children.pop(repo, None)
            child_logs.pop(repo, None)

        state = {
            "schema": "glm53-watch-matrix-state/1",
            "targets": status,
            "first_revision": first_revision,
            "active_downloads": sorted(children),
            "poll_interval_seconds": interval,
            "remaining_seconds": remaining,
            "checked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        }
        state["receipt_sha256"] = canonical_sha256(state)
        atomic_json(state_root / "state.json", state)
        with (state_root / "watch.log").open("a") as handle:
            handle.write(json.dumps(state, sort_keys=True) + "\n")
        if now - last_heartbeat >= args.heartbeat:
            summary = "; ".join(
                f"{repo.split('/')[-1]}={row['http']}/{row['revision'] or 'waiting'}"
                for repo, row in status.items()
            )
            notify(f"{summary}; active={sorted(children)}; next={interval}s", "GLM watch heartbeat")
            last_heartbeat = now
        time.sleep(30)


if __name__ == "__main__":
    raise SystemExit(main())
