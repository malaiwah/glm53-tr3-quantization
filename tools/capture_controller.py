#!/usr/bin/env python3
from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import shlex
import signal
import shutil
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse

from release_gate import canonical_sha256
from runpod_launch_b300 import request_json

WINDOWS = ["3-10", "11-18", "19-26", "27-34", "35-42", "43-50", "51-58", "59-66", "67-74", "75-77"]


def atomic_json(path: Path, value: dict, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".new-{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.chmod(tmp, mode)
    tmp.replace(path)


def endpoint_vast(vastai: str, private: dict) -> tuple[str, int, str]:
    contract = str(private["contract_id"])
    result = subprocess.run(
        [vastai, "ssh-url", contract], check=True, capture_output=True, text=True
    )
    parsed = urlparse(result.stdout.strip())
    if parsed.scheme != "ssh" or not parsed.hostname or not parsed.port:
        raise RuntimeError(f"invalid Vast SSH URL: {result.stdout.strip()!r}")
    return parsed.hostname, parsed.port, parsed.username or "root"


def endpoint_runpod(token: str, private: dict) -> tuple[str, int, str]:
    pod = request_json(token, "GET", f"/v2/pods/{private['pod_id']}")
    if pod.get("status") != "RUNNING":
        raise RuntimeError(f"RunPod pod status is {pod.get('status')}")
    ssh = pod.get("ssh") or {}
    endpoint = ssh.get("direct") or ssh.get("proxy")
    if not endpoint:
        raise RuntimeError("RunPod pod has no SSH endpoint")
    return endpoint["host"], int(endpoint["port"]), endpoint["username"]


def ssh_base(key: Path, host: str, port: int, user: str) -> list[str]:
    return [
        "ssh", "-i", str(key), "-p", str(port),
        "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=20",
        "-o", "ServerAliveCountMax=6", "-o", "ControlMaster=auto",
        "-o", "ControlPersist=600", "-o", "ControlPath=~/.ssh/cm-glm53-capture-%C",
        f"{user}@{host}",
    ]


def ssh_call(base: list[str], command: str, *, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(base + [command], check=check, capture_output=capture, text=True)


def rsync_shell(base: list[str]) -> str:
    return " ".join(shlex.quote(arg) for arg in base[:-1])


def copy_to_remote(base: list[str], sources: list[Path], destination: str) -> None:
    subprocess.run(
        ["rsync", "-a", "--chmod=F600", "-e", rsync_shell(base), *map(str, sources), f"{base[-1]}:{destination}"],
        check=True,
    )


def verify_window(root: Path) -> dict:
    ready_path = root / "READY.json"
    receipt = json.loads(ready_path.read_text())
    claimed = receipt.get("receipt_sha256")
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if not isinstance(claimed, str) or canonical_sha256(body) != claimed:
        raise RuntimeError(f"capture receipt seal differs: {root}")
    expected = receipt.get("files", {})
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*") if path.is_file() and path.name != "READY.json"
    }
    if actual != set(expected):
        raise RuntimeError(f"capture file census differs: missing={set(expected)-actual}, extra={actual-set(expected)}")
    for name, metadata in expected.items():
        path = root / name
        if path.stat().st_size != int(metadata["bytes"]):
            raise RuntimeError(f"capture size differs: {name}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 << 20), b""):
                digest.update(chunk)
        if digest.hexdigest() != metadata["sha256"]:
            raise RuntimeError(f"capture SHA-256 differs: {name}")
    return receipt


def terminate(provider: str, vastai: str, token: str | None, private: dict) -> None:
    if provider == "vast":
        subprocess.run(
            [vastai, "destroy", "instance", str(private["contract_id"]), "--yes", "--raw"],
            check=True,
        )
    else:
        request_json(token or "", "DELETE", f"/v2/pods/{private['pod_id']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap capture host, pull sealed windows, and terminate on closure")
    parser.add_argument("--provider", choices=["vast", "runpod"], required=True)
    parser.add_argument("--private-state", type=Path, required=True)
    parser.add_argument("--runpod-token-file", type=Path)
    parser.add_argument("--vastai", default="vastai")
    parser.add_argument("--ssh-key", type=Path, required=True)
    parser.add_argument("--adapter-archive", type=Path, required=True)
    parser.add_argument("--adapter-sha256", type=Path, required=True)
    parser.add_argument("--worker-script", type=Path, required=True)
    parser.add_argument("--hf-token-file", type=Path, required=True)
    parser.add_argument("--repo", default="zai-org/GLM-5.3")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--max-runtime-hours", type=float, default=8.0)
    args = parser.parse_args()
    if len(args.revision) != 40:
        raise SystemExit("revision must be a full commit SHA")
    private = json.loads(args.private_state.read_text())
    if private.get("release_revision") != args.revision:
        raise SystemExit("provider instance revision binding differs")
    token = args.runpod_token_file.read_text().strip() if args.runpod_token_file else None
    if args.provider == "runpod" and not token:
        raise SystemExit("--runpod-token-file is required for RunPod")
    cleanup_state = {"armed": True}

    def cleanup() -> None:
        if not cleanup_state["armed"]:
            return
        cleanup_state["armed"] = False
        try:
            terminate(args.provider, args.vastai, token, private)
        except Exception as exc:
            print(f"provider termination failed: {exc}", flush=True)

    def stop_from_signal(signum, _frame) -> None:
        raise SystemExit(f"capture controller received signal {signum}")

    atexit.register(cleanup)
    signal.signal(signal.SIGTERM, stop_from_signal)
    signal.signal(signal.SIGINT, stop_from_signal)
    deadline = time.time() + args.max_runtime_hours * 3600
    endpoint = None
    while time.time() < deadline and endpoint is None:
        try:
            endpoint = endpoint_vast(args.vastai, private) if args.provider == "vast" else endpoint_runpod(token or "", private)
        except Exception as exc:
            print(f"endpoint not ready: {exc}", flush=True)
            time.sleep(15)
    if endpoint is None:
        cleanup()
        raise SystemExit("capture host did not expose SSH before runtime deadline")
    host, port, user = endpoint
    base = ssh_base(args.ssh_key, host, port, user)
    while time.time() < deadline:
        result = ssh_call(base, "echo SSH_OK", check=False, capture=True)
        if result.returncode == 0 and result.stdout.strip() == "SSH_OK":
            break
        time.sleep(10)
    else:
        cleanup()
        raise SystemExit("capture host SSH did not become ready")
    ssh_call(base, "mkdir -p /workspace/glm53 /root && chmod 700 /root")
    copy_to_remote(base, [args.adapter_archive, args.adapter_sha256], "/workspace/glm53/")
    copy_to_remote(base, [args.worker_script], "/root/capture_worker.sh")
    copy_to_remote(base, [args.hf_token_file], "/root/.hf_token")
    ssh_call(base, "chmod 700 /root/capture_worker.sh; chmod 600 /root/.hf_token")
    if args.provider == "vast":
        ready_deadline = min(deadline, time.time() + 1200)
        while time.time() < ready_deadline:
            if ssh_call(base, "test -s /workspace/glm53/receipts/prepared-utc.txt", check=False).returncode == 0:
                break
            time.sleep(10)
    command = " ".join(map(shlex.quote, ["/root/capture_worker.sh", args.repo, args.revision]))
    tmux_command = f"tmux has-session -t glm53-capture 2>/dev/null || tmux new-session -d -s glm53-capture {shlex.quote(command)}"
    ssh_call(base, tmux_command)
    destination = args.capture_root / args.revision
    destination.mkdir(parents=True, exist_ok=True)
    windows = []
    total_bytes = 0
    for window in WINDOWS:
        final = destination / window
        if final.is_dir() and (final / "READY.json").is_file():
            window_receipt = verify_window(final)
        else:
            while time.time() < deadline:
                check = ssh_call(base, f"test -s /workspace/glm53/capture-export/{window}/READY.json", check=False)
                if check.returncode == 0:
                    break
                alive = ssh_call(base, "tmux has-session -t glm53-capture", check=False)
                if alive.returncode != 0:
                    log = ssh_call(base, "cat /workspace/glm53/logs/capture-worker.log", check=False, capture=True)
                    raise RuntimeError(f"capture worker exited before {window}: {log.stdout[-8000:]}")
                time.sleep(20)
            else:
                cleanup()
                raise SystemExit("capture runtime cap reached; instance terminated")
            incoming = destination / f"{window}.new"
            if incoming.exists():
                shutil.rmtree(incoming)
            incoming.mkdir(parents=True)
            subprocess.run([
                "rsync", "-a", "--partial", "--info=progress2", "-e", rsync_shell(base),
                f"{base[-1]}:/workspace/glm53/capture-export/{window}/", str(incoming) + "/",
            ], check=True)
            window_receipt = verify_window(incoming)
            incoming.replace(final)
        local_sha = window_receipt["receipt_sha256"]
        ssh_call(base, f"mkdir -p /workspace/glm53/capture-acks; printf '%s\\n' {shlex.quote(local_sha)} > /workspace/glm53/capture-acks/{window}.ack")
        window_bytes = sum(int(row["bytes"]) for row in window_receipt["files"].values())
        total_bytes += window_bytes
        windows.append({"window": window, "bytes": window_bytes, "receipt_sha256": local_sha})
        print(f"CAPTURE_WINDOW_LOCAL {window} bytes={window_bytes}", flush=True)
    cleanup()
    body = {
        "schema": "glm53-capture-controller/1",
        "provider": args.provider,
        "repo": args.repo,
        "revision": args.revision,
        "windows": windows,
        "total_bytes": total_bytes,
        "provider_terminated": True,
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    receipt = {**body, "receipt_sha256": canonical_sha256(body)}
    atomic_json(args.receipt, receipt)
    print(json.dumps({"complete": True, "total_bytes": total_bytes, "receipt_sha256": receipt["receipt_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
